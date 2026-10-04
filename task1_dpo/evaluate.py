from __future__ import annotations

import argparse
import json
import re

import numpy as np
import torch

from common.data import (
    load_yaml,
    prompt_messages,
    prompt_messages_from_preference,
    read_jsonl,
    write_jsonl,
)
from common.generation import batch_generate, response_sequence_logprobs, response_token_logprobs, score_reward_pairs
from common.logging_utils import save_json, set_seed
from common.metrics import (
    parse_word_limit,
    preference_accuracy,
    safe_corr,
    sampled_kl,
    word_count,
    word_limit_compliance,
)
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, reference_mode
from task1_dpo.dpo import dpo_loss
from task1_dpo.length_stats import find_stratum_key
from task1_dpo.train import make_collate, to_device


def settings(cfg, eval_batch_size=4):
    g = cfg["generation"]
    max_new = int(cfg["max_generation_tokens"])
    max_len = int(cfg["max_sequence_length"])
    return {
        "max_length": max_len,
        "max_new_tokens": max_new,
        "max_prompt_length": max_len - max_new,       # 768 - 256 = 512
        "temperature": float(g["temperature"]),
        "top_p": float(g["top_p"]),
        "do_sample": bool(g["do_sample"]),
        "eval_batch_size": eval_batch_size,
    }


def user_text(messages):
    return " ".join(m["content"] for m in messages if m.get("role") == "user")


def row_ids(rows):
    """Use a real ID field if present. Otherwise fall back to row index, and SAY SO."""
    ids, src = [], set()
    for i, r in enumerate(rows):
        for k in ("prompt_id", "id", "source_index"):
            if k in r:
                ids.append(str(r[k])); src.add(k); break
        else:
            ids.append(f"row{i}"); src.add("row_index")
    if "row_index" in src:
        print("WARNING: some rows have no prompt_id/id/source_index; using row index as the ID.")
    return ids, sorted(src)


# ------------------------------------------------------------ A / C: pair-based metrics (teacher-forced)
@torch.no_grad()
def pair_logps(model, tokenizer, rows, max_length, batch_size):
    device = next(model.parameters()).device
    collate = make_collate(tokenizer, max_length)
    model.eval()
    out = {"pc": [], "pr": [], "rc": [], "rr": []}
    for i in range(0, len(rows), batch_size):
        cb, rb = collate(rows[i:i + batch_size])
        cb, rb = to_device(cb, device), to_device(rb, device)
        pc, _, _ = response_sequence_logprobs(model, cb)
        pr, _, _ = response_sequence_logprobs(model, rb)
        with reference_mode(model):
            rc, _, _ = response_sequence_logprobs(model, cb)
            rr, _, _ = response_sequence_logprobs(model, rb)
        for k, v in zip(("pc", "pr", "rc", "rr"), (pc, pr, rc, rr)):
            out[k].append(v.float().cpu())
    return [torch.cat(out[k]) for k in ("pc", "pr", "rc", "rr")]


def heldout_pair_metrics(model, tokenizer, rows, s, beta):
    pc, pr, rc, rr = pair_logps(model, tokenizer, rows, s["max_length"], s["eval_batch_size"])
    loss, _ = dpo_loss(pc, pr, rc, rr, beta)
    lr_c, lr_r = pc - rc, pr - rr                    # log-ratios (policy - reference)
    margin = lr_c - lr_r
    summary = {
        "n_pairs": len(rows), "beta_for_loss": beta, "dpo_loss": float(loss),
        "pref_acc": preference_accuracy(lr_c, lr_r),  # fraction with margin > 0 (manual definition)
        "mean_margin": float(margin.mean()),
        "mean_logratio_chosen": float(lr_c.mean()), "mean_logratio_rejected": float(lr_r.mean()),
    }
    per_example = [{"idx": i, "margin": float(margin[i]), "logratio_chosen": float(lr_c[i]),
                    "logratio_rejected": float(lr_r[i])} for i in range(len(rows))]
    return summary, per_example, margin


def stratified_pair_metrics(model, tokenizer, rows, s, stratum_key):
    _, per_ex, margin = heldout_pair_metrics(model, tokenizer, rows, s, beta=1.0)  # sign of margin is beta-free
    groups = {}
    for i, row in enumerate(rows):
        groups.setdefault(str(row[stratum_key]), []).append(i)
    out = {"stratum_key": stratum_key}
    for name, ix in sorted(groups.items()):
        m = margin[torch.tensor(ix)]
        out[name] = {"n": len(ix), "pref_acc": float((m > 0).float().mean()), "mean_margin": float(m.mean())}
    out["overall"] = {"n": len(rows), "pref_acc": float((margin > 0).float().mean()),
                      "mean_margin": float(margin.mean())}
    return out, per_ex


# ------------------------------------------------------------ B: generation metrics
RM_MAX_LENGTH = 1024                       # default max_length of common.generation.score_reward_pairs (right-truncating)
ASSISTANT_MARKER = "<|im_start|>assistant\n"  # end of the Qwen chat template with add_generation_prompt=True
TRUNC_MARK = "\n[...]\n"                    # inserted where the middle of an over-long user message was removed
HEAD_FRAC = 0.5                            # fixed before looking at any result


def _n_ids(x):
    return len(x["input_ids"]) if hasattr(x, "keys") else len(x)


def _n_prompt_tokens(tokenizer, msgs):
    return _n_ids(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))


def truncate_prompt(tokenizer, messages, cap, head_frac=HEAD_FRAC):
    """Keep the START and END of the last user message, drop the middle, re-render the chat template around it.
    The system header and the assistant marker are therefore never cut.
    Returns (messages, method) with method in {"none", "head_tail", "left_fallback"}."""
    if _n_prompt_tokens(tokenizer, messages) <= cap:
        return messages, "none"
    ks = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if not ks:
        return messages, "left_fallback"
    k = ks[-1]
    ids = tokenizer(messages[k]["content"], add_special_tokens=False)["input_ids"]
    overhead = _n_prompt_tokens(tokenizer, messages) - len(ids)
    budget = cap - overhead - len(tokenizer(TRUNC_MARK, add_special_tokens=False)["input_ids"])
    while budget >= 32:
        h = int(budget * head_frac)
        t = budget - h
        head = tokenizer.decode(ids[:h]).rstrip("\ufffd")      # drop a half-decoded character at the seam
        tail = tokenizer.decode(ids[-t:]).lstrip("\ufffd")
        new = dict(messages[k]); new["content"] = head + TRUNC_MARK + tail
        out = list(messages); out[k] = new
        if _n_prompt_tokens(tokenizer, out) <= cap:
            return out, "head_tail"
        budget -= 8
    return messages, "left_fallback"      # e.g. a very long earlier turn: batch_generate then left-truncates


def prompt_trunc_info(tokenizer, rm_tok, orig, used, method, response, s):
    """What generation-time truncation did to this prompt, and whether the reward-model input was cut."""
    cap = s["max_prompt_length"]
    full = _n_prompt_tokens(tokenizer, orig)
    rendered = tokenizer.apply_chat_template(used, tokenize=False, add_generation_prompt=True)
    kept_ids = tokenizer(rendered, truncation=True, max_length=cap)["input_ids"]   # same call batch_generate makes
    rm_text = rm_tok.apply_chat_template(list(orig) + [{"role": "assistant", "content": response}],
                                         tokenize=False, add_generation_prompt=False)
    rm_len = len(rm_tok(rm_text)["input_ids"])
    return {
        "prompt_tokens_full": full,
        "prompt_tokens_used": len(kept_ids),
        "prompt_tokens_removed": max(0, full - len(kept_ids)),
        "prompt_truncated": full > cap,
        "prompt_truncation_method": method,
        "prompt_marker_intact": tokenizer.decode(kept_ids[-6:]).endswith(ASSISTANT_MARKER),
        "prompt_header_intact": tokenizer.decode(kept_ids[:3]).startswith("<|im_start|>"),
        "rm_input_tokens": rm_len,
        "rm_input_truncated": rm_len > RM_MAX_LENGTH,
    }


def summarize_prompt_trunc(records, s, tokenizer):
    full = np.array([r["prompt_tokens_full"] for r in records], dtype=float)
    removed = np.array([r["prompt_tokens_removed"] for r in records], dtype=float)
    over = np.array([r["prompt_truncated"] for r in records], dtype=bool)
    return {
        "rule": f"head+tail on last user message (head_frac={HEAD_FRAC}); left-truncation fallback",
        "fallback_truncation_side": tokenizer.truncation_side, "prompt_cap_tokens": s["max_prompt_length"],
        "n_prompts": len(records), "n_over_cap": int(over.sum()), "frac_over_cap": float(over.mean()),
        "methods": {m: int(sum(r["prompt_truncation_method"] == m for r in records))
                    for m in ("none", "head_tail", "left_fallback")},
        "prompt_tokens_full_p95": float(np.percentile(full, 95)), "prompt_tokens_full_max": int(full.max()),
        "tokens_removed_mean_over_truncated": float(removed[over].mean()) if over.any() else 0.0,
        "tokens_removed_max": int(removed.max()),
        "n_assistant_marker_intact": int(sum(r["prompt_marker_intact"] for r in records)),
        "n_system_header_intact": int(sum(r["prompt_header_intact"] for r in records)),
        "reward_input_cap_tokens": RM_MAX_LENGTH,
        "n_reward_input_over_cap": int(sum(r["rm_input_truncated"] for r in records)),
    }


def generate_eval(model, tokenizer, rm, rm_tok, prompts, ids, s):
    bs = s["eval_batch_size"]
    records, kl_num, kl_den = [], 0.0, 0.0
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        trunc = [truncate_prompt(tokenizer, m, s["max_prompt_length"]) for m in chunk]
        gen_chunk, methods = [t[0] for t in trunc], [t[1] for t in trunc]
        out = batch_generate(model, tokenizer, gen_chunk, s["max_prompt_length"], s["max_new_tokens"],
                             temperature=s["temperature"], top_p=s["top_p"], do_sample=s["do_sample"])
        with torch.no_grad():
            args = (out["sequences"], out["attention_mask"], out["prompt_width"], out["response_ids"])
            pol_lp, _ = response_token_logprobs(model, *args)
            with reference_mode(model):
                ref_lp, _ = response_token_logprobs(model, *args)
        mask = out["response_mask"]
        n_tok = float(mask.sum())
        kl_num += float(sampled_kl(pol_lp, ref_lp, mask)) * n_tok    # token-pooled mean
        kl_den += n_tok
        seq_kl = ((pol_lp - ref_lp) * mask).sum(-1).tolist()
        rewards = score_reward_pairs(rm, rm_tok, chunk, out["responses"]).tolist()
        for j in range(len(chunk)):
            pinfo = prompt_trunc_info(tokenizer, rm_tok, chunk[j], gen_chunk[j], methods[j], out["responses"][j], s)
            records.append({
                **pinfo,
                "prompt_id": ids[i + j], "prompt": user_text(chunk[j]), "response": out["responses"][j],
                "length_tokens": out["response_lengths"][j], "length_words": word_count(out["responses"][j]),
                "reward": rewards[j], "kl_seq_sum": seq_kl[j],
                "hit_cap_no_eos": out["truncated"][j], "ended_with_eos": out["terminated_with_eos"][j],
            })
    L = np.array([r["length_tokens"] for r in records], dtype=float)
    R = np.array([r["reward"] for r in records], dtype=float)
    summary = {
        "n_prompts": len(records),
        "kl_token_mean": kl_num / max(kl_den, 1.0),                  # official convention: token-pooled
        "kl_seq_sum_mean": float(np.mean([r["kl_seq_sum"] for r in records])),
        "reward_mean": float(R.mean()), "reward_std": float(R.std()),
        "length_mean": float(L.mean()), "length_std": float(L.std()), "length_median": float(np.median(L)),
        "length_iqr": float(np.percentile(L, 75) - np.percentile(L, 25)),
        "frac_hit_cap_no_eos": float(np.mean([r["hit_cap_no_eos"] for r in records])),
        "corr_length_reward": safe_corr(L, R),
        "prompt_truncation": summarize_prompt_trunc(records, s, tokenizer),
    }
    return summary, records


def qualitative_candidates(records, k=5):
    by_r = sorted(records, key=lambda r: r["reward"])
    med = float(np.median([r["reward"] for r in records]))
    high = [r for r in records if r["reward"] >= med]
    return {
        "highest_reward": by_r[-k:][::-1],
        "lowest_reward": by_r[:k],
        "high_reward_but_longest": sorted(high, key=lambda r: -r["length_tokens"])[:k],
    }


# ------------------------------------------------------------ D: word-limit prompts
def word_limit_eval(model, tokenizer, rows, s, n_samples=1):
    bs, recs = s["eval_batch_size"], []
    prompts = [prompt_messages(r) for r in rows] * n_samples
    pids = [r["prompt_id"] for r in rows] * n_samples            # strict: these files have prompt_id
    for i in range(0, len(prompts), bs):
        chunk = prompts[i:i + bs]
        out = batch_generate(model, tokenizer, chunk, s["max_prompt_length"], s["max_new_tokens"],
                             temperature=s["temperature"], top_p=s["top_p"], do_sample=s["do_sample"])
        for j, (p, resp, n_tok) in enumerate(zip(chunk, out["responses"], out["response_lengths"])):
            text = user_text(p)
            limit = parse_word_limit(text)
            words = word_count(resp)
            official = word_limit_compliance(text, resp)                 # course helper: words <= limit
            strict = None
            if limit is not None:                                       # "under N" strictly means < N
                strict = float(words < limit) if re.search(r"\bunder\s+\d+\s+words?", text.lower()) else float(words <= limit)
            recs.append({"prompt_id": pids[i + j], "prompt": text, "limit": limit, "response": resp,
                         "words": words, "tokens": n_tok, "compliant": official, "compliant_strict": strict,
                         "hit_cap_no_eos": out["truncated"][j]})
    parsed = [r for r in recs if r["compliant"] is not None]
    viol = [r for r in parsed if r["compliant"] == 0.0]
    summary = {
        "n_generations": len(recs), "samples_per_prompt": n_samples,
        "n_limit_parsed": len(parsed), "n_limit_unparsed": len(recs) - len(parsed),
        "compliance_rate": float(np.mean([r["compliant"] for r in parsed])) if parsed else None,
        "compliance_rate_strict_under": float(np.mean([r["compliant_strict"] for r in parsed])) if parsed else None,
        "mean_words": float(np.mean([r["words"] for r in recs])),
        "mean_tokens": float(np.mean([r["tokens"] for r in recs])),
        "n_hit_token_cap": int(sum(r["hit_cap_no_eos"] for r in recs)),
        "mean_overshoot_ratio_of_violators": float(np.mean([r["words"] / r["limit"] for r in viol])) if viol else None,
    }
    return summary, recs


# ------------------------------------------------------------ orchestration
def evaluate_adapter(cfg, adapter, name, tokenizer, reward_bundle, beta=None, stratified=False,
                     do_pairs=True, do_generation=True, max_gen_prompts=200, max_pairs=None,
                     eval_batch_size=4, stratum_key=None, wl_samples=1):
    tokenizer.truncation_side = "left"   # only used by the fallback path; the normal path never cuts the template
    s = settings(cfg, eval_batch_size)
    s["prompt_truncation"] = f"head+tail on last user message (head_frac={HEAD_FRAC}), left-truncation fallback"
    beta = float(cfg["beta"] if beta is None else beta)
    out_dir = f"{cfg['results_dir']}/{name}"
    rm, rm_tok = reward_bundle
    model = load_policy(cfg, adapter_path=None if adapter == "base" else adapter, trainable=False)
    summary = {"name": name, "adapter": adapter, "settings": s, "seed": int(cfg["seed"]),
               "max_gen_prompts": max_gen_prompts, "max_pairs": max_pairs}

    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])
    if do_pairs:
        pair_rows = rows[:max_pairs] if max_pairs else rows
        summary["heldout_pairs"], per_ex, _ = heldout_pair_metrics(model, tokenizer, pair_rows, s, beta)
        write_jsonl(f"{out_dir}/heldout_pair_margins.jsonl", per_ex)

    if do_generation:
        sub = rows[:max_gen_prompts]                      # fixed prefix -> same prompts for every condition
        ids, id_src = row_ids(sub)
        summary["prompt_id_source"] = id_src
        set_seed(int(cfg["seed"]))
        summary["generation"], recs = generate_eval(
            model, tokenizer, rm, rm_tok, [prompt_messages_from_preference(r) for r in sub], ids, s)
        write_jsonl(f"{out_dir}/generations.jsonl", recs)
        save_json(f"{out_dir}/qualitative_candidates.json", qualitative_candidates(recs))

    if stratified:
        srows = read_jsonl(cfg["paths"]["dpo_length_eval"])
        key = find_stratum_key(srows, stratum_key, required=True)
        if max_pairs:
            srows = srows[:max_pairs]
        summary["stratified_pairs"], s_per_ex = stratified_pair_metrics(model, tokenizer, srows, s, key)
        write_jsonl(f"{out_dir}/stratified_pair_margins.jsonl",
                    [dict(e, stratum=str(r[key])) for e, r in zip(s_per_ex, srows)])

    wl_path = cfg["paths"].get("word_limit_prompts")
    if wl_path:
        set_seed(int(cfg["seed"]))
        summary["word_limit"], wl_recs = word_limit_eval(model, tokenizer, read_jsonl(wl_path), s, wl_samples)
        write_jsonl(f"{out_dir}/word_limit_generations.jsonl", wl_recs)

    save_json(f"{out_dir}/eval_summary.json", summary)
    del model
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--adapter", default=None, help="adapter dir, or 'base' for the untouched model")
    ap.add_argument("--name", default="standard")
    ap.add_argument("--beta", type=float, default=None)
    ap.add_argument("--max-gen-prompts", type=int, default=200)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--stratified", action="store_true")
    ap.add_argument("--stratum-key", default=None)
    ap.add_argument("--wl-samples", type=int, default=1, help="generations per word-limit prompt")
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    rm = load_reward_model(cfg)
    summary = evaluate_adapter(cfg, args.adapter or cfg["standard_output"], args.name, tok, rm,
                               beta=args.beta, stratified=args.stratified,
                               max_gen_prompts=args.max_gen_prompts, max_pairs=args.max_pairs,
                               eval_batch_size=args.eval_batch_size,
                               stratum_key=args.stratum_key, wl_samples=args.wl_samples)
    print(json.dumps({k: v for k, v in summary.items() if k != "settings"}, indent=2))


if __name__ == "__main__":
    main()