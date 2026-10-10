from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse
import re
import time

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path, write_jsonl
from common.generation import score_reward_pairs
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task2_ppo import utils as U

MAX_EVAL_PROMPTS = 300   # use the whole file if it has <= 300 prompts, else the first 300
EVAL_BATCH = 16          # prompts per generation batch
LOGPROB_MB = 4           # sequences per log-prob forward pass


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    tok = load_tokenizer(cfg["base_model"])
    tok.truncation_side = "left"
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": tok,
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def distinct_n(words: list[str], n: int):
    grams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams) if grams else None


def evaluate_policy(bundle: dict, name: str, max_prompts: int = MAX_EVAL_PROMPTS, batch_size: int = EVAL_BATCH):
    cfg, tok, policy = bundle["cfg"], bundle["tokenizer"], bundle["policy"]
    rm, rm_tok = bundle["reward"]
    rows = bundle["rows"][:max_prompts]
    U.set_lora_dropout_zero(policy)
    device = next(policy.parameters()).device
    eval_seed = int(cfg["seed"])  # same decoding seed for every condition
    max_new = int(cfg["eval_max_response_length"])
    n_long = U.count_long_prompts(tok, [prompt_messages(r) for r in rows], int(cfg["max_prompt_length"]))

    torch.cuda.reset_peak_memory_stats()
    t0 = time.perf_counter()
    records = []
    kl_sum_tok, tok_total, ent_sum_tok = 0.0, 0.0, 0.0

    for b in range(0, len(rows), batch_size):
        chunk = rows[b:b + batch_size]
        msgs = [prompt_messages(r) for r in chunk]
        gen = U.rollout(policy, tok, msgs, cfg, max_new, seed=eval_seed + b)
        seqs, attn, resp, mask = gen["sequences"], gen["attention_mask"], gen["response_ids"], gen["response_mask"]
        scores = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg["reward_max_length"])).cpu()

        mbs = U.microbatches(seqs.shape[0], LOGPROB_MB)
        with torch.no_grad():
            pl, en = [], []
            for sl in mbs:
                lp, e = U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl], entropy=True)
                pl.append(lp)
                en.append(e)
            rl = []
            with reference_mode(policy):
                for sl in mbs:
                    rl.append(U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl])[0])
        pl, en, rl = torch.cat(pl), torch.cat(en), torch.cat(rl)
        kl_tok = (pl - rl) * mask
        n_per = mask.sum(-1).clamp_min(1.0)
        kl_sum_tok += float(kl_tok.sum())
        ent_sum_tok += float((en * mask).sum())
        tok_total += float(mask.sum())

        for i, r in enumerate(chunk):
            records.append({
                "prompt_id": r["prompt_id"], "source_index": r.get("source_index"),
                "response": gen["responses"][i], "reward": float(scores[i]),
                "length": int(gen["response_lengths"][i]),
                "terminated_with_eos": bool(gen["terminated_with_eos"][i]), "truncated": bool(gen["truncated"][i]),
                "kl_sum": float(kl_tok[i].sum()), "kl_mean": float(kl_tok[i].sum() / n_per[i]),
                "entropy_mean": float((en[i] * mask[i]).sum() / n_per[i]),
            })
        del gen, pl, en, rl, kl_tok
        torch.cuda.empty_cache()
        print(f"[eval {name}] {min(b + batch_size, len(rows))}/{len(rows)}", flush=True)

    # ---- aggregate (same conventions for every condition)
    rew = np.array([r["reward"] for r in records])
    length = np.array([r["length"] for r in records], dtype=float)
    d = {n: [] for n in (2, 3, 4)}
    pooled_words = []
    for r in records:
        words = re.findall(r"\w+", r["response"].lower())
        pooled_words.append(words)
        for n in d:
            v = distinct_n(words, n)
            if v is not None:
                d[n].append(v)
    summary = {
        "n_prompts": len(records),
        "reward_mean": float(rew.mean()), "reward_std": float(rew.std(ddof=1)),
        "reward_sem": float(rew.std(ddof=1) / np.sqrt(len(rew))),
        "kl_token": kl_sum_tok / tok_total,                        # token-pooled over the whole eval set
        "kl_seq": float(np.mean([r["kl_sum"] for r in records])),  # per-sequence summed KL, averaged
        "entropy": ent_sum_tok / tok_total,                        # exact full-distribution entropy, token-pooled
        "length_mean": float(length.mean()), "length_std": float(length.std(ddof=1)),
        "length_q25": float(np.percentile(length, 25)), "length_q75": float(np.percentile(length, 75)),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        # distinct-n: unique/total word n-grams *within* a response, averaged over responses (sensitive to loops)
        "distinct2": float(np.mean(d[2])), "distinct3": float(np.mean(d[3])), "distinct4": float(np.mean(d[4])),
        # same ratio on all responses pooled (corpus-level diversity)
        "distinct2_pooled": float(distinct_n([w for ws in pooled_words for w in ws], 2) or float("nan")),
        "corr_reward_length": float(np.corrcoef(rew, length)[0, 1]) if rew.std() > 0 and length.std() > 0 else float("nan"),
        "decoding": {"temperature": cfg["generation"]["temperature"], "top_p": cfg["generation"]["top_p"],
                     "do_sample": cfg["generation"]["do_sample"], "max_new_tokens": max_new, "seed": eval_seed,
                     "batch_size": batch_size},
        "n_prompts_over_max_prompt_length": n_long,
        "prompt_ids": [r["prompt_id"] for r in records],
        "eval_wall_clock_s": time.perf_counter() - t0,
        "eval_peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
    }
    write_jsonl(f"{U.OUTPUT_DIR}/{name}_eval.jsonl", records)
    U.update_result_json(name, "eval", summary)
    print(f"[eval {name}] reward={summary['reward_mean']:.3f}±{summary['reward_std']:.3f} KL={summary['kl_token']:.4f} "
          f"H={summary['entropy']:.3f} len={summary['length_mean']:.0f}±{summary['length_std']:.0f} "
          f"trunc={summary['truncation_rate']:.2f} d2={summary['distinct2']:.3f}", flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    ap.add_argument("--max-prompts", type=int, default=MAX_EVAL_PROMPTS)
    ap.add_argument("--batch-size", type=int, default=EVAL_BATCH)
    args = ap.parse_args()
    bundle = load_evaluation_bundle(args.config, args.adapter)
    evaluate_policy(bundle, args.name, args.max_prompts, args.batch_size)


if __name__ == "__main__":
    main()