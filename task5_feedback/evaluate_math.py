"""Task 5 evaluation: SFT vs RLVR vs RLAIF on GSM8K (in-domain) or SVAMP (transfer).

Run from repo root:
  python -m task5_feedback.evaluate_math --dataset gsm      --stage all
  python -m task5_feedback.evaluate_math --dataset transfer --stage all
Stages: generate -> judge -> score (each resumable; `all` runs the three in order).
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from common.data import load_yaml, read_jsonl, prompt_messages, write_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.metrics import word_count, safe_corr
from common.models import load_tokenizer, clear_gpu
from task5_feedback import task5_utils as U
from task5_feedback.rlvr import exact_reward, extract_designated_final

MAX_PROMPT_LEN = 512
GEN_BATCH = 16
PAIRS = [("rlaif", "sft"), ("rlvr", "sft"), ("rlaif", "rlvr")]  # (first, second); win = first preferred


def policy_specs(cfg):
    return {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}


def dataset_path(cfg, dataset: str):
    if dataset == "gsm":
        return cfg["paths"]["gsm_eval"]
    if dataset == "transfer":
        return cfg["paths"]["math_transfer_eval"]
    raise ValueError(dataset)


def load_math_evaluation(config_path: str, dataset: str):
    cfg = load_yaml(config_path)
    rows = read_jsonl(dataset_path(cfg, dataset))
    tokenizer = load_tokenizer(cfg["base_model"])
    return cfg, rows, tokenizer


def load_frozen_policy(cfg, name: str):
    return U.load_named_policy(cfg, name)


def row_id(r, i):
    return str(r.get("prompt_id", r.get("source_index", i)))


def gen_file(ds, pol): return U.out_path(f"{ds}_generations_{pol}.jsonl")
def judg_file(ds): return U.out_path(f"{ds}_judgments.jsonl")
def meta_file(ds): return U.out_path(f"{ds}_judge_meta.json")
def metrics_file(ds): return U.out_path(f"{ds}_metrics.json")


# ------------------------------------------------------------------ stage 1
def stage_generate(cfg, rows, tok, ds, force=False):
    U.assert_template_behaviour(tok)
    prompts = [prompt_messages(r) for r in rows]
    lens = [len(tok.apply_chat_template(p, tokenize=True, add_generation_prompt=True)) for p in prompts]
    assert max(lens) < MAX_PROMPT_LEN, f"prompt of {max(lens)} tokens would be truncated"
    order = sorted(range(len(rows)), key=lambda i: lens[i])
    max_new = int(cfg["math_max_new_tokens"])
    for pol in U.POLICY_NAMES:
        if U.exists(gen_file(ds, pol)) and not force:
            print(f"[generate] {pol}: exists, skip")
            continue
        set_seed(cfg["seed"])
        model = load_frozen_policy(cfg, pol)
        out = [None] * len(rows)
        t0 = time.perf_counter()
        for s in range(0, len(order), GEN_BATCH):
            idxs = order[s:s + GEN_BATCH]
            g = batch_generate(model, tok, [prompts[i] for i in idxs], MAX_PROMPT_LEN, max_new, do_sample=False)
            for k, i in enumerate(idxs):
                out[i] = {"idx": i, "prompt_id": row_id(rows[i], i), "policy": pol, "response": g["responses"][k],
                          "n_tokens": g["response_lengths"][k], "truncated": bool(g["truncated"][k]),
                          "terminated_with_eos": bool(g["terminated_with_eos"][k])}
            print(f"[generate] {pol} {min(s + GEN_BATCH, len(order))}/{len(order)}")
        secs = time.perf_counter() - t0
        for o in out:
            o["gen_seconds_total_for_policy"] = secs
        write_jsonl(gen_file(ds, pol), out)
        del model
        clear_gpu()


# ------------------------------------------------------------------ stage 2
def stage_judge(cfg, rows, ds):
    gens = {p: read_jsonl(gen_file(ds, p)) for p in U.POLICY_NAMES}
    judge = U.make_judge(cfg)
    recs, tie_items, secs_uncached = [], [], []
    for first, second in PAIRS:
        for i, r in enumerate(rows):
            a, b = gens[first][i]["response"], gens[second][i]["response"]
            res, meta = U.judge_compare(judge, r["question"], a, b)
            recs.append({"pair": f"{first}_vs_{second}", "idx": i, "prompt_id": row_id(r, i), "verdict": res,
                         "outcome": {"A": "win", "B": "loss", "TIE": "tie"}[res], "cached": meta["cached"],
                         "seconds": meta["seconds"], "identical_text": a == b})
            if not meta["cached"]:
                secs_uncached.append(meta["seconds"])
            if res == "TIE":
                tie_items.append((r["question"], a, b))
        print(f"[judge] {first} vs {second} done")
    write_jsonl(judg_file(ds), recs)
    audit = U.parse_audit(judge, tie_items)
    U.save(meta_file(ds), {
        "n_calls": len(recs), "n_uncached": len(secs_uncached),
        "mean_seconds_per_uncached_call": float(np.mean(secs_uncached)) if secs_uncached else None,
        "tie_rate_overall": float(np.mean([x["verdict"] == "TIE" for x in recs])),
        "identical_text_pairs": int(sum(x["identical_text"] for x in recs)),
        "parse_audit": audit,
        "judge_gpu_peak_mem_gb": (torch.cuda.max_memory_allocated() / 1e9) if torch.cuda.is_available() else None,
        "judge_hardcoded_max_new_tokens": 4,
        "note": "judge_max_new_tokens/judge_temperature in feedback.yaml are unused by the supplied PairwiseAIJudge",
    })


# ------------------------------------------------------------------ stage 3
def stage_score(cfg, rows, ds):
    gens = {p: read_jsonl(gen_file(ds, p)) for p in U.POLICY_NAMES}
    recs = read_jsonl(judg_file(ds))
    golds = [str(r["gold_final"]) for r in rows]
    rng = np.random.default_rng(cfg["seed"])
    M = {"dataset": ds, "n": len(rows), "policies": {}, "pairs": {}, "tests": {}}
    R, L = {}, {}
    for p in U.POLICY_NAMES:
        g = gens[p]
        t0 = time.perf_counter()
        r = [exact_reward(x["response"], golds[i]) for i, x in enumerate(g)]
        vt = (time.perf_counter() - t0) / len(g)
        fmt = [float(extract_designated_final(x["response"]) is not None) for x in g]
        toks = [x["n_tokens"] for x in g]
        fails = [U.classify_failure(x["response"], golds[i], x["truncated"]) for i, x in enumerate(g)]
        fc = {k: fails.count(k) for k in ["correct", "truncated", "no_designated_final",
                                          "wrong_final_gold_in_reasoning", "wrong_final_other"]}
        bad = [i for i, f in enumerate(fails) if f != "correct"]
        pick = rng.choice(bad, min(10, len(bad)), replace=False).tolist() if bad else []
        M["policies"][p] = {
            "accuracy": float(np.mean(r)), "accuracy_ci95": U.bootstrap_ci(r),
            "format_compliance": float(np.mean(fmt)), "format_ci95": U.bootstrap_ci(fmt),
            "truncation_rate": float(np.mean([x["truncated"] for x in g])),
            "mean_tokens": float(np.mean(toks)), "mean_words": float(np.mean([word_count(x["response"]) for x in g])),
            "failure_counts": fc, "verifier_seconds_per_call": vt,
            "failure_samples": [{"idx": i, "gold": golds[i], "type": fails[i],
                                 "response": U.trunc(g[i]["response"])} for i in pick]}
        R[p], L[p] = np.array(r), np.array(toks, float)
    for a, b in [("rlvr", "sft"), ("rlaif", "sft"), ("rlaif", "rlvr")]:
        M["tests"][f"acc_{a}_vs_{b}"] = {"mcnemar": U.mcnemar_exact(R[a], R[b]),
                                         "paired_bootstrap": U.paired_bootstrap(R[a], R[b])}
    for first, second in PAIRS:
        name = f"{first}_vs_{second}"
        jr = [x for x in recs if x["pair"] == name]
        jr.sort(key=lambda x: x["idx"])
        jo = [x["outcome"] for x in jr]
        n = len(jo)
        w, t, l = jo.count("win"), jo.count("tie"), jo.count("loss")
        d = R[first] - R[second]
        vo = ["win" if x > 0 else "loss" if x < 0 else "tie" for x in d]
        nt = [k for k in range(n) if vo[k] != "tie"]
        score = np.array([{"win": 1, "tie": .5, "loss": 0}[o] for o in jo])
        M["pairs"][name] = {
            "judge_win": w, "judge_tie": t, "judge_loss": l, "judge_win_rate": float((w + .5 * t) / n),
            "judge_win_rate_ci95": U.bootstrap_ci(score),
            "verifier_win": vo.count("win"), "verifier_tie": vo.count("tie"), "verifier_loss": vo.count("loss"),
            "agreement_raw": float(np.mean([a == b for a, b in zip(vo, jo)])),
            "agreement_on_verifier_nontied": (float(np.mean([vo[k] == jo[k] for k in nt])) if nt else None),
            "n_verifier_nontied": len(nt), "cohen_kappa": U.kappa(vo, jo),
            "confusion_rows_verifier_cols_judge_[win,tie,loss]": U.confusion(vo, jo),
            "corr_token_len_diff_vs_judge_score": safe_corr(L[first] - L[second], score),
            "judge_seconds_mean_uncached": None}
    M["env"] = U.env_info(cfg, {"decoding": "greedy", "max_new_tokens": cfg["math_max_new_tokens"],
                                "max_prompt_length": MAX_PROMPT_LEN, "gen_batch": GEN_BATCH})
    U.save(metrics_file(ds), M)
    for p, v in M["policies"].items():
        print(f"{p:6s} acc={v['accuracy']:.3f} fmt={v['format_compliance']:.3f} tok={v['mean_tokens']:.0f} fails={v['failure_counts']}")
    for k, v in M["pairs"].items():
        print(f"{k:16s} judge W/T/L={v['judge_win']}/{v['judge_tie']}/{v['judge_loss']} winrate={v['judge_win_rate']:.3f} agree={v['agreement_raw']:.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--dataset", choices=["gsm", "transfer"], default="gsm")
    ap.add_argument("--stage", choices=["generate", "judge", "score", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg, rows, tok = load_math_evaluation(args.config, args.dataset)
    set_seed(cfg["seed"])
    print("Rows:", len(rows), "| Policies:", U.POLICY_NAMES)
    if args.stage in ("generate", "all"):
        stage_generate(cfg, rows, tok, args.dataset, args.force)
    if args.stage in ("judge", "all"):
        stage_judge(cfg, rows, args.dataset)
    if args.stage in ("score", "all"):
        stage_score(cfg, rows, args.dataset)


if __name__ == "__main__":
    main()
