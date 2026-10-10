from __future__ import annotations

import argparse

import numpy as np
import torch

from common.data import load_yaml, read_jsonl
from common.generation import batch_generate, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import load_policy, load_reward_model, load_tokenizer, reference_mode
from task3_grpo.utils import (
    chunked_token_logps,
    normalize_gen,
    peak_memory_report,
    reset_peak_memory,
    results_dir,
    row_messages,
    row_prompt_id,
)


def load_evaluation_bundle(config_path: str, adapter: str):
    cfg = load_yaml(config_path)
    return {
        "cfg": cfg,
        "rows": read_jsonl(cfg["paths"]["rl_prompt_eval"]),
        "tokenizer": load_tokenizer(cfg["base_model"]),
        "policy": load_policy(cfg, adapter_path=adapter, trainable=False),
        "reward": load_reward_model(cfg),
    }


def evaluate_adapter(config_path: str, adapter: str, name: str = "standard") -> dict:
    timer = wall_timer()
    b = load_evaluation_bundle(config_path, adapter)
    cfg, rows, tok, policy = b["cfg"], b["rows"], b["tokenizer"], b["policy"]
    rm, rm_tok = b["reward"]
    gk = cfg["generation"]
    max_new, max_prompt = int(cfg["max_completion_length"]), int(cfg["max_prompt_length"])
    bs = int(cfg.get("eval_batch_size", 8))
    lp_chunk = int(cfg.get("eval_logp_batch_size", 4))
    rdir = results_dir(cfg)
    rec_path = rdir / f"eval_{name}_records.jsonl"
    rec_path.unlink(missing_ok=True)
    reset_peak_memory()

    records = []
    for bi, start in enumerate(range(0, len(rows), bs)):
        chunk = rows[start : start + bs]
        msgs = [row_messages(r) for r in chunk]
        set_seed(int(cfg["seed"]) + 10_000 + bi)  # identical noise stream for every condition
        gen = batch_generate(
            policy, tok, msgs, max_prompt, max_new,
            temperature=float(gk["temperature"]), top_p=float(gk["top_p"]), do_sample=bool(gk["do_sample"]),
        )
        gen = normalize_gen(gen)
        rewards = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=1024).float().cpu()
        pol_lp = chunked_token_logps(policy, gen, lp_chunk)  # policy is in eval mode
        with reference_mode(policy):
            ref_lp = chunked_token_logps(policy, gen, lp_chunk)  # adapter disabled -> base reference
        mask = gen["response_mask"].cpu().float()
        for i, r in enumerate(chunk):
            n = float(mask[i].sum())
            d = ((pol_lp[i] - ref_lp[i]) * mask[i]).sum().item()
            rec = {
                "prompt_id": row_prompt_id(r, start + i), "response": gen["responses"][i],
                "reward": float(rewards[i]), "length": int(gen["response_lengths"][i]),
                "truncated": bool(gen["truncated"][i]), "terminated_with_eos": bool(gen["terminated_with_eos"][i]),
                "n_tokens": n, "kl_sum": d, "kl_tok_mean": d / max(n, 1.0),
                "logp_sum": float((pol_lp[i] * mask[i]).sum()),
            }
            records.append(rec)
            append_jsonl(rec_path, rec)
        print(f"  [{name}] batch {bi + 1}/{(len(rows) + bs - 1) // bs}")
        del gen, pol_lp, ref_lp

    R = np.array([r["reward"] for r in records])
    L = np.array([r["length"] for r in records], float)
    ntok = np.array([r["n_tokens"] for r in records])
    kls = np.array([r["kl_sum"] for r in records])
    lps = np.array([r["logp_sum"] for r in records])
    result = {
        "name": name, "adapter": adapter, "n_prompts": len(records),
        "decoding": {"temperature": gk["temperature"], "top_p": gk["top_p"], "max_new_tokens": max_new,
                     "samples_per_prompt": 1, "noise_seed_base": int(cfg["seed"]) + 10_000, "batch_size": bs},
        "reward_mean": float(R.mean()), "reward_std": float(R.std(ddof=1)),
        "reward_se": float(R.std(ddof=1) / np.sqrt(len(R))),
        "kl_token_pooled": float(kls.sum() / max(ntok.sum(), 1.0)),  # course helper convention (primary)
        "kl_seq_mean": float(kls.mean()),  # sequence-summed (secondary, length-confounded)
        "entropy_sampled_token": float(-lps.sum() / max(ntok.sum(), 1.0)),
        "length_mean": float(L.mean()), "length_std": float(L.std(ddof=1)),
        "length_median": float(np.median(L)), "length_q25": float(np.percentile(L, 25)),
        "length_q75": float(np.percentile(L, 75)),
        "truncation_rate": float(np.mean([r["truncated"] for r in records])),
        "eos_rate": float(np.mean([r["terminated_with_eos"] for r in records])),
        "corr_length_reward": float(np.corrcoef(L, R)[0, 1]) if L.std() > 0 and R.std() > 0 else float("nan"),
        "peak_vram": peak_memory_report(), "wall_clock_s": timer(),
    }
    save_json(rdir / f"eval_{name}.json", result)
    print({k: result[k] for k in ("reward_mean", "reward_se", "kl_token_pooled", "length_mean", "truncation_rate")})
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--adapter", required=True)
    ap.add_argument("--name", default="standard")
    args = ap.parse_args()
    evaluate_adapter(args.config, args.adapter, args.name)


if __name__ == "__main__":
    main()
