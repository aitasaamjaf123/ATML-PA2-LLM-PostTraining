from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer, reference_mode
from task2_ppo import utils as U
from task2_ppo.ppo import compute_gae, normalize_advantages, ppo_policy_loss, shaped_rewards


def load_cached_rollouts(path):
    rows = torch.load(repo_path(path), map_location="cpu", weights_only=False)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a non-empty list in the supplied PPO rollout cache")

    # Instructor iterations used two equivalent names for these fields. Normalize once here so
    # the student analysis code sees one stable interface.
    normalized = []
    for row in rows:
        row = dict(row)
        if "old_logprobs" not in row and "old_policy_logprobs" in row:
            row["old_logprobs"] = row["old_policy_logprobs"]
        if "ref_logprobs" not in row and "reference_logprobs" in row:
            row["ref_logprobs"] = row["reference_logprobs"]
        normalized.append(row)

    required = {"source_index", "response", "old_logprobs", "ref_logprobs"}
    if not required.issubset(normalized[0]):
        raise ValueError(f"Unexpected PPO cache schema; need at least {sorted(required)}")
    return normalized


def _pad(seqs, fill=0.0):
    n, m = len(seqs), max(len(s) for s in seqs)
    out = torch.full((n, m), fill, dtype=torch.float32)
    for i, s in enumerate(seqs):
        out[i, :len(s)] = s.float() if torch.is_tensor(s) else torch.tensor(s, dtype=torch.float32)
    return out


def cached_stage(config_path: str):
    """Single forward pass of the midpoint policy over the fixed cached batch; no weights are updated.

    The cache stores no token ids, so each response is re-tokenised (+EOS if it terminated) and rows whose
    token count disagrees with the cached log-prob length are dropped and reported.
    """
    cfg = load_yaml(config_path)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    beta = float(cfg["kl_beta"])
    tok = load_tokenizer(cfg["base_model"])
    tok.truncation_side = "left"

    lookup = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            lookup.setdefault(r["prompt_id"], r)

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    U.set_lora_dropout_zero(policy)
    device = next(policy.parameters()).device

    kept, skipped = [], []
    new_lp_l, ref_new_l = [], []
    for row in rows:
        prow = lookup.get(row["prompt_id"])
        if prow is None:
            skipped.append({"prompt_id": row["prompt_id"], "reason": "prompt_id not found"})
            continue
        text = tok.apply_chat_template(prompt_messages(prow), tokenize=False, add_generation_prompt=True)
        p_ids = tok(text, add_special_tokens=False)["input_ids"][-int(cfg["max_prompt_length"]):]
        r_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos", False):
            r_ids = r_ids + [tok.eos_token_id]
        n_cached = len(row["old_logprobs"])
        if len(r_ids) != n_cached:
            skipped.append({"prompt_id": row["prompt_id"], "reason": f"token mismatch {len(r_ids)} vs {n_cached}"})
            continue
        seq = torch.tensor([p_ids + r_ids], device=device)
        attn = torch.ones_like(seq)
        resp = torch.tensor([r_ids], device=device)
        with torch.no_grad():
            lp, _ = U.response_logprobs(policy, seq, attn, resp)
            with reference_mode(policy):
                rlp, _ = U.response_logprobs(policy, seq, attn, resp)
        new_lp_l.append(lp[0].float().cpu())
        ref_new_l.append(rlp[0].float().cpu())
        kept.append(row)

    if not kept:
        raise RuntimeError("No cached rows could be aligned with their cached log-probs")
    print(f"cached rows used: {len(kept)}/{len(rows)}; skipped: {skipped}", flush=True)

    new_lp = _pad(new_lp_l)
    ref_new = _pad(ref_new_l)
    old_lp = _pad([r["old_logprobs"] for r in kept])
    ref_lp = _pad([r["ref_logprobs"] for r in kept])
    values = _pad([r["values"] for r in kept])
    mask = _pad([torch.ones(len(r["old_logprobs"])) for r in kept])
    task_r = torch.tensor([float(r["raw_terminal_reward"]) for r in kept])  # raw RM score (no EOS penalty, as in training)

    rewards = shaped_rewards(task_r, old_lp, ref_lp, mask, beta)
    adv_raw, returns = compute_gae(rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    adv_w = normalize_advantages(adv_raw, mask)

    log_r = U.clamp_new_logp(new_lp, old_lp)
    valid = mask.bool()
    rho = (new_lp - old_lp).exp()
    ev, v_mean, ret_mean, v_corr = U.value_stats(values, returns, mask)

    out = {"meta": {
        "n_cached_rows": len(rows), "n_used": len(kept), "skipped": skipped, "kl_beta": beta,
        "n_valid_tokens": int(mask.sum()), "n_clipped_at_max": int(sum(bool(r.get("clipped_at_max", False)) for r in kept)),
        "ratio_mean": float(rho[valid].mean()), "ratio_min": float(rho[valid].min()), "ratio_max": float(rho[valid].max()),
        "log_ratio_std": float((new_lp - old_lp)[valid].std()), "abs_log_ratio_mean": float((new_lp - old_lp).abs()[valid].mean()),
        "ref_logp_abs_diff_mean": float((ref_new - ref_lp).abs()[valid].mean()),
        "cached_critic_explained_variance": ev, "cached_value_mean": v_mean, "cached_return_mean": ret_mean,
        "cached_value_return_corr": v_corr,
        "adv_whitened_mean": float(adv_w[valid].mean()), "adv_whitened_std": float(adv_w[valid].std(unbiased=False)),
        "note": "theta == theta_old up to fp16 recomputation noise, so rho ~ 1 and the affected fraction is ~0 by construction",
    }}
    for eps in cfg["clip_values"]:
        loss_w, ratio, frac = ppo_policy_loss(log_r, old_lp, adv_w, mask, eps=float(eps))
        loss_r, _, _ = ppo_policy_loss(log_r, old_lp, adv_raw, mask, eps=float(eps))
        above = masked_mean((ratio > 1 + eps).float(), mask)
        below = masked_mean((ratio < 1 - eps).float(), mask)
        surr1 = ratio * adv_w
        surr2 = ratio.clamp(1 - eps, 1 + eps) * adv_w
        binding = masked_mean((surr2 < surr1).float(), mask)  # tokens where the clipped branch is actually selected
        out[f"{float(eps):g}"] = {
            "l_clip": float(-loss_w), "policy_loss": float(loss_w), "l_clip_raw_adv": float(-loss_r),
            "affected_frac": float(frac), "frac_above": float(above), "frac_below": float(below),
            "binding_frac": float(binding),
        }
        print(f"eps={eps}: L_clip={-float(loss_w):.6f} affected={float(frac):.6f} binding={float(binding):.6f}", flush=True)
    U.write_json(f"{U.RESULTS_DIR}/clipping_cached.json", out)
    return out


def fork_stage(cfg: dict, config_path: str, seeds, force: bool):
    beta = float(cfg["kl_beta"])
    conditions = [{"label": f"eps{float(e):g}", "eps": float(e), "beta": beta} for e in cfg["clip_values"]]
    failed = U.run_grid(conditions, seeds, config_path, int(cfg["fork_updates"]), force)
    summary = U.summarize_study(conditions, seeds, "clipping_study")
    print("clipping study summary -> results/task2_ppo/clipping_study.json")
    if any(v.get("flag_gt_10pct") for v in summary["token_budget"].values()):
        print("WARNING: generated-token budgets differ by >10% between conditions (see token_budget)")
    if failed:
        raise SystemExit(f"failed conditions: {failed}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--stage", choices=["cached", "forks", "all"], default="all")
    ap.add_argument("--seeds", type=int, nargs="+", default=U.SEEDS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)

    if args.stage == "cached":
        cached_stage(args.config)
        return
    print("Required epsilon values:", cfg["clip_values"], "| fork updates:", cfg["fork_updates"], "| seeds:", args.seeds)
    if args.stage == "all":  # separate process so the model is fully released before the forks start
        U.run_module("task2_ppo.analyze_clipping", ["--config", args.config, "--stage", "cached"])
    fork_stage(cfg, args.config, args.seeds, args.force)


if __name__ == "__main__":
    main()