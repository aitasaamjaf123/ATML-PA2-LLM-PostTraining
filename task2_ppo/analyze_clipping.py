from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse

import numpy as np
import torch

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.metrics import masked_mean
from common.models import load_policy, load_tokenizer, load_value_model, reference_mode
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


def _mean_abs(a, b, mask):
    return float(((a - b).abs() * mask).sum() / mask.sum())


def cached_stage(config_path: str):
    """Single forward pass of the midpoint policy over the fixed cached batch; no weights are updated.

    The cache stores no token ids, so each response is re-tokenised (+EOS if it terminated). Because the
    recomputed log-probs did not match the cache with left-truncated prompts, two conventions are compared
    against the cached values (over-long prompts truncated on the left vs on the right; log-probs always on
    raw T=1 logits) and the ratio table below uses whichever reproduces the cache best.
    """
    cfg = load_yaml(config_path)
    rows = load_cached_rollouts(cfg["cached_rollouts"])
    beta = float(cfg["kl_beta"])
    L = int(cfg["max_prompt_length"])
    tok = load_tokenizer(cfg["base_model"])

    lookup = {}
    for key in ("rl_prompt_train", "rl_prompt_eval"):
        for r in read_jsonl(cfg["paths"][key]):
            lookup.setdefault(r["prompt_id"], r)

    policy = load_policy(cfg, adapter_path=cfg["paths"]["ppo_midpoint_policy"], trainable=False)
    U.set_lora_dropout_zero(policy)
    vmodel = load_value_model(cfg, cfg["paths"]["ppo_midpoint_value"], train_mode="frozen")
    (vmodel.score if hasattr(vmodel, "score") else vmodel.classifier).float()
    device = next(policy.parameters()).device

    sides = ("left", "right")  # which end of an over-long prompt (> max_prompt_length) is truncated
    new_lp = {s: [] for s in sides}
    ref_new = {s: [] for s in sides}
    my_v = {"left": [], "right": []}
    kept, skipped, info = [], [], []

    for row in rows:
        prow = lookup.get(row["prompt_id"])
        if prow is None:
            skipped.append({"prompt_id": row["prompt_id"], "reason": "prompt_id not found"})
            continue
        text = tok.apply_chat_template(prompt_messages(prow), tokenize=False, add_generation_prompt=True)
        p_full = tok(text, add_special_tokens=False)["input_ids"]
        r_ids = tok(row["response"], add_special_tokens=False)["input_ids"]
        if row.get("terminated_with_eos", False):
            r_ids = r_ids + [tok.eos_token_id]
        n_cached = len(row["old_logprobs"])
        if len(r_ids) != n_cached:
            skipped.append({"prompt_id": row["prompt_id"], "reason": f"token mismatch {len(r_ids)} vs {n_cached}"})
            continue
        resp = torch.tensor([r_ids], device=device)
        with torch.no_grad():
            for side in sides:
                p_use = p_full[-L:] if side == "left" else p_full[:L]
                seq = torch.tensor([p_use + r_ids], device=device)
                attn = torch.ones_like(seq)
                lp, _ = U.response_logprobs(policy, seq, attn, resp)
                with reference_mode(policy):
                    rlp, _ = U.response_logprobs(policy, seq, attn, resp)
                new_lp[side].append(lp[0].float().cpu())
                ref_new[side].append(rlp[0].float().cpu())
                my_v[side].append(U.value_forward(vmodel, seq, attn, len(p_use), len(r_ids))[0].float().cpu())
        kept.append(row)
        info.append({"prompt_id": row["prompt_id"], "prompt_len": len(p_full), "n_tokens": n_cached,
                     "clipped_at_max": bool(row.get("clipped_at_max", False))})
    del vmodel
    if not kept:
        raise RuntimeError("No cached rows could be aligned with their cached log-probs")
    print(f"cached rows used: {len(kept)}/{len(rows)}; skipped: {skipped}", flush=True)

    old_lp = _pad([r["old_logprobs"] for r in kept])
    ref_lp = _pad([r["ref_logprobs"] for r in kept])
    values = _pad([r["values"] for r in kept])
    mask = _pad([torch.ones(len(r["old_logprobs"])) for r in kept])
    task_r = torch.tensor([float(r["raw_terminal_reward"]) for r in kept])  # raw RM score (no EOS penalty, as in training)
    long_rows = torch.tensor([i_["prompt_len"] > L for i_ in info])

    # ---- which convention reproduces the cached log-probs?
    alignment, padded_new, padded_ref = {}, {}, {}
    for name in sides:
        n_, r_ = _pad(new_lp[name]), _pad(ref_new[name])
        padded_new[name], padded_ref[name] = n_, r_
        per_row = [float(((n_[i] - old_lp[i]).abs() * mask[i]).sum() / mask[i].sum()) for i in range(len(kept))]
        for i_, v in zip(info, per_row):
            i_[f"abs_diff_{name}"] = v
        alignment[name] = {
            "policy_vs_cached_old_abs": _mean_abs(n_, old_lp, mask),
            "ref_vs_cached_ref_abs": _mean_abs(r_, ref_lp, mask),
            "policy_vs_cached_old_abs_long_prompts": _mean_abs(n_[long_rows], old_lp[long_rows], mask[long_rows]) if long_rows.any() else None,
            "policy_vs_cached_old_abs_short_prompts": _mean_abs(n_[~long_rows], old_lp[~long_rows], mask[~long_rows]) if (~long_rows).any() else None,
            "signed_mean_new_minus_old": float(((n_ - old_lp) * mask).sum() / mask.sum()),
            "median_row_abs_diff": float(np.median(per_row)),
        }
        print(f"[align] {name:10s} policy|diff|={alignment[name]['policy_vs_cached_old_abs']:.5f} "
              f"ref|diff|={alignment[name]['ref_vs_cached_ref_abs']:.5f} "
              f"long-prompt rows={alignment[name]['policy_vs_cached_old_abs_long_prompts']} "
              f"short-prompt rows={alignment[name]['policy_vs_cached_old_abs_short_prompts']}", flush=True)
    best = min(alignment, key=lambda n: alignment[n]["policy_vs_cached_old_abs"])
    best_side = "left" if best.endswith("left") else "right"
    print(f"[align] best-matching convention: {best}", flush=True)

    # ---- does my critic forward reproduce the cached values?
    value_check = {}
    for side in ("left", "right"):
        mv = _pad(my_v[side])
        valid_ = mask.bool()
        value_check[side] = {"abs_diff": _mean_abs(mv, values, mask),
                             "corr": float(np.corrcoef(mv[valid_].numpy(), values[valid_].numpy())[0, 1])}
    print(f"[value check] my critic vs cached values: {value_check}", flush=True)

    # ---- advantages from the cache (as in training: KL-shaped rewards, GAE, whitening)
    new_best = padded_new[best]
    rewards = shaped_rewards(task_r, old_lp, ref_lp, mask, beta)
    adv_raw, returns = compute_gae(rewards, values, mask, gamma=float(cfg["gamma"]), lam=float(cfg["gae_lambda"]))
    adv_w = normalize_advantages(adv_raw, mask)
    mc = torch.flip(torch.cumsum(torch.flip(rewards * mask, [1]), 1), [1]) * mask
    log_r = U.clamp_new_logp(new_best, old_lp)
    valid = mask.bool()
    rho = (new_best - old_lp).exp()
    ev, v_mean, ret_mean, v_corr = U.value_stats(values, returns, mask)
    ev_mc, _, mc_mean, mc_corr = U.value_stats(values, mc, mask)

    out = {"meta": {
        "n_cached_rows": len(rows), "n_used": len(kept), "skipped": skipped, "kl_beta": beta,
        "n_valid_tokens": int(mask.sum()), "n_clipped_at_max": int(sum(i_["clipped_at_max"] for i_ in info)),
        "n_prompts_over_max_prompt_length": int(long_rows.sum()),
        "alignment": alignment, "best_convention": best, "ratio_table_convention": best,
        "ratio_mean": float(rho[valid].mean()), "ratio_min": float(rho[valid].min()), "ratio_max": float(rho[valid].max()),
        "log_ratio_std": float((new_best - old_lp)[valid].std()), "abs_log_ratio_mean": float((new_best - old_lp).abs()[valid].mean()),
        "ref_logp_abs_diff_mean": float((padded_ref[best] - ref_lp).abs()[valid].mean()),
        "value_check": value_check,
        "terminal_reward_mean": float(task_r.mean()),
        "cached_critic_explained_variance": ev, "cached_value_mean": v_mean, "cached_lambda_return_mean": ret_mean,
        "cached_value_return_corr": v_corr,
        "cached_critic_ev_vs_mc_return": ev_mc, "cached_mc_return_mean": mc_mean, "cached_value_mc_corr": mc_corr,
        "adv_whitened_mean": float(adv_w[valid].mean()), "adv_whitened_std": float(adv_w[valid].std(unbiased=False)),
        "per_row": info,
        "note": "if the best convention reproduces the cache (|diff| ~1e-3) then rho ~ 1 and the affected fraction ~ 0 by construction",
    }}
    for eps in cfg["clip_values"]:
        loss_w, ratio, frac = ppo_policy_loss(log_r, old_lp, adv_w, mask, eps=float(eps))
        loss_r, _, _ = ppo_policy_loss(log_r, old_lp, adv_raw, mask, eps=float(eps))
        above = masked_mean((ratio > 1 + eps).float(), mask)
        below = masked_mean((ratio < 1 - eps).float(), mask)
        surr1 = ratio * adv_w
        surr2 = ratio.clamp(1 - eps, 1 + eps) * adv_w
        binding = masked_mean((surr2 < surr1).float(), mask)  # tokens where the clipped branch is actually selected
        affected = ((ratio < 1 - eps) | (ratio > 1 + eps)).float()
        m_short = mask * (~long_rows).float().unsqueeze(1)   # prompts that fit in max_prompt_length: convention is unambiguous
        m_long = mask * long_rows.float().unsqueeze(1)       # over-long prompts: depend on the truncation convention
        out[f"{float(eps):g}"] = {
            "l_clip": float(-loss_w), "policy_loss": float(loss_w), "l_clip_raw_adv": float(-loss_r),
            "affected_frac": float(frac), "frac_above": float(above), "frac_below": float(below),
            "binding_frac": float(binding),
            "affected_frac_short_prompts": float(masked_mean(affected, m_short)),
            "affected_frac_long_prompts": float(masked_mean(affected, m_long)),
        }
        e_ = out[f"{float(eps):g}"]
        print(f"eps={eps} [{best}]: L_clip={-float(loss_w):.6f} affected={float(frac):.6f} binding={float(binding):.6f} "
              f"| short-prompt rows={e_['affected_frac_short_prompts']:.6f} long-prompt rows={e_['affected_frac_long_prompts']:.6f}", flush=True)
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