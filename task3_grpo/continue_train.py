from __future__ import annotations

import argparse
import gc
import math
import time

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, read_jsonl, render_prompt, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.metrics import safe_corr
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
from task3_grpo.utils import (
    chunked_token_logps,
    eval_mode,
    kl_stats,
    length_weight_shares,
    normalize_gen,
    peak_memory_report,
    prompt_order,
    reset_peak_memory,
    results_dir,
    row_messages,
    row_prompt_id,
)


def prepare_grpo_continuation(config_path: str):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    tokenizer = load_tokenizer(cfg["base_model"])
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["grpo_midpoint_policy"],
        trainable=True,
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])
    optimizer = AdamW(trainable_parameters(policy), lr=float(cfg["learning_rate"]))
    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "optimizer": optimizer,
    }


# ----------------------------------------------------------------------------- internals
def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _rng_state():
    return torch.get_rng_state(), (torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def _set_rng_state(s):
    torch.set_rng_state(s[0])
    if s[1] is not None:
        torch.cuda.set_rng_state_all(s[1])


def _flat_norm(tensors) -> float:
    return math.sqrt(sum(float(t.float().pow(2).sum()) for t in tensors))


def _accumulate_grads(policy, optimizer, gen, adv, loss_mask, ref_lp, *, eps, beta, loss_type, max_len, mb_size):
    """Forward/backward over micro-batches. old_logp = new_logp.detach() (policy_epochs=1), so ratio == 1."""
    device = adv.device
    optimizer.zero_grad(set_to_none=True)
    N = gen["sequences"].shape[0]
    acc = {"loss": 0.0, "policy_term": 0.0, "k3_kl": 0.0, "clip_fraction": 0.0, "ratio_mean": 0.0}
    for i in range(0, N, mb_size):
        j = min(i + mb_size, N)
        lp, _ = response_token_logprobs(
            policy, gen["sequences"][i:j], gen["attention_mask"][i:j], gen["prompt_width"], gen["response_ids"][i:j]
        )
        del _
        loss, m = grpo_policy_loss(
            lp,
            lp.detach(),
            adv[i:j],
            loss_mask[i:j],
            ref_lp[i:j].to(device),
            eps,
            beta,
            loss_type=loss_type,
            max_completion_length=max_len,
        )
        w = (j - i) / N  # micro-batch rescaling so the sum matches the full-batch loss
        (loss * w).backward()
        acc["loss"] += float(loss.detach()) * w
        acc["policy_term"] += float(m["policy_term"]) * w
        acc["k3_kl"] += float(m["sampled_kl"]) * w
        acc["clip_fraction"] += float(m["clip_fraction"]) * w
        acc["ratio_mean"] += float(m["ratio_mean"]) * w
        del lp, loss, m
    return acc


def _accumulate_with_oom_retry(*args, mb_size, **kw):
    while True:
        try:
            return _accumulate_grads(*args, mb_size=mb_size, **kw), mb_size
        except torch.cuda.OutOfMemoryError:
            kw_opt = args[1]
            kw_opt.zero_grad(set_to_none=True)
            gc.collect()
            torch.cuda.empty_cache()
            if mb_size <= 1:
                raise
            mb_size = max(1, mb_size // 2)
            print(f"  [OOM] retrying with micro_batch_size={mb_size}")


# ----------------------------------------------------------------------------- main loop
def run_grpo(
    config_path: str,
    output: str | None = None,
    updates: int | None = None,
    loss_type: str = "grpo",
    run_name: str = "standard",
    seed: int | None = None,
    log_term_grads: bool = False,
):
    t_total = wall_timer()
    bundle = prepare_grpo_continuation(config_path)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if seed is not None:
        cfg["seed"] = int(seed)
        set_seed(int(seed))
    out = repo_path(output or cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)

    policy, tokenizer = bundle["policy"], bundle["tokenizer"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    rows, optimizer = bundle["prompt_rows"], bundle["optimizer"]
    params = trainable_parameters(policy)
    device = next(policy.parameters()).device

    K = int(cfg["num_generations"])
    P = int(cfg["prompts_per_update"])
    N = K * P
    n_updates = int(cfg["updates"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    max_new, max_prompt = int(cfg["max_completion_length"]), int(cfg["max_prompt_length"])
    gk = cfg["generation"]
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    max_gn = float(cfg["max_grad_norm"])
    tol = float(cfg.get("informative_tol", 1e-6))
    lp_chunk = int(cfg.get("eval_logp_batch_size", 4))
    mb_size = int(cfg.get("micro_batch_size", N))
    if int(cfg.get("policy_epochs", 1)) != 1:
        raise ValueError("old_logp = new_logp.detach() is only valid for policy_epochs == 1")

    order = prompt_order(len(rows), cfg["seed"])
    selected = order[: n_updates * P]
    rdir = results_dir(cfg)
    log_path = rdir / f"{run_name}_train_log.jsonl"
    comp_path = rdir / f"{run_name}_completions.jsonl"
    summary_path = rdir / f"{run_name}_summary.json"
    for p in (log_path, comp_path):
        p.unlink(missing_ok=True)

    try:  # generation must use the KV cache even though load_policy(trainable=True) disabled it in config
        policy.generation_config.use_cache = True
    except Exception:
        pass

    print(f"[{run_name}] loss_type={loss_type} seed={cfg['seed']} updates={n_updates} K={K} P={P} mb={mb_size}")
    reset_peak_memory()
    _sync()
    t_loop = wall_timer()
    skipped = 0

    for u in range(n_updates):
        t0 = time.perf_counter()
        idxs = selected[u * P : (u + 1) * P]
        prow = [rows[i] for i in idxs]
        msgs = [row_messages(r) for r in prow]
        pids = [row_prompt_id(r, i) for r, i in zip(prow, idxs)]
        ptoks = [len(tokenizer(render_prompt(tokenizer, m), add_special_tokens=False)["input_ids"]) for m in msgs]
        batch_msgs = [m for m in msgs for _ in range(K)]
        group_ids = torch.arange(P).repeat_interleave(K)

        # 1) rollouts (seed independent of loss_type -> GRPO and Dr.GRPO see identical first rollouts)
        set_seed(int(cfg["seed"]) + u)
        gen = batch_generate(
            policy, tokenizer, batch_msgs, max_prompt, max_new,
            temperature=float(gk["temperature"]), top_p=float(gk["top_p"]), do_sample=bool(gk["do_sample"]),
        )
        gen = normalize_gen(gen)
        _sync()
        t1 = time.perf_counter()

        # 2) rewards + group-relative advantages
        rewards = score_reward_pairs(rm, rm_tok, batch_msgs, gen["responses"], max_length=1024).float().cpu()
        adv_cpu = group_relative_advantages(rewards, group_ids)
        adv = adv_cpu.to(device)
        gstd = rewards.view(P, K).std(dim=1, unbiased=False)
        informative = (gstd > tol).float()
        full_mask = gen["response_mask"].float()
        loss_mask = mask_truncated_sequences(full_mask, gen["truncated"]) if mask_trunc else full_mask
        _sync()
        t2 = time.perf_counter()

        # 3) diagnostics on the rollout policy (eval mode: no dropout noise) and frozen reference (= base)
        with eval_mode(policy):
            pol_lp = chunked_token_logps(policy, gen, lp_chunk)
        with reference_mode(policy):
            ref_lp = chunked_token_logps(policy, gen, lp_chunk)
        st = kl_stats(pol_lp, ref_lp, full_mask)
        _sync()
        t3 = time.perf_counter()

        # 4) gradient step
        common = dict(eps=eps, loss_type=loss_type, max_len=max_new)
        gn_pol = gn_kl = None
        if log_term_grads:  # separate policy-term / KL-term gradient norms (same dropout masks via RNG restore)
            state = _rng_state()
            _, mb_size = _accumulate_with_oom_retry(
                policy, optimizer, gen, adv, loss_mask, ref_lp, beta=0.0, mb_size=mb_size, **common
            )
            g_pol = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            _set_rng_state(state)
        acc, mb_size = _accumulate_with_oom_retry(
            policy, optimizer, gen, adv, loss_mask, ref_lp, beta=beta, mb_size=mb_size, **common
        )
        if log_term_grads:
            g_tot = [p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p) for p in params]
            gn_pol = _flat_norm(g_pol)
            gn_kl = _flat_norm([t - q for t, q in zip(g_tot, g_pol)])
            del g_pol, g_tot
        gnorm = float(torch.nn.utils.clip_grad_norm_(params, max_gn))  # pre-clip norm
        step_skipped = not math.isfinite(gnorm)
        if step_skipped:
            skipped += 1
            optimizer.zero_grad(set_to_none=True)
        else:
            optimizer.step()  # also taken for uninformative groups (KL term + AdamW state still act)
            optimizer.zero_grad(set_to_none=True)
        _sync()
        t4 = time.perf_counter()

        # 5) bookkeeping
        lens = np.array(gen["response_lengths"], float)
        eff_len = loss_mask.sum(-1).cpu().numpy()
        advn = adv_cpu.numpy()
        shares = length_weight_shares(advn, eff_len, lens, K, max_new)
        corrs = [safe_corr(lens[g * K : (g + 1) * K], advn[g * K : (g + 1) * K]) for g in range(P)]
        n_trunc = int(sum(bool(x) for x in gen["truncated"]))
        rec = {
            "run": run_name, "loss_type": loss_type, "seed": int(cfg["seed"]), "update": u,
            "num_generations": K, "prompts_per_update": P,
            "prompt_ids": pids, "prompt_indices": [int(i) for i in idxs],
            "prompt_tokens": ptoks, "prompt_truncated": [t > max_prompt for t in ptoks],
            "reward_mean": float(rewards.mean()), "reward_min": float(rewards.min()),
            "reward_max": float(rewards.max()), "rewards": rewards.tolist(),
            "group_reward_std": float(gstd.mean()), "uninformative_fraction": float(1.0 - informative.mean()),
            "n_truncated": n_trunc, "truncation_rate": n_trunc / N,
            "loss": acc["loss"], "policy_term": acc["policy_term"], "k3_kl_loss": acc["k3_kl"],
            "grad_norm": gnorm, "grad_norm_policy_term": gn_pol, "grad_norm_kl_term": gn_kl,
            "kl": st["kl"], "kl_seq": st["kl_seq"], "kl_k3": st["kl_k3"], "entropy": st["entropy"],
            "length_mean": float(lens.mean()), "length_std": float(lens.std()), "lengths": lens.tolist(),
            "clip_fraction": acc["clip_fraction"], "ratio_mean": acc["ratio_mean"],
            "advantages": advn.tolist(),
            "weight_share_long_half_grpo": shares["grpo"], "weight_share_long_half_dr_grpo": shares["dr_grpo"],
            "len_adv_corr": float(np.nanmean(corrs)) if np.any(~np.isnan(corrs)) else float("nan"),
            "step_skipped": step_skipped, "micro_batch_size": mb_size,
            "time_generate": t1 - t0, "time_reward": t2 - t1, "time_diag": t3 - t2,
            "time_update": t4 - t3, "time_total": t4 - t0,
        }
        append_jsonl(log_path, rec)
        for n in range(N):
            append_jsonl(comp_path, {
                "run": run_name, "update": u, "prompt_id": pids[n // K], "k": n % K,
                "completion": gen["responses"][n], "reward": float(rewards[n]), "advantage": float(advn[n]),
                "length": int(lens[n]), "truncated": bool(gen["truncated"][n]),
            })
        print(
            f"  u{u:02d} R={rec['reward_mean']:.3f} gstd={rec['group_reward_std']:.3f} KL={rec['kl']:.4f} "
            f"len={rec['length_mean']:.0f} trunc={n_trunc}/{N} gn={gnorm:.3f} ent={rec['entropy']:.3f} "
            f"t={rec['time_total']:.1f}s"
        )
        del gen, pol_lp, ref_lp, adv, loss_mask, full_mask

    _sync()
    loop_s = t_loop()
    mem = peak_memory_report()
    policy.save_pretrained(str(out))
    total_s = t_total()
    summary = {
        "run": run_name, "loss_type": loss_type, "seed": int(cfg["seed"]), "updates": n_updates,
        "wall_clock_update_loop_s": loop_s, "wall_clock_total_incl_loading_s": total_s,
        "load_time_s": total_s - loop_s, "peak_vram": mem, "skipped_steps": skipped,
        "final_micro_batch_size": mb_size, "adapter_dir": str(out),
        "prompt_indices": [int(i) for i in selected],
        "hyperparameters": {
            "learning_rate": float(cfg["learning_rate"]), "clip_epsilon": eps, "kl_beta": beta,
            "num_generations": K, "prompts_per_update": P, "max_prompt_length": max_prompt,
            "max_completion_length": max_new, "mask_truncated_completions": mask_trunc,
            "max_grad_norm": max_gn, "generation": dict(gk), "optimizer": "AdamW(default betas, wd=0.01)",
            "policy_epochs": 1, "informative_tol": tol, "base_model": cfg["base_model"],
            "reward_model": cfg["reward_model"], "midpoint": cfg["paths"]["grpo_midpoint_policy"],
        },
    }
    save_json(summary_path, summary)
    print(f"[{run_name}] done: loop {loop_s:.1f}s, total {total_s:.1f}s, peak VRAM {mem.get('sum_over_devices')}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--loss-type", choices=["grpo", "dr_grpo"], default="grpo")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--log-term-grads", action="store_true")
    args = ap.parse_args()
    run_grpo(args.config, args.output, args.updates, args.loss_type, args.run_name, args.seed, args.log_term_grads)


if __name__ == "__main__":
    main()
