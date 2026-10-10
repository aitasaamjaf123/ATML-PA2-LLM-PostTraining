from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")  # single GPU so peak-VRAM numbers are meaningful

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from torch.optim import AdamW

from common.data import load_yaml, prompt_messages, read_jsonl, repo_path
from common.generation import score_reward_pairs
from common.logging_utils import append_jsonl, set_seed
from common.metrics import masked_mean, mean_response_length
from common.models import (
    load_policy,
    load_reward_model,
    load_tokenizer,
    load_value_model,
    reference_mode,
    trainable_parameters,
    value_parameter_groups,
)
from task2_ppo import utils as U
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)

# Task 2 design choices that are not in the release config (all logged in the "train" results block).
DEFAULT_ROLLOUT_PROMPTS = 4   # distinct prompts per update, 1 sample each (config's prompts_per_update=1 is overridden)
DEFAULT_MICRO_BATCH = 2       # sequences per forward/backward pass (gradient accumulation inside one PPO epoch)
LOG_RATIO_CLAMP = 20.0
GRAD_SCALER_INIT = 1024.0


def prepare_ppo_continuation(config_path: str, overrides: dict | None = None):
    cfg = load_yaml(config_path)
    if overrides:
        cfg.update(overrides)
    set_seed(int(cfg["seed"]))

    tokenizer = load_tokenizer(cfg["base_model"])
    tokenizer.truncation_side = "left"  # keep the generation prompt if a prompt exceeds max_prompt_length

    t0 = time.perf_counter()
    policy = load_policy(
        cfg,
        adapter_path=cfg["paths"]["ppo_midpoint_policy"],
        trainable=True,
    )
    value_model = load_value_model(
        cfg,
        cfg["paths"]["ppo_midpoint_value"],
        train_mode=cfg.get("value_train_mode", "head_only"),
    )
    reward_model, reward_tokenizer = load_reward_model(cfg)
    reward_tokenizer.truncation_side = "left"

    # Deliberate deviations from the config, documented in the report: LoRA dropout off (otherwise old and
    # new log-probs differ at update 0), trainable params in fp32 (fp16 AdamW is unstable).
    U.set_lora_dropout_zero(policy)
    U.set_lora_dropout_zero(value_model)
    U.cast_trainable_fp32(policy)
    U.cast_trainable_fp32(value_model)
    load_time = time.perf_counter() - t0

    prompts = read_jsonl(cfg["paths"]["rl_prompt_train"])

    policy_optimizer = AdamW(
        trainable_parameters(policy),
        lr=float(cfg["policy_learning_rate"]),
    )
    value_optimizer = AdamW(
        value_parameter_groups(
            value_model,
            lora_lr=float(cfg["value_lora_learning_rate"]),
            head_lr=float(cfg["value_head_learning_rate"]),
        ),
        weight_decay=0.0,
    )

    return {
        "cfg": cfg,
        "tokenizer": tokenizer,
        "policy": policy,
        "value_model": value_model,
        "reward_model": reward_model,
        "reward_tokenizer": reward_tokenizer,
        "prompt_rows": prompts,
        "policy_optimizer": policy_optimizer,
        "value_optimizer": value_optimizer,
        "load_time_s": load_time,
    }


def build_prompt_schedule(rows: list[dict], seed: int, n_prompts: int) -> list[dict]:
    """Seeded permutation of the training pool; shorter runs use a prefix of the same sequence."""
    perm = np.random.RandomState(int(seed)).permutation(len(rows))
    return [rows[int(i)] for i in perm[:n_prompts]]


def make_scaler():
    try:
        return torch.amp.GradScaler("cuda", init_scale=GRAD_SCALER_INIT, growth_interval=1000)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(init_scale=GRAD_SCALER_INIT, growth_interval=1000)


def scaled_step(scaler, optimizer, params, max_norm: float):
    scaler.unscale_(optimizer)
    gn = torch.nn.utils.clip_grad_norm_(params, max_norm)  # returns the pre-clip norm
    before = scaler.get_scale()
    scaler.step(optimizer)
    scaler.update()
    return float(gn), bool(scaler.get_scale() < before)  # (grad norm, step skipped because of inf/nan)


def run_ppo(config_path: str, output: str | None = None, updates: int | None = None, clip_epsilon: float | None = None, kl_beta: float | None = None, run_name: str = "standard", seed: int | None = None):
    bundle = prepare_ppo_continuation(config_path, {"seed": int(seed)} if seed is not None else None)
    cfg = bundle["cfg"]
    if updates is not None:
        cfg["updates"] = int(updates)
    if clip_epsilon is not None:
        cfg["clip_epsilon"] = float(clip_epsilon)
    if kl_beta is not None:
        cfg["kl_beta"] = float(kl_beta)
    out = repo_path(output or f"outputs/task2_ppo/{run_name}")
    out.mkdir(parents=True, exist_ok=True)
    metrics_path = out / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()  # append_jsonl appends; always start a run from a clean log

    tok, policy, value_model = bundle["tokenizer"], bundle["policy"], bundle["value_model"]
    rm, rm_tok = bundle["reward_model"], bundle["reward_tokenizer"]
    pol_opt, val_opt = bundle["policy_optimizer"], bundle["value_optimizer"]

    seed_ = int(cfg["seed"])
    eps, beta = float(cfg["clip_epsilon"]), float(cfg["kl_beta"])
    n_updates = int(cfg["updates"])
    ppo_epochs = int(cfg["ppo_epochs"])
    gamma, lam = float(cfg["gamma"]), float(cfg["gae_lambda"])
    value_coef, max_gn = float(cfg["value_coef"]), float(cfg["max_grad_norm"])
    per_update = int(cfg.get("rollout_prompts", DEFAULT_ROLLOUT_PROMPTS))
    mb_size = int(cfg.get("micro_batch_size", DEFAULT_MICRO_BATCH))

    schedule = build_prompt_schedule(bundle["prompt_rows"], seed_, n_updates * per_update)
    n_long = U.count_long_prompts(tok, [prompt_messages(r) for r in schedule], int(cfg["max_prompt_length"]))
    print(f"[{run_name}] seed={seed_} eps={eps} beta={beta} updates={n_updates} prompts/update={per_update} "
          f"(prompts over {cfg['max_prompt_length']} tokens: {n_long})", flush=True)

    device = next(policy.parameters()).device
    pol_params = trainable_parameters(policy)
    val_params = [p for g in val_opt.param_groups for p in g["params"]]
    scaler_p, scaler_v = make_scaler(), make_scaler()

    torch.cuda.synchronize()
    resident_gb = torch.cuda.memory_allocated() / 2**30
    loop_start = time.perf_counter()
    skipped_total, total_tokens, peak_gb_all = 0, 0, 0.0

    for u in range(n_updates):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t_u = time.perf_counter()

        rows_u = schedule[u * per_update:(u + 1) * per_update]
        msgs = [prompt_messages(r) for r in rows_u]

        # ---- 1. rollout + learned reward (raw RM score, no EOS penalty, no normalisation)
        gen = U.rollout(policy, tok, msgs, cfg, int(cfg["max_response_length"]), seed=seed_ * 1000 + u)
        seqs, attn, resp, mask = gen["sequences"], gen["attention_mask"], gen["response_ids"], gen["response_mask"]
        P, R, B = gen["prompt_width"], resp.shape[1], resp.shape[0]
        rm_scores = score_reward_pairs(rm, rm_tok, msgs, gen["responses"], max_length=int(cfg["reward_max_length"])).to(device)
        mbs = U.microbatches(B, mb_size)

        # ---- 2. old log-probs / entropy, reference log-probs, old values (no grad)
        policy.eval()
        value_model.eval()
        old_lp, ent, ref_lp, old_v = [], [], [], []
        with torch.no_grad():
            for sl in mbs:
                lp, en = U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl], entropy=True)
                old_lp.append(lp)
                ent.append(en)
            with reference_mode(policy):  # adapter disabled -> frozen base model is the reference
                for sl in mbs:
                    rlp, _ = U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl])
                    ref_lp.append(rlp)
            for sl in mbs:
                old_v.append(U.value_forward(value_model, seqs[sl], attn[sl], P, R))
        old_lp, ent, ref_lp, old_v = torch.cat(old_lp), torch.cat(ent), torch.cat(ref_lp), torch.cat(old_v)
        policy.train()
        value_model.train()

        # ---- 3. KL-shaped rewards, GAE, returns, whitened advantages
        rewards = shaped_rewards(rm_scores, old_lp, ref_lp, mask, beta)
        adv_raw, returns = compute_gae(rewards, old_v, mask, gamma=gamma, lam=lam)
        adv = normalize_advantages(adv_raw, mask)
        ev, v_mean, ret_mean, v_corr = U.value_stats(old_v, returns, mask)
        n_tok = mask.sum()

        # ---- 4. PPO epochs (full batch per epoch, micro-batched with exact token weighting)
        ep_policy_loss, ep_value_loss, ep_clip, ep_gn_p, ep_gn_v, skipped = [], [], [], [], [], 0
        for _ in range(ppo_epochs):
            pol_opt.zero_grad(set_to_none=True)
            l_acc, c_acc = 0.0, 0.0
            for sl in mbs:
                new_lp, _ = U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl])
                new_lp = U.clamp_new_logp(new_lp, old_lp[sl], LOG_RATIO_CLAMP)
                loss, _ratio, cfrac = ppo_policy_loss(new_lp, old_lp[sl], adv[sl], mask[sl], eps=eps)
                w = mask[sl].sum() / n_tok
                scaler_p.scale(loss * w).backward()
                l_acc += float(loss.detach()) * float(w)
                c_acc += float(cfrac) * float(w)
            gn_p, sk = scaled_step(scaler_p, pol_opt, pol_params, max_gn)
            skipped += int(sk)
            ep_policy_loss.append(l_acc)
            ep_clip.append(c_acc)
            ep_gn_p.append(gn_p)

            val_opt.zero_grad(set_to_none=True)
            v_acc = 0.0
            for sl in mbs:
                pred = U.value_forward(value_model, seqs[sl], attn[sl], P, R)
                vloss = value_mse_loss(pred, returns[sl], mask[sl])
                w = mask[sl].sum() / n_tok
                scaler_v.scale(value_coef * vloss * w).backward()
                v_acc += float(vloss.detach()) * float(w)
            gn_v, sk = scaled_step(scaler_v, val_opt, val_params, max_gn)
            skipped += int(sk)
            ep_value_loss.append(v_acc)
            ep_gn_v.append(gn_v)

        # ---- 5. post-update diagnostics: how far did theta move from theta_old on this batch?
        policy.eval()
        with torch.no_grad():
            new_lp_all = torch.cat([U.response_logprobs(policy, seqs[sl], attn[sl], resp[sl])[0] for sl in mbs])
        policy.train()
        log_r = (new_lp_all - old_lp).clamp(-LOG_RATIO_CLAMP, LOG_RATIO_CLAMP)
        r = log_r.exp()
        valid = mask.bool()

        torch.cuda.synchronize()
        wall = time.perf_counter() - t_u
        peak_gb = torch.cuda.max_memory_allocated() / 2**30
        peak_gb_all = max(peak_gb_all, peak_gb)
        skipped_total += skipped
        gen_tokens = int(mask.sum().item())
        total_tokens += gen_tokens

        rec = {
            "update": u + 1,
            "prompt_ids": [r_["prompt_id"] for r_ in rows_u],
            "reward_raw": float(rm_scores.mean()),
            "reward_std": float(rm_scores.std(unbiased=False)),
            "kl_token": float(masked_mean(old_lp - ref_lp, mask)),               # released token-pooled estimator
            "kl_seq": float(((old_lp - ref_lp) * mask).sum(-1).mean()),          # per-sequence summed KL
            "policy_loss": float(np.mean(ep_policy_loss)),
            "value_loss": float(np.mean(ep_value_loss)),
            "entropy": float(masked_mean(ent, mask)),                            # exact full-distribution entropy
            "clip_fraction": float(np.mean(ep_clip)),
            "clip_fraction_epochs": ep_clip,
            "grad_norm_policy": float(np.mean(ep_gn_p)),                         # pre-clip
            "grad_norm_value": float(np.mean(ep_gn_v)),
            "response_length": mean_response_length(mask),
            "truncation_rate": float(np.mean(gen["truncated"])),
            "approx_kl_old_new": float(masked_mean((r - 1.0) - log_r, mask)),    # k3 estimator of KL(old||new)
            "approx_kl_old_new_k1": float(masked_mean(-log_r, mask)),
            "max_ratio": float(r[valid].max()),
            "extreme_ratio_frac": float(masked_mean(((r < 0.5) | (r > 2.0)).float(), mask)),
            "explained_variance": ev,
            "value_mean": v_mean,
            "return_mean": ret_mean,
            "value_return_corr": v_corr,
            "wall_time_s": wall,
            "peak_vram_gb": peak_gb,
            "generated_tokens": gen_tokens,
            "skipped_optimizer_steps": skipped,
        }
        append_jsonl(metrics_path, rec)
        print(f"[{run_name}] u{u + 1:02d} R={rec['reward_raw']:.3f} KL={rec['kl_token']:.4f} H={rec['entropy']:.3f} "
              f"len={rec['response_length']:.0f} clip={rec['clip_fraction']:.4f} gn={rec['grad_norm_policy']:.3f} "
              f"vl={rec['value_loss']:.3f} EV={ev:.2f} t={wall:.0f}s vram={peak_gb:.1f}G skip={skipped}", flush=True)

        del gen, old_lp, ent, ref_lp, old_v, rewards, adv_raw, returns, adv, new_lp_all, log_r, r
        torch.cuda.empty_cache()

    torch.cuda.synchronize()
    loop_s = time.perf_counter() - loop_start
    policy.save_pretrained(str(out / "adapter"))

    summary = {
        "run_name": run_name, "seed": seed_, "clip_epsilon": eps, "kl_beta": beta, "updates": n_updates,
        "rollout_prompts_per_update": per_update, "samples_per_prompt": 1, "ppo_epochs": ppo_epochs,
        "micro_batch_size": mb_size, "policy_lr": float(cfg["policy_learning_rate"]),
        "value_lora_lr": float(cfg["value_lora_learning_rate"]), "value_head_lr": float(cfg["value_head_learning_rate"]),
        "gamma": gamma, "gae_lambda": lam, "value_coef": value_coef, "max_grad_norm": max_gn,
        "max_response_length": int(cfg["max_response_length"]), "generation": cfg["generation"],
        "log_ratio_clamp": LOG_RATIO_CLAMP, "grad_scaler_init": GRAD_SCALER_INIT, "lora_dropout_used": 0.0,
        "prompt_ids": [r_["prompt_id"] for r_ in schedule], "source_indices": [r_["source_index"] for r_ in schedule],
        "n_prompts_over_max_prompt_length": n_long,
        "load_time_s": bundle["load_time_s"], "loop_wall_clock_s": loop_s,
        "peak_vram_gb": peak_gb_all, "resident_vram_gb": resident_gb,
        "total_generated_tokens": total_tokens, "skipped_optimizer_steps": skipped_total,
        "adapter_path": str(out / "adapter"),
        "versions": _versions(),
    }
    U.update_result_json(run_name, "train", summary)
    print(f"[{run_name}] done: loop {loop_s / 60:.1f} min, peak {peak_gb_all:.2f} GiB, adapter -> {out / 'adapter'}", flush=True)
    return summary


def _versions():
    import peft
    import transformers
    return {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--output")
    ap.add_argument("--updates", type=int)
    ap.add_argument("--clip-epsilon", type=float)
    ap.add_argument("--kl-beta", type=float)
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--seed", type=int)
    args = ap.parse_args()
    run_ppo(args.config, args.output, args.updates, args.clip_epsilon, args.kl_beta, args.run_name, args.seed)


if __name__ == "__main__":
    main()