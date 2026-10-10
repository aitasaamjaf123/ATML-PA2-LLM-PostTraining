"""Shared helpers for Task 2 (PPO): numerics, rollouts, critic forward, logging, orchestration.

Kept free of heavy model imports (no peft / transformers at import time) so the unit tests can import it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.checkpoint import checkpoint

from common.data import repo_path
from common.generation import batch_generate
from common.metrics import safe_corr

SEEDS = [6304, 6305, 6306]
RESULTS_DIR = "results/task2_ppo"
OUTPUT_DIR = "outputs/task2_ppo"

EVAL_KEYS = ["reward_mean", "reward_std", "kl_token", "kl_seq", "entropy", "length_mean", "length_std",
             "truncation_rate", "distinct2", "distinct3", "distinct4"]
STAB_KEYS = ["grad_max", "grad_mean", "ratio_max", "extreme_frac", "approx_kl", "gen_tokens"]


# ------------------------------------------------------------------ naming / json
def fork_name(eps: float, beta: float, seed: int) -> str:
    return f"fork_eps{eps:g}_kl{beta:g}_s{seed}"


def result_path(name: str) -> Path:
    return repo_path(f"{RESULTS_DIR}/{name}.json")


def _json_default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, torch.Tensor):
        return o.detach().cpu().tolist()
    return str(o)


def write_json(path, obj) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_json_default), encoding="utf-8")


def read_json(path):
    p = repo_path(path)
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def update_result_json(name: str, key: str, value) -> None:
    """results/task2_ppo/<name>.json holds {"train": {...}, "eval": {...}}; each script owns one key."""
    d = read_json(result_path(name)) or {}
    d[key] = value
    write_json(result_path(name), d)


def result_has(name: str, key: str) -> bool:
    d = read_json(result_path(name))
    return bool(d) and key in d


def load_eval(name: str):
    d = read_json(result_path(name))
    return d.get("eval") if d else None


def load_metrics(name: str):
    p = repo_path(f"{OUTPUT_DIR}/{name}/metrics.jsonl")
    if not p.exists():
        return None
    rows = []
    with p.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows or None


# ------------------------------------------------------------------ numerics
def set_lora_dropout_zero(model) -> int:
    """Disable LoRA dropout without switching to eval() (which would disable gradient checkpointing)."""
    n = 0
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.p = 0.0
            n += 1
    return n


def cast_trainable_fp32(model) -> int:
    """Trainable params (LoRA, score head) in fp32; the frozen base stays fp16."""
    n = 0
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
            n += p.numel()
    return n


def clamp_new_logp(new_logp, old_logp, limit: float = 20.0):
    """old + clamp(new - old, +-limit). Identical to new_logp (value and gradient) unless |log rho| > limit."""
    return old_logp + (new_logp - old_logp).clamp(-limit, limit)


def microbatches(n: int, size: int):
    return [slice(i, min(i + size, n)) for i in range(0, n, size)]


def _lp(lg, lb):
    lg = lg.float()
    return lg.gather(-1, lb.unsqueeze(-1)).squeeze(-1) - torch.logsumexp(lg, -1)


def _lp_ent(lg, lb):
    lg = lg.float()
    lse = torch.logsumexp(lg, -1)
    lp = lg.gather(-1, lb.unsqueeze(-1)).squeeze(-1) - lse
    ent = lse - (torch.softmax(lg, -1) * lg).sum(-1)
    return lp, ent


def response_logprobs(model, sequences, attention_mask, response_ids, entropy: bool = False, chunk: int = 128):
    """Per-token log pi(a_t|s_t) over the response (raw T=1 logits, as in the released helper).

    Only the last R+1 logit positions are materialised, and the float32 log-softmax is computed in
    sequence chunks (recomputed in backward via checkpointing) so the 152k-vocab tensor never exists in
    full. Returns (logp [B,R], entropy [B,R] or None). Entropy is the exact full-distribution entropy
    and is only computed outside autograd.
    """
    R = response_ids.shape[1]
    kwargs = dict(input_ids=sequences, attention_mask=attention_mask, use_cache=False, return_dict=True)
    try:
        out = model(logits_to_keep=R + 1, **kwargs)
    except TypeError:
        out = model(**kwargs)
    logits = out.logits[:, -(R + 1):-1, :]  # positions P-1 .. P+R-2 predict response tokens 0 .. R-1

    use_ckpt = torch.is_grad_enabled() and logits.requires_grad and not entropy
    lps, ents = [], []
    for s in range(0, R, chunk):
        lg, lb = logits[:, s:s + chunk], response_ids[:, s:s + chunk]
        if entropy:
            lp, en = _lp_ent(lg, lb)
            ents.append(en)
        elif use_ckpt:
            lp = checkpoint(_lp, lg, lb, use_reentrant=False)
        else:
            lp = _lp(lg, lb)
        lps.append(lp)
    return torch.cat(lps, 1), (torch.cat(ents, 1) if entropy else None)


def value_forward(value_model, input_ids, attention_mask, prompt_width: int, R: int):
    """Per-token critic values [B,R] in fp32. Value for response token t is read at position
    prompt_width-1+t, i.e. the hidden state *before* action t (same slice as the logits).

    Bypasses the sequence-classification pooling: runs the backbone (LoRA is injected in place, so
    adapters are active) and applies the scalar head in fp32 to every response position.
    """
    base = value_model.get_base_model() if hasattr(value_model, "get_base_model") else value_model
    backbone = getattr(base, base.base_model_prefix)
    hidden = backbone(input_ids=input_ids, attention_mask=attention_mask, use_cache=False, return_dict=True).last_hidden_state
    hidden = hidden[:, prompt_width - 1: prompt_width - 1 + R, :]
    head = base.score if hasattr(base, "score") else base.classifier
    return head(hidden.float()).squeeze(-1).float()


def value_stats(values, returns, mask):
    """Critic diagnostics over valid tokens: explained variance, mean V, mean return, corr(V, return)."""
    m = mask.bool()
    v, r = values[m].float(), returns[m].float()
    ev = 1.0 - (r - v).var(unbiased=False) / r.var(unbiased=False).clamp_min(1e-8)
    corr = safe_corr(v.cpu().numpy(), r.cpu().numpy())
    return float(ev), float(v.mean()), float(r.mean()), corr


# ------------------------------------------------------------------ rollouts
def rollout(policy, tokenizer, msgs, cfg, max_new_tokens: int, seed: int | None = None):
    """Sample responses with the release decoding config and return plain (non-inference) tensors."""
    if seed is not None:
        torch.manual_seed(int(seed))
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(int(seed))
    gc_ = cfg.get("generation", {})
    prev = policy.config.use_cache
    policy.config.use_cache = True  # load_policy turns it off for training; generation needs the KV cache
    try:
        g = batch_generate(
            policy, tokenizer, msgs, int(cfg["max_prompt_length"]), int(max_new_tokens),
            temperature=float(gc_.get("temperature", 0.7)), top_p=float(gc_.get("top_p", 0.9)),
            do_sample=bool(gc_.get("do_sample", True)),
        )
    finally:
        policy.config.use_cache = prev
    # tensors made under inference_mode cannot be saved for backward -> clone into normal tensors
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        g[k] = g[k].clone()
    return g


def count_long_prompts(tokenizer, msgs_list, limit: int) -> int:
    n = 0
    for m in msgs_list:
        text = tokenizer.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        if len(tokenizer(text, add_special_tokens=False)["input_ids"]) > limit:
            n += 1
    return n


# ------------------------------------------------------------------ orchestration
def run_module(module: str, args, cuda_device: str = "0") -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = cuda_device
    cmd = [sys.executable, "-m", module] + [str(a) for a in args]
    print(">>", " ".join(cmd), flush=True)
    subprocess.run(cmd, cwd=str(repo_path(".")), env=env, check=True)


def run_condition(eps: float, beta: float, seed: int, config: str, updates: int, force: bool = False) -> str:
    """Train one fork from the supplied midpoint, then evaluate it; each in its own process.
    Finished stages are skipped (the shared eps=0.2, beta=0.1 run is therefore trained once)."""
    name = fork_name(eps, beta, seed)
    if force or not result_has(name, "train"):
        run_module("task2_ppo.continue_train", ["--config", config, "--run-name", name,
                                                "--output", f"{OUTPUT_DIR}/{name}", "--updates", updates,
                                                "--clip-epsilon", eps, "--kl-beta", beta, "--seed", seed])
    if force or not result_has(name, "eval"):
        run_module("task2_ppo.evaluate", ["--config", config, "--adapter", f"{OUTPUT_DIR}/{name}/adapter",
                                          "--name", name])
    return name


def run_grid(conditions, seeds, config: str, updates: int, force: bool = False):
    """conditions: list of dicts {label, eps, beta}. Seed-major order so partial runs give complete seed sets."""
    failed = []
    for s in seeds:
        for c in conditions:
            try:
                run_condition(c["eps"], c["beta"], s, config, updates, force)
            except subprocess.CalledProcessError as e:
                print(f"[FAILED] {c['label']} seed {s}: {e}", flush=True)
                failed.append((c["label"], s))
    return failed


# ------------------------------------------------------------------ summaries
def agg(values):
    arr = np.array([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": None, "std": None, "n": 0}
    return {"mean": float(arr.mean()), "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0, "n": int(arr.size)}


def stability_stats(rows):
    """H5 stability statistics for one training run (all from the per-update logs)."""
    def col(k):
        return np.array([r.get(k, np.nan) for r in rows], dtype=float)
    g = col("grad_norm_policy")
    return {
        "grad_max": float(np.nanmax(g)), "grad_mean": float(np.nanmean(g)),
        "ratio_max": float(np.nanmax(col("max_ratio"))),
        "extreme_frac": float(np.nanmean(col("extreme_ratio_frac"))),
        "approx_kl": float(np.nanmean(col("approx_kl_old_new"))),
        "gen_tokens": float(np.nansum(col("generated_tokens"))),
    }


def _eval_rows(name: str):
    p = repo_path(f"{OUTPUT_DIR}/{name}_eval.jsonl")
    if not p.exists():
        return None
    with p.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reward_length_corr(name: str):
    rows = _eval_rows(name)
    if not rows:
        return None
    return safe_corr([r["reward"] for r in rows], [r["length"] for r in rows])


def candidate_cases(name: str, baseline: str = "midpoint", k: int = 3):
    """Per-prompt reward/length changes vs the midpoint, to help pick qualitative examples (J1).
    Labels are only candidates; the final reward-vs-quality judgement is made by reading the text."""
    a, b = _eval_rows(name), _eval_rows(baseline)
    if not a or not b:
        return None
    base = {r["prompt_id"]: r for r in b}
    joined = []
    for r in a:
        o = base.get(r["prompt_id"])
        if o is None:
            continue
        joined.append({
            "prompt_id": r["prompt_id"], "d_reward": r["reward"] - o["reward"],
            "reward_base": o["reward"], "reward_new": r["reward"],
            "len_base": o["length"], "len_new": r["length"],
            "d_len_rel": (r["length"] - o["length"]) / max(o["length"], 1),
            "truncated_new": r.get("truncated"), "response_head": r["response"][:400],
            "baseline_head": o["response"][:400],
        })
    up_long = sorted([j for j in joined if j["d_reward"] > 0 and j["d_len_rel"] > 0.25], key=lambda j: -j["d_reward"])[:k]
    up_same = sorted([j for j in joined if j["d_reward"] > 0 and j["d_len_rel"] <= 0.05], key=lambda j: -j["d_reward"])[:k]
    down = sorted([j for j in joined if j["d_reward"] < 0], key=lambda j: j["d_reward"])[:k]
    return {"reward_up_and_longer": up_long, "reward_up_length_flat": up_same, "reward_down": down}


def summarize_study(conditions, seeds, out_name: str, baseline: str = "midpoint"):
    """Aggregate a fork study across seeds -> results/task2_ppo/<out_name>.json."""
    summary = {"seeds": list(seeds), "conditions": {}, "token_budget": {}}
    base = load_eval(baseline)
    if base:
        summary["baseline_midpoint"] = {k: base.get(k) for k in EVAL_KEYS}
    gen_tokens = {s: {} for s in seeds}
    for c in conditions:
        entry = {"epsilon": c["eps"], "beta": c["beta"], "per_seed": {}, "eval": {}, "stability": {}}
        for s in seeds:
            run = fork_name(c["eps"], c["beta"], s)
            ev, rows = load_eval(run), load_metrics(run)
            ps = {"run": run}
            if ev:
                ps["eval"] = {k: ev.get(k) for k in EVAL_KEYS}
                ps["corr_reward_length"] = reward_length_corr(run)
            if rows:
                ps["stability"] = stability_stats(rows)
                gen_tokens[s][c["label"]] = ps["stability"]["gen_tokens"]
            entry["per_seed"][str(s)] = ps
        for k in EVAL_KEYS:
            entry["eval"][k] = agg([entry["per_seed"][str(s)].get("eval", {}).get(k) for s in seeds])
        for k in STAB_KEYS:
            entry["stability"][k] = agg([entry["per_seed"][str(s)].get("stability", {}).get(k) for s in seeds])
        entry["corr_reward_length"] = agg([entry["per_seed"][str(s)].get("corr_reward_length") for s in seeds])
        first = next((s for s in seeds if entry["per_seed"][str(s)].get("eval")), None)
        if first is not None:
            entry["candidate_cases_seed"] = first
            entry["candidate_cases"] = candidate_cases(entry["per_seed"][str(first)]["run"], baseline)
        summary["conditions"][c["label"]] = entry
    for s, d in gen_tokens.items():
        if len(d) > 1:
            vals = np.array(list(d.values()), dtype=float)
            spread = float((vals.max() - vals.min()) / vals.mean())
            summary["token_budget"][str(s)] = {"generated_tokens": d, "relative_spread": spread, "flag_gt_10pct": spread > 0.10}
    write_json(f"{RESULTS_DIR}/{out_name}.json", summary)
    return summary
