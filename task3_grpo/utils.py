"""Shared helpers for Task 3 (additive; nothing in common/ is modified)."""
from __future__ import annotations

import math
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from common.data import prompt_messages, repo_path
from common.generation import response_token_logprobs
from common.metrics import masked_mean, sample_entropy, sampled_kl


# ----------------------------------------------------------------------------- prompts
def row_messages(row: dict) -> list[dict]:
    msgs = list(prompt_messages(row))
    if msgs and isinstance(msgs[-1], dict) and msgs[-1].get("role") == "assistant":
        msgs = msgs[:-1]
    return msgs


def row_prompt_id(row: dict, fallback) -> str:
    return str(row.get("prompt_id", fallback))


def prompt_order(n_rows: int, seed: int) -> list[int]:
    """Seeded permutation of the prompt pool (identical for every loss_type)."""
    return np.random.RandomState(int(seed)).permutation(n_rows).tolist()


def results_dir(cfg: dict) -> Path:
    p = repo_path(cfg["results_dir"])
    p.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- tensors
def normalize_gen(gen: dict) -> dict:
    """batch_generate runs under torch.inference_mode, so its tensors are *inference tensors*, which
    cannot be saved for backward (torch.gather saves its index). Clone them into normal tensors."""
    out = dict(gen)
    for k in ("sequences", "attention_mask", "response_ids", "response_mask"):
        out[k] = gen[k].clone()
    return out


@contextmanager
def eval_mode(model):
    was = model.training
    model.eval()
    try:
        yield
    finally:
        if was:
            model.train()


@torch.no_grad()
def chunked_token_logps(model, gen: dict, chunk: int = 4) -> torch.Tensor:
    """Per-token response log-probs (T=1 logits, as in common.generation) -> CPU float32 [N, T]."""
    seqs, attn, resp = gen["sequences"], gen["attention_mask"], gen["response_ids"]
    out = []
    for i in range(0, seqs.shape[0], chunk):
        lp, _ = response_token_logprobs(
            model, seqs[i : i + chunk], attn[i : i + chunk], gen["prompt_width"], resp[i : i + chunk]
        )
        out.append(lp.float().cpu())
        del lp, _
    return torch.cat(out, 0)


def kl_stats(pol_lp, ref_lp, mask) -> dict:
    """KL via the course helper (token-pooled log-prob difference) + secondary quantities."""
    mask = mask.detach().cpu().float()
    pol = pol_lp.detach().cpu().float()
    ref = ref_lp.detach().cpu().float()
    d = ref - pol
    return {
        "kl": float(sampled_kl(pol, ref, mask)),  # course helper (primary)
        "kl_seq": float(((pol - ref) * mask).sum(-1).mean()),  # sequence-summed (length-confounded)
        "kl_k3": float(masked_mean(torch.exp(d) - d - 1.0, mask)),  # estimator used inside the loss
        "entropy": float(sample_entropy(pol, mask)),  # course helper (sampled-token estimate)
    }


# ----------------------------------------------------------------------------- memory
def reset_peak_memory():
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)


def peak_memory_report() -> dict:
    if not torch.cuda.is_available():
        return {}
    rep, tot_a, tot_r = {}, 0.0, 0.0
    for d in range(torch.cuda.device_count()):
        a = torch.cuda.max_memory_allocated(d) / 2**30
        r = torch.cuda.max_memory_reserved(d) / 2**30
        rep[f"cuda:{d}"] = {"max_allocated_gb": a, "max_reserved_gb": r}
        tot_a += a
        tot_r += r
    rep["sum_over_devices"] = {"max_allocated_gb": tot_a, "max_reserved_gb": tot_r}
    return rep


# ----------------------------------------------------------------------------- statistics
def np_corr(a, b) -> float:
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def bootstrap_ci(values, stat=np.mean, reps=1000, seed=0):
    v = np.asarray(values, float)
    if len(v) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(v), size=(reps, len(v)))
    s = np.array([stat(v[i]) for i in idx])
    return [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))]


def bootstrap_ci_paired(a, b, stat, reps=1000, seed=0):
    a = np.asarray(a, float)
    b = np.asarray(b, float)
    if len(a) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.RandomState(seed)
    idx = rng.randint(0, len(a), size=(reps, len(a)))
    s = np.array([stat(a[i], b[i]) for i in idx])
    return [float(np.nanpercentile(s, 2.5)), float(np.nanpercentile(s, 97.5))]


def length_weight_shares(adv, eff_len, full_len, K: int, max_len: int) -> dict:
    """Analytic share of total |loss weight| that lies on the longer half (by length) of each group.

    Per-sequence total weight: GRPO  = |A_k| (T_k tokens x |A_k|/T_k), Dr.GRPO = |A_k| * T_k / max_len.
    Groups are consecutive blocks of K entries. Returns the mean over groups with non-zero weight.
    """
    adv, eff_len, full_len = (np.asarray(x, float) for x in (adv, eff_len, full_len))
    shares = {"grpo": [], "dr_grpo": []}
    for g in range(len(adv) // K):
        sl = slice(g * K, (g + 1) * K)
        a, e, L = np.abs(adv[sl]), eff_len[sl], full_len[sl]
        w = {"grpo": a * (e > 0), "dr_grpo": a * e / float(max_len)}
        long_idx = np.argsort(-L, kind="stable")[: max(1, K // 2)]
        for key, wv in w.items():
            tot = wv.sum()
            shares[key].append(float(wv[long_idx].sum() / tot) if tot > 0 else float("nan"))
    out = {}
    for key, v in shares.items():
        v = np.array(v, float)
        out[key] = float(np.nanmean(v)) if np.any(~np.isnan(v)) else float("nan")
    return out


def grad_norm(params) -> float:
    s = 0.0
    for p in params:
        if p.grad is not None:
            s += float(p.grad.detach().float().pow(2).sum())
    return math.sqrt(s)
