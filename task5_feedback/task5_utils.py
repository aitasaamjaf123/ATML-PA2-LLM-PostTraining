"""Shared helpers for Task 5. Course files (rlaif.py, rlvr.py, common/*) are NOT modified."""
from __future__ import annotations

import glob
import json
import math
import platform
import re
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

from common.data import REPO_ROOT, repo_path
from common.models import load_policy

OUT_DIR = "outputs/task5_feedback"
CACHE_PATH = f"{OUT_DIR}/judge_cache.json"
POLICY_NAMES = ["sft", "rlvr", "rlaif"]


def out_path(name: str) -> str:
    return f"{OUT_DIR}/{name}"


# ---------------------------------------------------------------- json io
def _default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def save(path: str, obj) -> None:
    p = repo_path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")


def load(path: str):
    return json.loads(repo_path(path).read_text(encoding="utf-8"))


def exists(path: str) -> bool:
    return repo_path(path).exists()


# ---------------------------------------------------------------- adapters
def resolve_adapter(path):
    """Search order: repo/<path> -> /kaggle/working/{adapters,checkpoints,.}/<name> -> /kaggle/input/**/<name>."""
    if path is None:
        return None
    name = Path(path).name
    cands = [repo_path(path),
             Path("/kaggle/working/adapters") / name,
             Path("/kaggle/working/checkpoints") / name,
             Path("/kaggle/working") / name]
    cands += [Path(p) for p in glob.glob(f"/kaggle/input/*/{name}")]
    cands += [Path(p) for p in glob.glob(f"/kaggle/input/*/*/{name}")]
    for c in cands:
        if (c / "adapter_config.json").exists():
            print(f"[adapter] {name} -> {c}")
            return str(c)
    raise FileNotFoundError(f"Adapter '{name}' not found. Looked in: " + ", ".join(map(str, cands)))


def load_named_policy(cfg: dict, name: str):
    specs = {"sft": None, "rlvr": cfg["policies"]["rlvr"], "rlaif": cfg["policies"]["rlaif"]}
    if name not in specs:
        raise KeyError(name)
    return load_policy(cfg, adapter_path=resolve_adapter(specs[name]), trainable=False)


# ---------------------------------------------------------------- B5 assertions
def assert_template_behaviour(tokenizer) -> None:
    """data.py needs list[int] from tokenize=True; rlaif.py needs a tensor from return_tensors='pt'."""
    msg = [{"role": "user", "content": "hi"}]
    a = tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
    assert isinstance(a, list) and all(isinstance(x, int) for x in a), \
        f"apply_chat_template(tokenize=True) returned {type(a)}; transformers version mismatch"
    b = tokenizer.apply_chat_template(msg, return_tensors="pt", add_generation_prompt=True)
    assert torch.is_tensor(b), f"apply_chat_template(return_tensors='pt') returned {type(b)}"


# ---------------------------------------------------------------- statistics
def bootstrap_ci(x, n_boot=10000, seed=6304, alpha=0.05):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return [float("nan"), float("nan")]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(n_boot, len(x)))
    m = x[idx].mean(1)
    return [float(np.quantile(m, alpha / 2)), float(np.quantile(m, 1 - alpha / 2))]


def paired_bootstrap(a, b, n_boot=10000, seed=6304):
    d = np.asarray(a, float) - np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n_boot, len(d)))
    m = d[idx].mean(1)
    p = 2 * min((m <= 0).mean(), (m >= 0).mean())
    return {"diff": float(d.mean()), "ci": [float(np.quantile(m, .025)), float(np.quantile(m, .975))],
            "p_boot": float(min(1.0, p))}


def mcnemar_exact(a, b):
    from scipy.stats import binomtest
    a = np.asarray(a, int)
    b = np.asarray(b, int)
    b01 = int(((a == 1) & (b == 0)).sum())
    b10 = int(((a == 0) & (b == 1)).sum())
    n = b01 + b10
    p = float(binomtest(b01, n, 0.5).pvalue) if n > 0 else 1.0
    return {"a_only": b01, "b_only": b10, "p": p}


def wilson(k, n, z=1.96):
    if n == 0:
        return [float("nan"), float("nan")]
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [float(c - h), float(c + h)]


def kappa(v, j, labels=("win", "tie", "loss")):
    try:
        from sklearn.metrics import cohen_kappa_score
        if len(set(v) | set(j)) < 2:
            return float("nan")
        return float(cohen_kappa_score(list(v), list(j), labels=list(labels)))
    except Exception:
        return float("nan")


def confusion(v, j, labels=("win", "tie", "loss")):
    m = [[0] * len(labels) for _ in labels]
    for a, b in zip(v, j):
        m[labels.index(a)][labels.index(b)] += 1
    return m


# ---------------------------------------------------------------- judge helpers
def make_judge(cfg: dict, cache_path: str = CACHE_PATH):
    from task5_feedback.rlaif import PairwiseAIJudge
    j = PairwiseAIJudge(cfg, cache_path)
    assert_template_behaviour(j.tokenizer)
    return j


def judge_compare(judge, problem: str, a: str, b: str):
    """Supplied single-call protocol. Returns (result in {A,B,TIE}, meta)."""
    cached = judge._key(problem, a, b) in judge.cache
    t = time.perf_counter()
    res = judge.compare(problem, a, b)
    return res, {"cached": bool(cached), "seconds": time.perf_counter() - t}


@torch.no_grad()
def raw_judge_output(judge, problem: str, a: str, b: str):
    """Replicates PairwiseAIJudge.compare generation to expose the raw decoded text (B3 audit)."""
    from task5_feedback.rlaif import PAIRWISE_RUBRIC
    key = judge._key(problem, a, b)
    swap = int(key[:8], 16) % 2 == 1
    aa, bb = (b, a) if swap else (a, b)
    text = PAIRWISE_RUBRIC.format(problem=problem, a=aa, b=bb)
    ids = judge.tokenizer.apply_chat_template(
        [{"role": "user", "content": text}], return_tensors="pt", add_generation_prompt=True
    ).to(next(judge.model.parameters()).device)
    out = judge.model.generate(ids, max_new_tokens=4, do_sample=False,
                               pad_token_id=judge.tokenizer.eos_token_id,
                               eos_token_id=judge.tokenizer.eos_token_id)
    dec = judge.tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip().upper()
    return dec, re.search(r"\b(A|B|TIE)\b", dec) is not None


def parse_audit(judge, tie_items, max_n=50, seed=6304):
    """tie_items: list of (problem, a, b) whose cached verdict was TIE. Reports how many were parse failures."""
    rng = np.random.default_rng(seed)
    items = list(tie_items)
    if len(items) > max_n:
        items = [items[i] for i in rng.choice(len(items), max_n, replace=False)]
    fails, raws = 0, []
    for p, a, b in items:
        dec, ok = raw_judge_output(judge, p, a, b)
        fails += int(not ok)
        raws.append(dec)
    return {"n_ties_total": len(tie_items), "n_audited": len(items), "n_parse_fail": fails,
            "parse_fail_rate_among_ties": (fails / len(items)) if items else 0.0,
            "raw_outputs_sample": raws[:10]}


# ---------------------------------------------------------------- misc
def env_info(cfg: dict, extra: dict | None = None) -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT,
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = None
    import transformers, peft
    info = {"seed": cfg["seed"], "git_commit": commit, "python": platform.python_version(),
            "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "base_model": cfg["base_model"], "ai_judge_model": cfg["ai_judge_model"],
            "policies": cfg["policies"], "config_math_max_new_tokens": cfg["math_max_new_tokens"]}
    if extra:
        info.update(extra)
    return info


def classify_failure(resp: str, gold: str, truncated: bool) -> str:
    from task5_feedback.rlvr import exact_reward, extract_designated_final, numerically_equal, _FINAL_RE
    if exact_reward(resp, gold) == 1.0:
        return "correct"
    if truncated:
        return "truncated"
    if extract_designated_final(resp) is None:
        return "no_designated_final"
    ms = list(_FINAL_RE.finditer(resp))
    body = resp[:ms[-1].start()] + resp[ms[-1].end():]
    for tok in re.findall(r"[-+]?\d[\d,]*(?:\.\d+)?", body):
        if numerically_equal(tok.replace(",", ""), gold):
            return "wrong_final_gold_in_reasoning"
    return "wrong_final_other"


def trunc(s: str, n: int = 600) -> str:
    return s if len(s) <= n else s[:n] + " ...[truncated]"
