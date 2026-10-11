from __future__ import annotations

import gc
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path

from common.data import repo_path

POLICIES = ["sft", "dpo", "ppo", "grpo"]
LABELS = ["SAFE_ANSWER", "JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE", "OVER_REFUSAL", "AMBIGUOUS"]
ADAPTER_TAG = {"dpo": "task1_dpo", "ppo": "task2_ppo", "grpo": "task3_grpo"}


def task4_dir(cfg) -> Path:
    override = os.environ.get("TASK4_RESULTS_DIR")
    p = Path(override) if override else repo_path(cfg["results_dir"]) / "task4_safety"
    p.mkdir(parents=True, exist_ok=True)
    return p


def _is_adapter_dir(p: Path) -> bool:
    return (p / "adapter_config.json").is_file() and (p / "adapter_model.safetensors").is_file()


def _candidate_roots():
    roots = []
    env = os.environ.get("ADAPTER_ROOT")
    if env:
        roots.append(Path(env))
    roots.append(repo_path("."))
    roots.append(Path("/kaggle/working"))
    inp = Path("/kaggle/input")
    if inp.is_dir():
        base = len(inp.parts)
        for dirpath, dirnames, _ in os.walk(inp):
            roots.append(Path(dirpath))
            if len(Path(dirpath).parts) - base >= 6:
                dirnames[:] = []
    return roots


def resolve_adapter(policy: str, configured):
    """Return a verified adapter directory, or None for SFT. HARD-FAILS if not found."""
    if configured is None:
        if policy != "sft":
            raise ValueError(f"Policy '{policy}' has no adapter configured; refusing to fall back to base model.")
        return None
    if policy == "sft":
        raise ValueError("SFT must be the untouched base model (no adapter).")
    norm = str(configured).replace("\\", "/").strip("/")
    tag = ADAPTER_TAG[policy]
    if not norm.endswith(f"{tag}/standard"):
        raise ValueError(f"Task 4 must use the STANDARD checkpoint ({tag}/standard); got '{configured}'.")
    parts = Path(norm).parts
    stripped = parts[1:] if parts[0] == "outputs" else parts
    variants = []
    for v in [parts, stripped, stripped[:-1], parts[:-1]]:
        v = Path(*v)
        if v not in variants:
            variants.append(v)
    roots = _candidate_roots()
    for v in variants:
        for r in roots:
            cand = r / v
            if _is_adapter_dir(cand):
                return cand.resolve()
    raise FileNotFoundError(
        f"Adapter for '{policy}' not found (needs adapter_config.json + adapter_model.safetensors).\n"
        f"Tried sub-paths: {[str(v) for v in variants]}\n"
        f"Under roots (first 15): {[str(r) for r in roots[:15]]}\n"
        "Set ADAPTER_ROOT or attach the Kaggle dataset."
    )


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_adapter(path: Path, base_model: str) -> dict:
    cfgj = json.loads((path / "adapter_config.json").read_text(encoding="utf-8"))
    if cfgj.get("base_model_name_or_path") != base_model:
        raise RuntimeError(f"Adapter {path} base model {cfgj.get('base_model_name_or_path')} != {base_model}")
    from safetensors.torch import load_file
    tensors = load_file(str(path / "adapter_model.safetensors"))
    bs = [t for k, t in tensors.items() if "lora_B" in k]
    if not bs:
        raise RuntimeError(f"No lora_B tensors in {path}")
    mx = max(float(t.float().abs().max()) for t in bs)
    if mx == 0.0:
        raise RuntimeError(f"All lora_B weights are zero in {path}: adapter is a no-op (same as SFT).")
    return {
        "path": str(path),
        "sha256": sha256_file(path / "adapter_model.safetensors"),
        "n_lora_B_tensors": len(bs),
        "max_abs_lora_B": mx,
        "r": cfgj.get("r"),
        "lora_alpha": cfgj.get("lora_alpha"),
        "target_modules": sorted(cfgj.get("target_modules", [])),
    }


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_path(".")),
                                       stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        return "unknown"


def env_meta() -> dict:
    import torch, transformers, peft
    return {
        "torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__,
        "cuda": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "git_commit": git_commit(), "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }


def update_json(path: Path, key: str, value) -> None:
    obj = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    obj[key] = value
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=str), encoding="utf-8")


def cleanup_gpu():
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass