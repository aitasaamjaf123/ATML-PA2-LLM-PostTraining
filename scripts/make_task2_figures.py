"""Build Task 2 figures and tables from saved logs.

Run from the repo root:  python -m scripts.make_task2_figures

Inputs (the Task 2 code must write exactly these):
  outputs/task2_ppo/<run>/metrics.jsonl     one JSON record per PPO update
  results/task2_ppo/<run>.json              eval summary (either top-level or under key "eval")
  results/task2_ppo/clipping_cached.json    {"0.05": {l_clip, affected_frac, frac_above, frac_below}, ...}

Run names:
  standard                                  20-update continuation (eps=0.2, beta=0.1)
  midpoint                                  eval-only run on the supplied midpoint
  fork_eps{eps:g}_kl{beta:g}_s{seed}        8-update forks, e.g. fork_eps0.2_kl0.1_s6304

Outputs: results/task2_ppo/figures/*.png and results/task2_ppo/tables.md
Missing files are skipped with a warning, so it can be run on partial results.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "task2_ppo"
RES = ROOT / "results" / "task2_ppo"
FIG = RES / "figures"

SEEDS = [6304, 6305, 6306]
EPS = [0.05, 0.20, 0.50]
BETAS = [0.0, 0.10, 0.20]
EPS_COLORS = {0.05: "#1b9e77", 0.20: "#7570b3", 0.50: "#d95f02"}
BETA_COLORS = {0.0: "#e7298a", 0.10: "#7570b3", 0.20: "#66a61e"}

plt.rcParams.update({
    "font.size": 9, "axes.titlesize": 9, "axes.labelsize": 9,
    "legend.fontsize": 8, "figure.dpi": 150, "axes.grid": True, "grid.alpha": 0.3,
})


# ---------------------------------------------------------------- loading
def fork_name(eps: float, beta: float, seed: int) -> str:
    return f"fork_eps{eps:g}_kl{beta:g}_s{seed}"


def read_jsonl(path: Path):
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_metrics(run: str):
    p = OUT / run / "metrics.jsonl"
    if not p.exists():
        print(f"[warn] missing {p}")
        return None
    rows = read_jsonl(p)
    return rows or None


def load_eval(run: str):
    p = RES / f"{run}.json"
    if not p.exists():
        print(f"[warn] missing {p}")
        return None
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("eval", d)


def series(rows, key):
    return np.array([r.get(key, np.nan) for r in rows], dtype=float)


def agg(vals):
    arr = np.array([v for v in vals if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), 0
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0, int(arr.size)


def fmt(vals, digits=3):
    m, s, n = agg(vals)
    if n == 0:
        return "n/a"
    return f"{m:.{digits}f}" if n == 1 else f"{m:.{digits}f} ± {s:.{digits}f}"


# ---------------------------------------------------------------- per-run stats
def stability_stats(rows):
    g = series(rows, "grad_norm_policy")
    return {
        "grad_max": np.nanmax(g), "grad_mean": np.nanmean(g),
        "ratio_max": np.nanmax(series(rows, "max_ratio")),
        "extreme_frac": np.nanmean(series(rows, "extreme_ratio_frac")),
        "approx_kl": np.nanmean(series(rows, "approx_kl_old_new")),
        "gen_tokens": np.nansum(series(rows, "generated_tokens")),
    }


def per_seed(eps, beta, getter):
    """Apply getter(run_name) for every seed and drop missing results."""
    out = []
    for s in SEEDS:
        v = getter(fork_name(eps, beta, s))
        if v is not None:
            out.append(v)
    return out


def stab_getter(key):
    def g(run):
        rows = load_metrics(run)
        return None if rows is None else stability_stats(rows)[key]
    return g


def eval_getter(key):
    def g(run):
        ev = load_eval(run)
        return None if ev is None else ev.get(key)
    return g


# ---------------------------------------------------------------- plotting helpers
def bar_panel(ax, labels, groups, colors, ylabel, title, logy=False):
    """groups: list of per-seed value lists, one per condition."""
    xs = np.arange(len(labels))
    for x, vals, c in zip(xs, groups, colors):
        m, s, n = agg(vals)
        if n == 0:
            continue
        ax.bar(x, m, yerr=s if n > 1 else None, color=c, alpha=0.65, capsize=3)
        ax.scatter([x] * n, [v for v in vals if np.isfinite(v)], color="k", s=10, zorder=3)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if logy:
        ax.set_yscale("log")


def curve_panel(ax, conds, key, ylabel, title):
    """conds: list of (label, color, [runs]) -> mean +- std across seeds over update index."""
    for label, color, runs in conds:
        arrs = []
        for r in runs:
            rows = load_metrics(r)
            if rows:
                arrs.append(series(rows, key))
        if not arrs:
            continue
        n = min(len(a) for a in arrs)
        a = np.stack([x[:n] for x in arrs])
        x = np.arange(1, n + 1)
        m = np.nanmean(a, 0)
        ax.plot(x, m, color=color, label=label, marker="o", ms=3)
        if len(arrs) > 1:
            ax.fill_between(x, m - np.nanstd(a, 0, ddof=1), m + np.nanstd(a, 0, ddof=1), color=color, alpha=0.15)
    ax.set_xlabel("PPO update")
    ax.set_ylabel(ylabel)
    ax.set_title(title)


# ---------------------------------------------------------------- figures
def fig_standard():
    rows = load_metrics("standard")
    if rows is None:
        return None
    panels = [
        ("reward_raw", "Mean RM reward (raw)"), ("kl_token", "KL to reference (token-pooled)"),
        ("policy_loss", "Policy loss"), ("value_loss", "Value loss"),
        ("entropy", "Entropy (full distribution)"), ("clip_fraction", "Clip fraction"),
        ("grad_norm_policy", "Policy grad norm (pre-clip)"), ("response_length", "Response length (tokens)"),
        ("explained_variance", "Critic explained variance (pre-update)"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(10, 7.5))
    x = np.arange(1, len(rows) + 1)
    for ax, (key, title) in zip(axes.ravel(), panels):
        y = series(rows, key)
        ax.plot(x, y, marker="o", ms=3, color="#7570b3")
        ax.set_title(title)
        ax.set_xlabel("PPO update")
        if key == "explained_variance":
            ax.axhline(0, color="k", lw=0.8)
        if key == "grad_norm_policy":
            ax.plot(x, series(rows, "grad_norm_value"), marker="s", ms=3, color="#d95f02", label="value")
            ax.legend()
    fig.tight_layout()
    p = FIG / "fig_standard_continuation.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    return p


def fig_clipping():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    labels = [f"ε={e:g}" for e in EPS]
    colors = [EPS_COLORS[e] for e in EPS]

    # (a) cached-rollout affected-token fraction (single forward pass, no seeds)
    ax = axes[0, 0]
    cp = RES / "clipping_cached.json"
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        xs = np.arange(len(EPS))
        up = [c.get(f"{e:g}", {}).get("frac_above", np.nan) for e in EPS]
        dn = [c.get(f"{e:g}", {}).get("frac_below", np.nan) for e in EPS]
        ax.bar(xs, up, color="#d95f02", alpha=0.7, label="ρ > 1+ε")
        ax.bar(xs, dn, bottom=up, color="#1b9e77", alpha=0.7, label="ρ < 1−ε")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels)
        ax.legend()
    else:
        print(f"[warn] missing {cp}")
    ax.set_ylabel("affected-token fraction")
    ax.set_title("Cached batch: affected tokens")

    getters = [
        (axes[0, 1], eval_getter("reward_mean"), "held-out RM reward", "Held-out reward", False),
        (axes[0, 2], None, "grad norm", "Policy grad norm (max / mean)", False),
        (axes[1, 0], stab_getter("ratio_max"), "max ρ", "Max probability ratio", True),
        (axes[1, 1], stab_getter("extreme_frac"), "fraction ρ∉[0.5,2]", "Extreme-ratio fraction", False),
        (axes[1, 2], stab_getter("approx_kl"), "approx KL(old‖new)", "Post-update approx KL", False),
    ]
    for ax, getter, ylabel, title, logy in getters:
        if getter is None:  # paired max / mean grad norm
            xs = np.arange(len(EPS))
            w = 0.38
            for off, key, hatch in [(-w / 2, "grad_max", ""), (w / 2, "grad_mean", "//")]:
                for x, e in zip(xs, EPS):
                    vals = per_seed(e, 0.10, stab_getter(key))
                    m, s, n = agg(vals)
                    if n:
                        ax.bar(x + off, m, w, yerr=s if n > 1 else None, color=EPS_COLORS[e],
                               alpha=0.65, hatch=hatch, capsize=3,
                               label=("max" if key == "grad_max" else "mean") if x == 0 else None)
            ax.set_xticks(xs)
            ax.set_xticklabels(labels)
            ax.set_ylabel(ylabel)
            ax.set_title(title)
            ax.legend()
            continue
        groups = [per_seed(e, 0.10, getter) for e in EPS]
        bar_panel(ax, labels, groups, colors, ylabel, title, logy)
    fig.tight_layout()
    p = FIG / "fig_clipping_study.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # training curves of the same forks
    fig, axes = plt.subplots(1, 3, figsize=(10, 3))
    conds = [(f"ε={e:g}", EPS_COLORS[e], [fork_name(e, 0.10, s) for s in SEEDS]) for e in EPS]
    curve_panel(axes[0], conds, "reward_raw", "RM reward", "Training reward")
    curve_panel(axes[1], conds, "clip_fraction", "clip fraction", "Training clip fraction")
    curve_panel(axes[2], conds, "grad_norm_policy", "grad norm", "Policy grad norm")
    axes[0].legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_clipping_curves.png", bbox_inches="tight")
    plt.close(fig)
    return p


def fig_kl():
    fig, axes = plt.subplots(2, 3, figsize=(10, 6))
    labels = [f"β={b:g}" for b in BETAS]
    colors = [BETA_COLORS[b] for b in BETAS]
    specs = [
        (axes[0, 0], "reward_mean", "held-out RM reward", "Held-out reward"),
        (axes[0, 1], "kl_token", "KL to reference", "Held-out KL"),
        (axes[0, 2], "entropy", "entropy", "Held-out entropy"),
        (axes[1, 0], "length_mean", "tokens", "Held-out response length"),
    ]
    for ax, key, ylabel, title in specs:
        groups = [per_seed(0.20, b, eval_getter(key)) for b in BETAS]
        bar_panel(ax, labels, groups, colors, ylabel, title)
    conds = [(f"β={b:g}", BETA_COLORS[b], [fork_name(0.20, b, s) for s in SEEDS]) for b in BETAS]
    curve_panel(axes[1, 1], conds, "reward_raw", "RM reward", "Training reward")
    curve_panel(axes[1, 2], conds, "kl_token", "KL to reference", "Training KL")
    axes[1, 1].legend()
    fig.tight_layout()
    p = FIG / "fig_kl_study.png"
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)

    # reward vs drift trade-off, one point per seed
    fig, ax = plt.subplots(figsize=(4, 3.4))
    for b in BETAS:
        r = per_seed(0.20, b, eval_getter("reward_mean"))
        k = per_seed(0.20, b, eval_getter("kl_token"))
        n = min(len(r), len(k))
        ax.scatter(k[:n], r[:n], color=BETA_COLORS[b], label=f"β={b:g}", s=25)
    ax.set_xlabel("held-out KL to reference")
    ax.set_ylabel("held-out RM reward")
    ax.set_title("Reward vs. drift")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG / "fig_kl_tradeoff.png", bbox_inches="tight")
    plt.close(fig)
    return p


# ---------------------------------------------------------------- tables
def tables():
    L = []

    # standard continuation + baseline
    L.append("## Standard continuation (ε=0.20, β=0.10, 20 updates)\n")
    rows = load_metrics("standard")
    if rows:
        wall = np.nansum(series(rows, "wall_time_s"))
        vram = np.nanmax(series(rows, "peak_vram_gb"))
        L.append("| metric | update 1 | update 20 | min | max |")
        L.append("|---|---|---|---|---|")
        for key in ["reward_raw", "kl_token", "policy_loss", "value_loss", "entropy", "clip_fraction",
                    "grad_norm_policy", "response_length", "explained_variance"]:
            y = series(rows, key)
            L.append(f"| {key} | {y[0]:.4g} | {y[-1]:.4g} | {np.nanmin(y):.4g} | {np.nanmax(y):.4g} |")
        L.append(f"\nLoop wall-clock: {wall / 60:.1f} min; peak VRAM (max_memory_allocated): {vram:.2f} GiB.\n")

    L.append("### Held-out: midpoint vs. final standard adapter\n")
    L.append("| model | reward | KL | entropy | length | trunc. rate | distinct-2 | distinct-3 | distinct-4 |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for run in ["midpoint", "standard"]:
        ev = load_eval(run)
        if ev:
            def g(k):
                v = ev.get(k)
                return f"{v:.3f}" if isinstance(v, (int, float)) else "n/a"
            rs = f"{g('reward_mean')} ± {g('reward_std')}"
            ls = f"{g('length_mean')} ± {g('length_std')}"
            L.append(f"| {run} | {rs} | {g('kl_token')} | {g('entropy')} | {ls} | {g('truncation_rate')} | "
                     f"{g('distinct2')} | {g('distinct3')} | {g('distinct4')} |")

    # clipping study
    L.append("\n## Clipping study\n")
    cp = RES / "clipping_cached.json"
    if cp.exists():
        c = json.loads(cp.read_text(encoding="utf-8"))
        L.append("### Cached rollout batch (single forward pass at the midpoint)\n")
        L.append("| ε | L_clip | affected frac | ρ>1+ε | ρ<1−ε |")
        L.append("|---|---|---|---|---|")
        for e in EPS:
            d = c.get(f"{e:g}", {})
            L.append(f"| {e:g} | {d.get('l_clip', float('nan')):.5g} | {d.get('affected_frac', float('nan')):.5g} | "
                     f"{d.get('frac_above', float('nan')):.5g} | {d.get('frac_below', float('nan')):.5g} |")
    L.append("\n### Matched 8-update forks (mean ± std over seeds, β=0.10)\n")
    L.append("| ε | held-out reward | held-out KL | length | max grad norm | mean grad norm | max ρ | extreme-ρ frac | approx KL(old‖new) | gen. tokens | n |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for e in EPS:
        n = len(per_seed(e, 0.10, stab_getter("grad_max")))
        L.append(
            f"| {e:g} | {fmt(per_seed(e, 0.10, eval_getter('reward_mean')))} | {fmt(per_seed(e, 0.10, eval_getter('kl_token')), 4)} | "
            f"{fmt(per_seed(e, 0.10, eval_getter('length_mean')), 1)} | {fmt(per_seed(e, 0.10, stab_getter('grad_max')))} | "
            f"{fmt(per_seed(e, 0.10, stab_getter('grad_mean')))} | {fmt(per_seed(e, 0.10, stab_getter('ratio_max')))} | "
            f"{fmt(per_seed(e, 0.10, stab_getter('extreme_frac')), 4)} | {fmt(per_seed(e, 0.10, stab_getter('approx_kl')), 5)} | "
            f"{fmt(per_seed(e, 0.10, stab_getter('gen_tokens')), 0)} | {n} |")

    # KL study
    L.append("\n## KL-pressure study (8-update forks, ε=0.20, mean ± std over seeds)\n")
    L.append("| β_KL | held-out reward | held-out KL | entropy | length | trunc. rate | n |")
    L.append("|---|---|---|---|---|---|---|")
    for b in BETAS:
        n = len(per_seed(0.20, b, eval_getter("reward_mean")))
        L.append(
            f"| {b:g} | {fmt(per_seed(0.20, b, eval_getter('reward_mean')))} | {fmt(per_seed(0.20, b, eval_getter('kl_token')), 4)} | "
            f"{fmt(per_seed(0.20, b, eval_getter('entropy')))} | {fmt(per_seed(0.20, b, eval_getter('length_mean')), 1)} | "
            f"{fmt(per_seed(0.20, b, eval_getter('truncation_rate')))} | {n} |")

    p = RES / "tables.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    for fn in (fig_standard, fig_clipping, fig_kl, tables):
        try:
            out = fn()
            print(f"[ok] {fn.__name__}: {out}")
        except Exception as e:  # keep going on partial results
            print(f"[fail] {fn.__name__}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
