"""Regenerate Task 3 figures from saved JSON logs (CPU only)."""
from __future__ import annotations

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.data import load_yaml, read_jsonl, repo_path


def _ma(x, w=5):
    x = np.asarray(x, float)
    if len(x) < w:
        return x
    return np.convolve(np.pad(x, (w // 2, w - 1 - w // 2), mode="edge"), np.ones(w) / w, mode="valid")


def fig_standard(rdir):
    p = rdir / "standard_train_log.jsonl"
    if not p.exists():
        return
    log = read_jsonl(p)
    u = [r["update"] for r in log]
    panels = [("reward_mean", "Reward (RM score)"), ("kl", "KL to reference (token-pooled)"),
              ("group_reward_std", "Within-group reward std"), ("policy_term", "Policy loss term"),
              ("grad_norm", "Grad norm (pre-clip)"), ("entropy", "Entropy (sampled-token est.)"),
              ("length_mean", "Mean response length"), ("truncation_rate", "Truncated fraction")]
    fig, axs = plt.subplots(2, 4, figsize=(16, 6.5))
    for ax, (k, t) in zip(axs.ravel(), panels):
        y = [r[k] for r in log]
        ax.plot(u, y, alpha=0.4, marker="o", ms=3, label="raw")
        ax.plot(u, _ma(y), lw=2, label="MA(5)")
        ax.set_title(t, fontsize=10)
        ax.set_xlabel("update")
        if k == "group_reward_std":
            ax2 = ax.twinx()
            ax2.plot(u, [r["uninformative_fraction"] for r in log], "r--", alpha=0.6)
            ax2.set_ylabel("uninformative frac", color="r")
    axs[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rdir / "fig_standard_diagnostics.png", dpi=150)
    plt.close(fig)


def fig_group(rdir):
    p = rdir / "group_size_results.json"
    if not p.exists():
        return
    res = json.loads(p.read_text())
    ks = list(res["by_k"])
    strata = list(res["by_k"][ks[0]]["strata"])
    metrics = [("informative_rate", "Informative-group rate"), ("mean_group_std", "Mean within-group std"),
               ("var_centered", "Var of centered advantage (r-mu)")]
    fig, axs = plt.subplots(1, 3, figsize=(15, 4))
    w = 0.8 / len(strata)
    for ax, (m, t) in zip(axs, metrics):
        for j, s in enumerate(strata):
            v = np.array([res["by_k"][k]["strata"][s][m]["value"] for k in ks])
            ci = np.array([res["by_k"][k]["strata"][s][m]["ci"] for k in ks])
            ax.bar(np.arange(len(ks)) + j * w, v, w, yerr=[v - ci[:, 0], ci[:, 1] - v], capsize=2, label=s)
        ax.set_xticks(np.arange(len(ks)) + 0.4 - w / 2)
        ax.set_xticklabels([f"K={k}" for k in ks])
        ax.set_title(t, fontsize=10)
    axs[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rdir / "fig_group_size.png", dpi=150)
    plt.close(fig)


def fig_norm(rdir):
    p = rdir / "normalization_summary.json"
    if not p.exists():
        return
    s = json.loads(p.read_text())
    ev = s.get("eval", {})
    metrics = [("reward_mean", "Held-out reward"), ("kl_token_pooled", "KL (token-pooled)"),
               ("length_mean", "Mean length")]
    probe = s.get("probe")
    ncol = len(metrics) + (1 if probe else 0)
    fig, axs = plt.subplots(1, ncol, figsize=(4.2 * ncol, 4))
    for ax, (m, t) in zip(axs, metrics):
        for i, loss in enumerate(["grpo", "dr_grpo"]):
            if loss in ev:
                v = ev[loss][m]
                ax.bar(i, v["mean"], yerr=v["std"], capsize=4, alpha=0.6)
                ax.scatter([i] * len(v["values"]), v["values"], c="k", s=14, zorder=3)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["GRPO", "Dr.GRPO"])
        ax.set_title(t, fontsize=10)
    if probe:
        ax = axs[-1]
        names = ["short", "mid", "long"]
        for j, (loss, lab) in enumerate([("grpo", "GRPO"), ("dr_grpo", "Dr.GRPO")]):
            ax.bar(np.arange(3) + j * 0.4, [probe["tercile_gradnorm"][loss][n] for n in names], 0.4, label=lab)
        ax.set_xticks(np.arange(3) + 0.2)
        ax.set_xticklabels(names)
        ax.set_yscale("log")
        ax.set_title("Per-sequence grad norm by length tercile", fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(rdir / "fig_normalization.png", dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    rdir = repo_path(cfg["results_dir"])
    fig_standard(rdir)
    fig_group(rdir)
    fig_norm(rdir)
    print("figures written to", rdir)


if __name__ == "__main__":
    main()
