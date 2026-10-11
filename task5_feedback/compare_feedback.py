"""Aggregate Task 5 results into tables (CSV + LaTeX), figures and a cost summary.
Run after evaluate_math (gsm + transfer) and score_perturbations:
  python -m task5_feedback.compare_feedback
"""
from __future__ import annotations

import argparse

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common.data import load_yaml, repo_path
from task5_feedback import task5_utils as U

POL = U.POLICY_NAMES


def latex(df: pd.DataFrame, caption: str) -> str:
    cols = "l" + "r" * (df.shape[1])
    lines = [r"\begin{table}[t]\centering\small", rf"\caption{{{caption}}}", rf"\begin{{tabular}}{{{cols}}}", r"\toprule",
             " & ".join([df.index.name or ""] + [str(c).replace("_", r"\_") for c in df.columns]) + r" \\", r"\midrule"]
    for idx, row in df.iterrows():
        lines.append(" & ".join([str(idx).replace("_", r"\_")] + [str(v) for v in row.values]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def write_table(df, name, caption):
    d = repo_path(U.out_path("tables"))
    d.mkdir(parents=True, exist_ok=True)
    df.to_csv(d / f"{name}.csv")
    (d / f"{name}.tex").write_text(latex(df, caption), encoding="utf-8")
    print(f"\n== {name} ==\n{df.to_string()}")


def f3(x): return "-" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--grpo_config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    gsm = U.load(U.out_path("gsm_metrics.json"))
    tr = U.load(U.out_path("transfer_metrics.json"))
    dg = U.load(U.out_path("diagnostics_metrics.json"))
    gmeta = U.load(U.out_path("gsm_judge_meta.json"))
    tmeta = U.load(U.out_path("transfer_judge_meta.json"))
    summary = {}

    # T1 / T3: policy-level tables
    for tag, M in [("indomain_gsm8k", gsm), ("transfer_svamp", tr)]:
        rows = {}
        for p in POL:
            v = M["policies"][p]
            wr = None if p == "sft" else M["pairs"][f"{p}_vs_sft"]["judge_win_rate"]
            rows[p] = {"accuracy": f"{v['accuracy']:.3f} [{v['accuracy_ci95'][0]:.2f},{v['accuracy_ci95'][1]:.2f}]",
                       "format": f3(v["format_compliance"]), "trunc": f3(v["truncation_rate"]),
                       "mean_tokens": f"{v['mean_tokens']:.0f}", "judge_winrate_vs_sft": f3(wr)}
        df = pd.DataFrame(rows).T
        df.index.name = "policy"
        write_table(df, tag, f"{tag}: policy metrics (greedy, n={M['n']}).")
        pr = {}
        for k, v in M["pairs"].items():
            pr[k] = {"judge_W/T/L": f"{v['judge_win']}/{v['judge_tie']}/{v['judge_loss']}",
                     "judge_winrate": f3(v["judge_win_rate"]),
                     "verifier_W/T/L": f"{v['verifier_win']}/{v['verifier_tie']}/{v['verifier_loss']}",
                     "agree_raw": f3(v["agreement_raw"]), "agree_nontied": f3(v["agreement_on_verifier_nontied"]),
                     "kappa": f3(v["cohen_kappa"]), "len_corr": f3(v["corr_token_len_diff_vs_judge_score"])}
        df = pd.DataFrame(pr).T
        df.index.name = "pair"
        write_table(df, f"{tag}_pairs", f"{tag}: judge vs verifier on pairwise comparisons.")
        fc = pd.DataFrame({p: M["policies"][p]["failure_counts"] for p in POL}).T
        fc.index.name = "policy"
        write_table(fc, f"{tag}_failures", f"{tag}: failure taxonomy counts.")

    # T3b: drops
    drop = {}
    for p in POL:
        a, b = gsm["policies"][p], tr["policies"][p]
        d = {"acc_drop_pp": 100 * (a["accuracy"] - b["accuracy"]),
             "acc_drop_rel_%": 100 * (a["accuracy"] - b["accuracy"]) / max(a["accuracy"], 1e-9),
             "format_drop_pp": 100 * (a["format_compliance"] - b["format_compliance"]),
             "len_change_tokens": b["mean_tokens"] - a["mean_tokens"]}
        if p != "sft":
            d["winrate_vs_sft_drop_pp"] = 100 * (gsm["pairs"][f"{p}_vs_sft"]["judge_win_rate"] - tr["pairs"][f"{p}_vs_sft"]["judge_win_rate"])
        drop[p] = {k: f"{v:.1f}" for k, v in d.items()}
    df = pd.DataFrame(drop).T
    df.index.name = "policy"
    write_table(df, "ood_drop", "In-domain (GSM8K) to transfer (SVAMP) change.")

    # T2: diagnostics
    dr = {}
    for name, v in dg["per_pair"].items():
        for mech in ("verifier", "judge"):
            d = v[mech]
            dr[f"{name}/{mech}"] = {"first_better": f"{d['first']}/20", "tie": f"{d['tie']}/20", "second_wrong": f"{d['second']}/20",
                                    "first_rate": f3(d["first_rate"]),
                                    "first_ci95": f"[{d['first_ci95'][0]:.2f},{d['first_ci95'][1]:.2f}]"}
    df = pd.DataFrame(dr).T
    df.index.name = "pair/mechanism"
    write_table(df, "diagnostics", "Controlled diagnostics. For `filler`, second = prefers filler (susceptibility).")
    sr = {m: {"S_reason": f3(dg[f"S_reason_{m}"]["first_rate"]), "S_outcome(n=40)": f3(dg[f"S_outcome_{m}"]["first_rate"]),
              "outcome_tie": f3(dg[f"S_outcome_{m}"]["tie_rate"]), "outcome_wrong": f3(dg[f"S_outcome_{m}"]["second_rate"]),
              "filler_suscept": f3(dg[f"filler_susceptibility_{m}"]), "distractor_robust": f3(dg[f"distractor_robustness_{m}"])}
          for m in ("verifier", "judge")}
    df = pd.DataFrame(sr).T
    df.index.name = "mechanism"
    write_table(df, "sensitivity", "S_reason and S_outcome.")

    # cost
    try:
        K = int(load_yaml(args.grpo_config)["num_generations"])
    except Exception:
        K = 4
    vs = np.mean([gsm["policies"][p]["verifier_seconds_per_call"] for p in POL])
    js = gmeta.get("mean_seconds_per_uncached_call")
    cost = {"K_train": K, "verifier_calls_per_prompt": K, "judge_calls_per_prompt": K * (K - 1) // 2,
            "judge_calls_per_prompt_K8": 28, "verifier_seconds_per_call": float(vs),
            "judge_seconds_per_call": js, "judge_gpu_peak_mem_gb": gmeta.get("judge_gpu_peak_mem_gb"),
            "judge_over_verifier_time_ratio": (js / vs) if js and vs else None,
            "judge_tie_rate_gsm": gmeta["tie_rate_overall"], "judge_tie_rate_svamp": tmeta["tie_rate_overall"],
            "judge_tie_parse_failure_rate_gsm": gmeta["parse_audit"]["parse_fail_rate_among_ties"],
            "judge_tie_parse_failure_rate_diag": dg["judge_parse_audit"]["parse_fail_rate_among_ties"]}
    summary["cost"] = cost
    summary["qualitative"] = dg["qualitative"]
    summary["length_bias_corr_diag"] = dg["length_bias_corr"]
    summary["verifier_label_mismatches"] = dg["verifier_label_mismatches"]
    U.save(U.out_path("final_comparison.json"), summary)
    print("\nCost:", cost)

    # figures
    fd = repo_path(U.out_path("figures"))
    fd.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.2))
    x = np.arange(len(POL))
    for k, (tag, M) in enumerate([("GSM8K", gsm), ("SVAMP", tr)]):
        ax[0].bar(x + (k - .5) * .35, [M["policies"][p]["accuracy"] for p in POL], .35, label=tag)
        ax[1].bar(x + (k - .5) * .35, [M["policies"][p]["format_compliance"] for p in POL], .35, label=tag)
    for a, t in zip(ax, ["Exact accuracy", "Format compliance"]):
        a.set_xticks(x); a.set_xticklabels(POL); a.set_title(t); a.set_ylim(0, 1); a.legend()
    fig.tight_layout(); fig.savefig(fd / "accuracy_format.png", dpi=200); plt.close(fig)

    names = list(dg["per_pair"])
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.2), sharey=True)
    for a, mech in zip(ax, ("verifier", "judge")):
        f = [dg["per_pair"][n][mech]["first_rate"] for n in names]
        t = [dg["per_pair"][n][mech]["tie_rate"] for n in names]
        s = [dg["per_pair"][n][mech]["second_rate"] for n in names]
        a.bar(names, f, label="prefers first"); a.bar(names, t, bottom=f, label="tie")
        a.bar(names, s, bottom=np.add(f, t), label="prefers second")
        a.set_title(mech); a.tick_params(axis="x", rotation=30)
    ax[0].legend(fontsize=7)
    fig.tight_layout(); fig.savefig(fd / "diagnostics.png", dpi=200); plt.close(fig)
    print("Saved tables to", U.out_path("tables"), "and figures to", U.out_path("figures"))


if __name__ == "__main__":
    main()
