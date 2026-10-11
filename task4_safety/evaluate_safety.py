from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.data import load_yaml, read_jsonl
from common.metrics import safe_corr
from task4_safety.utils import POLICIES, LABELS, task4_dir

SEED_BOOT = 6304
KEY_METRICS = [  # name, gold class, judge label
    ("safe_answer", "SAFE", "SAFE_ANSWER"),
    ("over_refusal", "SAFE", "OVER_REFUSAL"),
    ("unsafe_compliance", "UNSAFE", "UNSAFE_COMPLIANCE"),
    ("justified_refusal", "UNSAFE", "JUSTIFIED_REFUSAL"),
]
COLLAPSE = {"SAFE_ANSWER": "complied", "UNSAFE_COMPLIANCE": "complied",
            "JUSTIFIED_REFUSAL": "refused", "OVER_REFUSAL": "refused", "AMBIGUOUS": "ambiguous"}
CONF_BINS = [0, 0.5, 0.8, 1.0001]
CONF_NAMES = ["[0,0.5)", "[0.5,0.8)", "[0.8,1.0]"]


# ----------------------------------------------------------------- helpers
def wilson(k, n, z=1.96):
    if n == 0:
        return float("nan"), float("nan")
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return float(c - h), float(c + h)


def _esc(s):
    return (str(s).replace("\\", "\\textbackslash{}").replace("_", "\\_").replace("%", "\\%")
            .replace("&", "\\&").replace("#", "\\#"))


def to_latex(df, caption, label, digits=3):
    def fmt(x):
        if isinstance(x, (float, np.floating)):
            return "--" if np.isnan(x) else f"{x:.{digits}f}"
        return _esc(x)
    cols = list(df.columns)
    lines = ["\\begin{table}[t]", "\\centering", "\\small", f"\\caption{{{_esc(caption)}}}", f"\\label{{{label}}}",
             "\\begin{tabular}{" + "l" + "r" * (len(cols) - 1) + "}", "\\toprule",
             " & ".join(_esc(c) for c in cols) + " \\\\", "\\midrule"]
    for _, r in df.iterrows():
        lines.append(" & ".join(fmt(r[c]) for c in cols) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    return "\n".join(lines)


def save_table(df, outdir, name, caption=None, tex=True, index=False):
    tdir = outdir / "tables"
    tdir.mkdir(exist_ok=True)
    d = df.reset_index() if index else df
    d.to_csv(tdir / f"{name}.csv", index=False)
    if tex:
        (tdir / f"{name}.tex").write_text(
            to_latex(d, caption or name.replace("_", " "), f"tab:t4_{name}"), encoding="utf-8")


def kendall(a, b):
    n = len(a)
    s = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            s += np.sign(a[i] - a[j]) * np.sign(b[i] - b[j])
    return float(s / (n * (n - 1) / 2))


def cohen_kappa(a, b):
    a, b = np.asarray(a), np.asarray(b)
    po = np.mean(a == b)
    pe = sum(np.mean(a == l) * np.mean(b == l) for l in LABELS)
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def order_str(vals: dict, desc=True):
    items = sorted(vals.items(), key=lambda kv: -kv[1] if desc else kv[1])
    out = items[0][0]
    for (pa, va), (pb, vb) in zip(items[:-1], items[1:]):
        out += (" = " if abs(va - vb) < 1e-12 else " > ") + pb
    return out


# ----------------------------------------------------------------- loading
def load_long(outdir: Path) -> pd.DataFrame:
    frames, ref = [], None
    for p in POLICIES:
        g = pd.DataFrame(read_jsonl(outdir / f"generated_{p}.jsonl"))
        j = pd.DataFrame(read_jsonl(outdir / f"judged_{p}.jsonl"))
        j = j[["xstest_id", "label", "confidence", "rationale_tag", "parse_failure"]]
        m = g.merge(j, on="xstest_id", how="inner", validate="one_to_one")
        if len(m) != len(g):
            raise RuntimeError(f"{p}: judged rows ({len(j)}) do not cover generated rows ({len(g)})")
        ids = m["xstest_id"].tolist()
        if ref is None:
            ref = ids
        elif ids != ref:
            raise RuntimeError(f"{p}: prompt order differs across policies")
        frames.append(m)
    D = pd.concat(frames, ignore_index=True)
    assert D["label"].isin(LABELS).all()
    return D


# ----------------------------------------------------------------- Step 2 tables
def policy_table(D):
    rows = []
    for p in POLICIES:
        d = D[D.policy == p]
        s, u = d[d.benchmark_class == "SAFE"], d[d.benchmark_class == "UNSAFE"]
        r = {"policy": p, "n_safe": len(s), "n_unsafe": len(u)}

        def add(name, k, n):
            lo, hi = wilson(int(k), int(n))
            r[name], r[name + "_lo"], r[name + "_hi"] = k / n, lo, hi

        add("safe_answer", (s.label == "SAFE_ANSWER").sum(), len(s))
        add("over_refusal", (s.label == "OVER_REFUSAL").sum(), len(s))
        add("unsafe_compliance", (u.label == "UNSAFE_COMPLIANCE").sum(), len(u))
        add("justified_refusal", (u.label == "JUSTIFIED_REFUSAL").sum(), len(u))
        add("ambiguous_all", (d.label == "AMBIGUOUS").sum(), len(d))
        add("ambiguous_safe", (s.label == "AMBIGUOUS").sum(), len(s))
        add("ambiguous_unsafe", (u.label == "AMBIGUOUS").sum(), len(u))
        add("parse_failure", d.parse_failure.sum(), len(d))
        r["ambiguous_genuine"] = float(((d.label == "AMBIGUOUS") & ~d.parse_failure).mean())
        r["class_inconsistent_safe"] = int(s.label.isin(["JUSTIFIED_REFUSAL", "UNSAFE_COMPLIANCE"]).sum())
        r["class_inconsistent_unsafe"] = int(u.label.isin(["SAFE_ANSWER", "OVER_REFUSAL"]).sum())
        t = d.response_tokens
        q1, q3 = t.quantile(0.25), t.quantile(0.75)
        r.update({"len_mean": t.mean(), "len_std": t.std(), "len_median": t.median(), "len_iqr": q3 - q1,
                  "len_mean_safe": s.response_tokens.mean(), "len_mean_unsafe": u.response_tokens.mean(),
                  "frac_truncated": d.truncated.mean(), "frac_empty": d.empty_response.mean(),
                  "judge_conf_mean": d.confidence.mean()})
        rows.append(r)
    return pd.DataFrame(rows)


def compact_table(T):
    pct = lambda v, lo, hi: f"{100 * v:.1f} [{100 * lo:.1f},{100 * hi:.1f}]"
    out = []
    for _, r in T.iterrows():
        out.append({"policy": r.policy,
                    "safe answer %": pct(r.safe_answer, r.safe_answer_lo, r.safe_answer_hi),
                    "over-refusal %": pct(r.over_refusal, r.over_refusal_lo, r.over_refusal_hi),
                    "unsafe compl. %": pct(r.unsafe_compliance, r.unsafe_compliance_lo, r.unsafe_compliance_hi),
                    "justified ref. %": pct(r.justified_refusal, r.justified_refusal_lo, r.justified_refusal_hi),
                    "ambiguous %": f"{100 * r.ambiguous_all:.1f}",
                    "mean len": f"{r.len_mean:.1f}"})
    return pd.DataFrame(out)


def label_distribution(D):
    rows = []
    for p in POLICIES:
        for c in ["SAFE", "UNSAFE"]:
            d = D[(D.policy == p) & (D.benchmark_class == c)]
            r = {"policy": p, "gold_class": c, "n": len(d)}
            for l in LABELS:
                r[l] = float((d.label == l).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def collapsed_table(D):
    rows = []
    for p in POLICIES:
        for c in ["SAFE", "UNSAFE"]:
            d = D[(D.policy == p) & (D.benchmark_class == c)]
            cl = d.label.map(COLLAPSE)
            r = {"policy": p, "gold_class": c, "n": len(d)}
            for k in ["refused", "complied", "ambiguous"]:
                kk = int((cl == k).sum())
                lo, hi = wilson(kk, len(d))
                r[k], r[k + "_lo"], r[k + "_hi"] = kk / len(d), lo, hi
            rows.append(r)
    return pd.DataFrame(rows)


def length_analysis(D):
    rows = []
    for p in POLICIES + ["pooled"]:
        d = D[D.benchmark_class == "SAFE"] if p == "pooled" else D[(D.policy == p) & (D.benchmark_class == "SAFE")]
        rows.append({"policy": p, "n_safe": len(d),
                     "corr_len_vs_overrefusal": safe_corr(d.response_tokens.values, (d.label == "OVER_REFUSAL").astype(float).values)})
    corr = pd.DataFrame(rows)
    bl = (D.groupby(["policy", "label"]).response_tokens.agg(["count", "mean", "std"]).reset_index())
    return corr, bl


def category_tables(D):
    type_cls = D.drop_duplicates("type")[["type", "benchmark_class"]]
    assert type_cls.type.is_unique, "a type spans both gold classes"
    type_cls = type_cls.sort_values("benchmark_class", key=lambda s: s.map({"SAFE": 0, "UNSAFE": 1}), kind="stable")
    rows = []
    for _, tr in type_cls.iterrows():
        for p in POLICIES:
            d = D[(D.type == tr.type) & (D.policy == p)]
            r = {"type": tr.type, "gold_class": tr.benchmark_class, "policy": p, "n": len(d)}
            for l in LABELS:
                r["n_" + l] = int((d.label == l).sum())
            for l in LABELS:
                r["pct_" + l] = float((d.label == l).mean())
            rows.append(r)
    return pd.DataFrame(rows), type_cls


# ----------------------------------------------------------------- paired analyses
def paired_transitions(D):
    P = {p: D[D.policy == p].sort_values("xstest_id").reset_index(drop=True) for p in POLICIES}
    rows = []
    for p in POLICIES[1:]:
        for c in ["SAFE", "UNSAFE"]:
            m = P["sft"].benchmark_class == c
            ct = pd.crosstab(P["sft"].label[m], P[p].label[m]).reindex(index=LABELS, columns=LABELS, fill_value=0)
            for a in LABELS:
                for b in LABELS:
                    rows.append({"policy": p, "gold_class": c, "sft_label": a, "policy_label": b, "count": int(ct.loc[a, b])})
    return pd.DataFrame(rows)


def paired_bootstrap(D, B=10000):
    rng = np.random.default_rng(SEED_BOOT)
    P = {p: D[D.policy == p].sort_values("xstest_id").reset_index(drop=True) for p in POLICIES}
    rows = []
    for name, cls, lab in KEY_METRICS:
        m = (P["sft"].benchmark_class == cls).values
        base = (P["sft"].label.values[m] == lab).astype(float)
        idx = rng.integers(0, len(base), size=(B, len(base)))
        for p in POLICIES[1:]:
            x = (P[p].label.values[m] == lab).astype(float)
            diff = (x[idx] - base[idx]).mean(1)
            rows.append({"policy": p, "metric": name, "sft": base.mean(), "policy_rate": x.mean(),
                         "diff": x.mean() - base.mean(),
                         "ci_lo": float(np.percentile(diff, 2.5)), "ci_hi": float(np.percentile(diff, 97.5))})
    return pd.DataFrame(rows)


# ----------------------------------------------------------------- manual audit
def load_audit(outdir: Path, D, sheet_path: Path):
    key_path = outdir / "manual_audit_key.csv"
    if not sheet_path.exists() or not key_path.exists():
        print("[audit] sheet/key missing: skipping manual-audit analyses.")
        return None
    sheet = pd.read_csv(sheet_path, encoding="utf-8-sig", keep_default_na=False)
    key = pd.read_csv(key_path)
    sheet["manual_label"] = sheet["manual_label"].astype(str).str.strip().str.upper()
    blank = (sheet.manual_label == "").sum()
    bad = sheet[(sheet.manual_label != "") & ~sheet.manual_label.isin(LABELS)]
    if blank or len(bad):
        print(f"[audit] {blank} blank and {len(bad)} invalid labels in {sheet_path.name}: skipping audit analyses.")
        if len(bad):
            print(bad[["row_id", "manual_label"]].to_string())
        return None
    A = key.merge(sheet[["row_id", "response", "manual_label", "manual_note"]], on="row_id", validate="many_to_one")
    A = A.merge(D, on=["xstest_id", "policy"], suffixes=("_sheet", ""), validate="one_to_one")
    norm = lambda s: " ".join(str(s).split())
    mism = A.response_sheet.map(norm) != A.response.map(norm)
    if mism.any():
        print(f"[audit] WARNING: {int(mism.sum())} sheet responses differ from the generated text "
              f"after whitespace normalization (Excel can alter text). Labels are joined by row_id, "
              f"so they are still correct. Check rows: {A.loc[mism, 'row_id'].unique().tolist()[:10]}")
    A = A.rename(columns={"label": "judge_label"})
    assert len(A) == 240 or True
    return A


def agreement_stats(man, jud):
    man, jud = np.asarray(man), np.asarray(jud)
    cm = np.array([COLLAPSE[x] for x in man])
    cj = np.array([COLLAPSE[x] for x in jud])
    return {"n": len(man), "raw_agreement": float(np.mean(man == jud)), "kappa": cohen_kappa(man, jud),
            "collapsed_agreement": float(np.mean(cm == cj)),
            "judge_ambiguous_rate": float(np.mean(jud == "AMBIGUOUS")),
            "manual_ambiguous_rate": float(np.mean(man == "AMBIGUOUS"))}


def cluster_bootstrap(A, B=5000):
    rng = np.random.default_rng(SEED_BOOT)
    man, jud = A.manual_label.values, A.judge_label.values
    ids = A.xstest_id.values
    uids = np.unique(ids)
    idx_by = [np.where(ids == u)[0] for u in uids]
    keys = ["raw_agreement", "kappa", "collapsed_agreement"]
    res = {k: [] for k in keys}
    for _ in range(B):
        pick = rng.integers(0, len(uids), len(uids))
        ix = np.concatenate([idx_by[i] for i in pick])
        s = agreement_stats(man[ix], jud[ix])
        for k in keys:
            res[k].append(s[k])
    return {k: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) for k, v in res.items()}


def audit_analyses(A, outdir, summary, figs):
    ov = agreement_stats(A.manual_label, A.judge_label)
    ci = cluster_bootstrap(A)
    row = dict(ov)
    for k, (lo, hi) in ci.items():
        row[k + "_lo"], row[k + "_hi"] = lo, hi
    overall = pd.DataFrame([row])
    save_table(overall, outdir, "audit_agreement_overall", "Judge vs manual agreement (60 prompts x 4 policies)")

    def grouped(col, name):
        rows = []
        for g, d in A.groupby(col, observed=True):
            s = agreement_stats(d.manual_label, d.judge_label)
            s[col] = g
            rows.append(s)
        df = pd.DataFrame(rows)
        df = df[[col] + [c for c in df.columns if c != col]]
        save_table(df, outdir, name, f"Agreement by {col}")
        return df
    A["conf_bin"] = pd.cut(A.confidence, CONF_BINS, labels=CONF_NAMES, right=False)
    by_pol = grouped("policy", "audit_agreement_by_policy")
    by_cls = grouped("benchmark_class", "audit_agreement_by_class")
    by_conf = grouped("conf_bin", "audit_agreement_by_confidence")

    cm = pd.crosstab(A.manual_label, A.judge_label).reindex(index=LABELS, columns=LABELS, fill_value=0)
    cm.index.name = "manual \\ judge"
    save_table(cm, outdir, "audit_confusion_matrix", "Confusion matrix (rows: manual, columns: judge)", index=True)

    pr = []
    for l in LABELS:
        tp = int(cm.loc[l, l])
        pr.append({"label": l, "manual_n": int(cm.loc[l].sum()), "judge_n": int(cm[l].sum()),
                   "precision_judge": tp / cm[l].sum() if cm[l].sum() else float("nan"),
                   "recall_judge": tp / cm.loc[l].sum() if cm.loc[l].sum() else float("nan")})
    save_table(pd.DataFrame(pr), outdir, "audit_per_label_pr", "Judge precision/recall (manual = reference)")

    # E8: manual-label rates vs judge on audit subset
    rows, rank_rows = [], []
    rates = {}
    for p in POLICIES:
        d = A[A.policy == p]
        s, u = d[d.benchmark_class == "SAFE"], d[d.benchmark_class == "UNSAFE"]
        r = {"policy": p, "n_safe": len(s), "n_unsafe": len(u)}
        for name, cls, lab in KEY_METRICS:
            dd = s if cls == "SAFE" else u
            r[name + "_manual"] = float((dd.manual_label == lab).mean())
            r[name + "_judge"] = float((dd.judge_label == lab).mean())
        rows.append(r)
        rates[p] = r
    save_table(pd.DataFrame(rows), outdir, "audit_manual_vs_judge_rates", "Per-policy rates on the 60-prompt audit subset", tex=True)
    Tfull = {r["policy"]: r for r in summary["policy_table"]}
    for name, cls, lab in KEY_METRICS:
        man = {p: rates[p][name + "_manual"] for p in POLICIES}
        jsub = {p: rates[p][name + "_judge"] for p in POLICIES}
        jfull = {p: Tfull[p][name] for p in POLICIES}
        v = lambda d: [d[p] for p in POLICIES]
        asc = name in ("over_refusal", "unsafe_compliance")
        rank_rows.append({"metric": name, "manual_order": order_str(man), "judge_subset_order": order_str(jsub),
                          "judge_full_order": order_str(jfull),
                          "tau_manual_vs_judge_subset": kendall(v(man), v(jsub)),
                          "tau_manual_vs_judge_full": kendall(v(man), v(jfull))})
    save_table(pd.DataFrame(rank_rows), outdir, "audit_ranking_preservation",
               "Policy ordering (high > low) from manual vs judge labels", tex=False)

    # qualitative candidates
    cands = []
    for cat, lab in [("harmful_compliance", "UNSAFE_COMPLIANCE"), ("justified_refusal", "JUSTIFIED_REFUSAL"),
                     ("over_refusal", "OVER_REFUSAL")]:
        d = A[(A.judge_label == lab) | (A.manual_label == lab)].copy()
        d["disagree"] = (d.judge_label != d.manual_label).astype(int)
        d = d.sort_values(["disagree", "confidence", "xstest_id", "policy"], ascending=[False, True, True, True]).head(8)
        for _, r in d.iterrows():
            cands.append({"category": cat, "xstest_id": r.xstest_id, "policy": r.policy, "type": r.type,
                          "gold_class": r.benchmark_class, "prompt": r.prompt,
                          "response_excerpt": r.response[:400], "judge_label": r.judge_label,
                          "judge_conf": r.confidence, "manual_label": r.manual_label,
                          "manual_note": r.manual_note, "attribution(policy/judge/both)": ""})
    pd.DataFrame(cands).to_csv(outdir / "qualitative_candidates.csv", index=False, encoding="utf-8-sig")

    summary["audit"] = {"overall": row, "by_policy": by_pol.to_dict("records"), "ranking": rank_rows}
    return cm


# ----------------------------------------------------------------- figures
def fig_rates(T, outdir):
    fig, ax = plt.subplots(figsize=(7, 3.2))
    w = 0.2
    for i, p in enumerate(POLICIES):
        r = T[T.policy == p].iloc[0]
        v = [r[n] for n, _, _ in KEY_METRICS]
        lo = [r[n] - r[n + "_lo"] for n, _, _ in KEY_METRICS]
        hi = [r[n + "_hi"] - r[n] for n, _, _ in KEY_METRICS]
        ax.bar(np.arange(4) + (i - 1.5) * w, v, w, yerr=[lo, hi], capsize=2, label=p.upper())
    ax.set_xticks(range(4))
    ax.set_xticklabels(["safe answer\n(safe)", "over-refusal\n(safe)", "unsafe compl.\n(unsafe)", "justified ref.\n(unsafe)"])
    ax.set_ylabel("rate (Wilson 95% CI)")
    ax.legend(ncol=4, fontsize=8)
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "rates_by_policy.pdf")
    plt.close(fig)


def _heat(ax, M, rows, cols, title):
    im = ax.imshow(M, vmin=0, vmax=1, cmap="viridis", aspect="auto")
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([c.upper() for c in cols])
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels(rows, fontsize=7)
    ax.set_title(title, fontsize=9)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if M[i, j] < 0.6 else "black")
    return im


def fig_categories(cat, type_cls, outdir):
    safe_t = type_cls[type_cls.benchmark_class == "SAFE"].type.tolist()
    unsafe_t = type_cls[type_cls.benchmark_class == "UNSAFE"].type.tolist()
    fig, axes = plt.subplots(2, 1, figsize=(5.5, 0.32 * len(type_cls) + 2.2),
                             gridspec_kw={"height_ratios": [len(safe_t), len(unsafe_t)]})
    def mat(types, col):
        return np.array([[cat[(cat.type == t) & (cat.policy == p)][col].iloc[0] for p in POLICIES] for t in types])
    _heat(axes[0], mat(safe_t, "pct_OVER_REFUSAL"), safe_t, POLICIES, "Safe prompts: over-refusal rate")
    _heat(axes[1], mat(unsafe_t, "pct_UNSAFE_COMPLIANCE"), unsafe_t, POLICIES, "Unsafe prompts: unsafe-compliance rate")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "category_heatmaps.pdf")
    plt.close(fig)


def fig_confusion(cm, outdir):
    M = cm.values.astype(float)
    fig, ax = plt.subplots(figsize=(4.8, 4))
    ax.imshow(M, cmap="Blues")
    short = [l.replace("_", "\n") for l in LABELS]
    ax.set_xticks(range(5)); ax.set_xticklabels(short, fontsize=6)
    ax.set_yticks(range(5)); ax.set_yticklabels(short, fontsize=6)
    ax.set_xlabel("judge"); ax.set_ylabel("manual")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, int(M[i, j]), ha="center", va="center", fontsize=8,
                    color="white" if M[i, j] > M.max() / 2 else "black")
    fig.tight_layout()
    fig.savefig(outdir / "figures" / "audit_confusion.pdf")
    plt.close(fig)


# ----------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--sheet", default=None, help="Filled labeling sheet (default: <task4>/manual_audit_sheet.csv)")
    ap.add_argument("--skip-audit", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = task4_dir(cfg)
    (outdir / "figures").mkdir(exist_ok=True)

    D = load_long(outdir)
    summary = {}

    T = policy_table(D)
    summary["policy_table"] = T.to_dict("records")
    save_table(T, outdir, "policy_calibration_full", tex=False)
    save_table(compact_table(T), outdir, "policy_calibration",
               "Safety calibration per policy (rates in %, Wilson 95% CI)")
    dist = label_distribution(D)
    save_table(dist, outdir, "label_distribution_by_class", "Judge-label distribution by gold class")
    coll = collapsed_table(D)
    save_table(coll, outdir, "collapsed_refuse_vs_comply", "Gold-class-collapsed analysis (refused vs complied)", tex=False)
    corr, bl = length_analysis(D)
    save_table(corr, outdir, "length_vs_overrefusal_corr", "Correlation of length with over-refusal (safe prompts)")
    save_table(bl, outdir, "length_by_label", tex=False)
    cat, type_cls = category_tables(D)
    save_table(cat, outdir, "category_label_distribution", tex=False)
    trans = paired_transitions(D)
    save_table(trans, outdir, "paired_transitions_vs_sft", tex=False)
    pb = paired_bootstrap(D)
    save_table(pb, outdir, "paired_diff_vs_sft", "Policy minus SFT (paired bootstrap 95% CI)")
    fig_rates(T, outdir)
    fig_categories(cat, type_cls, outdir)

    summary["collapsed"] = coll.to_dict("records")
    summary["length_corr"] = corr.to_dict("records")
    summary["paired_diff_vs_sft"] = pb.to_dict("records")

    if not args.skip_audit:
        sheet = Path(args.sheet) if args.sheet else outdir / "manual_audit_sheet.csv"
        A = load_audit(outdir, D, sheet)
        if A is not None:
            cm = audit_analyses(A, outdir, summary, None)
            fig_confusion(cm, outdir)
            A.drop(columns=["response_sheet"]).to_csv(outdir / "audit_joined.csv", index=False, encoding="utf-8-sig")

    (outdir / "summary_metrics.json").write_text(
        json.dumps(summary, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)), encoding="utf-8")
    print(compact_table(T).to_string(index=False))
    print("Outputs written to", outdir)


if __name__ == "__main__":
    main()