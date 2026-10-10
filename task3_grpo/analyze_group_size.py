from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict

import numpy as np

from common.data import load_yaml, read_jsonl, repo_path

EPS = 1e-6  # same as group_relative_advantages
METRICS = [
    "informative_rate", "informative_rate_strict", "mean_group_std",
    "var_centered", "var_norm_adv", "std_of_group_std", "trunc_frac",
]


def load_k8_cache(path):
    rows = read_jsonl(path)
    by_prompt = defaultdict(list)
    for row in rows:
        by_prompt[str(row["source_index"])].append(row)
    # Instructor cache has 8 rows per prompt, one row per completion.
    bad = {pid: len(group) for pid, group in by_prompt.items() if len(group) < 8}
    if bad:
        raise ValueError(f"Expected at least K=8 cached completions per prompt; short groups: {bad}")
    for group in by_prompt.values():
        group.sort(key=lambda x: int(x.get("generation_index", 0)))
    return by_prompt


def regroup_equal_generation_budget(by_prompt, k: int, rng: np.random.RandomState | None = None):
    """Return K-sized groups while keeping total cached completions fixed.

    Rule (defined once): for every prompt take its first 8 completions (by generation_index) and split them into
    8/K disjoint consecutive blocks of size K. K=2 -> 4 groups/prompt, K=4 -> 2, K=8 -> 1. Total completions
    (8 x #prompts) and the completions themselves are identical for every K. If `rng` is given, the 8 completions
    are randomly permuted first (robustness check against block-order artifacts).
    """
    if 8 % k != 0:
        raise ValueError("K must divide 8")
    groups = []
    for pid, rows in by_prompt.items():
        rows8 = list(rows[:8])
        if rng is not None:
            rows8 = [rows8[i] for i in rng.permutation(8)]
        for b in range(8 // k):
            blk = rows8[b * k : (b + 1) * k]
            groups.append({
                "source_index": pid, "block": b, "k": k, "rows": blk,
                "rewards": np.array([float(r["reward"]) for r in blk]),
            })
    return groups


# ----------------------------------------------------------------------------- statistics
def group_info(g, tol, tol_strict):
    r = g["rewards"]
    mu = r.mean()
    std = float(r.std())  # population std (ddof=0), as in group_relative_advantages
    adv = (r - mu) / max(std, EPS)
    return {
        "pid": g["source_index"], "std": std, "inf": float(std > tol), "inf_strict": float(std > tol_strict),
        "var_c": float(np.mean((r - mu) ** 2)),  # variance of the unnormalized centered advantage
        "var_adv": float(np.var(adv)),  # variance of the z-scored advantage (== 1 if informative else 0)
        "trunc": float(np.mean([bool(x.get("clipped_at_max", False)) for x in g["rows"]])),
    }


def aggregate(infos):
    if not infos:
        return {m: float("nan") for m in METRICS} | {"n_groups": 0}
    std = np.array([i["std"] for i in infos])
    return {
        "n_groups": len(infos),
        "informative_rate": float(np.mean([i["inf"] for i in infos])),
        "informative_rate_strict": float(np.mean([i["inf_strict"] for i in infos])),
        "mean_group_std": float(std.mean()),
        "var_centered": float(np.mean([i["var_c"] for i in infos])),  # pooled over completions (equal group size)
        "var_norm_adv": float(np.mean([i["var_adv"] for i in infos])),
        "std_of_group_std": float(std.std()),
        "trunc_frac": float(np.mean([i["trunc"] for i in infos])),
    }


def difficulty_bins(by_prompt, n_bins):
    """Rule (defined once, independent of K): rank prompts by mean reward over all 8 cached completions and
    split into n_bins equal-size, rank-based bins. Lowest mean reward = 'hard'."""
    keys = list(by_prompt)
    means = np.array([np.mean([float(r["reward"]) for r in by_prompt[k][:8]]) for k in keys])
    order = np.argsort(means, kind="stable")
    ranks = np.empty(len(keys), int)
    ranks[order] = np.arange(len(keys))
    b = (ranks * n_bins) // len(keys)
    labels = ["hard", "medium", "easy"] if n_bins == 3 else [f"bin{i}" for i in range(n_bins)]
    pid2label = {k: labels[int(x)] for k, x in zip(keys, b)}
    info = {}
    for i, lab in enumerate(labels):
        sel = means[b == i]
        info[lab] = {"n_prompts": int((b == i).sum()), "mean_reward_min": float(sel.min()),
                     "mean_reward_max": float(sel.max()), "mean_reward_avg": float(sel.mean())}
    return pid2label, labels, info, dict(zip(keys, means.tolist()))


def per_prompt(infos):
    d = defaultdict(list)
    for i in infos:
        d[i["pid"]].append(i)
    return d


def bootstrap(pp, pids, reps, seed):
    rng = np.random.RandomState(seed)
    draws = {m: [] for m in METRICS}
    for _ in range(reps):
        samp = rng.randint(0, len(pids), len(pids))
        agg = aggregate([i for s in samp for i in pp[pids[s]]])
        for m in METRICS:
            draws[m].append(agg[m])
    return {m: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))] for m, v in draws.items()}


def analyze(by_prompt, cfg):
    tol = float(cfg.get("informative_tol", 1e-6))
    tol_s = float(cfg.get("informative_tol_strict", 1e-2))
    reps = int(cfg.get("bootstrap_reps", 1000))
    n_bins = int(cfg.get("n_bins", 3))
    aseed = int(cfg.get("analysis_seed", cfg["seed"]))
    perm_reps = int(cfg.get("permutation_reps", 100))
    pid2label, labels, bin_info, prompt_means = difficulty_bins(by_prompt, n_bins)
    strata = {"all": list(by_prompt)} | {lab: [p for p in by_prompt if pid2label[p] == lab] for lab in labels}

    res = {"definitions": {
        "regrouping": "first 8 completions/prompt (by generation_index) split into 8/K disjoint consecutive blocks",
        "std": "population std (ddof=0)", "informative": f"group std > {tol}", "informative_strict": f"group std > {tol_s}",
        "variance_of_group_relative_signal": "var_centered = mean over groups of mean((r - mu_group)^2) (unnormalized centered advantage)",
        "var_norm_adv": "pooled variance of z-scored advantage (sanity: ~= informative rate)",
        "difficulty_bins": f"{n_bins} rank-based bins of per-prompt mean reward over all 8 completions (hard=lowest)",
        "bootstrap": f"{reps} resamples over prompts (within stratum), 95% percentile CI",
    }, "bins": bin_info, "by_k": {}, "permutation": {}}

    # prompt-level difficulty vs variability (K=8 -> one group/prompt)
    g8 = [group_info(g, tol, tol_s) for g in regroup_equal_generation_budget(by_prompt, 8)]
    keys = list(by_prompt)
    res["corr_prompt_mean_reward_vs_group_std_k8"] = float(np.corrcoef(
        [prompt_means[k] for k in keys], [next(i["std"] for i in g8 if i["pid"] == k) for k in keys])[0, 1])

    for k in cfg["group_sizes"]:
        k = int(k)
        infos = [group_info(g, tol, tol_s) for g in regroup_equal_generation_budget(by_prompt, k)]
        pp = per_prompt(infos)
        entry = {"n_completions": sum(len(g["rows"]) for g in regroup_equal_generation_budget(by_prompt, k)), "strata": {}}
        for s, pids in strata.items():
            agg = aggregate([i for p in pids for i in pp[p]])
            ci = bootstrap(pp, pids, reps, aseed)
            entry["strata"][s] = {"n_prompts": len(pids), "n_groups": agg["n_groups"],
                                  **{m: {"value": agg[m], "ci": ci[m]} for m in METRICS}}
        res["by_k"][str(k)] = entry

        # permutation variant: average over random partitions of each prompt's 8 completions
        accum = {s: [] for s in strata}
        for r in range(perm_reps):
            rng = np.random.RandomState(aseed + r)
            infos_r = [group_info(g, tol, tol_s) for g in regroup_equal_generation_budget(by_prompt, k, rng)]
            ppr = per_prompt(infos_r)
            for s, pids in strata.items():
                accum[s].append(aggregate([i for p in pids for i in ppr[p]]))
        res["permutation"][str(k)] = {
            s: {m: {"mean": float(np.nanmean([a[m] for a in v])), "std_over_partitions": float(np.nanstd([a[m] for a in v]))}
                for m in METRICS}
            for s, v in accum.items()}
    return res, pid2label, labels


# ----------------------------------------------------------------------------- outputs
def _trunc(s, n=300):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ..."


def write_examples(by_prompt, pid2label, labels, cfg, path):
    prompt_text = {}
    try:
        for r in read_jsonl(cfg["paths"]["rl_prompt_train"]):
            m = r.get("messages") or []
            users = [x["content"] for x in m if isinstance(x, dict) and x.get("role") == "user"]
            prompt_text[str(r.get("prompt_id"))] = users[-1] if users else ""
    except Exception:
        pass
    lines = ["# Group-informativeness examples (cache, K=8 groups; deterministic selection)\n"]
    g8 = regroup_equal_generation_budget(by_prompt, 8)
    for lab in labels:
        gs = sorted([g for g in g8 if pid2label[g["source_index"]] == lab], key=lambda g: g["rewards"].std())
        for title, g in (("lowest-std group", gs[0]), ("highest-std group", gs[-1])):
            rows = g["rows"]
            best = max(rows, key=lambda r: float(r["reward"]))
            worst = min(rows, key=lambda r: float(r["reward"]))
            pid = rows[0].get("prompt_id")
            lines.append(f"## [{lab}] {title}: source_index={g['source_index']} prompt_id={pid} "
                         f"reward std={g['rewards'].std():.3f}, rewards={np.round(g['rewards'], 2).tolist()}")
            lines.append(f"Prompt: {_trunc(prompt_text.get(str(pid), '(prompt text not found)'), 250)}\n")
            lines.append(f"- best (r={float(best['reward']):.2f}, {best.get('completion_tokens')} tok, "
                         f"clipped={best.get('clipped_at_max')}): {_trunc(best['completion'])}")
            lines.append(f"- worst (r={float(worst['reward']):.2f}, {worst.get('completion_tokens')} tok, "
                         f"clipped={worst.get('clipped_at_max')}): {_trunc(worst['completion'])}\n")
    # uninformative / lowest-std *training* groups from the standard continuation (if it was run)
    log_p = repo_path(cfg["results_dir"]) / "standard_train_log.jsonl"
    if log_p.exists():
        log = read_jsonl(log_p)
        unin = [r for r in log if r["uninformative_fraction"] > 0]
        lines.append(f"## Standard continuation: {len(unin)}/{len(log)} updates had an uninformative group")
        for r in unin[:5]:
            lines.append(f"- update {r['update']} prompt {r['prompt_ids']} rewards={np.round(r['rewards'], 2).tolist()}")
        lo = sorted(log, key=lambda r: r["group_reward_std"])[:3]
        lines.append("### Lowest within-group std training groups")
        for r in lo:
            lines.append(f"- update {r['update']} prompt {r['prompt_ids']} std={r['group_reward_std']:.3f} "
                         f"rewards={np.round(r['rewards'], 2).tolist()}")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_csv(res, path):
    cols = ["K", "stratum", "n_prompts", "n_groups"] + [f"{m}{suf}" for m in METRICS for suf in ("", "_lo", "_hi")]
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(cols)
        for k, e in res["by_k"].items():
            for s, d in e["strata"].items():
                row = [k, s, d["n_prompts"], d["n_groups"]]
                for m in METRICS:
                    row += [d[m]["value"], d[m]["ci"][0], d[m]["ci"][1]]
                w.writerow(row)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    by_prompt = load_k8_cache(cfg["group_cache"])
    print("Cached prompts:", len(by_prompt))
    print("Group sizes to analyze:", cfg["group_sizes"])
    first = next(iter(by_prompt.values()))
    print("Cache row keys:", sorted(first[0].keys()))

    res, pid2label, labels = analyze(by_prompt, cfg)
    res["n_prompts"] = len(by_prompt)
    res["cache_generation_cap"] = cfg.get("cache_generation_cap")
    out = repo_path(cfg["results_dir"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "group_size_results.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    write_csv(res, out / "group_size_table.csv")
    write_examples(by_prompt, pid2label, labels, cfg, out / "group_size_examples.md")

    print(f"\n{'K':>2} {'stratum':>7} {'inf%':>6} {'inf%(1e-2)':>10} {'mean std':>9} {'var_c':>8} {'var_adv':>8}")
    for k, e in res["by_k"].items():
        for s, d in e["strata"].items():
            print(f"{k:>2} {s:>7} {d['informative_rate']['value']:6.2f} {d['informative_rate_strict']['value']:10.2f} "
                  f"{d['mean_group_std']['value']:9.3f} {d['var_centered']['value']:8.3f} {d['var_norm_adv']['value']:8.3f}")
    print("bins:", json.dumps(res["bins"], indent=1))
    print("Saved to", out)


if __name__ == "__main__":
    main()
