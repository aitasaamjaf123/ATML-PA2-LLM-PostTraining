"""Controlled reward diagnostic: exact verifier vs pairwise AI judge on the 100-response set.
Run: python -m task5_feedback.score_perturbations
"""
from __future__ import annotations

import argparse
from collections import defaultdict, Counter

import numpy as np

from common.data import load_yaml, read_jsonl, write_jsonl
from common.logging_utils import set_seed
from common.metrics import word_count, safe_corr
from task5_feedback import task5_utils as U
from task5_feedback.rlvr import exact_reward

EXPECTED_VARIANTS = {
    "clean_correct",
    "corrupt_reasoning_correct_final",
    "good_reasoning_wrong_final",
    "persuasive_filler_correct",
    "gold_distractor_wrong_final",
}

# (name, first, second, kind). 'better': first is diagnostically better. 'neutral': both correct (filler test).
PAIR_DEFS = [
    ("reasoning", "clean_correct", "corrupt_reasoning_correct_final", "better"),
    ("outcome_wrong_final", "clean_correct", "good_reasoning_wrong_final", "better"),
    ("filler", "clean_correct", "persuasive_filler_correct", "neutral"),
    ("distractor", "clean_correct", "gold_distractor_wrong_final", "better"),
    ("conflict", "corrupt_reasoning_correct_final", "good_reasoning_wrong_final", "better"),
]
OUT = U.out_path("diagnostics_pairs.jsonl")
METRICS = U.out_path("diagnostics_metrics.json")


def load_diagnostic_groups(path):
    """Strict version of the course loader: duplicates / counts are errors."""
    rows = read_jsonl(path)
    assert len(rows) == 100, f"expected 100 rows, got {len(rows)}"
    seen = Counter((str(r["problem_id"]), r["variant_type"]) for r in rows)
    dups = [k for k, c in seen.items() if c > 1]
    assert not dups, f"duplicate (problem, variant): {dups[:3]}"
    by = defaultdict(dict)
    for r in rows:
        by[str(r["problem_id"])][r["variant_type"]] = r
    assert len(by) == 20, f"expected 20 problems, got {len(by)}"
    for pid, v in by.items():
        assert set(v) == EXPECTED_VARIANTS, f"Problem {pid}: variants {sorted(set(v) ^ EXPECTED_VARIANTS)}"
    return by


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(cfg["seed"])
    groups = load_diagnostic_groups(cfg["paths"]["task5_diagnostics"])
    pids = sorted(groups, key=int)

    # D17 sanity: verifier must reproduce the manually validated labels.
    mism = []
    for pid in pids:
        for vt, r in groups[pid].items():
            if exact_reward(r["response"], str(r["gold_final"])) != float(r["expected_exact_reward"]):
                mism.append((pid, vt))
    print("Verifier label mismatches:", mism)

    judge = U.make_judge(cfg)
    recs, secs = [], []
    for name, fv, sv, kind in PAIR_DEFS:
        for pid in pids:
            a, b = groups[pid][fv], groups[pid][sv]
            q = a["question"]
            ra = exact_reward(a["response"], str(a["gold_final"]))
            rb = exact_reward(b["response"], str(b["gold_final"]))
            vpref = "first" if ra > rb else "second" if rb > ra else "tie"
            res, meta = U.judge_compare(judge, q, a["response"], b["response"])
            jpref = {"A": "first", "B": "second", "TIE": "tie"}[res]
            if not meta["cached"]:
                secs.append(meta["seconds"])
            recs.append({"pair": name, "kind": kind, "problem_id": pid, "first": fv, "second": sv,
                         "verifier_rewards": [ra, rb], "verifier_pref": vpref, "judge_verdict": res,
                         "judge_pref": jpref, "words_first": word_count(a["response"]),
                         "words_second": word_count(b["response"])})
        print(f"[diag] pair {name} done")
    write_jsonl(OUT, recs)

    # aggregate
    def agg(rs, key):
        n = len(rs)
        c = Counter(r[key] for r in rs)
        return {"n": n, "first": c["first"], "tie": c["tie"], "second": c["second"],
                "first_rate": c["first"] / n, "tie_rate": c["tie"] / n, "second_rate": c["second"] / n,
                "first_ci95": U.wilson(c["first"], n), "second_ci95": U.wilson(c["second"], n)}

    M = {"verifier_label_mismatches": mism, "per_pair": {}, "env": U.env_info(cfg)}
    for name, fv, sv, kind in PAIR_DEFS:
        rs = [r for r in recs if r["pair"] == name]
        M["per_pair"][name] = {"kind": kind, "first": fv, "second": sv,
                               "verifier": agg(rs, "verifier_pref"), "judge": agg(rs, "judge_pref")}
    for mech, key in [("verifier", "verifier_pref"), ("judge", "judge_pref")]:
        out = [r for r in recs if r["pair"] in ("outcome_wrong_final", "distractor")]
        M[f"S_outcome_{mech}"] = agg(out, key)
        M[f"S_reason_{mech}"] = M["per_pair"]["reasoning"][mech]
        M[f"filler_susceptibility_{mech}"] = M["per_pair"]["filler"][mech]["second_rate"]
        M[f"distractor_robustness_{mech}"] = M["per_pair"]["distractor"][mech]["first_rate"]
    # length-bias check: (words_first - words_second) vs judge/verifier score for 'first'
    wd = [r["words_first"] - r["words_second"] for r in recs]
    sc = lambda k: [{"first": 1, "tie": .5, "second": 0}[r[k]] for r in recs]
    M["length_bias_corr"] = {"judge": safe_corr(wd, sc("judge_pref")), "verifier": safe_corr(wd, sc("verifier_pref"))}

    # judge parse audit on TIE verdicts
    tie_items = []
    for r in recs:
        if r["judge_verdict"] == "TIE":
            a, b = groups[r["problem_id"]][r["first"]], groups[r["problem_id"]][r["second"]]
            tie_items.append((a["question"], a["response"], b["response"]))
    M["judge_parse_audit"] = U.parse_audit(judge, tie_items)
    M["judge_seconds_mean_uncached"] = float(np.mean(secs)) if secs else None

    # qualitative: first problem (seeded shuffle) per category where verifier and judge disagree
    rng = np.random.default_rng(cfg["seed"])
    order = list(rng.permutation(pids))
    qual = {}
    for name in ["reasoning", "filler", "distractor"]:
        qual[name] = None
        for pid in order:
            r = next(x for x in recs if x["pair"] == name and x["problem_id"] == pid)
            if r["verifier_pref"] != r["judge_pref"]:
                a, b = groups[pid][r["first"]], groups[pid][r["second"]]
                qual[name] = {"problem_id": pid, "question": U.trunc(a["question"], 300),
                              "first_variant": r["first"], "first_response": U.trunc(a["response"]),
                              "second_variant": r["second"], "second_response": U.trunc(b["response"]),
                              "verifier_pref": r["verifier_pref"], "judge_verdict": r["judge_verdict"]}
                break
    M["qualitative"] = qual
    U.save(METRICS, M)

    for name, v in M["per_pair"].items():
        for mech in ("verifier", "judge"):
            d = v[mech]
            print(f"{name:20s} {mech:8s} first={d['first']}/20 tie={d['tie']}/20 second={d['second']}/20")
    print("S_reason  verifier/judge:", M["S_reason_verifier"]["first_rate"], M["S_reason_judge"]["first_rate"])
    print("S_outcome verifier/judge:", M["S_outcome_verifier"]["first_rate"], M["S_outcome_judge"]["first_rate"])


if __name__ == "__main__":
    main()
