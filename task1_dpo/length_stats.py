from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from common.data import (
    encode_prompt_response,
    load_yaml,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
)
from common.logging_utils import save_json
from common.models import load_tokenizer

STRATUM_KEYS = ("stratum", "length_stratum", "length_bucket", "bucket", "length_bin",
                "length_category", "category", "length_relation")


def find_stratum_key(rows, override=None, required=True):
    """Return the name of the stratum field. Never guesses silently."""
    first = rows[0]
    if override and override in first:
        return override
    for k in STRATUM_KEYS:
        if k in first:
            return k
    if required:
        raise KeyError(f"No stratum field found. Row keys: {list(first)}. "
                       f"Re-run with --stratum-key <name>.")
    return None


def _ids(x):
    return list(x["input_ids"]) if hasattr(x, "keys") else list(x)


def side_lengths(tokenizer, prompt_msgs, response, max_length):
    """Replicates the truncation rule of common.data.encode_prompt_response, but only measures."""
    p = len(_ids(tokenizer.apply_chat_template(prompt_msgs, tokenize=True, add_generation_prompt=True)))
    r = len(tokenizer(response + (tokenizer.eos_token or ""), add_special_tokens=False)["input_ids"])
    total = p + r
    if total <= max_length:
        status, p_kept, r_kept = "intact", p, r
    elif r >= max_length:                       # response alone fills the window
        status, p_kept, r_kept = "prompt_lost", 0, max_length
    else:                                       # prompt shortened from the left, response intact
        status, p_kept, r_kept = "prompt_truncated", max_length - r, r
    return {"prompt_len": p, "resp_len": r, "total_len": total, "status": status,
            "prompt_kept": p_kept, "resp_kept": r_kept, "excess": max(0, total - max_length)}


def describe(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return None
    return {"mean": round(float(x.mean()), 2), "std": round(float(x.std()), 2),
            "min": round(float(x.min()), 2), "median": round(float(np.median(x)), 2),
            "p90": round(float(np.percentile(x, 90)), 2), "p95": round(float(np.percentile(x, 95)), 2),
            "max": round(float(x.max()), 2)}


def summarize_side(df: pd.DataFrame, max_length: int):
    n = len(df)
    trunc = df[df.status != "intact"]
    pt = df[df.status == "prompt_truncated"]
    pl = df[df.status == "prompt_lost"]
    resp_lost = df.resp_len - df.resp_kept
    return {
        "n": n,
        "n_truncated": len(trunc),
        "pct_truncated": round(100 * len(trunc) / max(n, 1), 2),
        "n_prompt_truncated_response_intact": len(pt),
        "n_prompt_lost_entirely": len(pl),
        "n_response_alone_ge_max_length": int((df.resp_len >= max_length).sum()),
        "n_response_tokens_cut": int((resp_lost > 0).sum()),
        "n_prompt_left_under_64_tokens": int((pt.prompt_kept < 64).sum()),
        "excess_tokens_over_limit (truncated rows)": describe(trunc.excess),
        "prompt_fraction_removed (prompt_truncated rows)":
            describe((pt.prompt_len - pt.prompt_kept) / pt.prompt_len) if len(pt) else None,
        "response_tokens_cut (rows where response was cut)": describe(resp_lost[resp_lost > 0]),
        "prompt_len": describe(df.prompt_len),
        "resp_len": describe(df.resp_len),
        "total_len": describe(df.total_len),
    }


def truncation_report(tokenizer, rows, max_length, tag, validate=True, stratum_key=None):
    chosen, rejected = [], []
    for row in rows:
        pm = prompt_messages_from_preference(row)
        yc, yr = preference_responses(row)
        chosen.append(side_lengths(tokenizer, pm, yc, max_length))
        rejected.append(side_lengths(tokenizer, pm, yr, max_length))
    dfc, dfr = pd.DataFrame(chosen), pd.DataFrame(rejected)

    if validate:  # check my replica against the real encoder on a few rows
        for i in range(min(25, len(rows))):
            pm = prompt_messages_from_preference(rows[i])
            yc, _ = preference_responses(rows[i])
            ids, mask = encode_prompt_response(tokenizer, pm, yc, max_length)
            c = chosen[i]
            assert len(ids) == min(c["total_len"], max_length), f"length mismatch row {i}"
            assert int(sum(mask)) == c["resp_kept"], f"mask mismatch row {i}"

    diff = dfc.resp_len - dfr.resp_len
    any_t = (dfc.status != "intact") | (dfr.status != "intact")
    both_t = (dfc.status != "intact") & (dfr.status != "intact")
    summary = {
        "tag": tag, "max_length": max_length, "n_pairs": len(rows),
        "chosen": summarize_side(dfc, max_length),
        "rejected": summarize_side(dfr, max_length),
        "pairs": {
            "chosen_longer_frac": round(float((diff > 0).mean()), 4),
            "equal_len_frac": round(float((diff == 0).mean()), 4),
            "rejected_longer_frac": round(float((diff < 0).mean()), 4),
            "mean_resp_len_diff (chosen - rejected)": round(float(diff.mean()), 2),
            "median_resp_len_diff": round(float(diff.median()), 2),
            "pairs_with_any_truncation": int(any_t.sum()),
            "pairs_truncated_on_one_side_only": int((any_t & ~both_t).sum()),
        },
    }
    if stratum_key:
        strata = pd.Series([str(r[stratum_key]) for r in rows])
        summary["stratum_key"] = stratum_key
        summary["by_stratum"] = {
            name: {"n": int(m.sum()),
                   "mean_resp_len_diff (chosen - rejected)": round(float(diff[m.values].mean()), 2),
                   "chosen_longer_frac": round(float((diff[m.values] > 0).mean()), 4),
                   "rejected_longer_frac": round(float((diff[m.values] < 0).mean()), 4)}
            for name in sorted(strata.unique()) for m in [strata == name]
        }
    print_report(summary)
    return summary


def print_report(s):
    print(f"\n=== Truncation report: {s['tag']}  (max_length={s['max_length']}, pairs={s['n_pairs']}) ===")
    for side in ("chosen", "rejected"):
        d = s[side]
        print(f"[{side}] truncated: {d['n_truncated']}/{d['n']} ({d['pct_truncated']}%)")
        print(f"   prompt shortened, response intact : {d['n_prompt_truncated_response_intact']}")
        print(f"   prompt lost entirely (resp>=max)  : {d['n_prompt_lost_entirely']}")
        print(f"   rows where response itself is cut : {d['n_response_tokens_cut']}")
        print(f"   prompt left with <64 tokens       : {d['n_prompt_left_under_64_tokens']}")
        print(f"   excess tokens over limit          : {d['excess_tokens_over_limit (truncated rows)']}")
        print(f"   response length (tokens)          : {d['resp_len']}")
    print(f"[pairs] {s['pairs']}")
    if "by_stratum" in s:
        print(f"[by {s['stratum_key']}] {s['by_stratum']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--stratum-key", default=None)
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    tok = load_tokenizer(cfg["base_model"])
    max_len = int(cfg["max_sequence_length"])
    out = {}
    for key in ("dpo_standard_train", "dpo_standard_eval", "dpo_length_train", "dpo_length_eval"):
        rows = read_jsonl(cfg["paths"][key])
        sk = find_stratum_key(rows, args.stratum_key, required=False) if key.startswith("dpo_length") else None
        out[key] = truncation_report(tok, rows, max_len, key, stratum_key=sk)
    save_json(f"{cfg['results_dir']}/length_stats.json", out)


if __name__ == "__main__":
    main()