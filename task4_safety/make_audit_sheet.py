from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from common.data import load_yaml, read_jsonl, repo_path
from task4_safety.utils import POLICIES, task4_dir


def fixed_audit_ids(base_rows, per_class: int, seed: int):
    rng = np.random.default_rng(seed)
    meta = pd.DataFrame(base_rows)
    ids = []
    for label in ["SAFE", "UNSAFE"]:
        pool = meta.loc[meta["benchmark_class"] == label, "xstest_id"].to_numpy()
        if len(pool) < per_class:
            raise ValueError(f"Not enough {label} rows for audit")
        ids.extend(rng.choice(pool, size=per_class, replace=False).tolist())
    return sorted(int(x) for x in ids)


def build_sheet(outdir: Path, ids, seed: int, overwrite: bool):
    sheet_path = outdir / "manual_audit_sheet.csv"
    key_path = outdir / "manual_audit_key.csv"
    if sheet_path.exists() and not overwrite:
        raise FileExistsError(f"{sheet_path} exists (it may contain your labels). Use --overwrite only if sure.")
    gens = {}
    for p in POLICIES:
        f = outdir / f"generated_{p}.jsonl"
        if not f.exists():
            raise FileNotFoundError(f"Need all four generated files; missing {f}")
        gens[p] = {r["xstest_id"]: r for r in read_jsonl(f)}

    groups = {}  # (id, stripped response) -> {"row":..., "policies":[...]}
    for i in ids:
        for p in POLICIES:
            r = gens[p][i]
            g = groups.setdefault((i, r["response"].strip()), {"row": r, "policies": []})
            g["policies"].append(p)
    items = list(groups.values())
    perm = np.random.default_rng(seed).permutation(len(items))  # fixed shuffle, policy hidden

    sheet, key = [], []
    for new_idx, j in enumerate(perm, start=1):
        it = items[j]
        rid = f"A{new_idx:03d}"
        r = it["row"]
        sheet.append({"row_id": rid, "prompt": r["prompt"], "response": r["response"],
                      "benchmark_class": r["benchmark_class"], "type": r["type"],
                      "manual_label": "", "manual_note": ""})
        for p in it["policies"]:
            key.append({"row_id": rid, "xstest_id": r["xstest_id"], "policy": p})
    pd.DataFrame(sheet).to_csv(sheet_path, index=False, encoding="utf-8-sig")
    pd.DataFrame(key).to_csv(key_path, index=False)
    n_total = len(ids) * len(POLICIES)
    print(f"Wrote {sheet_path}: {len(sheet)} unique texts to label (from {n_total} (id, policy) pairs; "
          f"{n_total - len(sheet)} duplicates removed).")
    print("Wrote key:", key_path, "(do not open until labeling is complete)")
    meta = pd.DataFrame(sheet)
    print("Type coverage of audit IDs:\n", meta.drop_duplicates(["type"]).shape[0], "of 18 types present")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--build-sheet", action="store_true",
                    help="Build blinded, deduplicated labeling sheet + key from the four generated files")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    outdir = task4_dir(cfg)
    src = outdir / "generated_sft.jsonl"
    if not src.exists():
        raise FileNotFoundError("Generate/save SFT responses first: " + str(src))
    ids = fixed_audit_ids(read_jsonl(src), int(cfg["manual_audit_per_class"]), int(cfg["seed"]))
    pd.DataFrame({"xstest_id": ids, "manual_label": [""] * len(ids)}).to_csv(outdir / "manual_audit_ids.csv", index=False)
    print("Wrote fixed audit IDs:", outdir / "manual_audit_ids.csv")
    if args.build_sheet:
        build_sheet(outdir, ids, int(cfg["seed"]), args.overwrite)
    else:
        print("Run again with --build-sheet to create the labeling sheet.")


if __name__ == "__main__":
    main()