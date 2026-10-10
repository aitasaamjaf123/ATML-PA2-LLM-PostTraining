from __future__ import annotations

import os

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import argparse

from common.data import load_yaml
from task2_ppo import utils as U


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ppo.yaml")
    ap.add_argument("--seeds", type=int, nargs="+", default=U.SEEDS)
    ap.add_argument("--force", action="store_true", help="re-run finished conditions")
    ap.add_argument("--summarize-only", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("KL beta conditions:", cfg["kl_values"])
    print("Fork update budget:", cfg["fork_updates"])

    eps = float(cfg["clip_epsilon"])  # 0.20, shared with the clipping study's beta=0.10 fork
    conditions = [{"label": f"beta{float(b):g}", "eps": eps, "beta": float(b)} for b in cfg["kl_values"]]

    failed = []
    if not args.summarize_only:
        # every fork restarts from the supplied midpoint in its own process; finished runs are skipped
        failed = U.run_grid(conditions, args.seeds, args.config, int(cfg["fork_updates"]), args.force)

    summary = U.summarize_study(conditions, args.seeds, "kl_study")
    print("\nKL study (mean over seeds):")
    for label, e in summary["conditions"].items():
        ev = e["eval"]
        f = lambda k: "n/a" if ev[k]["mean"] is None else f"{ev[k]['mean']:.3f}"
        print(f"  {label:9s} reward={f('reward_mean')} KL={f('kl_token')} entropy={f('entropy')} "
              f"length={f('length_mean')} trunc={f('truncation_rate')} (n={ev['reward_mean']['n']})")
    print("Summary + reward/length candidate cases -> results/task2_ppo/kl_study.json")
    if any(v.get("flag_gt_10pct") for v in summary["token_budget"].values()):
        print("WARNING: generated-token budgets differ by >10% between conditions (see token_budget)")
    if failed:
        raise SystemExit(f"failed conditions: {failed}")


if __name__ == "__main__":
    main()