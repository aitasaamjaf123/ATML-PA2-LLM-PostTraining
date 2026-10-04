from __future__ import annotations

import argparse

from common.data import load_yaml
from common.logging_utils import save_json
from common.models import clear_gpu, load_reward_model, load_tokenizer
from task1_dpo.evaluate import evaluate_adapter
from task1_dpo.train import run_training


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--max-gen-prompts", type=int, default=200)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--stratum-key", default=None)
    ap.add_argument("--wl-samples", type=int, default=1)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    balanced_adapter = cfg["length_output"]

    # 1. Train the length-balanced model from the ORIGINAL init: same beta/lr/seed, balanced file.
    if not args.skip_train:
        run_training(args.config, "length_study/balanced_train", cfg["paths"]["dpo_length_train"], balanced_adapter)
        clear_gpu()

    # 2. Evaluate standard vs balanced under an identical protocol.
    tok = load_tokenizer(cfg["base_model"])
    rm = load_reward_model(cfg)
    ev = dict(stratified=True, max_gen_prompts=args.max_gen_prompts, max_pairs=args.max_pairs,
              eval_batch_size=args.eval_batch_size, stratum_key=args.stratum_key, wl_samples=args.wl_samples)
    summaries = {}
    for name, adapter in (("standard", cfg["standard_output"]), ("length_balanced", balanced_adapter)):
        summaries[name] = evaluate_adapter(cfg, adapter, f"length_study/{name}", tok, rm, **ev)
        clear_gpu()
    save_json(f"{cfg['results_dir']}/length_study/summary.json", summaries)

    # 3. Comparison table
    for name, r in summaries.items():
        st = {k: v for k, v in r["stratified_pairs"].items() if k != "stratum_key"}
        strata = "  ".join(f"{k}: acc={v['pref_acc']:.3f} (n={v['n']})" for k, v in st.items())
        wl = r.get("word_limit", {})
        print(f"\n[{name}] {strata}")
        print(f"   gen length = {r['generation']['length_mean']:.1f} tok | "
              f"word-limit compliance = {wl.get('compliance_rate')} "
              f"(strict-under: {wl.get('compliance_rate_strict_under')}, "
              f"parsed {wl.get('n_limit_parsed')}/{wl.get('n_generations')})")


if __name__ == "__main__":
    main()