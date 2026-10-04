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
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--max-gen-prompts", type=int, default=200)
    ap.add_argument("--max-pairs", type=int, default=None)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    n = int(cfg["short_ablation_examples"])
    print("Beta values:", cfg["betas"], "| short-run examples:", n)

    tok = load_tokenizer(cfg["base_model"])
    rm = None if args.skip_eval else load_reward_model(cfg)
    ev = dict(max_gen_prompts=args.max_gen_prompts, max_pairs=args.max_pairs, eval_batch_size=args.eval_batch_size)
    results = {}

    if not args.skip_eval:   # untouched-model anchor row (KL = 0, accuracy = 0 by construction)
        results["base_init"] = evaluate_adapter(cfg, "base", "beta_study/base_init", tok, rm, **ev)
        clear_gpu()

    for beta in cfg["betas"]:
        name = f"beta_{beta:g}"
        adapter = f"outputs/task1_dpo/{name}"
        if not args.skip_train:
            run_training(args.config, f"beta_study/{name}", None, adapter, beta=float(beta), max_examples=n)
            clear_gpu()
        if not args.skip_eval:
            results[name] = evaluate_adapter(cfg, adapter, f"beta_study/{name}", tok, rm, beta=float(beta), **ev)
            clear_gpu()

    if results:
        save_json(f"{cfg['results_dir']}/beta_study/summary.json", results)
        print(f"{'run':<12}{'loss':>8}{'acc':>8}{'margin':>9}{'KL':>9}{'reward':>9}{'len':>8}")
        for k, r in results.items():
            h, g = r.get("heldout_pairs", {}), r.get("generation", {})
            nan = float("nan")
            print(f"{k:<12}{h.get('dpo_loss', nan):>8.3f}{h.get('pref_acc', nan):>8.3f}"
                  f"{h.get('mean_margin', nan):>9.3f}{g.get('kl_token_mean', nan):>9.4f}"
                  f"{g.get('reward_mean', nan):>9.3f}{g.get('length_mean', nan):>8.1f}")


if __name__ == "__main__":
    main()