from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict

import numpy as np
import torch

from common.data import REPO_ROOT, load_yaml, read_jsonl, repo_path
from common.generation import batch_generate, response_token_logprobs, score_reward_pairs
from common.logging_utils import load_json, save_json, set_seed
from common.models import clear_gpu, load_policy, load_reward_model, load_tokenizer, trainable_parameters
from task3_grpo.grpo import group_relative_advantages, grpo_policy_loss, mask_truncated_sequences
from task3_grpo.utils import (
    bootstrap_ci,
    bootstrap_ci_paired,
    grad_norm,
    length_weight_shares,
    normalize_gen,
    np_corr,
    prompt_order,
    results_dir,
    row_messages,
    row_prompt_id,
)

LOSSES = ["grpo", "dr_grpo"]


def run_name(loss: str, seed: int) -> str:
    return f"norm_{loss}_s{seed}"


def _seeds(cfg):
    return [int(s) for s in cfg.get("normalization_seeds", [cfg["seed"]])]


def _run(cmd):
    print("$", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=str(REPO_ROOT))


# ----------------------------------------------------------------------------- stage: train
def stage_train(config_path, cfg, force):
    rdir = results_dir(cfg)
    for s in _seeds(cfg):
        for loss in LOSSES:  # interleaved per seed
            name = run_name(loss, s)
            if (rdir / f"{name}_summary.json").exists() and not force:
                print(f"skip {name} (done)")
                continue
            _run([sys.executable, "-m", "task3_grpo.continue_train", "--config", config_path,
                  "--output", f"outputs/task3_grpo/{name}", "--updates", str(cfg["fork_updates"]),
                  "--loss-type", loss, "--run-name", name, "--seed", str(s), "--log-term-grads"])


# ----------------------------------------------------------------------------- stage: eval
def stage_eval(config_path, cfg, force):
    rdir = results_dir(cfg)
    targets = [("midpoint", cfg["paths"]["grpo_midpoint_policy"])]
    if (repo_path(cfg["output"]) / "adapter_config.json").exists():
        targets.append(("standard", cfg["output"]))
    for s in _seeds(cfg):
        for loss in LOSSES:
            name = run_name(loss, s)
            if (repo_path(f"outputs/task3_grpo/{name}") / "adapter_config.json").exists():
                targets.append((name, f"outputs/task3_grpo/{name}"))
    for name, adapter in targets:
        if (rdir / f"eval_{name}.json").exists() and not force:
            print(f"skip eval {name} (done)")
            continue
        _run([sys.executable, "-m", "task3_grpo.evaluate", "--config", config_path, "--adapter", adapter, "--name", name])


# ----------------------------------------------------------------------------- stage: probe
def run_probe(config_path, cfg, force=False):
    """Empirical per-sequence gradient norms on a fixed probe set, both losses on identical rollouts."""
    rdir = results_dir(cfg)
    out_json = rdir / "probe_gradient_stats.json"
    if out_json.exists() and not force:
        print("skip probe (done)")
        return load_json(out_json)
    seed = int(cfg.get("probe_seed", cfg["seed"]))
    set_seed(seed)
    tok = load_tokenizer(cfg["base_model"])
    policy = load_policy(cfg, adapter_path=cfg["paths"]["grpo_midpoint_policy"], trainable=True)
    try:
        policy.generation_config.use_cache = True
    except Exception:
        pass
    rm, rm_tok = load_reward_model(cfg)
    rows = read_jsonl(cfg["paths"]["rl_prompt_train"])
    P, K = int(cfg.get("probe_prompts", 16)), int(cfg["num_generations"])
    max_new, max_prompt = int(cfg["max_completion_length"]), int(cfg["max_prompt_length"])
    eps = float(cfg["clip_epsilon"])
    gk = cfg["generation"]
    mask_trunc = bool(cfg.get("mask_truncated_completions", True))
    idxs = prompt_order(len(rows), seed)[-P:]  # last P of the seeded permutation
    params = trainable_parameters(policy)
    device = next(policy.parameters()).device

    recs = []
    for i, idx in enumerate(idxs):
        msgs = row_messages(rows[idx])
        pid = row_prompt_id(rows[idx], idx)
        set_seed(seed + 1000 + i)
        gen = normalize_gen(batch_generate(
            policy, tok, [msgs] * K, max_prompt, max_new,
            temperature=float(gk["temperature"]), top_p=float(gk["top_p"]), do_sample=bool(gk["do_sample"])))
        rewards = score_reward_pairs(rm, rm_tok, [msgs] * K, gen["responses"], max_length=1024).float().cpu()
        adv = group_relative_advantages(rewards, torch.zeros(K, dtype=torch.long)).to(device)
        full = gen["response_mask"].float()
        mask = mask_truncated_sequences(full, gen["truncated"]) if mask_trunc else full
        policy.eval()  # no dropout -> deterministic gradients
        for k in range(K):
            kept = float(mask[k].sum()) > 0
            gns = {}
            for lt in LOSSES:
                if not kept:
                    gns[lt] = 0.0
                    continue
                for p in params:
                    p.grad = None
                lp, _ = response_token_logprobs(policy, gen["sequences"][k : k + 1], gen["attention_mask"][k : k + 1],
                                                gen["prompt_width"], gen["response_ids"][k : k + 1])
                loss, _m = grpo_policy_loss(lp, lp.detach(), adv[k : k + 1], mask[k : k + 1], lp.detach(),
                                            eps, 0.0, loss_type=lt, max_completion_length=max_new)
                loss.backward()
                gns[lt] = grad_norm(params)
                del lp, loss, _m, _
            recs.append({"prompt_id": pid, "group": i, "k": k, "length": int(gen["response_lengths"][k]),
                         "eff_length": float(mask[k].sum()), "kept": kept, "truncated": bool(gen["truncated"][k]),
                         "reward": float(rewards[k]), "advantage": float(adv[k]),
                         "grad_norm_grpo": gns["grpo"], "grad_norm_dr_grpo": gns["dr_grpo"],
                         "completion": gen["responses"][k]})
        for p in params:
            p.grad = None
        print(f"  probe {i + 1}/{P}")
        del gen

    # ---- aggregate
    kept = [r for r in recs if r["kept"]]
    L = np.array([r["length"] for r in kept], float)
    g = {"grpo": np.array([r["grad_norm_grpo"] for r in kept]), "dr_grpo": np.array([r["grad_norm_dr_grpo"] for r in kept])}
    ranks = np.argsort(np.argsort(L, kind="stable"), kind="stable")
    terc = (ranks * 3) // max(len(L), 1)
    stats = {"n_total": len(recs), "n_kept": len(kept), "n_masked_truncated": len(recs) - len(kept),
             "probe_prompt_ids": [row_prompt_id(rows[i], i) for i in idxs],
             "length_tercile_bounds": [float(L[terc == t].min()) if (terc == t).any() else None for t in range(3)] +
                                       [float(L.max())],
             "corr_length_gradnorm": {}, "tercile_gradnorm": {}, "long_over_short_ratio": {}}
    for lt in LOSSES:
        stats["corr_length_gradnorm"][lt] = {"value": np_corr(L, g[lt]),
                                              "ci": bootstrap_ci_paired(L, g[lt], np_corr, 1000, seed)}
        stats["tercile_gradnorm"][lt] = {n: float(g[lt][terc == t].mean()) for t, n in enumerate(["short", "mid", "long"])}
        ratio = lambda idx_arr, lt=lt: float(g[lt][idx_arr][terc[idx_arr] == 2].mean() / g[lt][idx_arr][terc[idx_arr] == 0].mean())
        rng = np.random.RandomState(seed)
        draws = []
        for _ in range(1000):
            b = rng.randint(0, len(L), len(L))
            if (terc[b] == 2).any() and (terc[b] == 0).any():
                draws.append(ratio(b))
        stats["long_over_short_ratio"][lt] = {"value": ratio(np.arange(len(L))),
                                               "ci": [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]}
    # sanity: ||grad_dr|| / ||grad_grpo|| must equal T_eff/max_len exactly (same forward, scalar rescaling)
    errs = [abs(r["grad_norm_dr_grpo"] / max(r["grad_norm_grpo"], 1e-12) - r["eff_length"] / max_new) /
            (r["eff_length"] / max_new) for r in kept if r["grad_norm_grpo"] > 0]
    stats["proportionality_max_rel_err"] = float(max(errs)) if errs else None
    adv_all = np.array([r["advantage"] for r in recs])
    len_all = np.array([r["length"] for r in recs], float)
    eff_all = np.array([r["eff_length"] for r in recs], float)
    stats["weight_share_long_half"] = length_weight_shares(adv_all, eff_all, len_all, K, max_new)
    cen = np.concatenate([len_all[j * K : (j + 1) * K] - len_all[j * K : (j + 1) * K].mean() for j in range(P)])
    stats["len_adv_corr_pooled_within_group"] = np_corr(cen, adv_all)
    stats["mean_gradnorm"] = {lt: float(g[lt].mean()) for lt in LOSSES}
    save_json(out_json, stats)
    with (rdir / "probe_records.jsonl").open("w", encoding="utf-8") as f:
        import json
        for r in recs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    clear_gpu(policy, rm)
    return stats


# ----------------------------------------------------------------------------- stage: analyze
def _ms(vals):
    a = np.array(vals, float)
    return {"mean": float(np.nanmean(a)), "std": float(np.nanstd(a, ddof=1)) if len(a) > 1 else 0.0,
            "values": [float(x) for x in a]}


TRAIN_KEYS = ["reward_mean", "kl", "kl_seq", "length_mean", "grad_norm", "entropy", "truncation_rate",
              "uninformative_fraction", "group_reward_std", "policy_term", "grad_norm_policy_term",
              "grad_norm_kl_term", "weight_share_long_half_grpo", "weight_share_long_half_dr_grpo"]
EVAL_KEYS = ["reward_mean", "kl_token_pooled", "kl_seq_mean", "length_mean", "length_std", "length_median",
             "truncation_rate", "entropy_sampled_token", "corr_length_reward"]


def _train_metrics(log):
    out = {k: float(np.nanmean([r[k] if r[k] is not None else np.nan for r in log])) for k in TRAIN_KEYS}
    cen, adv = [], []
    for r in log:
        K, L, A = r["num_generations"], np.array(r["lengths"], float), np.array(r["advantages"], float)
        for g in range(len(L) // K):
            cen.extend((L[g * K : (g + 1) * K] - L[g * K : (g + 1) * K].mean()).tolist())
            adv.extend(A[g * K : (g + 1) * K].tolist())
    out["len_adv_corr_pooled"] = np_corr(cen, adv)
    return out


def _trunc(s, n=350):
    s = str(s).replace("\n", " ")
    return s if len(s) <= n else s[:n] + " ..."


def stage_analyze(cfg):
    rdir = results_dir(cfg)
    seeds = _seeds(cfg)
    summary = {"seeds": seeds, "fork_updates": cfg["fork_updates"], "train": {}, "eval": {}, "reference_eval": {}, "paired": {}}

    for loss in LOSSES:
        logs = [read_jsonl(rdir / f"{run_name(loss, s)}_train_log.jsonl") for s in seeds
                if (rdir / f"{run_name(loss, s)}_train_log.jsonl").exists()]
        if logs:
            per = [_train_metrics(l) for l in logs]
            summary["train"][loss] = {k: _ms([p[k] for p in per]) for k in per[0]}
        evs = [load_json(rdir / f"eval_{run_name(loss, s)}.json") for s in seeds
               if (rdir / f"eval_{run_name(loss, s)}.json").exists()]
        if evs:
            summary["eval"][loss] = {k: _ms([e[k] for e in evs]) for k in EVAL_KEYS}
    for ref in ("midpoint", "standard"):
        if (rdir / f"eval_{ref}.json").exists():
            e = load_json(rdir / f"eval_{ref}.json")
            summary["reference_eval"][ref] = {k: e[k] for k in EVAL_KEYS} | {"reward_se": e["reward_se"]}

    # ---- paired GRPO - Dr.GRPO on identical held-out prompts (averaged over seeds, bootstrap over prompts)
    recs = {}
    for loss in LOSSES:
        for s in seeds:
            p = rdir / f"eval_{run_name(loss, s)}_records.jsonl"
            if p.exists():
                recs[(loss, s)] = {r["prompt_id"]: r for r in read_jsonl(p)}
    ok = [s for s in seeds if ("grpo", s) in recs and ("dr_grpo", s) in recs]
    examples = []
    if ok:
        pids = sorted(set.intersection(*[set(recs[(l, s)]) for l in LOSSES for s in ok]))
        fields = {"reward": "reward", "length": "length", "kl_tok_mean": "kl_tok_mean"}
        arr = {f: {l: np.array([[recs[(l, s)][p][key] for p in pids] for s in ok], float) for l in LOSSES}
               for f, key in fields.items()}
        for f in fields:
            d_seed = arr[f]["grpo"] - arr[f]["dr_grpo"]  # [seeds, prompts]
            d = d_seed.mean(0)
            summary["paired"][f] = {"mean_delta_grpo_minus_dr": float(d.mean()), "ci95": bootstrap_ci(d, np.mean, 1000, 0),
                                    "per_seed_mean_delta": [float(x) for x in d_seed.mean(1)], "n_prompts": len(pids)}
        # qualitative: deterministic selection by seed-averaged |delta length|
        dl = arr["length"]["grpo"].mean(0) - arr["length"]["dr_grpo"].mean(0)
        order = np.argsort(-dl)  # GRPO longer first
        pick = [("GRPO longer", int(i)) for i in order[:2]] + [("Dr.GRPO longer", int(i)) for i in order[::-1][:2]]
        lines = ["# Normalization examples (selected by largest seed-averaged |delta length| on held-out prompts)\n"]
        for tag, i in pick:
            pid = pids[i]
            s0 = ok[0]
            a, b = recs[("grpo", s0)][pid], recs[("dr_grpo", s0)][pid]
            lines += [f"## {tag}: prompt_id={pid}, mean length GRPO={arr['length']['grpo'][:, i].mean():.0f} vs "
                      f"Dr.GRPO={arr['length']['dr_grpo'][:, i].mean():.0f} (seed {s0} shown)",
                      f"- GRPO (len {a['length']}, r={a['reward']:.2f}): {_trunc(a['response'])}",
                      f"- Dr.GRPO (len {b['length']}, r={b['reward']:.2f}): {_trunc(b['response'])}\n"]
        # training-time view: longest negative-advantage completions per loss
        for loss in LOSSES:
            cp = rdir / f"{run_name(loss, ok[0])}_completions.jsonl"
            if cp.exists():
                comps = [c for c in read_jsonl(cp) if c["advantage"] < 0]
                comps.sort(key=lambda c: -c["length"])
                if comps:
                    c = comps[0]
                    lines.append(f"### [{loss}] longest negative-advantage training completion: update {c['update']}, "
                                 f"len {c['length']}, A={c['advantage']:.2f}, r={c['reward']:.2f}, truncated={c['truncated']}\n"
                                 f"{_trunc(c['completion'])}\n")
        (rdir / "normalization_examples.md").write_text("\n".join(lines), encoding="utf-8")

    if (rdir / "probe_gradient_stats.json").exists():
        summary["probe"] = load_json(rdir / "probe_gradient_stats.json")
    save_json(rdir / "normalization_summary.json", summary)

    with (rdir / "normalization_table.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["phase", "loss", "metric", "mean_over_seeds", "std_over_seeds"])
        for ph in ("train", "eval"):
            for loss, d in summary[ph].items():
                for k, v in d.items():
                    w.writerow([ph, loss, k, v["mean"], v["std"]])
    print("train :", {l: round(d["reward_mean"]["mean"], 3) for l, d in summary["train"].items()})
    print("eval  :", {l: {k: round(d[k]["mean"], 3) for k in ("reward_mean", "kl_token_pooled", "length_mean")}
                      for l, d in summary["eval"].items()})
    print("paired:", {k: (round(v["mean_delta_grpo_minus_dr"], 4), [round(x, 4) for x in v["ci95"]])
                      for k, v in summary["paired"].items()})
    print("saved normalization_summary.json / normalization_table.csv / normalization_examples.md")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo.yaml")
    ap.add_argument("--stage", choices=["train", "eval", "probe", "analyze", "all"], default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    print("Fork updates:", cfg["fork_updates"])
    print("Compare loss_type='grpo' vs loss_type='dr_grpo' from the identical supplied midpoint.")
    print("Seeds:", _seeds(cfg))
    if args.stage in ("train", "all"):
        stage_train(args.config, cfg, args.force)
    if args.stage in ("eval", "all"):
        stage_eval(args.config, cfg, args.force)
    if args.stage in ("probe", "all"):
        run_probe(args.config, cfg, args.force)
    if args.stage in ("analyze", "all"):
        stage_analyze(cfg)


if __name__ == "__main__":
    main()
