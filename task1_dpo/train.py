from __future__ import annotations

import argparse
import math

import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from common.data import (
    encode_prompt_response,
    load_yaml,
    pad_batch,
    preference_responses,
    prompt_messages_from_preference,
    read_jsonl,
    repo_path,
)
from common.generation import response_sequence_logprobs
from common.logging_utils import append_jsonl, save_json, set_seed, wall_timer
from common.models import (
    clear_gpu,
    count_parameters,
    load_policy,
    load_tokenizer,
    reference_mode,
    trainable_parameters,
)
from task1_dpo.dpo import dpo_loss, validate_dpo_loss
from task1_dpo.length_stats import truncation_report


def to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


def make_collate(tokenizer, max_length):
    def collate(rows):
        chosen, rejected = [], []
        for row in rows:
            prompt = prompt_messages_from_preference(row)
            yc, yr = preference_responses(row)
            chosen.append(encode_prompt_response(tokenizer, prompt, yc, max_length))
            rejected.append(encode_prompt_response(tokenizer, prompt, yr, max_length))
        return pad_batch(tokenizer, chosen), pad_batch(tokenizer, rejected)
    return collate


def prepare_dpo_run(config_path: str, dataset_path: str | None = None, beta: float | None = None, max_examples: int | None = None):
    cfg = load_yaml(config_path)
    set_seed(int(cfg["seed"]))
    path = dataset_path or cfg["paths"]["dpo_standard_train"]
    rows = read_jsonl(path)
    if max_examples is not None:
        rows = rows[: int(max_examples)]          # fixed prefix -> identical subset for every beta

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, trainable=True, fresh_lora=True)
    loader = DataLoader(
        rows,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        collate_fn=make_collate(tokenizer, int(cfg["max_sequence_length"])),
    )
    optimizer = AdamW(
        trainable_parameters(model),
        lr=float(cfg["learning_rate"]),
        weight_decay=float(cfg.get("weight_decay", 0.0)),
    )
    return {
        "cfg": cfg, "rows": rows, "tokenizer": tokenizer, "model": model,
        "loader": loader, "optimizer": optimizer, "path": path,
        "beta": float(cfg["beta"] if beta is None else beta),
    }


def run_training(config_path: str, run_name: str, dataset_path: str | None = None,
                 output_path: str | None = None, beta: float | None = None,
                 max_examples: int | None = None):
    validate_dpo_loss()                      # abort BEFORE loading any model if the objective is wrong

    bundle = prepare_dpo_run(config_path, dataset_path, beta, max_examples)
    cfg, model, loader = bundle["cfg"], bundle["model"], bundle["loader"]
    optimizer, tokenizer, rows, beta = bundle["optimizer"], bundle["tokenizer"], bundle["rows"], bundle["beta"]

    output = repo_path(output_path or cfg["standard_output"])
    output.parent.mkdir(parents=True, exist_ok=True)
    res_dir = f"{cfg['results_dir']}/{run_name}"
    log_file = repo_path(f"{res_dir}/train_log.jsonl")
    if log_file.exists():
        log_file.unlink()                    # fresh log for this run

    max_len = int(cfg["max_sequence_length"])
    save_json(f"{res_dir}/truncation_train.json",
              truncation_report(tokenizer, rows, max_len, tag=f"{run_name}/train"))

    accum = int(cfg["grad_accum_steps"])
    max_norm = float(cfg["max_grad_norm"])
    epochs = int(cfg["epochs"])
    bs = int(cfg["batch_size"])
    device = next(model.parameters()).device
    total_p, train_p = count_parameters(model)
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    elapsed = wall_timer()

    model.train()
    optimizer.zero_grad(set_to_none=True)
    n_micro = len(loader)
    step, n_bad, seen = 0, 0, 0
    buf = {"loss": [], "acc": [], "margin": [], "logit": []}
    history = []
    mean = lambda v: sum(v) / len(v) if v else float("nan")

    for epoch in range(epochs):
        for i, (cb, rb) in enumerate(loader):
            cb, rb = to_device(cb, device), to_device(rb, device)
            seen += cb["input_ids"].shape[0]

            # reference log-probs: adapter OFF, no gradients
            with torch.no_grad(), reference_mode(model):
                ref_c, _, _ = response_sequence_logprobs(model, cb)
                ref_r, _, _ = response_sequence_logprobs(model, rb)

            # policy log-probs: adapter ON, gradients tracked
            pol_c, _, _ = response_sequence_logprobs(model, cb)
            pol_r, _, _ = response_sequence_logprobs(model, rb)

            loss, diag = dpo_loss(pol_c, pol_r, ref_c, ref_r, beta)

            # sanity check on the very first micro-batch: LoRA B-matrix starts at 0,
            # so policy == reference and the loss must be ln 2.
            if step == 0 and i == 0:
                assert abs(loss.item() - math.log(2)) < 0.05, (
                    f"Step-0 loss {loss.item():.4f} != ln2={math.log(2):.4f}. "
                    "Reference/policy log-probs disagree at init: check reference_mode / masks.")
                print(f"[sanity] step-0 loss = {loss.item():.4f} (ln2 = {math.log(2):.4f})  OK")

            if torch.isfinite(loss):
                (loss / accum).backward()
                buf["loss"].append(loss.item())
                buf["acc"].append(diag["preference_accuracy"].item())
                buf["margin"].append(diag["reward_margin_mean"].item())
                buf["logit"].append(diag["logit_mean"].item())
            else:
                n_bad += 1

            if (i + 1) % accum == 0 or (i + 1) == n_micro:
                gnorm = torch.nn.utils.clip_grad_norm_(trainable_parameters(model), max_norm)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                step += 1
                rec = {
                    "step": step, "epoch": epoch, "examples_seen": seen,
                    "loss": mean(buf["loss"]), "pref_acc": mean(buf["acc"]),
                    "reward_margin": mean(buf["margin"]), "logit_mean": mean(buf["logit"]),
                    "grad_norm": float(gnorm), "nonfinite_microbatches_so_far": n_bad,
                    "elapsed_s": round(elapsed(), 1),
                }
                history.append(rec)
                append_jsonl(f"{res_dir}/train_log.jsonl", rec)
                if step == 1 or step % 10 == 0:
                    print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in rec.items()})
                buf = {k: [] for k in buf}

    model.save_pretrained(str(output))

    k = min(10, len(history))
    summary = {
        "run_name": run_name, "beta": beta, "n_examples": len(rows), "epochs": epochs,
        "dataset": bundle["path"], "batch_size": bs, "grad_accum_steps": accum,
        "effective_batch": bs * accum, "optimizer_steps": step,
        "learning_rate": float(cfg["learning_rate"]), "seed": int(cfg["seed"]),
        "max_sequence_length": max_len, "total_params": total_p, "trainable_params": train_p,
        "loss_first10_mean": mean([r["loss"] for r in history[:k]]),
        "loss_last10_mean": mean([r["loss"] for r in history[-k:]]),
        "acc_last10_mean": mean([r["pref_acc"] for r in history[-k:]]),
        "nonfinite_microbatches": n_bad,
        "wall_clock_s": round(elapsed(), 1),
        "peak_vram_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2) if torch.cuda.is_available() else None,
        "adapter_path": str(output),
    }
    save_json(f"{res_dir}/train_summary.json", summary)
    print(summary)

    del model, optimizer, loader, bundle
    clear_gpu()
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--run-name", default="standard")
    ap.add_argument("--dataset")
    ap.add_argument("--output")
    ap.add_argument("--beta", type=float)
    ap.add_argument("--max-examples", type=int)
    args = ap.parse_args()
    run_training(args.config, args.run_name, args.dataset, args.output, args.beta, args.max_examples)


if __name__ == "__main__":
    main()