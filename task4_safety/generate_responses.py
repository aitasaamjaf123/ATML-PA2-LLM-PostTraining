from __future__ import annotations

import argparse
import gc

import pandas as pd
from peft import PeftModel

from common.data import load_yaml, repo_path, write_jsonl, read_jsonl
from common.generation import batch_generate
from common.logging_utils import set_seed
from common.models import load_policy, load_tokenizer
from task4_safety.utils import (POLICIES, task4_dir, resolve_adapter, verify_adapter,
                                env_meta, update_json, cleanup_gpu)

MAX_PROMPT_LEN = 256


def policy_specs(cfg):
    return {
        "sft": None,
        "dpo": cfg["policies"]["dpo"],
        "ppo": cfg["policies"]["ppo"],
        "grpo": cfg["policies"]["grpo"],
    }


def load_xstest(cfg):
    return pd.read_csv(repo_path(cfg["paths"]["xstest"]))


def generate_for_policy(cfg, policy_name: str, batch_size: int = 4, meta: dict | None = None):
    specs = policy_specs(cfg)
    if policy_name not in specs:
        raise KeyError(policy_name)
    adapter = resolve_adapter(policy_name, specs[policy_name])  # hard-fails if missing
    info = verify_adapter(adapter, cfg["base_model"]) if adapter else None
    print(f"[{policy_name}] adapter: {adapter}")

    tokenizer = load_tokenizer(cfg["base_model"])
    model = load_policy(cfg, adapter_path=str(adapter) if adapter else None, trainable=False)
    if adapter:
        assert isinstance(model, PeftModel), "adapter policy did not load as PeftModel"
    else:
        assert not isinstance(model, PeftModel), "SFT must be the untouched base model"

    if meta is not None:
        gc_obj = getattr(model, "generation_config", None)
        meta.update({
            "adapter": info,
            "generation_config": gc_obj.to_dict() if gc_obj is not None else None,
            "decoding": {"do_sample": False, "temperature": 0.0, "top_p": 1.0,
                         "max_new_tokens": int(cfg["safety_max_new_tokens"]),
                         "max_prompt_length": MAX_PROMPT_LEN, "batch_size": batch_size},
            "env": env_meta(),
        })

    df = load_xstest(cfg)
    assert df["xstest_id"].is_unique
    records = []
    for start in range(0, len(df), batch_size):
        chunk = df.iloc[start:start + batch_size]
        prompts = [[{"role": "user", "content": str(x)}] for x in chunk["prompt"].tolist()]
        for p in prompts:  # B7: ensure no prompt gets truncated
            rendered = tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            n = len(tokenizer(rendered, add_special_tokens=False)["input_ids"])
            assert n <= MAX_PROMPT_LEN, f"prompt too long ({n} tokens): {p[0]['content'][:80]}"
        gen = batch_generate(
            model, tokenizer, prompts,
            max_prompt_length=MAX_PROMPT_LEN,
            max_new_tokens=int(cfg["safety_max_new_tokens"]),
            temperature=0.0, top_p=1.0, do_sample=False,
        )
        for (_, row), response, n_tok, trunc, eos in zip(
            chunk.iterrows(), gen["responses"], gen["response_lengths"],
            gen["truncated"], gen["terminated_with_eos"],
        ):
            records.append({
                "xstest_id": int(row["xstest_id"]),
                "policy": policy_name,
                "prompt": str(row["prompt"]),
                "benchmark_class": str(row["benchmark_class"]),
                "type": str(row["type"]),
                "response": response,
                "response_tokens": int(n_tok),  # includes the EOS token (starter convention)
                "truncated": bool(trunc),
                "terminated_with_eos": bool(eos),
                "empty_response": len(response.strip()) == 0,
            })
        if (start // batch_size) % 25 == 0:
            print(f"[{policy_name}] {start + len(chunk)}/{len(df)}")

    del model
    gc.collect()
    cleanup_gpu()
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/feedback.yaml")
    ap.add_argument("--policies", nargs="+", default=POLICIES, choices=POLICIES)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()
    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))
    outdir = task4_dir(cfg)
    n_rows = len(load_xstest(cfg))
    print("Policies:", list(policy_specs(cfg)), "| XSTest rows:", n_rows)

    # Resolve all adapters up-front so a missing one fails before any GPU time is spent.
    for p in args.policies:
        resolve_adapter(p, policy_specs(cfg)[p])

    for p in args.policies:
        out = outdir / f"generated_{p}.jsonl"
        if out.exists() and not args.overwrite:
            print(f"[{p}] exists, skipping (use --overwrite): {out}")
            continue
        meta = {}
        records = generate_for_policy(cfg, p, batch_size=args.batch_size, meta=meta)
        assert len(records) == n_rows
        write_jsonl(out, records)
        meta["n_records"] = len(records)
        meta["frac_truncated"] = sum(r["truncated"] for r in records) / len(records)
        meta["frac_empty"] = sum(r["empty_response"] for r in records) / len(records)
        update_json(outdir / "generation_meta.json", p, meta)
        print(f"[{p}] wrote {out}")

    # Sanity: RL policies must not be identical to SFT; prompt order must match.
    if all((outdir / f"generated_{p}.jsonl").exists() for p in POLICIES):
        base = read_jsonl(outdir / "generated_sft.jsonl")
        base_ids = [r["xstest_id"] for r in base]
        for p in POLICIES[1:]:
            rows = read_jsonl(outdir / f"generated_{p}.jsonl")
            assert [r["xstest_id"] for r in rows] == base_ids, f"{p}: prompt order differs from SFT"
            same = sum(a["response"] == b["response"] for a, b in zip(base, rows)) / len(rows)
            print(f"[{p}] fraction of responses identical to SFT: {same:.3f}")
            if same == 1.0:
                raise RuntimeError(f"{p} is identical to SFT: adapter was not applied.")


if __name__ == "__main__":
    main()