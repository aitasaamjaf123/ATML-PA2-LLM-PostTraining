"""Evidence for the generation-time prompt-truncation rule (prompt cap = max_sequence_length - max_generation_tokens).

Compares three rules on the held-out prompts WITHOUT running any model, so the rule is not chosen by looking at outputs:
  right     : tokenizer cuts the END of the rendered prompt   (what the starter's batch_generate did by default)
  left      : tokenizer cuts the START of the rendered prompt
  head_tail : keep start+end of the last user message, re-render the template (task1_dpo.evaluate.truncate_prompt)
For every prompt over the cap it records whether each rule keeps the assistant marker, the system header, and the
first / last WINDOW tokens of the user's message. This shows WHAT each rule preserves; it does not measure output quality.

Run:  python -m task1_dpo.audit_truncation          (CPU only, about a minute)
Out:  results/task1_dpo/truncation_audit/summary.json, per_prompt.jsonl     (exit code 1 if a gate fails)
"""
from __future__ import annotations

import argparse
import sys

from common.data import load_yaml, prompt_messages_from_preference, read_jsonl, write_jsonl
from common.logging_utils import save_json
from common.models import load_tokenizer
from task1_dpo.evaluate import ASSISTANT_MARKER, HEAD_FRAC, _n_prompt_tokens, truncate_prompt

WINDOW = 48
RULES = ("right", "left", "head_tail")


def apply_rule(tok, msgs, cap, rule):
    if rule == "head_tail":
        new, method = truncate_prompt(tok, msgs, cap)
        tok.truncation_side = "left"            # fallback side, as in evaluate_adapter
    else:
        new, method = msgs, rule
        tok.truncation_side = rule
    rendered = tok.apply_chat_template(new, tokenize=False, add_generation_prompt=True)
    ids = tok(rendered, truncation=True, max_length=cap)["input_ids"]   # the call batch_generate makes
    return ids, method


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/dpo.yaml")
    ap.add_argument("--n-prompts", type=int, default=300)
    args = ap.parse_args()

    cfg = load_yaml(args.config)
    cap = int(cfg["max_sequence_length"]) - int(cfg["max_generation_tokens"])
    tok = load_tokenizer(cfg["base_model"])
    rows = read_jsonl(cfg["paths"]["dpo_standard_eval"])[: args.n_prompts]

    per = []
    for idx, r in enumerate(rows):
        msgs = prompt_messages_from_preference(r)
        full = _n_prompt_tokens(tok, msgs)
        users = [m["content"] for m in msgs if m.get("role") == "user"]
        user = users[-1] if users else ""
        uids = tok(user, add_special_tokens=False)["input_ids"]
        head = tok.decode(uids[:WINDOW]).rstrip("\ufffd")
        tail = tok.decode(uids[-WINDOW:]).lstrip("\ufffd")
        rec = {"idx": idx, "prompt_id": str(r.get("prompt_id", idx)), "tokens_full": full, "over_cap": full > cap,
               "tokens_over_cap": max(0, full - cap), "user_start": user[:150], "user_end": user[-150:],
               "rules": {}}
        for rule in RULES:
            ids, method = apply_rule(tok, msgs, cap, rule)
            text = tok.decode(ids)
            rec["rules"][rule] = {
                "method": method, "tokens_used": len(ids), "fits_cap": len(ids) <= cap,
                "marker_intact": text.endswith(ASSISTANT_MARKER), "header_intact": text.startswith("<|im_start|>"),
                "head_kept": bool(head) and head in text, "tail_kept": bool(tail) and tail in text,
            }
        per.append(rec)

    over = [p for p in per if p["over_cap"]]
    summary = {"cap_tokens": cap, "window_tokens": WINDOW, "head_frac": HEAD_FRAC, "n_prompts": len(per),
               "n_over_cap": len(over), "frac_over_cap": len(over) / max(len(per), 1),
               "max_tokens_full": max(p["tokens_full"] for p in per),
               "note": "Preservation audit only: shows what each rule keeps, not which gives better outputs.",
               "rules": {}}
    for rule in RULES:
        S = [p["rules"][rule] for p in over]
        summary["rules"][rule] = {
            "marker_intact": sum(s["marker_intact"] for s in S), "header_intact": sum(s["header_intact"] for s in S),
            "head_kept": sum(s["head_kept"] for s in S), "tail_kept": sum(s["tail_kept"] for s in S),
            "head_and_tail_kept": sum(s["head_kept"] and s["tail_kept"] for s in S),
            "fits_cap": sum(s["fits_cap"] for s in S),
            "methods": {m: sum(s["method"] == m for s in S) for m in sorted({s["method"] for s in S})},
        }
    n_untouched = sum(p["rules"]["head_tail"]["method"] == "none" for p in per if not p["over_cap"])
    ht, n = summary["rules"]["head_tail"], len(over)
    gates = {
        "every_over_cap_prompt_handled_by_head_tail_no_fallback": ht["methods"].get("head_tail", 0) == n,
        "marker_and_header_intact_for_all_over_cap": ht["marker_intact"] == n and ht["header_intact"] == n,
        "all_over_cap_fit_the_cap": ht["fits_cap"] == n,
        "head_and_tail_kept_for_all_over_cap": ht["head_and_tail_kept"] == n,
        "prompts_under_cap_left_untouched": n_untouched == len(per) - n,
    }
    summary["gates"] = gates
    summary["all_gates_pass"] = all(gates.values())

    out = f"{cfg['results_dir']}/truncation_audit"
    save_json(f"{out}/summary.json", summary)
    write_jsonl(f"{out}/per_prompt.jsonl", per)

    print(f"\nprompt cap = {cap} tokens | {n} of {len(per)} prompts exceed it | max prompt = {summary['max_tokens_full']}")
    print(f"{'rule':<10}{'marker':>8}{'header':>8}{'head':>7}{'tail':>7}{'both':>7}{'fits':>7}   (counts out of {n})")
    for rule in RULES:
        v = summary["rules"][rule]
        print(f"{rule:<10}{v['marker_intact']:>8}{v['header_intact']:>8}{v['head_kept']:>7}{v['tail_kept']:>7}"
              f"{v['head_and_tail_kept']:>7}{v['fits_cap']:>7}")
    for g, ok in gates.items():
        print(("PASS  " if ok else "FAIL  ") + g)
    sys.exit(0 if summary["all_gates_pass"] else 1)


if __name__ == "__main__":
    main()
