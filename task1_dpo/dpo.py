from __future__ import annotations

import torch
import torch.nn.functional as F


def dpo_loss(
    policy_chosen_logp: torch.Tensor,
    policy_rejected_logp: torch.Tensor,
    ref_chosen_logp: torch.Tensor,
    ref_rejected_logp: torch.Tensor,
    beta: float,
):
    """Return scalar DPO loss plus lightweight diagnostics.

    Validate this implementation against the equation in the assignment manual before using it.
    """
    policy_margin = policy_chosen_logp - policy_rejected_logp
    ref_margin = ref_chosen_logp - ref_rejected_logp

    # Starter implementation: students must validate the objective carefully.
    logits = beta * (policy_margin - ref_margin)

    loss = -F.logsigmoid(logits).mean()
    return loss, {
        "logit_mean": logits.detach().mean(),
        "policy_margin_mean": policy_margin.detach().mean(),
        "preference_accuracy": (policy_margin - ref_margin > 0).float().mean().detach(),
    }


def validate_dpo_loss(verbose: bool = True) -> None:
    """Check dpo_loss against the manual. Raises RuntimeError on failure (training then aborts)."""
    def t(x, grad=False):
        return torch.tensor([x], dtype=torch.float32, requires_grad=grad)

    def check(cond, msg):
        if not cond:
            raise RuntimeError(f"DPO loss validation FAILED: {msg}")

    # 1. policy == reference -> logit 0 -> loss = ln 2
    loss, _ = dpo_loss(t(-40.), t(-60.), t(-40.), t(-60.), beta=0.1)
    check(abs(loss.item() - math.log(2)) < 1e-6, f"policy==ref gave {loss.item()}, expected ln2")

    # 2. hand-computed: log-ratios +5 and -2 -> margin 7 -> logit 0.7
    loss, d = dpo_loss(t(-40.), t(-60.), t(-45.), t(-58.), beta=0.1)
    expected = -math.log(1 / (1 + math.exp(-0.7)))
    check(abs(loss.item() - expected) < 1e-6, f"hand example gave {loss.item()}, expected {expected}")
    check(d["preference_accuracy"].item() == 1.0, "accuracy should be 1 when margin > 0")

    # 3. raw policy margin > 0 but reference-adjusted margin < 0 -> must NOT count as correct
    _, d = dpo_loss(t(-40.), t(-50.), t(-30.), t(-60.), beta=0.1)
    check(d["preference_accuracy"].item() == 0.0, "accuracy ignores the reference term")

    # 4. gradient directions: raising chosen / lowering rejected must reduce the loss
    pc, pr = t(-40., True), t(-60., True)
    loss, _ = dpo_loss(pc, pr, t(-45.), t(-58.), beta=0.1)
    loss.backward()
    check(pc.grad.item() < 0 and pr.grad.item() > 0, "gradient signs wrong (chosen/rejected swapped?)")

    # 5. beta scaling: doubling beta must change the logit by exactly 2x
    _, d1 = dpo_loss(t(-40.), t(-60.), t(-45.), t(-58.), beta=0.1)
    _, d2 = dpo_loss(t(-40.), t(-60.), t(-45.), t(-58.), beta=0.2)
    check(abs(d2["logit_mean"].item() - 2 * d1["logit_mean"].item()) < 1e-5, "beta is not a plain multiplier")

    if verbose:
        print("[validate_dpo_loss] all 5 checks passed")