"""Validation tests for the PPO objective pieces (Task 2, step 1: "validate the implementation").

Run from the repo root:   python -m pytest tests/test_ppo.py -q     (or)     python tests/test_ppo.py
"""
from __future__ import annotations

import math
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from common.metrics import masked_mean
from task2_ppo import utils as U
from task2_ppo.ppo import (
    compute_gae,
    normalize_advantages,
    ppo_policy_loss,
    shaped_rewards,
    value_mse_loss,
)


# ------------------------------------------------------------------ clipped surrogate geometry
def _grad(ratio, adv, eps=0.2):
    """d(loss)/d(new_logp) for a single valid token with old_logp=0, so rho = ratio."""
    new = torch.tensor([[math.log(ratio)]], requires_grad=True)
    old = torch.zeros(1, 1)
    loss, _, _ = ppo_policy_loss(new, old, torch.tensor([[adv]]), torch.ones(1, 1), eps=eps)
    loss.backward()
    return float(loss), float(new.grad)


def test_positive_adv_ratio_above_band_is_clipped_zero_grad():
    loss, g = _grad(1.5, 1.0)
    assert abs(g) < 1e-9
    assert abs(loss - (-1.2)) < 1e-5            # -min(1.5, 1.2)


def test_negative_adv_ratio_below_band_is_clipped_zero_grad():
    loss, g = _grad(0.5, -1.0)
    assert abs(g) < 1e-9
    assert abs(loss - 0.8) < 1e-5               # -min(-0.5, -0.8)


def test_negative_adv_ratio_above_band_keeps_gradient():
    loss, g = _grad(1.5, -1.0)                  # pessimistic branch: min picks rho*A = -1.5
    assert abs(loss - 1.5) < 1e-5
    assert abs(g - 1.5) < 1e-4                  # d(-rho*A)/d logp = -A*rho


def test_positive_adv_ratio_below_band_keeps_gradient():
    loss, g = _grad(0.5, 1.0)                   # min picks rho*A = 0.5
    assert abs(loss + 0.5) < 1e-5
    assert abs(g + 0.5) < 1e-4


def test_inside_band_gradient_is_unclipped():
    _, g = _grad(1.1, 2.0)
    assert abs(g - (-2.2)) < 1e-4


def test_other_epsilons():
    assert abs(_grad(1.3, 1.0, eps=0.5)[1] + 1.3) < 1e-4   # inside the wide band -> gradient
    assert abs(_grad(1.3, 1.0, eps=0.05)[1]) < 1e-9        # outside the narrow band -> clipped


def test_clip_fraction_counts_valid_tokens_only():
    ratios = torch.tensor([[0.4, 0.9, 1.0, 1.1, 1.6, 9.0]])
    mask = torch.tensor([[1, 1, 1, 1, 1, 0.0]])          # last token is padding with a huge ratio
    adv = torch.ones_like(ratios)
    for eps, expected in [(0.2, 2 / 5), (0.5, 2 / 5), (0.05, 4 / 5)]:
        _, _, frac = ppo_policy_loss(ratios.log(), torch.zeros_like(ratios), adv, mask, eps=eps)
        assert abs(float(frac) - expected) < 1e-6, (eps, float(frac))


def test_masked_tokens_get_no_gradient():
    new = torch.tensor([[0.0, 3.0]], requires_grad=True)
    loss, _, _ = ppo_policy_loss(new, torch.zeros(1, 2), torch.ones(1, 2), torch.tensor([[1.0, 0.0]]))
    loss.backward()
    assert float(new.grad[0, 1]) == 0.0


def test_microbatch_token_weighting_equals_full_batch():
    torch.manual_seed(0)
    new, old = torch.randn(4, 7) * 0.1, torch.randn(4, 7) * 0.1
    adv = torch.randn(4, 7)
    mask = (torch.rand(4, 7) > 0.3).float()
    mask[:, 0] = 1
    full, _, _ = ppo_policy_loss(new, old, adv, mask)
    acc = 0.0
    for sl in U.microbatches(4, 2):
        l, _, _ = ppo_policy_loss(new[sl], old[sl], adv[sl], mask[sl])
        acc += float(l) * float(mask[sl].sum() / mask.sum())
    assert abs(acc - float(full)) < 1e-6


def test_clamp_new_logp_identity_and_overflow_guard():
    old = torch.zeros(1, 3)
    new = torch.tensor([[0.5, -0.5, 80.0]], requires_grad=True)
    out = U.clamp_new_logp(new, old, 20.0)
    assert torch.allclose(out[0, :2], new[0, :2]) and float(out[0, 2]) == 20.0
    assert torch.isfinite(out.exp()).all()
    out.sum().backward()
    assert float(new.grad[0, 0]) == 1.0 and float(new.grad[0, 2]) == 0.0


# ------------------------------------------------------------------ GAE
def _brute_gae(r, v, gamma, lam):
    T = len(r)
    v_ext = list(v) + [0.0]
    delta = [r[t] + gamma * v_ext[t + 1] - v_ext[t] for t in range(T)]
    return [sum((gamma * lam) ** k * delta[t + k] for k in range(T - t)) for t in range(T)]


def test_gae_matches_definition():
    r, v = [0.1, -0.2, 0.0, 0.5], [0.3, 0.1, -0.4, 0.2]
    for gamma, lam in [(1.0, 0.95), (0.9, 0.7), (1.0, 1.0), (1.0, 0.0)]:
        adv, ret = compute_gae(torch.tensor([r]), torch.tensor([v]), torch.ones(1, 4), gamma, lam)
        assert torch.allclose(adv[0], torch.tensor(_brute_gae(r, v, gamma, lam)), atol=1e-6)
        assert torch.allclose(ret, adv + torch.tensor([v]))


def test_gae_lambda_one_is_monte_carlo_return_minus_value():
    r, v = [0.1, -0.2, 0.5], [0.3, 0.1, -0.4]
    adv, _ = compute_gae(torch.tensor([r]), torch.tensor([v]), torch.ones(1, 3), 1.0, 1.0)
    G = [sum(r[t:]) for t in range(3)]
    assert torch.allclose(adv[0], torch.tensor([G[t] - v[t] for t in range(3)]), atol=1e-6)


def test_gae_padding_invariance():
    r1, v1 = [0.1, 0.2, 1.0], [0.5, 0.4, 0.3]
    r2, v2 = [0.0, -0.1, 0.0, 0.2, 2.0], [0.1, 0.2, 0.3, 0.4, 0.5]
    rew = torch.tensor([r1 + [0, 0], r2])
    val = torch.tensor([v1 + [7.0, -9.0], v2])          # garbage values at padded positions
    mask = torch.tensor([[1, 1, 1, 0, 0], [1, 1, 1, 1, 1.0]])
    adv, ret = compute_gae(rew, val, mask, 1.0, 0.95)
    a1, _ = compute_gae(torch.tensor([r1]), torch.tensor([v1]), torch.ones(1, 3), 1.0, 0.95)
    a2, _ = compute_gae(torch.tensor([r2]), torch.tensor([v2]), torch.ones(1, 5), 1.0, 0.95)
    assert torch.allclose(adv[0, :3], a1[0], atol=1e-6) and torch.allclose(adv[1], a2[0], atol=1e-6)
    assert float(adv[0, 3:].abs().sum()) == 0.0


# ------------------------------------------------------------------ reward shaping / helpers
def test_shaped_rewards_sign_and_terminal_position():
    pol = torch.tensor([[-1.0, -2.0, -3.0, 0.0], [-1.0, -1.0, 0.0, 0.0]])
    ref = torch.tensor([[-1.5, -2.0, -2.0, 0.0], [-1.0, -2.0, 0.0, 0.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0.0]])
    out = shaped_rewards(torch.tensor([2.0, -1.0]), pol, ref, mask, beta_kl=0.1)
    expected0 = [-0.1 * 0.5, 0.0, -0.1 * (-1.0) + 2.0, 0.0]   # policy above ref -> penalty; terminal at idx 2
    expected1 = [0.0, -0.1 * 1.0 - 1.0, 0.0, 0.0]
    assert torch.allclose(out[0], torch.tensor(expected0), atol=1e-6)
    assert torch.allclose(out[1], torch.tensor(expected1), atol=1e-6)
    assert torch.allclose(shaped_rewards(torch.tensor([2.0, -1.0]), pol, ref, mask, 0.0)[0], torch.tensor([0, 0, 2.0, 0]))


def test_normalize_advantages():
    adv = torch.tensor([[1.0, 2.0, 3.0, 99.0], [-4.0, 0.5, 99.0, 99.0]])
    mask = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0.0]])
    out = normalize_advantages(adv, mask)
    valid = out[mask.bool()]
    assert abs(float(valid.mean())) < 1e-5 and abs(float(valid.std(unbiased=False)) - 1.0) < 1e-4
    assert float(out[mask == 0].abs().sum()) == 0.0


def test_value_mse_masking():
    pred, ret = torch.tensor([[1.0, 5.0]]), torch.tensor([[0.0, 0.0]])
    assert abs(float(value_mse_loss(pred, ret, torch.tensor([[1.0, 0.0]]))) - 1.0) < 1e-6


def test_value_stats_perfect_critic():
    v = torch.tensor([[1.0, 2.0, 3.0, 0.0]])
    ev, vm, rm, corr = U.value_stats(v, v.clone(), torch.tensor([[1, 1, 1, 0.0]]))
    assert abs(ev - 1.0) < 1e-6 and abs(corr - 1.0) < 1e-6


# ------------------------------------------------------------------ log-prob / critic plumbing
class _DummyLM(torch.nn.Module):
    def __init__(self, V=11, d=5):
        super().__init__()
        self.emb, self.out = torch.nn.Embedding(V, d), torch.nn.Linear(d, V)

    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True, logits_to_keep=0):
        logits = self.out(self.emb(input_ids))
        if logits_to_keep:
            logits = logits[:, -logits_to_keep:]
        return types.SimpleNamespace(logits=logits)


def test_response_logprobs_matches_full_log_softmax_and_entropy():
    torch.manual_seed(0)
    m, B, P, R = _DummyLM(), 2, 4, 6
    seq = torch.randint(0, 11, (B, P + R))
    resp = seq[:, P:]
    full = torch.log_softmax(m(seq).logits[:, P - 1:P - 1 + R].float(), -1)
    ref_lp = full.gather(-1, resp.unsqueeze(-1)).squeeze(-1)
    ref_ent = -(full.exp() * full).sum(-1)
    with torch.no_grad():
        lp, ent = U.response_logprobs(m, seq, torch.ones_like(seq), resp, entropy=True, chunk=4)
    assert torch.allclose(lp, ref_lp, atol=1e-5) and torch.allclose(ent, ref_ent, atol=1e-5)


def test_chunked_checkpointed_logprob_gradients_match():
    torch.manual_seed(1)
    m, P, R = _DummyLM(), 3, 7
    seq = torch.randint(0, 11, (2, P + R))
    resp = seq[:, P:]
    grads = []
    for chunk in (R, 2):
        m.zero_grad()
        lp, _ = U.response_logprobs(m, seq, torch.ones_like(seq), resp, chunk=chunk)
        lp.sum().backward()
        grads.append(m.out.weight.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5)


class _DummyBackbone(torch.nn.Module):
    def forward(self, input_ids, attention_mask=None, use_cache=False, return_dict=True):
        B, T = input_ids.shape
        h = torch.zeros(B, T, 3)
        h[:, :, 0] = torch.arange(T).float()             # feature 0 encodes the absolute position
        return types.SimpleNamespace(last_hidden_state=h)


class _DummyValueModel(torch.nn.Module):
    base_model_prefix = "model"

    def __init__(self):
        super().__init__()
        self.model = _DummyBackbone()
        self.score = torch.nn.Linear(3, 1, bias=False)
        with torch.no_grad():
            self.score.weight.copy_(torch.tensor([[1.0, 0.0, 0.0]]))


def test_value_positions_are_state_before_action():
    P, R = 5, 4
    ids = torch.zeros(2, P + R, dtype=torch.long)
    v = U.value_forward(_DummyValueModel(), ids, torch.ones_like(ids), P, R)
    assert v.shape == (2, R)
    assert torch.allclose(v[0], torch.tensor([float(P - 1 + t) for t in range(R)]))   # position P-1+t


def test_fork_naming_matches_figure_script():
    assert U.fork_name(0.2, 0.1, 6304) == "fork_eps0.2_kl0.1_s6304"
    assert U.fork_name(0.05, 0.0, 6305) == "fork_eps0.05_kl0_s6305"
    assert U.fork_name(0.5, 0.2, 6306) == "fork_eps0.5_kl0.2_s6306"


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for n, f in tests:
        try:
            f()
            print(f"PASS {n}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"FAIL {n}: {type(e).__name__}: {e}")
    print(f"{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
