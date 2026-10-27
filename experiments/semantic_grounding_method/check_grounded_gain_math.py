"""Synthetic logit arithmetic only: no neural model, dataset inference or GPU."""
import json
from pathlib import Path
import numpy as np


def logprob(x, target):
    z = x - np.max(x)
    return float(z[target] - np.log(np.exp(z).sum()))


def account(base, parts, target, order=64):
    base = np.asarray(base, dtype=float)
    parts = np.asarray(parts, dtype=float)
    assert parts.ndim == 2 and parts.shape[1:] == base.shape
    assert np.isfinite(base).all() and np.isfinite(parts).all()
    nodes, weights = np.polynomial.legendre.leggauss(order)
    path = base + ((nodes + 1) / 2)[:, None] * parts.sum(axis=0)
    path -= path.max(axis=1, keepdims=True)
    probs = np.exp(path)
    probs /= probs.sum(axis=1, keepdims=True)
    average = (weights / 2) @ probs
    contribution = parts[:, target] - parts @ average
    delta = logprob(base + parts.sum(axis=0), target) - logprob(base, target)
    return delta, contribution


def penalties(delta, grouped, margin=.05, rho=.8, kappa=2):
    support, rest = grouped
    return dict(gain=max(margin-delta, 0),
                groundgain=max(rho*max(delta, 0)-support, 0),
                cancel=max(abs(support)+abs(rest)-kappa*max(delta, 0), 0))


def main():
    rng = np.random.default_rng(42)
    base = rng.normal(size=7)
    parts = rng.normal(size=(4, 7)) * .3
    delta, c = account(base, parts, 2)
    residual = abs(c.sum()-delta)
    assert residual < 1e-12
    _, c128 = account(base, parts, 2, order=128)
    assert np.max(abs(c-c128)) < 1e-12
    # Per-patch vocabulary-constant offsets have no probability contribution.
    dshift, cshift = account(base, parts + np.arange(4)[:, None], 2)
    assert abs(delta-dshift) < 1e-12 and np.max(abs(c-cshift)) < 1e-12
    zero_delta, zero_c = account([0, 0], [[0, 0], [0, 0]], 0)
    assert zero_delta == 0 and np.all(zero_c == 0)
    assert penalties(zero_delta, zero_c)['gain'] > 0
    # Positive attribution can arise entirely from suppressing a competing word.
    suppress_delta, suppress_c = account([0, 0], [[0, -1], [0, 0]], 0)
    assert suppress_delta > 0 and suppress_c[0] > 0
    # Same total logits, different patch decomposition: attribution changes.
    da, ca = account([0, 0], [[1, 0], [0, 0]], 0)
    db, cb = account([0, 0], [[0, 0], [1, 0]], 0)
    assert abs(da-db) < 1e-12 and abs(ca[0]-cb[0]) > .1
    # Cancellation can happen inside a group and escape the two-group budget.
    di, ci = account([0, 0], [[10, 0], [-9, 0], [0, 0]], 0)
    group_penalty = penalties(di, [ci[:2].sum(), ci[2]])
    assert group_penalty['cancel'] == 0
    assert np.abs(ci[:2]).sum() > 10*abs(ci[:2].sum())
    # Explicitly catch the intended cross-group cancellation example.
    assert penalties(.05, [5, -4.95])['cancel'] > 9
    assert abs(penalties(.05, [5, -4.95])['gain']) < 1e-12
    assert penalties(.05, [0, .05])['groundgain'] > 0
    result = dict(
        scope=__doc__, seed=42, quadrature_order=64,
        random_case_completeness_residual=residual,
        common_logit_offset_invariance=True, zero_branch_gain_penalized=True,
        competitor_only_suppression=dict(target_logit_change=0,
            delta=suppress_delta, support_contribution=float(suppress_c[0]),
            interpretation='Discriminative probability support, not proof of local object recognition.'),
        identical_total_logits_different_allocation=dict(delta=da,
            support_a=float(ca[0]), support_b=float(cb[0]),
            interpretation='Attribution is conditional on the chosen branch decomposition.'),
        within_group_cancellation=dict(contributions=ci.tolist(),
            penalties=group_penalty,
            interpretation='Two-group budget cannot bound cancellation within either group.'),
        cross_group_cancellation_detected=True, outside_only_gain_penalized=True,
        limitations='Synthetic finite logits only. No gradients, phrase trie, model runtime, grounding, caption quality or novelty validated.')
    Path(__file__).with_name('GROUNDED_GAIN_MATH.json').write_text(
        json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
