"""Synthetic guards for acar/v3/training.py + normalizers.py + the FittedCandidate artifact. TOY ONLY (toy source
state + GIVEN synthetic ΔR targets; no DEV cohorts, no DEV gate, no lockbox). Run: python -m acar.v3.tests.test_training
"""
import math
import numpy as np
import torch

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS
from acar.v3.set_features import build_action_sets, NON_IDENTITY, PER_WINDOW_FEATURES, CONTEXT_FEATURES
from acar.v3.normalizers import InputNormalizer, TargetNormalizer, SD_FLOOR
from acar.v3.predictors import FittedCandidate, CANDIDATES
from acar.v3.training import (fit_candidate, TrainExample, _huber, _beta_nll, _pinball, _subject_balanced_loss)


def _state(d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(160) < 0.5).astype(int)
    z = rng.standard_normal((160, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    return fit_source_state(z, y, N_CLS, rho=0.1)


def _examples(state, n_subj=16, nwin=12, d=8, seed=1):
    rng = np.random.default_rng(seed); ex = []
    for s in range(n_subj):
        z = rng.standard_normal((nwin, d)) + 0.2 * rng.standard_normal()
        keys = [f"d/s{s}/r0/{w}" for w in range(nwin)]
        sets = build_action_sets(state, z, keys)
        sig = float(z.mean())
        for a in NON_IDENTITY:
            ex.append(TrainExample(sets[a], sig + 0.1 * rng.standard_normal(), f"d|s{s}"))
    return ex


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_loss_exactness():
    y = torch.tensor(0.3)
    p = torch.tensor(1.0)
    assert abs(float(_huber(p, y, 1.0)) - 0.5 * (0.7 ** 2)) < 1e-6                  # |e|<=delta -> quadratic
    p2 = torch.tensor(3.0)
    assert abs(float(_huber(p2, y, 1.0)) - (1.0 * (abs(2.7) - 0.5))) < 1e-6         # |e|>delta -> linear
    mu, sig = torch.tensor(0.5), torch.tensor(2.0)
    v = 4.0; base = 0.5 * ((0.3 - 0.5) ** 2 / v + math.log(v)); w = v ** 0.5
    assert abs(float(_beta_nll(mu, sig, y, 0.5)) - base * w) < 1e-6
    q = torch.tensor(0.0)
    assert abs(float(_pinball(y, q, 0.9)) - 0.9 * 0.3) < 1e-6                       # y>q
    assert abs(float(_pinball(torch.tensor(-0.3), q, 0.9)) - (0.1 * 0.3)) < 1e-6    # y<q -> (1-tau)|e|
    print("  [ok] Huber / β-NLL / pinball loss formulas exact")


def test_beta_nll_stop_gradient():
    y = torch.tensor(0.3)
    s1 = torch.tensor(2.0, requires_grad=True)
    _beta_nll(torch.tensor(0.5), s1, y, 0.5).backward()
    s2 = torch.tensor(2.0, requires_grad=True)
    v = s2 * s2; base = 0.5 * ((y - 0.5) ** 2 / v + torch.log(v))
    (base * (v ** 0.5)).backward()                              # weight NOT detached
    assert abs(float(s1.grad) - float(s2.grad)) > 1e-6, "β-NLL weight is not stop-gradient'd"
    print("  [ok] β-NLL variance weight is stop-gradient'd (grad differs from non-detached)")


def test_subject_balanced_reduction():
    state = _state()
    z = np.random.default_rng(0).standard_normal((12, 8))
    was = build_action_sets(state, z, [f"d/s/r/{i}" for i in range(12)])["matched_coral"]
    net_seed = 0
    from acar.v3.predictors import DeepSetsNet
    net = DeepSetsNet("C1", net_seed); net.eval()
    # subject A: 1 example; subject B: 50 identical examples. Subject-balanced -> A and B weighted equally.
    win = torch.tensor(np.concatenate([was.values, was.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
    ctx = torch.tensor(np.concatenate([was.context_values, was.context_mask.astype(np.float64)]), dtype=torch.float32)
    exA = [(win, ctx, "matched_coral", torch.tensor(0.0, dtype=torch.float32))]
    exB = [(win, ctx, "matched_coral", torch.tensor(2.0, dtype=torch.float32))] * 50
    by = {"A": exA, "B": exB}
    bal = float(_subject_balanced_loss(net, by, "C1", ["A", "B"]))
    la = float(_subject_balanced_loss(net, {"A": exA}, "C1", ["A"]))
    lb = float(_subject_balanced_loss(net, {"B": exB}, "C1", ["B"]))
    assert abs(bal - 0.5 * (la + lb)) < 1e-5, "subject-balanced loss not equal-weighted across subjects"
    print("  [ok] subject-balanced reduction: 50-batch subject does not dominate a 1-batch subject")


def test_deterministic_train_twice():
    state = _state(); ex = _examples(state); tr, va = ex[:36], ex[36:]
    a1 = fit_candidate("C2", "PD", tr, va, seed=0)
    a2 = fit_candidate("C2", "PD", tr, va, seed=0)
    assert a1.artifact_sha256 == a2.artifact_sha256 and len(a1.artifact_sha256) == 64
    print(f"  [ok] deterministic train-twice -> identical 64-char artifact hash (epoch={a1.training_epoch})")


def test_early_stop_best_state():
    state = _state(); ex = _examples(state)
    from acar.v3.predictors import HP
    a = fit_candidate("C1", "PD", ex[:36], ex[36:], seed=0)
    assert 0 <= a.training_epoch < HP["max_epochs"], "no early stop / best epoch not recorded"
    was = build_action_sets(state, np.random.default_rng(9).standard_normal((10, 8)),
                            [f"d/s/r/{i}" for i in range(10)])["matched_coral"]
    assert a.predict(was) == a.predict(was)                     # deterministic restored state
    print(f"  [ok] early stopping + best-epoch restoration (best_epoch={a.training_epoch})")


def test_normalizers():
    state = _state()
    sets = [build_action_sets(state, np.random.default_rng(s).standard_normal((12, 8)),
                              [f"d/s{s}/r/{i}" for i in range(12)])["matched_coral"] for s in range(8)]
    inorm = InputNormalizer.fit(sets)
    nw = inorm.transform(sets[0])
    assert np.array_equal(nw.availability_mask, sets[0].availability_mask)
    assert np.all(nw.values[nw.availability_mask == 0] == 0.0)
    tn = TargetNormalizer.fit([0.0, 1.0, 2.0, 3.0])
    assert abs(float(tn.destandardize(tn.standardize(1.7))) - 1.7) < 1e-9
    _expect(ValueError, lambda: TargetNormalizer(0.0, SD_FLOOR / 2))
    print("  [ok] FIT-only mask-aware input normalizer; target normalizer round-trip; SD floor enforced")


def test_artifact_sigma_min_and_hash_coverage():
    state = _state(); ex = _examples(state)
    a = fit_candidate("C2", "PD", ex[:36], ex[36:], seed=0)
    assert set(dict(a.sigma_min)) == set(NON_IDENTITY), "C2 sigma_min must cover all non-identity actions"
    # hash covers sigma_min: rebuild artifact with one floor changed -> different hash
    sm = dict(a.sigma_min); first = sorted(sm)[0]; sm[first] = sm[first] + 1.0
    b = FittedCandidate("C2", "PD", a.net, a.input_norm, a.target_norm, tuple(sorted(sm.items())),
                        a.training_epoch, a.env)
    assert b.artifact_sha256 != a.artifact_sha256, "artifact hash does not cover sigma_min"
    # C2 with incomplete sigma_min -> reject
    partial = tuple(sorted(sm.items())[:-1])
    _expect(ValueError, lambda: FittedCandidate("C2", "PD", a.net, a.input_norm, a.target_norm, partial,
                                                a.training_epoch, a.env))
    print("  [ok] C2 sigma_min covers all actions; artifact hash covers sigma_min; incomplete floor rejected")


def main():
    print("ACAR v3 training/normalizer/artifact guards:")
    for t in (test_loss_exactness, test_beta_nll_stop_gradient, test_subject_balanced_reduction,
              test_deterministic_train_twice, test_early_stop_best_state, test_normalizers,
              test_artifact_sigma_min_and_hash_coverage):
        t()
    print("ALL V3 TRAINING GUARDS PASS")


if __name__ == "__main__":
    main()
