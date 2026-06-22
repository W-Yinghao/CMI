"""Synthetic guards for acar/v3 training + artifact (Amendment 4). TOY ONLY (toy source state + GIVEN synthetic ΔR;
no DEV cohorts/gate/lockbox). Run: python -m acar.v3.tests.test_training
"""
import hashlib
import math
import numpy as np
import torch

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS
from acar.v3.set_features import build_action_sets, WindowKey, NON_IDENTITY
from acar.v3.data import SubjectKey
from acar.v3.normalizers import InputNormalizer, TargetNormalizer, TARGET_SD_FLOOR
from acar.v3.predictors import (FittedCandidateArtifact, make_artifact, DeepSetsNet, state_items, build_net, HP)
from acar.v3 import training as T
from acar.v3.training import (fit_candidate_earlystop, refit_candidate_fixed_epochs, final_epochs, TrainExample,
                              _huber, _beta_nll, _pinball, _subject_balanced_loss, _sigma_min)


def _state(d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(160) < 0.5).astype(int)
    z = rng.standard_normal((160, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    return fit_source_state(z, y, N_CLS, rho=0.1)


def _examples_for_subjects(state, ds, subjects, nwin=12, d=8, seed=1):
    rng = np.random.default_rng(seed); ex = []
    for s in subjects:
        z = rng.standard_normal((nwin, d)) + 0.2 * rng.standard_normal()
        keys = [WindowKey(ds, s, "rec-00", w) for w in range(nwin)]
        sets = build_action_sets(state, z, keys)
        digest = hashlib.sha256(f"{ds}|{s}".encode()).hexdigest()
        sk = SubjectKey(ds, s)
        for a in NON_IDENTITY:
            ex.append(TrainExample(sk, digest, a, sets[a], float(z.mean() + 0.1 * rng.standard_normal())))
    return ex


def _split(state, n_tr=12, n_va=6):
    tr = _examples_for_subjects(state, "ds", [f"sub-{i:03d}" for i in range(n_tr)], seed=1)
    va = _examples_for_subjects(state, "ds", [f"sub-{i:03d}" for i in range(n_tr, n_tr + n_va)], seed=2)
    return tr, va


def _expect(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__}")


def test_loss_exactness():
    y = torch.tensor(0.3)
    assert abs(float(_huber(torch.tensor(1.0), y, 1.0)) - 0.5 * 0.7 ** 2) < 1e-6
    assert abs(float(_huber(torch.tensor(3.0), y, 1.0)) - (abs(2.7) - 0.5)) < 1e-6
    v = 4.0; base = 0.5 * ((0.3 - 0.5) ** 2 / v + math.log(v))
    assert abs(float(_beta_nll(torch.tensor(0.5), torch.tensor(2.0), y, 0.5)) - base * v ** 0.5) < 1e-6
    assert abs(float(_pinball(y, torch.tensor(0.0), 0.9)) - 0.9 * 0.3) < 1e-6
    print("  [ok] Huber / β-NLL / pinball formulas exact")


def test_beta_nll_stop_gradient():
    y = torch.tensor(0.3)
    s1 = torch.tensor(2.0, requires_grad=True); _beta_nll(torch.tensor(0.5), s1, y, 0.5).backward()
    s2 = torch.tensor(2.0, requires_grad=True); v = s2 * s2
    (0.5 * ((y - 0.5) ** 2 / v + torch.log(v)) * (v ** 0.5)).backward()
    assert abs(float(s1.grad) - float(s2.grad)) > 1e-6
    print("  [ok] β-NLL variance weight is stop-gradient'd")


def test_subject_balanced():
    state = _state()
    was = build_action_sets(state, np.random.default_rng(0).standard_normal((12, 8)),
                            [WindowKey("d", "s", "r", i) for i in range(12)])["matched_coral"]
    net = DeepSetsNet("C1", 0); net.eval()
    win = torch.tensor(np.concatenate([was.values, was.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
    ctx = torch.tensor(np.concatenate([was.context_values, was.context_mask.astype(np.float64)]), dtype=torch.float32)
    A = [(win, ctx, "matched_coral", torch.tensor(0.0))]; Bb = [(win, ctx, "matched_coral", torch.tensor(2.0))] * 50
    by = {"A": A, "B": Bb}
    bal = float(_subject_balanced_loss(net, by, "C1", ["A", "B"]))
    la = float(_subject_balanced_loss(net, {"A": A}, "C1", ["A"]))
    lb = float(_subject_balanced_loss(net, {"B": Bb}, "C1", ["B"]))
    assert abs(bal - 0.5 * (la + lb)) < 1e-5
    print("  [ok] subject-balanced reduction (50-batch subject does not dominate)")


def test_train_val_subject_disjoint_enforced():
    state = _state(); tr, va = _split(state)
    fit_candidate_earlystop("C1", "PD", tr, va, seed=0)                      # disjoint -> ok
    _expect(ValueError, lambda: fit_candidate_earlystop("C1", "PD", tr, tr[:6], seed=0))   # overlap -> raise
    print("  [ok] TRAIN/VAL subject overlap rejected")


def test_deterministic_and_early_stop():
    state = _state(); tr, va = _split(state)
    a1 = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    a2 = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    assert a1.artifact_sha256 == a2.artifact_sha256 and len(a1.artifact_sha256) == 64
    assert 0 <= a1.training_epochs < HP["max_epochs"]
    print(f"  [ok] deterministic earlystop (identical artifact hash; best_epoch={a1.training_epochs})")


def test_c2_sigma_min_and_artifact_integrity():
    state = _state(); tr, va = _split(state)
    a = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    assert set(dict(a.sigma_min)) == set(NON_IDENTITY) and all(v > 0 for _, v in a.sigma_min)
    net = build_net("C2", a.state_items)
    # 1e-13 floor change -> different artifact hash (no rounding)
    sm = dict(a.sigma_min); k0 = sorted(sm)[0]; sm[k0] = sm[k0] + 1e-13
    b = make_artifact("C2", "PD", net, a.input_norm, a.target_norm, sm, a.training_epochs, dict(a.env))
    assert b.artifact_sha256 != a.artifact_sha256, "artifact hash insensitive to 1e-13 floor change"
    # missing / non-positive floor -> reject
    _expect(ValueError, lambda: make_artifact("C2", "PD", net, a.input_norm, a.target_norm,
                                              {kk: vv for kk, vv in list(sm.items())[:-1]}, 1, dict(a.env)))
    _expect(ValueError, lambda: make_artifact("C2", "PD", net, a.input_norm, a.target_norm,
                                              {kk: 0.0 for kk in NON_IDENTITY}, 1, dict(a.env)))
    # post-hash tampering -> integrity failure
    import copy
    tampered = copy.deepcopy(a)
    bad_items = list(tampered.state_items); name, dt, shape, buf = bad_items[0]
    arr = np.frombuffer(buf, dtype=np.dtype(dt)).copy(); arr[0] = arr[0] + 1.0
    bad_items[0] = (name, dt, shape, arr.tobytes()); object.__setattr__(tampered, "state_items", tuple(bad_items))
    _expect(ValueError, lambda: tampered.verify_integrity())
    # candidate / architecture mismatch -> reject
    _expect(ValueError, lambda: make_artifact("C1", "PD", net, a.input_norm, a.target_norm, {}, 1, dict(a.env)))
    print("  [ok] C2 sigma_min complete+positive; hash covers 1e-13 floor; tamper->integrity fail; arch mismatch->raise")


def test_nonfinite_loss_failclosed():
    net = DeepSetsNet("C1", 0)
    orig = T._example_loss
    T._example_loss = lambda *a, **k: torch.tensor(float("nan"))
    try:
        _expect(ValueError, lambda: _subject_balanced_loss(net, {"s": [(None, None, "matched_coral", None)]}, "C1", ["s"]))
    finally:
        T._example_loss = orig
    print("  [ok] non-finite loss -> deterministic candidate failure (ValueError)")


def test_refit_fixed_epochs_and_final_epochs():
    assert final_epochs([0, 1, 2]) == 2 and final_epochs([0, 1, 2, 3]) == 3      # round-half-up of median(best+1)
    state = _state(); tr, va = _split(state)
    oof = {a: 0.05 for a in NON_IDENTITY}
    r1 = refit_candidate_fixed_epochs("C2", "PD", tr + va, n_epochs=3, sigma_min_oof=oof, seed=0)
    r2 = refit_candidate_fixed_epochs("C2", "PD", tr + va, n_epochs=3, sigma_min_oof=oof, seed=0)
    assert r1.artifact_sha256 == r2.artifact_sha256
    assert dict(r1.sigma_min) == oof and r1.training_epochs == 2               # uses supplied OOF floor; epoch=n-1
    _expect(ValueError, lambda: refit_candidate_fixed_epochs("C2", "PD", tr + va, 3, {"matched_coral": 0.05}, 0))
    print("  [ok] fixed-epoch refit deterministic; uses supplied OOF σ_min; final_epochs round-half-up")


def test_normalizers():
    state = _state()
    sets = [build_action_sets(state, np.random.default_rng(s).standard_normal((12, 8)),
                              [WindowKey("d", f"s{s}", "r", i) for i in range(12)])["matched_coral"] for s in range(8)]
    nw = InputNormalizer.fit(sets).transform(sets[0])
    assert np.all(nw.values[nw.availability_mask == 0] == 0.0)
    tn = TargetNormalizer.fit([0.0, 1.0, 2.0, 3.0]); assert abs(float(tn.destandardize(tn.standardize(1.7))) - 1.7) < 1e-9
    _expect(ValueError, lambda: TargetNormalizer(0.0, TARGET_SD_FLOOR / 2))
    print("  [ok] FIT-only mask-aware input normalizer; target round-trip; target SD floor 1e-3 enforced")


def main():
    print("ACAR v3 training/artifact guards:")
    for t in (test_loss_exactness, test_beta_nll_stop_gradient, test_subject_balanced,
              test_train_val_subject_disjoint_enforced, test_deterministic_and_early_stop,
              test_c2_sigma_min_and_artifact_integrity, test_nonfinite_loss_failclosed,
              test_refit_fixed_epochs_and_final_epochs, test_normalizers):
        t()
    print("ALL V3 TRAINING GUARDS PASS")


if __name__ == "__main__":
    main()
