"""Synthetic guards for acar/v3 training + artifact (Amendments 4+5). TOY ONLY. Run: python -m acar.v3.tests.test_training"""
import hashlib
import math
import numpy as np
import torch

from cmi.eval.source_state import fit_source_state
from acar.config import N_CLS
from acar.v3.set_features import build_action_sets, WindowKey, NON_IDENTITY
from acar.v3.data import SubjectKey
from acar.v3.normalizers import InputNormalizer, TargetNormalizer, TARGET_SD_FLOOR
from acar.v3.predictors import FittedCandidateArtifact, make_artifact, DeepSetsNet, state_items, build_net, HP
from acar.v3 import training as T
from acar.v3.training import (fit_candidate_earlystop, refit_candidate_fixed_epochs, final_epochs, TrainExample,
                              DeploymentFeatureRecord, _huber, _beta_nll, _pinball, _subject_balanced_loss)


def _state(d=8, seed=0):
    rng = np.random.default_rng(seed)
    y = (rng.random(160) < 0.5).astype(int)
    z = rng.standard_normal((160, d)) + np.where(y[:, None] == 1, 0.8, -0.8)
    return fit_source_state(z, y, N_CLS, rho=0.1)


def _ex(state, ds, subjects, disease="PD", nwin=12, d=8, seed=1):
    rng = np.random.default_rng(seed); ex = []
    for s in subjects:
        z = rng.standard_normal((nwin, d)) + 0.2 * rng.standard_normal()
        keys = [WindowKey(ds, s, "rec-00", w) for w in range(nwin)]
        sets = build_action_sets(state, z, keys)
        dg = hashlib.sha256(f"{ds}{s}".encode()).hexdigest(); sk = SubjectKey(ds, s)
        for a in NON_IDENTITY:
            ex.append(TrainExample(disease, sk, dg, a, sets[a], float(z.mean() + 0.1 * rng.standard_normal())))
    return ex


def _split(state, n_tr=12, n_va=6):
    tr = _ex(state, "ds", [f"sub-{i:03d}" for i in range(n_tr)], seed=1)
    va = _ex(state, "ds", [f"sub-{i:03d}" for i in range(n_tr, n_tr + n_va)], seed=2)
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
    bal = float(_subject_balanced_loss(net, {"A": A, "B": Bb}, "C1", ["A", "B"]))
    la = float(_subject_balanced_loss(net, {"A": A}, "C1", ["A"])); lb = float(_subject_balanced_loss(net, {"B": Bb}, "C1", ["B"]))
    assert abs(bal - 0.5 * (la + lb)) < 1e-5
    print("  [ok] subject-balanced reduction (50-batch subject does not dominate)")


def test_disease_binding_and_subject_disjoint():
    state = _state(); tr, va = _split(state)
    fit_candidate_earlystop("C1", "PD", tr, va, seed=0)                        # ok
    _expect(ValueError, lambda: fit_candidate_earlystop("C1", "SCZ", tr, va, seed=0))   # examples are PD
    mixed = tr[:-1] + [TrainExample("SCZ", tr[-1].subject_key, tr[-1].deployment_batch_digest, tr[-1].action,
                                    tr[-1].window_action_set, tr[-1].delta_r)]
    _expect(ValueError, lambda: fit_candidate_earlystop("C1", "PD", mixed, va, seed=0))  # mixed disease (valid candidate)
    _expect(ValueError, lambda: fit_candidate_earlystop("C1", "PD", tr, tr[:6], seed=0)) # TRAIN/VAL overlap
    print("  [ok] training disease-bound (TrainExample.disease) + TRAIN/VAL subject-disjoint enforced")


def test_deterministic_earlystop_and_epoch_semantics():
    state = _state(); tr, va = _split(state)
    a1, be1 = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    a2, be2 = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    assert a1.artifact_sha256 == a2.artifact_sha256 and len(a1.artifact_sha256) == 64
    assert a1.best_epoch_zero_based == be1 and a1.checkpoint_epoch_count == be1 + 1
    assert a1.n_epochs_executed >= a1.checkpoint_epoch_count and a1.n_epochs_executed <= HP["max_epochs"]
    print(f"  [ok] deterministic earlystop; checkpoint=best+1={a1.checkpoint_epoch_count}, executed={a1.n_epochs_executed}")


def test_canonical_bytes_and_integrity():
    state = _state(); tr, va = _split(state)
    a, _ = fit_candidate_earlystop("C2", "PD", tr, va, seed=0)
    assert all(dt == "<f4" for _, dt, _, _ in a.state_items), "state bytes not canonical '<f4'"
    assert set(dict(a.sigma_min)) == set(NON_IDENTITY) and all(v > 0 for _, v in a.sigma_min)
    net = build_net("C2", a.state_items)
    sm = dict(a.sigma_min); k0 = sorted(sm)[0]; sm[k0] += 1e-13
    b = make_artifact("C2", "PD", net, a.input_norm, a.target_norm, sm, a.best_epoch_zero_based, a.n_epochs_executed, dict(a.env))
    assert b.artifact_sha256 != a.artifact_sha256                              # 1e-13 floor -> different hash
    _expect(ValueError, lambda: make_artifact("C2", "PD", net, a.input_norm, a.target_norm,
                                              {kk: 0.0 for kk in NON_IDENTITY}, 0, 1, dict(a.env)))   # floor<=0
    _expect(ValueError, lambda: make_artifact("C1", "PD", net, a.input_norm, a.target_norm, {}, 0, 1, dict(a.env)))  # arch mismatch
    # duplicate sigma_min / env keys rejected (no dict() collision)
    _expect(ValueError, lambda: FittedCandidateArtifact("C2", "PD", a.state_items, a.input_norm, a.target_norm,
            (("matched_coral", 0.1), ("matched_coral", 0.1), ("spdim", 0.1), ("t3a", 0.1)), 0, 1, 1, tuple(a.env)))
    # tamper state bytes after construction -> integrity failure (no live cache to bypass)
    import copy
    t = copy.deepcopy(a); bad = list(t.state_items); nm, dt, sh, buf = bad[0]
    arr = np.frombuffer(buf, dtype=np.dtype(dt)).copy(); arr[0] += np.float32(1.0)
    bad[0] = (nm, dt, sh, arr.tobytes()); object.__setattr__(t, "state_items", tuple(bad))
    _expect(ValueError, lambda: t.verify_integrity())
    print("  [ok] canonical '<f4'; hash covers 1e-13 floor + dup-key reject; floor<=0/arch reject; tamper->integrity fail")


def test_nonfinite_loss_failclosed():
    net = DeepSetsNet("C1", 0); orig = T._example_loss
    T._example_loss = lambda *a, **k: torch.tensor(float("nan"))
    try:
        _expect(ValueError, lambda: _subject_balanced_loss(net, {"s": [(None, None, "matched_coral", None)]}, "C1", ["s"]))
    finally:
        T._example_loss = orig
    print("  [ok] non-finite loss -> deterministic candidate failure")


def test_refit_and_final_epochs():
    assert final_epochs([0, 1, 2]) == 2 and final_epochs([0, 1, 2, 3]) == 3
    for bad in ([True], [0.9], [-1]):                                          # strict non-bool, non-neg int
        _expect(ValueError, lambda b=bad: final_epochs(b))
    state = _state(); tr, va = _split(state); oof = {a: 0.05 for a in NON_IDENTITY}
    r1 = refit_candidate_fixed_epochs("C2", "PD", tr + va, 3, oof, seed=0)
    r2 = refit_candidate_fixed_epochs("C2", "PD", tr + va, 3, oof, seed=0)
    assert r1.artifact_sha256 == r2.artifact_sha256 and dict(r1.sigma_min) == oof
    assert r1.n_epochs_executed == 3 and r1.checkpoint_epoch_count == 3 and r1.best_epoch_zero_based == 2
    _expect(ValueError, lambda: refit_candidate_fixed_epochs("C2", "PD", tr + va, 3, {"matched_coral": 0.05}, 0))
    _expect(ValueError, lambda: refit_candidate_fixed_epochs("C2", "PD", tr + va, True, oof, 0))   # bool n_epochs
    print("  [ok] fixed-epoch refit deterministic; epoch provenance fields; OOF σ_min required; final_epochs strict")


def test_deployment_feature_record_consistency():
    state = _state()
    z = np.random.default_rng(0).standard_normal((12, 8)); keys = [WindowKey("ds", "s0", "r", w) for w in range(12)]
    sets = build_action_sets(state, z, keys); dg = hashlib.sha256(b"b").hexdigest(); sk = SubjectKey("ds", "s0")
    rec = DeploymentFeatureRecord("PD", sk, dg, tuple((a, sets[a], 0.1) for a in NON_IDENTITY))
    exs = rec.to_train_examples(); assert len(exs) == 3 and all(e.subject_key == sk for e in exs)
    # mismatched window_keys across actions -> raise (different natural batch)
    z2 = np.random.default_rng(1).standard_normal((12, 8)); keys2 = [WindowKey("ds", "s0", "r", w + 100) for w in range(12)]
    sets2 = build_action_sets(state, z2, keys2)
    bad = (("matched_coral", sets["matched_coral"], 0.1), ("spdim", sets2["spdim"], 0.1), ("t3a", sets["t3a"], 0.1))
    _expect(ValueError, lambda: DeploymentFeatureRecord("PD", sk, dg, bad))
    print("  [ok] DeploymentFeatureRecord enforces one window_key set across the 3 actions -> consistent TrainExamples")


def test_normalizers():
    state = _state()
    sets = [build_action_sets(state, np.random.default_rng(s).standard_normal((12, 8)),
                              [WindowKey("d", f"s{s}", "r", i) for i in range(12)])["matched_coral"] for s in range(8)]
    nw = InputNormalizer.fit(sets).transform(sets[0]); assert np.all(nw.values[nw.availability_mask == 0] == 0.0)
    tn = TargetNormalizer.fit([0.0, 1.0, 2.0, 3.0]); assert abs(float(tn.destandardize(tn.standardize(1.7))) - 1.7) < 1e-9
    _expect(ValueError, lambda: TargetNormalizer(0.0, TARGET_SD_FLOOR / 2))
    assert HP["target_sd_floor"] == 1e-3
    print("  [ok] FIT-only mask-aware input normalizer; target round-trip; target SD floor 1e-3 (HP synced)")


def main():
    print("ACAR v3 training/artifact guards:")
    for t in (test_loss_exactness, test_beta_nll_stop_gradient, test_subject_balanced,
              test_disease_binding_and_subject_disjoint, test_deterministic_earlystop_and_epoch_semantics,
              test_canonical_bytes_and_integrity, test_nonfinite_loss_failclosed, test_refit_and_final_epochs,
              test_deployment_feature_record_consistency, test_normalizers):
        t()
    print("ALL V3 TRAINING GUARDS PASS")


if __name__ == "__main__":
    main()
