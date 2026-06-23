"""
CSC-P1.4.2 regression tests (the review's required list): null accounting, paired-unit semantics,
operational separability, executable-manifest fail-closed. Fast COMPONENT tests of the invariants.
"""
import inspect
import warnings
import dataclasses
import numpy as np
warnings.filterwarnings("ignore")

from csc.protocol import ProtocolConfig, ProtocolError
from csc.certificate.residual_test import (
    _conservative_p, aggregate_subject_loss, _subject_condition_weights,
)
from csc.certificate.atlas import (
    support_signature_strata, stratified_subject_resample, cov_concept_angle,
)
from csc.calibration.lodo import oracle_boundary_effect, _subject_mean
from csc.sim.shift_simulator import SimConfig, make_geom, make_source, make_paired_subjects, GenGeom


# 1 -------------------------------------------------------------------------------------
def test_invalid_replicates_cannot_lower_p():
    valid = [0.1, 0.2, -0.3, 0.05]
    T = 0.15
    p0 = _conservative_p(valid, 0, T, n_boot=10)
    for n_inv in (1, 3, 6):
        p = _conservative_p(valid, n_inv, T, n_boot=10)
        assert p >= p0, f"invalid replicates LOWERED p ({p} < {p0})"
    # invalids are charged as extreme: p strictly increases with n_invalid
    assert _conservative_p(valid, 5, T, 10) > _conservative_p(valid, 0, T, 10)
    print("OK invalid null replicates can only RAISE the p-value (charged as extreme)")


# 2 -------------------------------------------------------------------------------------
def test_bootstrap_budget_vs_alpha_fail_closed():
    # B must reach alpha: min bootstrap p = 1/(B+1) <= alpha  =>  B >= ceil(1/alpha)-1
    import math
    a = 0.05
    bmin = math.ceil(1 / a) - 1
    ProtocolConfig(alpha=a, n_boot=bmin, n_dir_boot=bmin).validate()           # exactly enough
    for bad in (bmin - 1, 5, 0):
        try:
            ProtocolConfig(alpha=a, n_boot=bad, n_dir_boot=200).validate()
            raise AssertionError(f"n_boot={bad} should fail (cannot reach alpha)")
        except ProtocolError:
            pass
    print(f"OK bootstrap budget validated against alpha (need B >= {bmin} at alpha={a})")


# 3 -------------------------------------------------------------------------------------
def test_duplicate_epochs_within_cell_leaves_loss_invariant():
    # one subject (0) in two conditions; duplicate ALL epochs of cell (subj0, cond1)
    g = np.array([0, 0, 0, 0, 1, 1, 1])
    D = np.array([0, 0, 1, 1, 0, 0, 1])
    loss = np.array([1.0, 3.0, 2.0, 4.0, 0.5, 1.5, 2.0])
    base = aggregate_subject_loss(loss, g, D)
    dup = (g == 0) & (D == 1)
    g2 = np.concatenate([g, g[dup]]); D2 = np.concatenate([D, D[dup]])
    loss2 = np.concatenate([loss, loss[dup]])
    after = aggregate_subject_loss(loss2, g2, D2)
    assert abs(base[0] - after[0]) < 1e-12 and abs(base[1] - after[1]) < 1e-12, (base, after)
    print(f"OK duplicating epochs within a subject-condition cell leaves l_s invariant ({base[0]:.3f})")


# 4 -------------------------------------------------------------------------------------
def test_unequal_condition_epochs_do_not_reweight():
    # subject 0: cond0 has 100 epochs (mean loss 1.0), cond1 has 2 epochs (mean loss 3.0)
    g = np.zeros(102, int)
    D = np.concatenate([np.zeros(100, int), np.ones(2, int)])
    loss = np.concatenate([np.full(100, 1.0), np.full(2, 3.0)])
    ls = aggregate_subject_loss(loss, g, D)[0]
    assert abs(ls - 2.0) < 1e-9, f"condition-balanced l_s should be 2.0, got {ls} (epoch-weighted=1.04)"
    # fit weights: each (subject,condition) cell carries equal TOTAL weight
    w = _subject_condition_weights(g, D)
    assert abs(w[D == 0].sum() - w[D == 1].sum()) < 1e-9
    print("OK unequal ON/OFF epoch counts do NOT reweight conditions (l_s=2.0; equal cell weight)")


# 5 -------------------------------------------------------------------------------------
def test_paired_subjects_stay_intact_in_bootstrap():
    Z, Y, D, G = make_paired_subjects(SimConfig(seed=0), n_subjects=20, concept_delta=0.0, seed=0)
    idx_by, strata = support_signature_strata(G, D, Y)
    rng = np.random.default_rng(1)
    for _ in range(20):
        idx, gid = stratified_subject_resample(idx_by, strata, rng)
        # every resampled cluster id maps to ONE biological subject AND spans BOTH conditions
        for u in np.unique(gid):
            rows = idx[gid == u]
            assert len(np.unique(G[rows])) == 1, "a bootstrap cluster mixed two biological subjects"
            assert set(D[rows].tolist()) == {0, 1}, "a paired subject lost a condition in bootstrap"
    print("OK paired subjects remain INTACT (one biological subject, both conditions) in every bootstrap")


# 6 -------------------------------------------------------------------------------------
def test_domain_class_cells_cannot_disappear():
    src = make_source(SimConfig(seed=2), n_domains=6, concept_domains=2, seed=2)
    idx_by, strata = support_signature_strata(src.group_ids, src.D, src.Y)
    orig = set(zip(src.D.tolist(), src.Y.tolist()))
    rng = np.random.default_rng(3)
    for _ in range(30):
        idx, _ = stratified_subject_resample(idx_by, strata, rng)
        cells = set(zip(src.D[idx].tolist(), src.Y[idx].tolist()))
        assert orig <= cells, f"domain-class cells DISAPPEARED: {orig - cells}"
    print(f"OK no domain-class cell can silently disappear ({len(orig)} cells preserved every replicate)")


# 7 -------------------------------------------------------------------------------------
def test_oracle_and_decoder_same_subject_estimand():
    # oracle now accepts group_tr + group_g (subject-level), like the decoder
    params = inspect.signature(oracle_boundary_effect).parameters
    assert "group_tr" in params and "group_g" in params, "oracle must accept group_tr + group_g"
    # both estimands = mean over subjects of per-subject mean CE (identical on single-condition data)
    g = np.array([0, 0, 0, 1, 1, 2])
    D = np.zeros(6, int)                       # single condition
    loss = np.array([1.0, 2.0, 3.0, 0.0, 4.0, 5.0])
    dec = float(np.mean(list(aggregate_subject_loss(loss, g, D).values())))
    orc = _subject_mean(loss, g)
    assert abs(dec - orc) < 1e-12, f"oracle/decoder subject estimand differ: {orc} vs {dec}"
    print(f"OK oracle and decoder use the SAME subject-level CE estimand ({dec:.3f})")


# 8 -------------------------------------------------------------------------------------
def test_overlap_reacts_to_true_cov_concept_angle():
    # the operational separability diagnostic = angle between the nuisance offset A_cov and the
    # concept direction. Drive the TRUE angle and confirm the diagnostic tracks it monotonically
    # (so the planned principal-angle difficulty axis genuinely moves this gate).
    d = 6
    u = np.zeros(d); u[0] = 1.0                       # nuisance (covariate) direction
    rng = np.random.default_rng(0)
    A_cov = (4.0 * rng.standard_normal(8))[:, None] * u[None, :]   # 8 domain offsets along u
    prev = None
    measured = {}
    for theta in (90.0, 60.0, 30.0, 10.0):
        th = np.radians(theta)
        w = np.cos(th) * u + np.sin(th) * np.eye(d)[1]            # concept at angle theta from u
        ang = cov_concept_angle(A_cov, (w / np.linalg.norm(w))[:, None])
        measured[theta] = ang
        assert abs(ang - theta) < 2.0, f"diagnostic should read ~{theta}, got {ang:.1f}"
        if prev is not None:
            assert ang < prev, "angle must DECREASE as the true cov/concept angle decreases"
        prev = ang
    assert measured[90.0] > 20.0 >= measured[10.0], "gate must separate wide vs narrow angle"
    print(f"OK overlap diagnostic REACTS to true cov/concept angle: "
          f"{ {k: round(v,1) for k,v in measured.items()} }")


# 9 -------------------------------------------------------------------------------------
def test_unsupported_manifest_fails_closed():
    for bad in (dict(group_aware=False), dict(analysis_unit="epoch"),
                dict(tau_group_resampling=False),
                dict(analysis_unit="epoch", group_aware=False)):
        try:
            ProtocolConfig(**bad).validate()
            raise AssertionError(f"unsupported combo {bad} must fail closed")
        except ProtocolError:
            pass
    ProtocolConfig(analysis_unit="subject", group_aware=True, tau_group_resampling=True).validate()
    print("OK unsupported manifest combinations fail closed (only subject+group_aware+tau_group_resampling)")


# 10 ------------------------------------------------------------------------------------
def test_exact_pair_identical_not_necessarily_unidentifiable():
    # byte-identical Z (clean vs relabel-only pure) -> IDENTICAL certifier output, by construction.
    # The output need not be UNIDENTIFIABLE (that is a finite-sample statistical claim).
    from csc.sim.shift_simulator import make_paired_clean_pure
    from csc.certificate import analyze_source, certify
    src = make_source(SimConfig(seed=5), n_domains=8, concept_domains=3, seed=5)
    clean, pure = make_paired_clean_pure(SimConfig(seed=5), geom=src.geom, seed=55)
    assert np.array_equal(clean.Z, pure.Z), "paired clean/pure must be byte-identical in Z"
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40,
                        group_ids=src.group_ids, seed=5)
    c1 = certify(sa, clean.Z, group_ids=clean.group_ids)
    c2 = certify(sa, pure.Z, group_ids=pure.group_ids)
    assert c1.state == c2.state, f"byte-identical targets gave different states: {c1.state} vs {c2.state}"
    print(f"OK exact-pair indistinguishability: identical output ({c1.state}) -- NOT asserted UNIDENTIFIABLE")


if __name__ == "__main__":
    test_invalid_replicates_cannot_lower_p()
    test_bootstrap_budget_vs_alpha_fail_closed()
    test_duplicate_epochs_within_cell_leaves_loss_invariant()
    test_unequal_condition_epochs_do_not_reweight()
    test_paired_subjects_stay_intact_in_bootstrap()
    test_domain_class_cells_cannot_disappear()
    test_oracle_and_decoder_same_subject_estimand()
    test_overlap_reacts_to_true_cov_concept_angle()
    test_unsupported_manifest_fails_closed()
    test_exact_pair_identical_not_necessarily_unidentifiable()
    print("\nall CSC-P1.4.2 paired/accounting regression tests passed")
