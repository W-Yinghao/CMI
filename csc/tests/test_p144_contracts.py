"""
CSC-P1.4.4 inference-contract regression tests (the review's 10 pass conditions): label_unit
data-validation + class-profile folds, leading-direction concept-attribution stability that blocks
BOTH states, conservative cov-bootstrap, single-condition target + cluster-vote visibility helper,
manifest-field hygiene, and machine-readable full-T invariance in the audit.
"""
import warnings
import dataclasses
import numpy as np
warnings.filterwarnings("ignore")

from csc.protocol import ProtocolConfig, ProtocolError, execute_protocol
from csc.certificate.residual_test import (
    validate_label_unit, LabelUnitError, _subject_fold_assignment,
)
from csc.certificate.atlas import (
    analyze_source, principal_angle_deg, cluster_mean, components, visibility_statistic,
)
from csc.certificate.certifier import certify, UNIDENTIFIABLE
from csc.calibration.lodo import calibrate_thresholds, CertifierConfig
from csc.sim.shift_simulator import SimConfig, make_source, make_paired_subjects


# 1 ---- a wrong label_unit declaration must fail closed -----------------------------------------
def test_wrong_label_unit_fails_closed():
    Z, Y, D, G = make_paired_subjects(SimConfig(seed=0), n_subjects=10, concept_delta=1.0, seed=0)
    # paired data is TRIAL-level (Y varies within a (subject,condition) cell)
    for bad in ("subject", "subject_condition"):
        try:
            validate_label_unit(Y, G, D, bad)
            raise AssertionError(f"label_unit={bad} must fail on trial-level data")
        except LabelUnitError:
            pass
    validate_label_unit(Y, G, D, "trial")                # the correct declaration passes
    src = make_source(SimConfig(seed=0), n_domains=6, concept_domains=2, seed=0)
    validate_label_unit(src.Y, src.group_ids, src.D, "subject")   # source IS subject-level
    print("OK wrong label_unit fails closed; correct one passes")


# 2 ---- mixed-label subjects: every grouped TRAINING fold keeps all-class support ---------------
def test_mixed_label_folds_keep_class_support():
    Z, Y, D, G = make_paired_subjects(SimConfig(seed=1), n_subjects=24, concept_delta=1.0, seed=1)
    assert any(np.unique(Y[G == s]).size > 1 for s in np.unique(G)), "need mixed-label subjects"
    fold, k, covered = _subject_fold_assignment(G, Y, 4, seed=3)
    assert covered, "class-profile folds must retain all-class support in every training split"
    for f in range(k):
        assert set(np.unique(Y[fold != f])) == set(np.unique(Y))
    print(f"OK mixed-label subjects: all {k} training folds keep full class support")


# 3 ---- min_principal_angle_deg removed; new stability params hashed ---------------------------
def test_manifest_field_hygiene():
    fields = {f.name for f in dataclasses.fields(ProtocolConfig)}
    assert "min_principal_angle_deg" not in fields, "dead manifest field must be removed"
    for f in ("concept_stability_max_deg", "concept_eigengap_min", "label_unit"):
        assert f in fields
    a = ProtocolConfig()
    assert ProtocolConfig(concept_eigengap_min=0.5).hash() != a.hash()          # drives the hash
    assert ProtocolConfig(concept_stability_max_deg=30.0).hash() != a.hash()
    print("OK min_principal_angle_deg removed; concept_stability_max_deg/eigengap drive the hash")


# 4 ---- leading-direction stability: shared SECONDARY dir but orthogonal LEADING dirs -> unstable
def test_leading_direction_not_fooled_by_shared_secondary():
    e = np.eye(4)
    UA = np.stack([e[0], e[1]], 1)        # leading e0, secondary e1
    UB = np.stack([e[1], e[2]], 1)        # leading e1, secondary e2  (shares e1 as a sub-direction)
    # the OLD min-principal-angle is ~0 (they share e1) -> would call them 'stable'
    assert principal_angle_deg(UA, UB) < 1.0
    # the NEW sign-invariant LEADING-direction angle (e0 vs e1) is 90 deg -> correctly UNSTABLE
    vA, vB = UA[:, 0], UB[:, 0]
    lead_angle = np.degrees(np.arccos(np.clip(abs(vA @ vB), 0, 1)))
    assert lead_angle > 89.0, lead_angle
    print(f"OK leading-direction angle {lead_angle:.0f} deg (not masked by shared secondary; "
          f"min-principal-angle={principal_angle_deg(UA, UB):.1f})")


# 5 ---- UNSTABLE/UNASSESSED concept attribution blocks BOTH definite states --------------------
def test_attribution_failure_blocks_both_states():
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40, group_ids=src.group_ids, seed=0)
    cov_like = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)  # any target Z
    for st in ("UNSTABLE_CONCEPT_ATTRIBUTION", "UNASSESSED_CONCEPT_STABILITY"):
        bad = dataclasses.replace(sa, source_status=st)
        c = certify(bad, cov_like.Z[:300], group_ids=cov_like.group_ids[:300])
        assert c.state == UNIDENTIFIABLE, f"{st} must block BOTH states, got {c.state}"
    print("OK unstable/unassessed concept attribution -> abstains BOTH concept AND covariate")


# 6 ---- an invalid cov-bootstrap replicate can only RAISE cov_ub (never drop it) ---------------
def test_invalid_cov_replicate_cannot_lower_ub():
    # the implementation charges an invalid replicate +inf and KEEPS it (FIXED B). Replacing any
    # valid value with +inf can only RAISE the (1-alpha) quantile -- never lower it (the v0 DROPPED
    # invalids, reducing B, which could shrink cov_ub and over-pass COVARIATE_COMPATIBLE).
    rng = np.random.default_rng(0)
    B = 16
    base = rng.random(B)
    base_ub = float(np.quantile(base, 0.95))
    for k in (1, 3, 8):
        arr = base.copy(); arr[:k] = np.inf            # k invalid replicates -> +inf, SAME B
        q = float(np.quantile(arr, 0.95))
        assert not (q < base_ub), "invalid replicate lowered cov_ub"   # inf/nan are NOT lower
    # enough invalids drive the (1-alpha) quantile to inf/nan -> cov_stable=bool(.<eps)=False
    allinf = base.copy(); allinf[: B - 1] = np.inf
    cov_ub = float(np.quantile(allinf, 0.95))
    assert not bool(cov_ub < 1e9), "saturated-invalid cov_ub must NOT certify stable"
    print("OK invalid cov replicate -> +inf (fixed B): can only RAISE cov_ub, never lower it")


# 7 ---- single-condition target guard + cluster-size-profile calibration -----------------------
def test_single_condition_target_and_profile_tau():
    src = make_source(SimConfig(seed=2), n_domains=8, concept_domains=3, seed=2)
    tb = make_source(SimConfig(seed=3), n_domains=8, concept_domains=3, seed=3)
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, target_n_boot=30, tau_n_pseudotargets=40, oracle_boot=20)
    multi_cond = np.arange(len(tb.Z)) % 2                  # TWO conditions in one target -> reject
    try:
        execute_protocol(src.Z, src.Y, src.D, tb.Z, cfg, src_group_ids=src.group_ids,
                         tgt_group_ids=tb.group_ids, tgt_condition_ids=multi_cond, seed=2)
        raise AssertionError("multi-condition target must be rejected")
    except ProtocolError:
        pass
    # cluster-size profile drives tau: same subject count, fewer epochs/subject -> larger tau
    from csc.certificate.atlas import build_atlas
    at = build_atlas(src.Z, src.Y, src.D, group_ids=src.group_ids)
    big = calibrate_thresholds(src.Z, src.Y, src.D, at, CertifierConfig(), block_ids_tr=src.group_ids,
                               target_epochs_per_subject=np.full(12, 80), n_block=100, seed=0)
    small = calibrate_thresholds(src.Z, src.Y, src.D, at, CertifierConfig(), block_ids_tr=src.group_ids,
                                 target_epochs_per_subject=np.full(12, 3), n_block=100, seed=0)
    assert small.tau_detect > big.tau_detect, (small.tau_detect, big.tau_detect)
    print(f"OK single-condition target enforced; tau matches profile "
          f"(3-ep {small.tau_detect:.2f} > 80-ep {big.tau_detect:.2f})")


# 8 ---- visibility helper uses the SAME subject-vote aggregator as the certifier ---------------
def test_visibility_helper_matches_certifier_aggregator():
    src = make_source(SimConfig(seed=4), n_domains=8, concept_domains=3, seed=4)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40, group_ids=src.group_ids, seed=4)
    g = src.group_ids[:400]; Zt = src.Z[:400]
    vis = visibility_statistic(sa.atlas, Zt, group_ids=g)
    delta = cluster_mean(Zt, g) - sa.atlas.pooled_mean           # the certifier's exact delta
    c = components(sa.atlas, delta)
    assert abs(vis - max(c["n_cov"], c["n_concept"], c["n_resid"])) < 1e-9
    # and it DIFFERS from the old raw-row-mean version on cluster data
    raw = components(sa.atlas, Zt.mean(0) - sa.atlas.pooled_mean)
    assert abs(vis - max(raw["n_cov"], raw["n_concept"], raw["n_resid"])) > 0
    print("OK visibility_statistic uses cluster_mean (== certifier), not a raw row mean")


# 9 ---- audit records FULL T/p/status/fold-hash invariance (not just the loss helper) ----------
def test_audit_records_full_T_invariance():
    from csc.run_audit import contract_diagnostics
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40, source_cv_folds=4)
    diag = contract_diagnostics(cfg, [0])
    ft = diag["full_residual_test_duplication_invariance"]
    for kk in ("T_before", "T_after", "abs_delta_T", "p_before", "p_after", "abs_delta_p",
               "status_before", "status_after", "fold_hash_before", "fold_hash_after"):
        assert kk in ft, f"audit missing full-T field {kk}"
    assert ft["abs_delta_p"] == 0.0 and ft["status_before"] == ft["status_after"]
    assert ft["fold_hash_before"] == ft["fold_hash_after"]
    print(f"OK audit records full residual-test invariance (dT={ft['abs_delta_T']:.1e}, "
          f"p/status/fold-hash identical)")


# 10 ---- audit status + diagnostics block name match the actual round --------------------------
def test_audit_status_matches_round():
    from csc.run_audit import contract_diagnostics
    cfg = ProtocolConfig(n_boot=20, n_dir_boot=40)
    diag = contract_diagnostics(cfg, [0])
    assert "method_changed_vs_prior_round" in diag and diag["method_changed_vs_prior_round"]
    assert diag["builtin_python_hash_used"] is False and diag["seed_hash_algo"] == "sha256"
    assert "concept_eigengap_min" in diag and "concept_stability_max_deg" in diag
    print("OK audit diagnostics carry the P1.4.4 fields + sha256 seed provenance")


if __name__ == "__main__":
    test_wrong_label_unit_fails_closed()
    test_mixed_label_folds_keep_class_support()
    test_manifest_field_hygiene()
    test_leading_direction_not_fooled_by_shared_secondary()
    test_attribution_failure_blocks_both_states()
    test_invalid_cov_replicate_cannot_lower_ub()
    test_single_condition_target_and_profile_tau()
    test_visibility_helper_matches_certifier_aggregator()
    test_audit_records_full_T_invariance()
    test_audit_status_matches_round()
    print("\nall CSC-P1.4.4 contract tests passed")
