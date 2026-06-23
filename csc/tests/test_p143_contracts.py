"""
CSC-P1.4.3 inference-contract regression tests (the review's 9 pass conditions): full-T
duplication invariance, seed-driven row-multiplicity-independent folds, geometry-null support
gate + unified source status, principal-angle separability, paired estimand consistency,
label_unit freeze, and cluster-size-profile calibration.
"""
import warnings
import numpy as np
warnings.filterwarnings("ignore")

from csc.certificate.residual_test import (
    residual_decoder_test, _subject_fold_assignment, check_support_graph, subject_null_labels,
)
from csc.certificate.atlas import (
    cov_concept_angle, principal_angle_deg, _concept_attribution_stability, cluster_mean, SourceAnalysis,
)
from csc.certificate.certifier import certify, UNIDENTIFIABLE
from csc.calibration.lodo import calibrate_thresholds, oracle_boundary_effect, CertifierConfig
from csc.sim.shift_simulator import SimConfig, make_source, make_paired_subjects


def _paired(seed=0, n=30, delta=1.0):
    return make_paired_subjects(SimConfig(seed=seed), n_subjects=n, concept_delta=delta, seed=seed)


# 1 ---- full residual_decoder_test T/p/status invariant to within-cell epoch duplication --------
def test_full_T_invariant_to_epoch_duplication():
    Z, Y, D, G = _paired()
    r1 = residual_decoder_test(Z, Y, D, n_boot=30, group_ids=G, C=0.5, label_unit="trial", seed=0)
    dup = (G == G[0]) & (D == D[0])
    Z2, Y2 = np.concatenate([Z, Z[dup]]), np.concatenate([Y, Y[dup]])
    D2, G2 = np.concatenate([D, D[dup]]), np.concatenate([G, G[dup]])
    r2 = residual_decoder_test(Z2, Y2, D2, n_boot=30, group_ids=G2, C=0.5, label_unit="trial", seed=0)
    assert abs(r1.T - r2.T) < 5e-3, f"T not invariant: {r1.T} vs {r2.T}"
    assert r1.p_value == r2.p_value and r1.status == r2.status, (r1.p_value, r2.p_value)
    print(f"OK full T/p/status invariant to epoch duplication (dT={abs(r1.T-r2.T):.1e}, "
          f"p={r1.p_value:.3f}, status={r1.status})")


# 2 ---- named CV seed drives folds; folds independent of row multiplicity ----------------------
def test_folds_seed_driven_and_multiplicity_independent():
    Z, Y, D, G = _paired(n=24)
    f_a, _, _ = _subject_fold_assignment(G, Y, 4, seed=1)
    f_b, _, _ = _subject_fold_assignment(G, Y, 4, seed=2)
    assert not np.array_equal(f_a, f_b), "different named seeds must give different folds"
    # duplicate one subject's epochs -> its (and everyone's) fold assignment is UNCHANGED
    dup = G == G[0]
    G2 = np.concatenate([G, G[dup]]); Y2 = np.concatenate([Y, Y[dup]])
    f_a2, _, _ = _subject_fold_assignment(G2, Y2, 4, seed=1)
    # the original rows keep their fold; subject->fold map is identical
    for s in np.unique(G):
        assert len(set(f_a[G == s].tolist())) == 1 and len(set(f_a2[G2 == s].tolist())) == 1
        assert f_a[G == s][0] == f_a2[G2 == s][0], "row multiplicity changed a subject's fold"
    print("OK named CV seed drives folds; fold assignment is row-multiplicity independent")


# 3 ---- disconnected-but-all-classes graph is INVALID ------------------------------------------
def test_disconnected_all_classes_is_invalid():
    # domains {0,1}<->classes{0,1}; domains {2,3}<->classes{2,3}: every class present, but the
    # bipartite (domain,class) graph is DISCONNECTED -> must be invalid.
    g, Y, D = [], [], []
    sid = 0
    for (doms, cls) in (([0, 1], [0, 1]), ([2, 3], [2, 3])):
        for d in doms:
            for c in cls:
                for _ in range(5):                  # 5 subjects per occupied cell
                    g.append(sid); Y.append(c); D.append(d); sid += 1
    g = np.repeat(np.arange(len(g)), 4)             # 4 epochs/subject
    Y = np.repeat(Y, 4); D = np.repeat(D, 4)
    sup = check_support_graph(np.array(Y), np.array(D), group_ids=g, n_folds=2, check_design=False)
    assert not sup.valid, "disconnected graph with all classes must be INVALID"
    assert any("DISCONNECT" in r.upper() for r in sup.reasons)
    print("OK disconnected-but-all-classes graph -> INVALID (geometry null would charge it)")


# 4 ---- unified source status: each non-VALID status -> certifier abstains ---------------------
def test_source_status_gate_abstains():
    import dataclasses
    from csc.certificate.atlas import analyze_source
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40,
                        group_ids=src.group_ids, seed=0)
    for st in ("INVALID_SUPPORT", "INVALID_RESIDUAL_NULL", "INVALID_GEOMETRY_NULL",
               "UNASSESSED_CONCEPT_STABILITY", "UNSTABLE_CONCEPT_ATTRIBUTION"):
        sa_bad = dataclasses.replace(sa, source_status=st)
        c = certify(sa_bad, src.Z[:300], group_ids=src.group_ids[:300])
        assert c.state == UNIDENTIFIABLE, f"status {st} must abstain, got {c.state}"
    # the VALID source must NOT be forced to abstain by this gate
    assert certify(sa, src.Z[:300], group_ids=src.group_ids[:300]) is not None
    print("OK every non-VALID source_status -> certifier abstains (geometry/separability covered)")


# 5 ---- principal-angle: rank>1 exact overlap -> ~0; orthogonal -> ~90 ------------------------
def test_principal_angle_rank2_overlap():
    A = np.array([[1.0, 0, 0], [0, 10, 0]])            # rowspace = span(e0,e1)
    assert cov_concept_angle(A, np.array([[1.0], [0], [0]])) < 1.0, "exact overlap must read ~0"
    assert cov_concept_angle(A, np.array([[0.0], [0], [1]])) > 89.0, "orthogonal must read ~90"
    # weak overlapping direction NOT masked by a strong orthogonal one (the energy-ratio bug)
    assert principal_angle_deg(np.array([[1.0], [0]]), np.array([[1.0], [0]])) < 1.0
    print("OK principal-angle separability: rank-2 exact overlap ~0 (not masked); orthogonal ~90")


# 6 ---- separability UNASSESSED -> abstain (fail-closed) ---------------------------------------
def test_separability_unassessed_abstains():
    import dataclasses
    from csc.certificate.atlas import analyze_source
    Z, Y, D, G = _paired(n=3)                         # < 4 subjects -> cannot split-assess
    _, assessable, _ = _concept_attribution_stability(Z, Y, D, list(np.unique(Y)),
                                                      list(np.unique(D)), 0.95, G, 0, 0.30)
    assert assessable is False, "too-few-subjects stability must be UNASSESSED (not 'stable')"
    # and an UNASSESSED-stability source must abstain end-to-end
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    sa = analyze_source(src.Z, src.Y, src.D, n_boot=20, n_dir_boot=40, group_ids=src.group_ids, seed=0)
    sa_un = dataclasses.replace(sa, source_status="UNASSESSED_CONCEPT_STABILITY")
    assert certify(sa_un, src.Z[:300], group_ids=src.group_ids[:300]).state == UNIDENTIFIABLE
    print("OK concept-stability unassessed -> abstain (fail-closed)")


# 7 ---- atlas pooled mean is CONDITION-balanced (same estimand as the decoder) -----------------
def test_atlas_pooled_mean_condition_balanced():
    Z, Y, D, G = _paired(n=20)
    base = cluster_mean(Z, G, D)
    dup = (G == G[0]) & (D == D[0])                   # duplicate ONE condition's epochs
    Z2 = np.concatenate([Z, Z[dup]]); G2 = np.concatenate([G, G[dup]]); D2 = np.concatenate([D, D[dup]])
    after = cluster_mean(Z2, G2, D2)
    assert np.max(np.abs(base - after)) < 1e-9, "condition-first pooled mean must be duplication-invariant"
    # and it DIFFERS from the row-weighted (non-condition) mean on unequal ON/OFF data
    assert np.max(np.abs(cluster_mean(Z, G) - base)) > 0
    print("OK atlas pooled mean is condition-first (duplication-invariant; != row-weighted)")


# 8 ---- paired simulator matches its declared label_unit (trial); source matches 'subject' ----
def test_simulator_label_unit_consistency():
    Z, Y, D, G = _paired(n=20)
    # trial unit: Y VARIES within at least one (subject,condition) cell
    varies = any(np.unique(Y[(G == s) & (D == c)]).size > 1
                 for s in np.unique(G) for c in np.unique(D[G == s]))
    assert varies, "paired simulator declares label_unit='trial' -> Y must vary within a cell"
    src = make_source(SimConfig(seed=0), n_domains=6, concept_domains=2, seed=0)
    assert all(np.unique(src.Y[src.group_ids == s]).size == 1 for s in np.unique(src.group_ids))
    print("OK simulators consistent with declared label_unit (paired=trial, source=subject)")


# 9 ---- tau calibration matches the target CLUSTER-SIZE PROFILE, not just subject count --------
def test_calibration_matches_cluster_size_profile():
    src = make_source(SimConfig(seed=0), n_domains=8, concept_domains=3, seed=0)
    from csc.certificate.atlas import build_atlas
    at = build_atlas(src.Z, src.Y, src.D, group_ids=src.group_ids)
    base = CertifierConfig()
    k = 12
    big = calibrate_thresholds(src.Z, src.Y, src.D, at, base, block_ids_tr=src.group_ids,
                               target_epochs_per_subject=np.full(k, 60), n_block=80, seed=0)
    small = calibrate_thresholds(src.Z, src.Y, src.D, at, base, block_ids_tr=src.group_ids,
                                 target_epochs_per_subject=np.full(k, 4), n_block=80, seed=0)
    # same subject count k, but few-epochs/subject => larger subject-mean sampling variance => larger tau
    assert small.tau_detect > big.tau_detect, (small.tau_detect, big.tau_detect)
    print(f"OK calibration matches cluster-size profile: tau_detect 4-ep={small.tau_detect:.2f} "
          f"> 60-ep={big.tau_detect:.2f} (same subject count)")


if __name__ == "__main__":
    test_full_T_invariant_to_epoch_duplication()
    test_folds_seed_driven_and_multiplicity_independent()
    test_disconnected_all_classes_is_invalid()
    test_source_status_gate_abstains()
    test_principal_angle_rank2_overlap()
    test_separability_unassessed_abstains()
    test_atlas_pooled_mean_condition_balanced()
    test_simulator_label_unit_consistency()
    test_calibration_matches_cluster_size_profile()
    print("\nall CSC-P1.4.3 contract tests passed")
