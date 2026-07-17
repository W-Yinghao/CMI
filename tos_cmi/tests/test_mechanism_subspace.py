"""Stage-B tests for the Mechanism-Subspace Oracle (M0.2 contract). Algebraic + firewall properties; synthetic
data is used ONLY to check the implementation, never for a scientific verdict."""
import numpy as np
import pytest

from tos_cmi.eval import mechanism_subspace as MS


def _src(seed=0, C=4, m=6, per=60, p=12, missing=False):
    rng = np.random.default_rng(seed)
    Z, ys, ds = [], [], []
    for d in range(m):
        for c in range(C):
            if missing and d == 0 and c == C - 1:
                continue
            n = per * (3 if d == 1 else 1)              # subject 1 has 3x trials (weighting test)
            mu = rng.standard_normal(p) + (2.0 if c == 1 else 0) * np.eye(p)[0] + 0.5 * d * np.eye(p)[1]
            Z.append(mu + 0.3 * rng.standard_normal((n, p))); ys += [c] * n; ds += [d] * n
    return np.vstack(Z), np.array(ys), np.array(ds)


def test_helmert_orthonormal_and_sum_zero():
    for C in (2, 3, 4):
        H = MS.build_helmert_contrast_matrices(C)
        assert H.shape == (C - 1, C)
        assert np.allclose(H @ H.T, np.eye(C - 1), atol=1e-10)          # orthonormal
        assert np.allclose(H @ np.ones(C), 0, atol=1e-10)              # H 1 = 0


def test_contrast_disagreement_weighting_independent_of_trial_count():
    Zw, ys, ds = _src(0)
    out = MS.build_contrast_disagreement(Zw, ys, ds)
    assert not out["fail_closed"]
    # G_dis built from per-subject CLASS MEANS (equal weight), not trial-weighted: subject-1's 3x trials must not
    # dominate. Recompute with subject 1 downsampled to 1x and check G_dis is close (weighting is by subject/class).
    keep = ~((ds == 1) & (np.arange(len(ds)) % 3 != 0))
    out2 = MS.build_contrast_disagreement(Zw[keep], ys[keep], ds[keep])
    assert np.linalg.norm(out["G_dis"] - out2["G_dis"]) / (np.linalg.norm(out["G_dis"]) + 1e-9) < 0.5


def test_missing_source_class_fails_closed():
    Zw, ys, ds = _src(0, missing=True)
    out = MS.build_contrast_disagreement(Zw, ys, ds)
    assert out["fail_closed"] and "missing_class" in out["reason"]


def test_generalized_eig_orthonormal_basis_and_eigenvalues():
    Zw, ys, ds = _src(1)
    cd = MS.build_contrast_disagreement(Zw, ys, ds)
    gm = MS.solve_generalized_mechanism_basis(cd["G_dis"], cd["G_shared"])
    assert not gm["below_resolution"]
    B = gm["orthonormal_basis"]
    assert B.shape[0] == gm["numerical_rank"] and B.shape[0] <= MS.DICT_MAX_RANK
    assert np.allclose(B @ B.T, np.eye(B.shape[0]), atol=1e-8)         # ambient-orthonormal dictionary
    assert len(gm["generalized_eigenvalues"]) >= 1 and "eta" in gm


def test_zero_shared_signal_below_resolution():
    p = 8; G_dis = np.eye(p); G_shared = np.zeros((p, p))
    gm = MS.solve_generalized_mechanism_basis(G_dis, G_shared)
    assert gm["below_resolution"] and gm["reason"] == "TASK_MECHANISM_BELOW_RESOLUTION"


def test_exhaustive_action_family_count_and_identity():
    acts = MS.build_exhaustive_action_family(8, 3)
    assert acts[0] == []                                              # identity included
    assert len(acts) == 1 + 8 + 28 + 56                              # 93 = identity + 92 non-empty
    assert all(len(S) <= 3 for S in acts)


def test_ambient_and_matched_random_same_rank_budget():
    D, r = 12, 8
    amb = MS.build_ambient_random_dictionaries(D, r, 5, 0)
    assert all(Q.shape == (r, D) and np.allclose(Q @ Q.T, np.eye(r), atol=1e-8) for Q in amb)
    # matched dictionaries are also rank r and go through the SAME 92-action exhaustive family downstream
    G_shared = np.eye(D)
    B = MS.build_ambient_random_dictionaries(D, r, 1, 99)[0]
    mm = MS.build_shared_profile_matched_dictionaries(B, G_shared, D, r, 5, 0, n_pool=50)
    assert all(Q.shape == (r, D) for Q in mm["dictionaries"])


def test_shared_overlap_matching_uses_no_target_outcome():
    # build_shared_profile_matched_dictionaries signature takes ONLY (B, G_shared, D, rank, ...) -> no y/query args
    import inspect
    params = set(inspect.signature(MS.build_shared_profile_matched_dictionaries).parameters)
    assert not ({"yq", "Xq", "ycal", "query", "target"} & params)


def test_shared_match_fail_closed_verdict():
    D, r = 16, 8; G_shared = np.diag(np.arange(1, D + 1).astype(float))   # anisotropic -> hard to match
    B = np.eye(D)[:r]                                                  # coordinate dict (extreme profile)
    mm = MS.build_shared_profile_matched_dictionaries(B, G_shared, D, r, 5, 0, n_pool=80)
    assert mm["verdict"] in ("OK", "SHARED_MATCH_CONTROL_FAILED")
    assert (mm["verdict"] == "OK") == mm["match_ok"]


def test_source_safety_uses_source_loso():
    Zw, ys, ds = _src(2, p=10)
    U = np.eye(10)[[0]]                                               # delete the class-1 discriminative axis
    s = MS.source_loso_safety(Zw, ys, ds, U)
    assert set(s) >= {"mean_drop", "median_drop", "worst_drop", "n_positive", "n_negative"}
    assert np.isfinite(s["mean_drop"])


def test_constrained_vs_unconstrained_selection_separate():
    Zw, ys, ds = _src(3, p=10)
    cd = MS.build_contrast_disagreement(Zw, ys, ds)
    B = MS.solve_generalized_mechanism_basis(cd["G_dis"], cd["G_shared"])["orthonormal_basis"]
    acts = MS.build_exhaustive_action_family(B.shape[0], 3)
    Xcal, ycal = Zw[:80], ys[:80]
    S_unc = MS.select_on_target_cal(Zw, ys, B, acts, Xcal, ycal, source_safe=False)
    S_safe = MS.select_on_target_cal(Zw, ys, B, acts, Xcal, ycal, source_safe=True, ds=ds, safety_max=0.02)
    assert isinstance(S_unc, list) and isinstance(S_safe, list)       # both computed independently


def test_query_labels_not_used_in_selection():
    import inspect
    p = set(inspect.signature(MS.select_on_target_cal).parameters)
    assert not ({"Xq_w", "yq", "sq", "query"} & p)                   # selection sees only cal


def test_ridge_rule_shared_plus_residual():
    Zw, ys, ds = _src(4, p=10)
    out = MS.fit_shared_residual_ridge(Zw, ys, ds, lam0=1.0, lamD=10.0)
    assert not out["fail_closed"]
    assert out["G_rule"].shape == (10, 10) and len(out["raw_singular_values"]) >= 1


def test_class_conditional_gradient_grouped_by_dy():
    Zw, ys, ds = _src(5, p=10)
    out = MS.build_class_conditional_gradient_disagreement(Zw, ys, ds)
    assert not out["fail_closed"]
    assert out["G_grad"].shape == (10, 10) and len(out["raw_singular_values"]) >= 1


def test_builders_save_raw_spectrum_and_numerical_rank():
    Zw, ys, ds = _src(6)
    cd = MS.build_contrast_disagreement(Zw, ys, ds)
    gm = MS.solve_generalized_mechanism_basis(cd["G_dis"], cd["G_shared"])
    assert "raw_singular_values" in gm and "numerical_rank" in gm
    gr = MS.basis_from_gram(MS.fit_shared_residual_ridge(Zw, ys, ds)["G_rule"])
    assert "raw_singular_values" in gr and gr["numerical_rank"] == gr["orthonormal_basis"].shape[0]


def test_result_routing_deterministic_and_failure_record():
    a = MS.route_stage_result(0.02, 0.05, 0.01, 0.0, "contrast_disagreement", "EEGNet")
    assert a["verdict"] == "MECHANISM_ENRICHED_OVER_RANDOM"
    b = MS.route_stage_result(-0.02, -0.005, 0.5, 0.01, "contrast_disagreement", "EEGNet")
    assert b["verdict"] == "NO_DETECTED_MECHANISM_ENRICHMENT"
    assert all(k in b for k in ("failure_layer", "learned_lesson", "next_hypothesis", "next_experiment"))
    # DGCNN-only positive -> backbone specific
    c = MS.route_stage_result(0.01, 0.03, 0.2, 0.0, "contrast_disagreement", "DGCNN")
    assert c["verdict"] == "BACKBONE_SPECIFIC"


def test_routing_gated_on_shared_null_specificity_control():
    # SAME stats that would clear route A, but only a fallback ran (shared-null-Haar control did not run):
    # ENRICHMENT must NOT be granted (fail-closed), so M2 cannot unlock on an ambient-only comparison.
    g = MS.route_stage_result(0.02, 0.05, 0.01, 0.0, "contrast_disagreement", "EEGNet", specificity_control="AMBIENT_ONLY")
    assert g["verdict"] == "ENRICHED_VS_AMBIENT_ONLY_MATCHED_CONTROL_UNAVAILABLE"
    assert g["next"] == "resolve_specificity_control_before_M2"
    # with the PRIMARY shared-null-Haar control the same stats DO route A (default control)
    m = MS.route_stage_result(0.02, 0.05, 0.01, 0.0, "contrast_disagreement", "EEGNet", specificity_control="SHARED_NULL_HAAR")
    assert m["verdict"] == "MECHANISM_ENRICHED_OVER_RANDOM"


def test_score_on_query_is_session_macro():
    Zw, ys, ds = _src(7, p=10)
    U = np.eye(10)[[2]]
    Xq, yq = Zw[:120], ys[:120]; sq = np.where(np.arange(120) < 60, "A", "B")
    g = MS.score_on_target_query(Zw, ys, U, Xq, yq, sq)
    assert np.isfinite(g)


# ============================================================ amendment 03 (shared-null conditional estimand)
def _null_setup(seed=0, C=4, p=12):
    Zw, ys, ds = _src(seed, C=C, p=p)
    cd = MS.build_contrast_disagreement(Zw, ys, ds)
    b = MS.build_shared_null_contrast_basis(cd)
    return Zw, ys, ds, cd, b


def test_shared_null_is_orthogonal_to_Cbar():
    _, _, _, cd, b = _null_setup()
    N = b["N"]
    assert np.allclose(cd["Cbar"] @ N, 0, atol=1e-8)                 # span(N) = null(Cbar)
    assert b["shared_null_dim"] == N.shape[1]
    # B_contrast lives in span(N): projecting onto N leaves it unchanged, and its shared_overlap ~ 0
    B = b["orthonormal_basis"]
    assert np.allclose(N @ (N.T @ B.T), B.T, atol=1e-8)
    assert abs(MS.shared_overlap(B, cd["G_shared"])) < 1e-9


def test_informed_and_random_live_in_same_null():
    _, _, _, cd, b = _null_setup()
    N, B = b["N"], b["orthonormal_basis"]; r = B.shape[0]
    Qs = MS.build_shared_null_haar_dictionaries(N, r, 4, MS.cell_seed("d", "bb", "s", 0, "contrast", "SHARED_NULL_HAAR", 0))
    for Q in Qs:
        assert np.allclose(N @ (N.T @ Q.T), Q.T, atol=1e-8)          # random control also in span(N)
        assert abs(MS.shared_overlap(Q, cd["G_shared"])) < 1e-9      # shared_overlap = 0 (matched to informed)


def test_null_control_non_degenerate():
    # q >> r -> the shared-null Haar dictionaries genuinely vary (subspace overlap with B below 1, with spread)
    _, _, _, cd, b = _null_setup()
    N, B = b["N"], b["orthonormal_basis"]; r = B.shape[0]
    assert N.shape[1] > r                                            # non-degenerate control space
    Qs = MS.build_shared_null_haar_dictionaries(N, r, 8, 12345)
    ov = [float(np.linalg.norm(B @ Q.T, "fro") ** 2 / r) for Q in Qs]
    assert max(ov) < 0.999 and (max(ov) - min(ov)) > 1e-3           # real variation, not a collapsed control


def test_joint_ridge_KKT_residual():
    Zw, ys, ds = _src(4, p=10)
    out = MS.fit_shared_residual_ridge(Zw, ys, ds, lam0=1.0, lamD=10.0)
    assert not out["fail_closed"]
    assert out["G_rule"].shape == (10, 10)
    assert out["kkt_ok"] and out["kkt_residual"] < 1e-6             # exact joint solution (stationarity ~ 0)


def test_gradient_invariant_to_trial_duplication():
    Zw, ys, ds = _src(5, p=10)
    g1 = MS.build_class_conditional_gradient_disagreement(Zw, ys, ds)["G_grad"]
    # duplicate ALL of subject 2's trials (trial-count change); equal-subject weighting must keep the basis stable
    m = ds == 2
    Zw2 = np.vstack([Zw, Zw[m]]); ys2 = np.concatenate([ys, ys[m]]); ds2 = np.concatenate([ds, ds[m]])
    g2 = MS.build_class_conditional_gradient_disagreement(Zw2, ys2, ds2)["G_grad"]
    # top eigenvector of the (subject-equal) gradient Gram should be near-invariant to the duplication
    v1 = np.linalg.eigh(g1)[1][:, -1]; v2 = np.linalg.eigh(g2)[1][:, -1]
    assert abs(abs(float(v1 @ v2)) - 1.0) < 0.05


def test_cell_specific_random_seeds():
    a = MS.cell_seed("BNCI2014_001", "EEGNet", "1", 0, "contrast", "SHARED_NULL_HAAR", 0)
    b = MS.cell_seed("BNCI2014_001", "EEGNet", "2", 0, "contrast", "SHARED_NULL_HAAR", 0)   # different subject
    c = MS.cell_seed("BNCI2014_001", "EEGNet", "1", 0, "contrast", "SHARED_NULL_HAAR", 0)   # identical -> same
    assert a != b and a == c
    # different cells -> different ambient dictionaries
    Qa = MS.build_ambient_random_dictionaries(12, 4, 1, a)[0]; Qb = MS.build_ambient_random_dictionaries(12, 4, 1, b)[0]
    assert not np.allclose(Qa, Qb)


def test_two_independent_control_blocks():
    _, _, _, cd, b = _null_setup()
    N, r = b["N"], b["orthonormal_basis"].shape[0]
    s0 = MS.cell_seed("d", "bb", "s", 0, "contrast", "SHARED_NULL_HAAR", 0)
    s1 = MS.cell_seed("d", "bb", "s", 0, "contrast", "SHARED_NULL_HAAR", 1)                # block 1
    Q0 = MS.build_shared_null_haar_dictionaries(N, r, 3, s0)
    Q1 = MS.build_shared_null_haar_dictionaries(N, r, 3, s1)
    assert not np.allclose(Q0[0], Q1[0])                             # independent blocks differ


def test_all_families_use_common_Gshared():
    Zw, ys, ds, cd, b = _null_setup()
    N = b["N"]
    rr = MS.fit_shared_residual_ridge(Zw, ys, ds); gd = MS.build_class_conditional_gradient_disagreement(Zw, ys, ds)
    br = MS.build_shared_null_gram_basis(rr["G_rule"], N); bg = MS.build_shared_null_gram_basis(gd["G_grad"], N)
    # rule & grad null-projected bases share the contrast N -> both orthogonal to the shared span
    for basis in (br["orthonormal_basis"], bg["orthonormal_basis"]):
        if basis.shape[0]:
            assert abs(MS.shared_overlap(basis, cd["G_shared"])) < 1e-8


def test_builder_contract_complete():
    _, _, _, _, b = _null_setup()
    for k in ("raw_matrix", "raw_singular_values", "numerical_rank", "orthonormal_basis", "generalized_eigenvalues"):
        assert k in b
    assert b["numerical_rank"] == b["orthonormal_basis"].shape[0]


def test_sign_flip_p_exact():
    # all-positive n=5 -> only the all-positive sign pattern has mean >= observed -> 1/32
    assert abs(MS.exact_sign_flip_p([0.03, 0.04, 0.02, 0.05, 0.01]) - 1.0 / 32) < 1e-9
    # symmetric-ish mixed -> larger p; and matches a brute-force enumeration
    v = [0.03, -0.04, 0.02, -0.01, 0.05, 0.02]
    import itertools
    obs = float(np.mean(v))
    brute = np.mean([np.mean([s * abs(x) for s, x in zip(sg, v)]) >= obs - 1e-12
                     for sg in itertools.product([-1, 1], repeat=len(v))])
    assert abs(MS.exact_sign_flip_p(v) - brute) < 1e-9


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
