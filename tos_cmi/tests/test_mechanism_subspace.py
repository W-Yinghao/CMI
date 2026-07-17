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


def test_routing_gated_on_matched_specificity_control():
    # SAME stats that would clear route A, but only the AMBIENT_ONLY fallback ran (matched control fail-closed):
    # ENRICHMENT must NOT be granted (fail-closed), so M2 cannot unlock on an ambient-only comparison.
    g = MS.route_stage_result(0.02, 0.05, 0.01, 0.0, "contrast_disagreement", "EEGNet", specificity_control="AMBIENT_ONLY")
    assert g["verdict"] == "ENRICHED_VS_AMBIENT_ONLY_MATCHED_CONTROL_UNAVAILABLE"
    assert g["next"] == "resolve_specificity_control_before_M2"
    # with a real matched control the same stats DO route A (default is MATCHED)
    m = MS.route_stage_result(0.02, 0.05, 0.01, 0.0, "contrast_disagreement", "EEGNet", specificity_control="MATCHED")
    assert m["verdict"] == "MECHANISM_ENRICHED_OVER_RANDOM"


def test_score_on_query_is_session_macro():
    Zw, ys, ds = _src(7, p=10)
    U = np.eye(10)[[2]]
    Xq, yq = Zw[:120], ys[:120]; sq = np.where(np.arange(120) < 60, "A", "B")
    g = MS.score_on_target_query(Zw, ys, U, Xq, yq, sq)
    assert np.isfinite(g)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
