"""CMI-Trace P0.2 tests — objective->effect audit columns + fold/subject-cluster inference."""
import numpy as np
import pytest

from cmi.eval import objective_effect_report as oer


# --------------------------------------------------------------- moment gaps
def test_marginal_moment_gap_zero_matched_positive_gap():
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, 5))
    d0 = np.r_[np.zeros(100), np.ones(100)].astype(int)
    zz = np.vstack([z[:100], z[:100]])                       # identical domains
    assert oer.marginal_moment_gap(zz, d0) == pytest.approx(0.0, abs=1e-9)
    zz2 = np.vstack([z[:100], z[:100] + 4.0])                # shifted
    assert oer.marginal_moment_gap(zz2, d0) > 1e-2


def test_class_conditional_moment_gap_ignores_label_prior():
    # constant-per-class features, different class proportions across domains -> C-CORAL gap == 0
    v0 = np.array([1.0, 0, 0, 0]); v1 = np.array([0, 1.0, 0, 0])
    z = np.vstack([np.tile(v0, (40, 1)), np.tile(v1, (10, 1)),
                   np.tile(v0, (10, 1)), np.tile(v1, (40, 1))])
    y = np.array([0] * 40 + [1] * 10 + [0] * 10 + [1] * 40)
    d = np.array([0] * 50 + [1] * 50)
    gap, support = oer.class_conditional_moment_gap(z, y, d)
    assert gap == pytest.approx(0.0, abs=1e-9)
    assert oer.marginal_moment_gap(z, d) > 1e-2               # marginal still fires


def test_class_conditional_moment_gap_detects_and_logs():
    rng = np.random.default_rng(1)
    n = 100
    z = np.vstack([rng.standard_normal((n, 4)), rng.standard_normal((n, 4)) + np.array([3.0, 0, 0, 0])])
    y = np.zeros(2 * n, int)
    d = np.r_[np.zeros(n), np.ones(n)].astype(int)
    gap, support = oer.class_conditional_moment_gap(z, y, d)
    assert gap > 0.5
    # add an under-supported cell -> recorded in skipped_cells
    z2 = np.vstack([z, rng.standard_normal((2, 4))])
    y2 = np.r_[y, [1, 1]]
    d2 = np.r_[d, [0, 1]]
    _, support2 = oer.class_conditional_moment_gap(z2, y2, d2, min_n=4)
    assert support2["n_qualifying_classes"] >= 1


# --------------------------------------------------------------- risk / IRM / geometry
def test_per_domain_risk_variance_and_irm_finite():
    rng = np.random.default_rng(2)
    logits = rng.standard_normal((150, 3))
    y = rng.integers(0, 3, 150); d = rng.integers(0, 3, 150)
    assert oer.per_domain_risk_variance(logits, y, d) >= 0.0
    v = oer.irmv1_diagnostic(logits, y, d)
    assert np.isfinite(v) and v >= 0.0


def test_per_domain_risk_variance_zero_when_equal():
    logits = np.tile(np.array([2.0, 0.0, 0.0]), (60, 1))
    y = np.zeros(60, int)
    d = np.r_[np.zeros(30), np.ones(30)].astype(int)
    assert oer.per_domain_risk_variance(logits, y, d) == pytest.approx(0.0, abs=1e-9)


def test_feature_geometry_effective_rank():
    rng = np.random.default_rng(3)
    # rank-2 signal embedded in 8 dims -> effective rank close to 2
    basis = rng.standard_normal((2, 8))
    coeff = rng.standard_normal((400, 2))
    z = coeff @ basis
    g = oer.feature_geometry(z)
    assert 1.5 < g["effective_rank"] < 2.6
    assert g["top_singular_value"] > 0 and g["feature_norm"] > 0


# --------------------------------------------------------------- cluster inference
def test_cluster_bootstrap_ci_basic():
    m, lo, hi, n = oer.cluster_bootstrap_ci([5.0, 5.0, 5.0, 5.0])
    assert m == pytest.approx(5.0) and lo == pytest.approx(5.0) and hi == pytest.approx(5.0) and n == 4
    m2, lo2, hi2, n2 = oer.cluster_bootstrap_ci([0.0, 1.0, 2.0, 3.0, 4.0])
    assert lo2 < m2 < hi2 and n2 == 5
    # single cluster is degenerate, not an error
    assert oer.cluster_bootstrap_ci([7.0])[0] == pytest.approx(7.0)


def _rows(method, base, delta):
    rows = []
    for ds in ("A", "B"):
        for fold in range(3):
            for seed in range(2):
                rows.append({"dataset": ds, "fold": fold, "seed": seed, "method": method,
                             "m": base + delta + 0.01 * fold + 0.001 * seed})
    return rows


def test_paired_delta_vs_baseline_pairs_by_fold():
    rows = _rows("erm", 0.5, 0.0) + _rows("coral", 0.5, 0.1)
    out = oer.paired_delta_vs_baseline(rows, "m", baseline_method="erm")
    assert "coral" in out
    assert out["coral"]["delta_mean"] == pytest.approx(0.1, abs=1e-6)
    # deltas are constant (0.1) across clusters here -> CI collapses to 0.1 (up to fp)
    assert out["coral"]["cluster_ci_lo"] == pytest.approx(0.1, abs=1e-6)
    assert out["coral"]["cluster_ci_hi"] == pytest.approx(0.1, abs=1e-6)
    assert out["coral"]["n_clusters"] == 6


def test_summarize_metric_reports_clusters():
    rows = _rows("erm", 0.5, 0.0)
    s = oer.summarize_metric(rows, "m")
    assert s["n_clusters"] == 6 and np.isfinite(s["raw_mean"])
    assert s["cluster_ci_lo"] <= s["raw_mean"] <= s["cluster_ci_hi"]


# --------------------------------------------------------------- end-to-end on a synthetic audit npz
def test_objective_effect_row_end_to_end(tmp_path):
    from cmi.eval.audit_npz import save_audit_npz
    rng = np.random.default_rng(7)
    Ns, Nt, Zg, C, Zn, ncls = 120, 30, 16, 8, 5, 3
    N = Ns + Nt
    graph_z = rng.standard_normal((N, Zg)).astype("float32")
    node_z = rng.standard_normal((N, C, Zn)).astype("float32")
    y = rng.integers(0, ncls, N).astype("int64")
    W = rng.standard_normal((ncls, Zg)).astype("float32"); b = rng.standard_normal(ncls).astype("float32")
    model_logits = (graph_z @ W.T + b).astype("float32")     # linear head -> replay must verify
    # 3 source domains (0,1,2) + target domain id 3 (eval-only, appended)
    d = np.r_[rng.integers(0, 3, Ns), np.full(Nt, 3)].astype("int64")
    src_idx = np.arange(Ns); tgt_idx = np.arange(Ns, N)
    p = tmp_path / "fold0.audit.npz"
    save_audit_npz(str(p), graph_z=graph_z, node_z=node_z, y=y, d=d, model_logits=model_logits,
                   fold=0, seed=0, target_subject="S3", method="coral", dataset="BNCI2014_001",
                   task_head_weight=W, task_head_bias=b, task_head_input="graph_z",
                   source_indices=src_idx, target_indices=tgt_idx)
    leakage = {"graph_kl": 0.05, "graph_null": 0.02, "graph_perm_p": 0.01,
               "node_kl": 0.03, "node_null": 0.015, "node_perm_p": 0.04}
    row = oer.objective_effect_row(str(p), leakage=leakage, primary_k=2, seed=0)
    for col in ("marginal_moment_gap", "class_conditional_moment_gap", "per_domain_risk_variance",
                "irmv1_diagnostic", "feature_norm", "top_singular_value", "effective_rank",
                "R_rel_k2", "R_rel_k2_random_control", "graph_kl", "node_kl"):
        assert col in row and np.isfinite(row[col]), f"missing/nonfinite column {col}"
    assert row["head_replay_available"] is True               # linear head -> head-replay path
    assert row["reliance_firewall_passed"] is True            # source/target indices verify the firewall
    assert row["method"] == "coral" and row["dataset"] == "BNCI2014_001"
