"""CIGL Phase 2-real tests: node/edge leakage-map seed-stability (CPU, synthetic)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.eval.graph_map_stability import (  # noqa: E402
    flatten_node_map, flatten_edge_map, spearman_or_pearson_stability, random_map_stability_null)


def test_node_flatten_length_C():
    C = 22
    v = flatten_node_map(np.arange(C, dtype=float))
    assert v.shape == (C,)


def test_edge_flatten_strict_upper_triangle_ignores_diagonal():
    C = 5
    m = np.arange(C * C, dtype=float).reshape(C, C)
    m = m + m.T                                   # symmetric
    np.fill_diagonal(m, 999.0)                    # diagonal must be ignored
    v = flatten_edge_map(m)
    assert v.shape == (C * (C - 1) // 2,)
    assert 999.0 not in v.tolist(), "diagonal must not appear in the flattened edge vector"


def test_identical_maps_high_stability():
    rng = np.random.default_rng(0)
    base = rng.standard_normal(30)
    vecs = [base.copy() for _ in range(4)]
    s = spearman_or_pearson_stability(vecs, method="spearman")
    assert s["mean_corr"] > 0.99, "identical maps must be near-perfectly stable"


def test_consistent_maps_above_random_null():
    rng = np.random.default_rng(1)
    base = rng.standard_normal(40)
    # consistent across seeds (same signal + small per-seed noise)
    vecs = [base + 0.05 * rng.standard_normal(40) for _ in range(3)]
    obs = spearman_or_pearson_stability(vecs)["mean_corr"]
    null = random_map_stability_null(vecs, n_perm=200, seed=0)
    assert np.isfinite(null["null_mean"])
    assert obs > null["null_q95"], "consistent maps must exceed the random-map null"
    assert null["above_random"] is True


def test_random_null_finite_and_low_for_noise():
    rng = np.random.default_rng(2)
    vecs = [rng.standard_normal(40) for _ in range(3)]   # independent noise -> not stable
    obs = spearman_or_pearson_stability(vecs)["mean_corr"]
    null = random_map_stability_null(vecs, n_perm=200, seed=0)
    assert np.isfinite(null["null_mean"]) and np.isfinite(null["null_std"])
    assert isinstance(null["above_random"], bool) and null["degenerate"] is False
    # independent-noise maps are not consistently correlated -> observed near null, not above it
    assert abs(obs) < 0.4, f"independent noise should not be strongly correlated (got {obs:.3f})"
    assert null["above_random"] is False, "independent noise must not be flagged above the random null"
    assert abs(null["null_mean"]) < 0.5


def test_single_vector_is_graceful():
    s = spearman_or_pearson_stability([np.arange(10.0)])
    assert s["n_pairs"] == 0
    null = random_map_stability_null([np.arange(10.0)])
    assert null["n_perm"] == 0
    assert null["degenerate"] is True


def test_constant_maps_flagged_degenerate():
    # all-constant maps -> every correlation 0, null collapses -> uninformative, must be flagged
    vecs = [np.zeros(20), np.zeros(20), np.zeros(20)]
    null = random_map_stability_null(vecs, n_perm=50, seed=0)
    assert null["degenerate"] is True
    assert null["above_random"] is False, "a collapsed null must not report above_random"
