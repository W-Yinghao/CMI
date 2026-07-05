"""R3 descriptive — spatial correlation (Spearman/Pearson) + cluster bootstrap CI; NaN/constant-map safety."""
import numpy as np
from cmi.eval.spatial_correlation import (
    spatial_correlation, bootstrap_correlation_ci, spatial_correlation_report,
)


def test_perfect_and_anti_correlation():
    a = np.arange(10.0)
    assert spatial_correlation(a, 2 * a + 1, "pearson") > 0.999
    assert spatial_correlation(a, 2 * a + 1, "spearman") > 0.999
    assert spatial_correlation(a, -a, "spearman") < -0.999


def test_constant_map_returns_nan_not_crash():
    a = np.arange(8.0); const = np.ones(8)
    assert np.isnan(spatial_correlation(a, const, "spearman"))
    assert np.isnan(spatial_correlation(a, const, "pearson"))


def test_nan_entries_are_dropped():
    a = np.array([1.0, 2, 3, 4, np.nan]); b = np.array([1.0, 2, 3, 4, 100])
    assert spatial_correlation(a, b, "spearman") > 0.99


def test_bootstrap_ci_structure_and_brackets_point():
    rng = np.random.default_rng(0)
    pairs = []
    for _ in range(6):                                          # 6 (fold,seed) groups, correlated maps
        x = rng.standard_normal(12); pairs.append((x, x + 0.1 * rng.standard_normal(12)))
    out = bootstrap_correlation_ci(pairs, method="spearman", n_boot=500, seed=0)
    assert out["n_groups"] == 6 and out["ci"][0] <= out["point"] <= out["ci"][1]
    assert out["point"] > 0.5


def test_single_group_ci_is_nan():
    out = bootstrap_correlation_ci([(np.arange(5.0), np.arange(5.0))], seed=0)
    assert out["n_groups"] == 1 and np.isnan(out["ci"][0])


def test_report_has_both_methods():
    pairs = [(np.arange(6.0), np.arange(6.0)), (np.arange(6.0), np.arange(6.0)[::-1])]
    rep = spatial_correlation_report(pairs, seed=0)
    assert set(rep) == {"spearman", "pearson"}
