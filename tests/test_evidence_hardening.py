"""CIGL R1 — tests for exact permutation p, Benjamini-Hochberg FDR, hierarchical bootstrap."""
import numpy as np
import pytest

from cmi.eval.evidence_hardening import (exact_permutation_pvalue, benjamini_hochberg,
                                         hierarchical_bootstrap, harden_leakage_table)


def test_exact_permutation_pvalue_bounds_and_formula():
    null = np.arange(100).astype(float)                 # 0..99
    # observed above all null -> p = (1+0)/(100+1)
    assert exact_permutation_pvalue(1000.0, null) == pytest.approx(1 / 101)
    # observed below all null (greater tail) -> every null >= observed -> p = (1+100)/101 = 1
    assert exact_permutation_pvalue(-5.0, null) == pytest.approx(1.0)
    # observed == 50 -> #{null>=50}=50 -> p=(1+50)/101
    assert exact_permutation_pvalue(50.0, null) == pytest.approx(51 / 101)
    assert 0.0 < exact_permutation_pvalue(50.0, null) <= 1.0    # never 0


def test_benjamini_hochberg_known_case():
    # p=[.001,.008,.04,.5], m=4, alpha=.05 -> thresholds .0125/.025/.0375/.05 -> reject ranks 1,2
    bh = benjamini_hochberg([0.001, 0.008, 0.04, 0.5], alpha=0.05)
    assert bh["n_rejected"] == 2
    assert bh["rejected"].tolist() == [True, True, False, False]
    assert bh["critical_p"] == pytest.approx(0.008)
    # adjusted p monotone, in [0,1], and >= raw p
    assert np.all(bh["adjusted_p"] >= np.array([0.001, 0.008, 0.04, 0.5]) - 1e-12)
    # all-null: nothing rejected
    assert benjamini_hochberg([0.6, 0.7, 0.9], alpha=0.05)["n_rejected"] == 0


def test_hierarchical_bootstrap_covers_mean_and_respects_nesting():
    rng = np.random.default_rng(0)
    recs = [{"dataset": ds, "fold": f, "seed": s, "value": 0.4 + 0.02 * f + rng.normal(0, 0.01)}
            for ds in ("A", "B") for f in range(5) for s in range(3)]
    out = hierarchical_bootstrap(recs, n_boot=1000, seed=1)
    assert out["lo"] < out["point"] < out["hi"]                 # CI brackets the point estimate
    assert out["point"] == pytest.approx(np.mean([r["value"] for r in recs]))
    # a cluster (hierarchical) CI is >= a naive flat bootstrap CI (accounts for within-cluster correlation)
    flat = hierarchical_bootstrap(recs, levels=(), n_boot=1000, seed=1)   # 0 levels = flat resample
    assert (out["hi"] - out["lo"]) >= (flat["hi"] - flat["lo"]) - 1e-6


def test_harden_leakage_table_end_to_end():
    # 6 rows: 4 with strong leakage (observed >> null), 2 null (observed within null)
    rng = np.random.default_rng(0)
    rows = []
    for i in range(6):
        null = rng.normal(0.1, 0.02, 200).tolist()
        obs = 1.2 if i < 4 else 0.1                              # 4 leaky, 2 null
        rows.append({"dataset": "2a", "fold": i, "seed": 0, "observed": obs, "null": null,
                     "value": (obs - 0.1)})
    h = harden_leakage_table(rows, alpha=0.05, n_boot=500)
    assert h["n_tests"] == 6
    assert h["bh"]["n_rejected"] == 4                            # exactly the 4 leaky rows clear FDR
    assert h["frac_cleared_fdr"] == pytest.approx(4 / 6)
    assert h["bootstrap_ci"]["point"] > 0                        # mean reduction positive
