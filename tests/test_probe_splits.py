"""CIGL Phase 2-real tests: support-aware (Y,D) trial-level probe split (CPU, synthetic)."""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.eval.probe_splits import stratified_trial_split_by_y_d  # noqa: E402


def _balanced(n_per_cell=20, n_cls=2, n_dom=4, seed=0):
    """Every (Y,D) cell well populated -> both splits should contain every domain."""
    rng = np.random.default_rng(seed)
    ys, ds = [], []
    for c in range(n_cls):
        for g in range(n_dom):
            ys += [c] * n_per_cell
            ds += [g] * n_per_cell
    y = np.array(ys); d = np.array(ds)
    perm = rng.permutation(len(y))
    return y[perm], d[perm]


def test_split_is_trial_level_disjoint_and_covering():
    y, d = _balanced()
    tr, va, diag = stratified_trial_split_by_y_d(y, d, train_frac=0.7, seed=0)
    assert len(np.intersect1d(tr, va)) == 0, "train/val overlap"
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(y))), "lost/duplicated trials"
    assert diag["n_train"] + diag["n_val"] == len(y) == diag["n_trials"]


def test_adequate_cells_put_every_domain_in_both_splits():
    y, d = _balanced(n_per_cell=20)
    tr, va, diag = stratified_trial_split_by_y_d(y, d, train_frac=0.7, seed=1)
    assert diag["missing_train_domains"] == [], "every domain must appear in train"
    assert diag["missing_val_domains"] == [], "every domain must appear in val (support-aware)"
    assert min(diag["val_domain_support"]) > 0 and min(diag["train_domain_support"]) > 0
    assert diag["n_cells_low_support"] == 0


def test_low_support_cells_go_to_train_and_are_reported():
    # domain 3 has only 1 trial for class 0 -> low-support cell -> train, flagged
    y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 0])
    d = np.array([0, 0, 1, 1, 0, 0, 1, 1, 3])
    tr, va, diag = stratified_trial_split_by_y_d(y, d, train_frac=0.7, seed=0, min_per_cell=2)
    assert diag["n_cells_low_support"] >= 1
    assert any(cell["d"] == 3 for cell in diag["low_support_cells"])
    assert 8 in tr and 8 not in va, "the single domain-3 trial must be kept in train"
    assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(y)))


def test_no_sample_lost_or_duplicated_across_seeds():
    y, d = _balanced(n_per_cell=7, seed=3)
    for s in range(5):
        tr, va, _ = stratified_trial_split_by_y_d(y, d, train_frac=0.6, seed=s)
        assert np.array_equal(np.sort(np.concatenate([tr, va])), np.arange(len(y)))
        assert len(np.intersect1d(tr, va)) == 0


def test_diagnostics_keys_present():
    y, d = _balanced()
    _, _, diag = stratified_trial_split_by_y_d(y, d)
    for k in ("n_trials", "n_train", "n_val", "n_classes", "n_domains", "n_cells_total",
              "n_cells_split", "n_cells_low_support", "train_domain_support", "val_domain_support",
              "missing_val_domains", "missing_train_domains"):
        assert k in diag, f"diagnostics missing '{k}'"


def test_support_aware_at_cell_level():
    """Every adequate (Y,D) cell is represented in BOTH splits (stronger than the aggregate check)."""
    y, d = _balanced(n_per_cell=10, n_cls=2, n_dom=3, seed=7)
    tr, va, diag = stratified_trial_split_by_y_d(y, d, train_frac=0.7, seed=0)
    for c in range(diag["n_classes"]):
        for g in range(diag["n_domains"]):
            in_tr = int(np.sum((y[tr] == c) & (d[tr] == g)))
            in_va = int(np.sum((y[va] == c) & (d[va] == g)))
            assert in_tr > 0 and in_va > 0, f"cell (y={c},d={g}) not in both splits ({in_tr},{in_va})"


def test_negative_labels_rejected():
    """Negative sentinel labels (e.g. -1) must fail loud, not be silently dropped from (Y,D) cells."""
    import pytest
    y = np.array([0, 1, 0, 1]); d = np.array([0, -1, 1, 0])
    with pytest.raises(ValueError):
        stratified_trial_split_by_y_d(y, d)
