"""CMI-Trace P1.1 tests — flat-feature conditional-subject-leakage ruler + three-way cross-fitting."""
import numpy as np
import pytest

from cmi.eval import conditional_subject_leakage as csl


def _make(n_per_cell=60, n_cls=2, n_dom=3, dim=6, subject_strength=3.0, seed=0):
    """Trials with class signal + (optional) within-label subject offset. Returns Z, y, d and the subject
    offset directions so a test can 'erase' them."""
    rng = np.random.default_rng(seed)
    class_means = rng.standard_normal((n_cls, dim)) * 2.0
    subj_dirs = rng.standard_normal((n_dom, dim))
    subj_dirs /= np.linalg.norm(subj_dirs, axis=1, keepdims=True)
    Z, y, d = [], [], []
    for c in range(n_cls):
        for g in range(n_dom):
            z = class_means[c] + subject_strength * subj_dirs[g] + 0.5 * rng.standard_normal((n_per_cell, dim))
            Z.append(z); y += [c] * n_per_cell; d += [g] * n_per_cell
    return np.vstack(Z), np.array(y), np.array(d), subj_dirs


def test_three_way_split_disjoint_and_support():
    Z, y, d, _ = _make()
    er, pt, pe, diag = csl.three_way_support_split(y, d, seed=0)
    assert diag["disjoint"] is True
    assert len(np.intersect1d(er, pt)) == 0 and len(np.intersect1d(pt, pe)) == 0
    # every subject appears in the posterior pool (support preserved)
    pool = np.r_[pt, pe]
    assert set(np.unique(d[pool]).tolist()) == set(np.unique(d).tolist())


def test_conditional_entropy_known():
    # 2 classes, uniform over 2 domains within each class -> H(D|Y) = ln 2
    y = np.array([0, 0, 1, 1] * 50)
    d = np.array([0, 1, 0, 1] * 50)
    assert csl.conditional_entropy_d_given_y(y, d, 2) == pytest.approx(np.log(2), abs=0.02)


def test_ruler_detects_leakage_and_null_is_clean():
    Z, y, d, _ = _make(subject_strength=3.0, seed=1)
    er, pt, pe, _ = csl.three_way_support_split(y, d, seed=1)
    leaky = csl.flat_conditional_cmi(Z, y, d, 2, 3, pt, pe, n_perm=20, seed=1, epochs=60, with_residual=True)
    assert leaky["posterior_kl_nats"] > leaky["null_mean"]
    assert leaky["excess_over_null"] > 0 and leaky["perm_p"] <= 0.1
    assert leaky["subject_residual_linear"] > 0.6          # subjects decodable within label
    assert 0.0 <= leaky["normalized_leakage"]

    # no-subject features -> leakage collapses toward the null, residual toward chance
    Zc, yc, dc, _ = _make(subject_strength=0.0, seed=2)
    erc, ptc, pec, _ = csl.three_way_support_split(yc, dc, seed=2)
    clean = csl.flat_conditional_cmi(Zc, yc, dc, 2, 3, ptc, pec, n_perm=20, seed=2, epochs=60)
    assert clean["excess_over_null"] < leaky["excess_over_null"]
    assert clean["subject_residual_linear"] < leaky["subject_residual_linear"]


def test_erasure_reduces_measured_leakage():
    Z, y, d, subj_dirs = _make(subject_strength=3.0, seed=3)
    er, pt, pe, _ = csl.three_way_support_split(y, d, seed=3)
    # 'erase' the subject subspace: fit the subject-offset directions on the ERASER split only, project out
    fitZ, fity, fitd = Z[er], y[er], d[er]
    rows = []
    for c in np.unique(fity):
        mu = fitZ[fity == c].mean(0)
        for g in np.unique(fitd[fity == c]):
            m = (fity == c) & (fitd == g)
            rows.append(fitZ[m].mean(0) - mu)
    M = np.vstack(rows)
    Vt = np.linalg.svd(M, full_matrices=False)[2]
    S = Vt[:min(3, Vt.shape[0])]
    P = np.eye(Z.shape[1]) - S.T @ S           # remove subject subspace (fit on eraser split only)
    transformed = {"full": Z, "leace_like": Z @ P.T}
    table = csl.cmi_ruler_across_transforms(transformed, y, d, 2, 3, pt, pe, n_perm=15, seed=3, epochs=50)
    assert table["leace_like"]["posterior_kl_nats"] < table["full"]["posterior_kl_nats"]
    assert table["leace_like"]["subject_residual_linear"] < table["full"]["subject_residual_linear"]


def test_ruler_uses_only_posterior_split_not_eraser():
    # structural firewall: flat_conditional_cmi only indexes ptrain/peval; assert an eraser-only row cannot
    # change the result (pass features with garbage on eraser rows; ruler must be unaffected).
    Z, y, d, _ = _make(subject_strength=2.0, seed=4)
    er, pt, pe, _ = csl.three_way_support_split(y, d, seed=4)
    Z2 = Z.copy(); Z2[er] = 1e6 * np.random.default_rng(0).standard_normal(Z2[er].shape)  # corrupt eraser rows
    r1 = csl.flat_conditional_cmi(Z, y, d, 2, 3, pt, pe, n_perm=10, seed=4, epochs=40, with_residual=False)
    r2 = csl.flat_conditional_cmi(Z2, y, d, 2, 3, pt, pe, n_perm=10, seed=4, epochs=40, with_residual=False)
    assert r1["posterior_kl_nats"] == pytest.approx(r2["posterior_kl_nats"], abs=1e-9)
