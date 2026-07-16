"""CMI-Trace Stage 4 tests — task-direction consistency + task/subject subspace overlap + geometry.

Synthetic data with a KNOWN answer: shared vs independent per-subject task directions, one-consistent /
one-inconsistent class pair, orthogonal vs coincident task/subject subspaces, and rank-2 / ill-conditioned
geometry. Statistical units are SUBJECTS (cluster bootstrap resamples subjects, not windows).
"""
import numpy as np
import pytest

from cmi.eval import task_direction_consistency as tdc


# --------------------------------------------------------------- data generators
def _binary(n_subj=8, n_win=40, dim=12, shared=True, noise=0.1, offset=3.0, half=1.5, seed=0):
    """Binary within-subject contrast: each subject gets a distinct identity offset (which cancels in the
    contrast). If shared, every subject's pos-vs-neg direction is the SAME unit vector v; else it is an
    independent random unit vector per subject."""
    rng = np.random.default_rng(seed)
    v = rng.standard_normal(dim); v /= np.linalg.norm(v)
    Z, y, subj = [], [], []
    for s in range(n_subj):
        off = offset * rng.standard_normal(dim)
        if shared:
            vs = v
        else:
            vs = rng.standard_normal(dim); vs /= np.linalg.norm(vs)
        for lab, sign in ((1, +1.0), (0, -1.0)):
            base = off + sign * half * vs
            Z.append(base + noise * rng.standard_normal((n_win, dim)))
            y += [lab] * n_win; subj += [s] * n_win
    return np.vstack(Z), np.array(y), np.array(subj)


def _multiclass(n_subj=8, n_win=30, dim=12, noise=0.1, seed=0):
    """4 classes: pair (0,1) shares one task direction across subjects (consistent); pair (2,3) uses an
    independent per-subject direction (inconsistent)."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal(dim); a /= np.linalg.norm(a)
    Z, y, subj = [], [], []
    for s in range(n_subj):
        off = 3.0 * rng.standard_normal(dim)
        b = rng.standard_normal(dim); b /= np.linalg.norm(b)
        means = {0: off + 1.5 * a, 1: off - 1.5 * a, 2: off + 1.5 * b, 3: off - 1.5 * b}
        for c in (0, 1, 2, 3):
            Z.append(means[c] + noise * rng.standard_normal((n_win, dim)))
            y += [c] * n_win; subj += [s] * n_win
    return np.vstack(Z), np.array(y), np.array(subj)


def _overlap_data(coincide, n_subj=3, n_cls=3, n_win=40, dim=8, noise=0.05, A=3.0, B=3.0, seed=0):
    """Class offsets span dims {0,1}. Subject offsets span dims {2,3} (ORTHOGONAL) or dims {0,1} (COINCIDE)."""
    rng = np.random.default_rng(seed)
    class_off = np.zeros((n_cls, dim)); class_off[1, 0] = A; class_off[2, 1] = A
    subj_off = np.zeros((n_subj, dim))
    if coincide:
        subj_off[1, 0] = B; subj_off[2, 1] = B
    else:
        subj_off[1, 2] = B; subj_off[2, 3] = B
    Z, y, subj = [], [], []
    for s in range(n_subj):
        for c in range(n_cls):
            Z.append(class_off[c] + subj_off[s] + noise * rng.standard_normal((n_win, dim)))
            y += [c] * n_win; subj += [s] * n_win
    return np.vstack(Z), np.array(y), np.array(subj)


# --------------------------------------------------------------- 1. binary consistency
def test_binary_high_consistency():
    Z, y, subj = _binary(shared=True, seed=1)
    r = tdc.direction_consistency_binary(Z, y, subj, pos_label=1, neg_label=0, n_boot=2000, n_perm=100, seed=0)
    assert r["mean_pairwise_cosine"] > 0.9          # every subject shares the task direction
    assert r["ci_lo"] > 0.5                          # CI stays high
    assert r["perm_p"] <= 0.05                        # far above the within-subject label-permutation null
    assert r["perm_null_mean"] < 0.4
    assert r["contrast_snr"] > 1.0                    # class shift >> within-class scatter along the axis
    assert r["n_used"] == 8 and r["n_skipped"] == 0


def test_binary_no_shared_direction():
    Z, y, subj = _binary(n_subj=10, dim=20, shared=False, seed=2)
    r = tdc.direction_consistency_binary(Z, y, subj, pos_label=1, neg_label=0, n_boot=2000, n_perm=100, seed=0)
    assert abs(r["mean_pairwise_cosine"]) < 0.3       # independent random directions -> ~0
    assert r["perm_p"] > 0.05                          # NOT above the permutation null
    assert r["n_used"] == 10


def test_binary_skips_subject_missing_a_class():
    Z, y, subj = _binary(n_subj=6, shared=True, seed=3)
    # wipe class 0 for subject 5 -> it must be skipped and recorded
    y = y.copy()
    y[(subj == 5) & (y == 0)] = 1
    r = tdc.direction_consistency_binary(Z, y, subj, pos_label=1, neg_label=0, n_boot=1000, n_perm=30, seed=0)
    assert r["n_used"] == 5 and r["n_skipped"] == 1
    assert any(s == 5 for s, _ in r["subjects_skipped"])


# --------------------------------------------------------------- 2. four-class (no arbitrary collapse)
def test_multiclass_per_pair_preserved_and_macro_between():
    Z, y, subj = _multiclass(seed=4)
    r = tdc.direction_consistency_multiclass(Z, y, subj, classes=[0, 1, 2, 3],
                                             n_boot=1500, n_perm=60, seed=0)
    per = r["per_pair"]
    # every unordered pair preserved
    assert set(per.keys()) == {(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)}
    c_consistent = per[(0, 1)]["mean_pairwise_cosine"]
    c_inconsistent = per[(2, 3)]["mean_pairwise_cosine"]
    assert c_consistent > 0.8                          # shared direction
    assert c_inconsistent < 0.4                         # per-subject random direction
    assert c_consistent > c_inconsistent + 0.3          # pair values genuinely differ
    # macro-average lies between the smallest and largest per-pair value
    vals = [v["mean_pairwise_cosine"] for v in per.values()]
    assert min(vals) - 1e-9 <= r["macro_avg_consistency"] <= max(vals) + 1e-9
    # OVR sensitivity kept separate and labeled, never folded into the pairwise macro-average
    assert set(r["ovr_sensitivity"].keys()) == {0, 1, 2, 3}
    assert r["n_valid_pairs"] == 6


# --------------------------------------------------------------- 3. task/subject subspace overlap
def test_overlap_orthogonal_near_null():
    Z, y, subj = _overlap_data(coincide=False, seed=5)
    r = tdc.task_subject_overlap(Z, y, subj, n_random=60, seed=0)
    assert r["rank_Y"] == 2 and r["rank_D"] == 2
    assert r["normalized_overlap"] < 0.1                       # orthogonal task/subject subspaces
    assert r["normalized_overlap"] <= r["null_mean"] + 1e-6     # not above the random-subspace null
    assert np.all(r["cos_principal_angles"] < 0.2)             # all principal angles ~ 90 deg
    assert len(r["cos_principal_angles"]) == 2


def test_overlap_coincident_far_above_null():
    Z, y, subj = _overlap_data(coincide=True, seed=6)
    r = tdc.task_subject_overlap(Z, y, subj, n_random=60, seed=0)
    assert r["normalized_overlap"] > 0.9                        # subspaces coincide
    assert r["normalized_overlap"] > r["null_ci_hi"]            # far above the matched-rank random null
    assert np.all(r["cos_principal_angles"] > 0.9)             # all principal angles ~ 0 deg


def test_overlap_documents_centering_and_whitening():
    Z, y, subj = _overlap_data(coincide=False, seed=7)
    r = tdc.task_subject_overlap(Z, y, subj, n_random=20, seed=0)
    assert "grand-mean" in r["centering"].lower()
    assert "zca" in r["whitening"].lower() and "cov" in r["whitening"].lower()


# --------------------------------------------------------------- 4. per-representation geometry
def test_geometry_effective_rank_and_condition_number():
    rng = np.random.default_rng(8)
    basis = rng.standard_normal((2, 8))
    coeff = rng.standard_normal((400, 2))
    z = coeff @ basis + 1e-3 * rng.standard_normal((400, 8))    # rank-2 signal + tiny noise
    g = tdc.representation_geometry(z, deleted_rank=3, latent_dim=8)
    assert 1.5 < g["effective_rank"] < 2.6                      # effectively rank 2
    assert g["cov_condition_number"] > 1e3                       # ill-conditioned (2 strong dims, rest tiny)
    assert g["top_singular_value"] > 0 and g["feature_norm"] > 0
    assert g["deleted_rank_ratio"] == pytest.approx(3 / 8)


def test_geometry_well_conditioned_and_ratio_none():
    rng = np.random.default_rng(9)
    z = rng.standard_normal((2000, 6))                           # isotropic -> well conditioned
    g = tdc.representation_geometry(z)                            # no deleted_rank/latent_dim
    assert g["cov_condition_number"] < 10.0
    assert g["deleted_rank_ratio"] is None and g["deleted_rank"] is None
    assert 5.0 < g["effective_rank"] <= 6.0


# --------------------------------------------------------------- cluster bootstrap resamples SUBJECTS
def test_cluster_bootstrap_unit_is_subject_not_window():
    n_subj, n_win = 7, 50
    Z, y, subj = _binary(n_subj=n_subj, n_win=n_win, shared=True, seed=10)
    r = tdc.direction_consistency_binary(Z, y, subj, pos_label=1, neg_label=0, n_boot=1000, n_perm=20, seed=0)
    assert r["n_clusters"] == n_subj                             # one cluster per SUBJECT
    assert r["n_clusters"] != len(y)                             # NOT the number of windows
    assert r["n_clusters"] != (subj == 0).sum() * n_subj


def test_cluster_bootstrap_ci_helper_matches_n_subjects():
    m, lo, hi, n = tdc.cluster_bootstrap_ci([0.1, 0.2, 0.3, 0.4, 0.5], n_boot=2000, seed=0)
    assert n == 5 and lo <= m <= hi
    assert tdc.cluster_bootstrap_ci([0.7])[0] == pytest.approx(0.7)
