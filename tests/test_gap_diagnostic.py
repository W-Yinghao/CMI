"""CIGL_66 gap-diagnostic engineering tests (synthetic fixtures = engineering only, no scientific claim)."""
import numpy as np
import pytest

from cmi.eval.gap_diagnostic import (
    subject_offset_matrix, spectrum_diagnostics, effective_rank, topk_energy_fraction,
    subject_subspace, task_head_alignment, alignment_curve, DEFAULT_KS,
)


def _syn(n_dom=4, n_cls=2, n_per=30, Z=8, seed=0):
    rng = np.random.default_rng(seed)
    z, y, d = [], [], []
    for dd in range(n_dom):
        for cc in range(n_cls):
            b = rng.standard_normal((n_per, Z)) * 0.2
            b[:, 0] += 3.0 * cc                                 # class signal on dim0
            b[:, 1] += (dd - 1.5) * 2.0                         # subject signal on dim1
            z.append(b); y += [cc] * n_per; d += [dd] * n_per
    return np.vstack(z), np.array(y), np.array(d)


def test_1_effective_rank_finite_under_rank_deficient():
    assert effective_rank([5.0, 0, 0, 0]) == pytest.approx(1.0)   # one direction -> erank 1
    assert 0.0 <= effective_rank([1.0, 1, 1, 1]) <= 4.0
    assert effective_rank([0.0, 0, 0]) == 0.0                     # degenerate -> 0, no crash
    assert np.isfinite(effective_rank(np.zeros(5)))


def test_2_topk_energy_monotonic_and_bounded():
    s = [4.0, 2, 1, 0.5]
    fracs = [topk_energy_fraction(s, k) for k in (0, 1, 2, 4, 8)]
    assert fracs[0] == 0.0 and fracs[-1] == pytest.approx(1.0)
    assert all(0.0 <= f <= 1.0 for f in fracs)
    assert all(fracs[i] <= fracs[i + 1] + 1e-12 for i in range(len(fracs) - 1))   # monotonic
    assert topk_energy_fraction(np.zeros(4), 2) == 0.0


def test_3_alignment_bounded():
    z, y, d = _syn()
    M = subject_offset_matrix(z, y, d)
    W = np.random.default_rng(1).standard_normal((2, 8))
    for k in DEFAULT_KS:
        a = task_head_alignment(W, subject_subspace(M, k))
        assert 0.0 <= a <= 1.0 + 1e-9
    assert task_head_alignment(np.zeros((2, 8)), subject_subspace(M, 2)) == 0.0   # zero head fails to 0


def test_4_alignment_invariant_to_basis_rotation():
    z, y, d = _syn()
    M = subject_offset_matrix(z, y, d)
    S = subject_subspace(M, 4)                                   # [4, 8]
    W = np.random.default_rng(2).standard_normal((2, 8))
    # rotate the subspace basis by a random orthogonal R (same subspace, different basis)
    A = np.random.default_rng(3).standard_normal((4, 4))
    R, _ = np.linalg.qr(A)
    S_rot = R @ S
    assert task_head_alignment(W, S) == pytest.approx(task_head_alignment(W, S_rot), abs=1e-10)


def test_5_target_labels_do_not_affect_source_subspace():
    z, y, d = _syn()
    src = d != 3
    M0 = subject_offset_matrix(z[src], y[src], d[src])
    y2 = y.copy(); y2[d == 3] = np.random.default_rng(4).integers(0, 2, (d == 3).sum())   # corrupt TARGET labels
    M1 = subject_offset_matrix(z[src], y2[src], d[src])         # fit still source-only
    assert np.allclose(M0, M1)


def test_6_random_subspace_baseline_deterministic():
    rng_dirs = lambda s: np.linalg.svd(np.random.default_rng(s).standard_normal((8, 8)), full_matrices=False)[2][:2]
    assert np.allclose(rng_dirs(7), rng_dirs(7)) and not np.allclose(rng_dirs(7), rng_dirs(8))


def test_7_graph_and_node_handled_separately():
    z, y, d = _syn(Z=8)                                         # "graph_z" [N,8]
    nz = np.random.default_rng(5).standard_normal((len(z), 6, 4))   # "node_z" [N,6,4]
    Mg = subject_offset_matrix(z, y, d)
    Mn = subject_offset_matrix(nz.reshape(len(nz), -1), y, d)   # node flattened -> different Zdim
    assert Mg.shape[1] == 8 and Mn.shape[1] == 24
    assert spectrum_diagnostics(Mg)["total_subject_energy"] >= 0
    assert spectrum_diagnostics(Mn)["total_subject_energy"] >= 0


def test_8_alignment_curve_monotonic_and_spectrum_schema():
    z, y, d = _syn()
    M = subject_offset_matrix(z, y, d)
    W = np.random.default_rng(6).standard_normal((2, 8))
    ac = alignment_curve(M, W)
    ks = sorted(ac)
    assert all(ac[ks[i]] <= ac[ks[i + 1]] + 1e-9 for i in range(len(ks) - 1))   # bigger subspace -> >= alignment
    sp = spectrum_diagnostics(M)
    for key in ("singular_values", "total_subject_energy", "effective_rank", "top1_energy_fraction", "top8_energy_fraction"):
        assert key in sp
