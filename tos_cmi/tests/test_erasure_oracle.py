"""Tests for the exact-head nullspace oracle — the ALGEBRAIC softmax-invariance guarantee + subject removal."""
import numpy as np
import pytest

from tos_cmi.eeg import erasure_oracle as EO


def _softmax(logits):
    m = logits - logits.max(1, keepdims=True); e = np.exp(m); return e / e.sum(1, keepdims=True)


def test_centered_head_rows_sum_zero():
    W = np.random.default_rng(0).standard_normal((4, 16))
    Wc = EO.centered_head(W)
    assert np.allclose(Wc.mean(0), 0.0)                    # class-centered


def test_nullspace_annihilates_centered_head():
    W = np.random.default_rng(1).standard_normal((4, 20))
    N = EO.head_nullspace_basis(W)
    Wc = EO.centered_head(W)
    assert np.allclose(Wc @ N, 0.0, atol=1e-8)             # W_c N = 0
    assert N.shape[1] >= 20 - (4 - 1)                       # dim ker(W_c) >= d - (C-1)
    assert np.allclose(N.T @ N, np.eye(N.shape[1]), atol=1e-8)  # orthonormal


def test_removing_nullspace_subspace_preserves_softmax_exactly():
    rng = np.random.default_rng(2)
    C, d, N_s = 3, 12, 200
    W = rng.standard_normal((C, d)); b = rng.standard_normal(C)
    Z = rng.standard_normal((N_s, d))
    Nb = EO.head_nullspace_basis(W)
    # remove an ARBITRARY 3-dim subspace inside ker(W_c)
    Q = EO.random_axes_in_span(Nb, 3, seed=0)
    P = Nb @ (Q.T @ Q) @ Nb.T
    Zrm = Z @ (np.eye(d) - P).T
    p0, p1 = _softmax(Z @ W.T + b), _softmax(Zrm @ W.T + b)
    assert np.abs(p0 - p1).max() < 1e-10                    # softmax ALGEBRAICALLY unchanged
    # predictions identical
    assert np.array_equal((Z @ W.T + b).argmax(1), (Zrm @ W.T + b).argmax(1))


def test_head_null_projector_is_within_kernel_and_low_rank():
    rng = np.random.default_rng(3)
    C, d, k = 4, 24, 5
    W = rng.standard_normal((C, d))
    Z = rng.standard_normal((300, d)); y = rng.integers(0, C, 300); dsub = rng.integers(0, 6, 300)
    P, rank, Nb = EO.exact_head_null_projector(Z, y, dsub, W, k)
    assert rank <= k
    assert np.allclose(EO.centered_head(W) @ P, 0.0, atol=1e-7)   # range(P) subseteq ker(W_c)
    # P is a projector (idempotent, symmetric)
    assert np.allclose(P, P.T, atol=1e-8) and np.allclose(P @ P, P, atol=1e-7)


def test_oracle_removes_subject_leakage_placed_in_nullspace():
    # place subject-structured signal INSIDE ker(W_c); the oracle should capture it and (by construction)
    # leave the classifier exactly unchanged. random-null of the same rank should capture much less.
    rng = np.random.default_rng(4)
    C, d = 3, 16
    W = rng.standard_normal((C, d)); b = rng.standard_normal(C)
    Nb = EO.head_nullspace_basis(W)                         # [d, r]
    # subject directions = 2 fixed axes within the nullspace
    subj_axes = Nb[:, :2]                                   # [d, 2]
    n_subj, per = 5, 60
    Z, y, dsub = [], [], []
    task_dir = np.linalg.pinv(EO.centered_head(W)) @ np.eye(C)   # something task-relevant
    for s in range(n_subj):
        coeff = rng.standard_normal(2) * 3.0
        for c in range(C):
            base = rng.standard_normal((per, d)) * 0.3
            base += (subj_axes @ coeff)                     # subject signal in nullspace
            base[:, :] += (W[c] / np.linalg.norm(W[c]))     # weak task signal along W row
            Z.append(base); y += [c] * per; dsub += [s] * per
    Z = np.vstack(Z); y = np.array(y); dsub = np.array(dsub)
    P, rank, _ = EO.exact_head_null_projector(Z, y, dsub, W, k=2)
    Zrm = Z @ (np.eye(d) - P).T
    # exact task-safety
    assert np.abs(_softmax(Z @ W.T + b) - _softmax(Zrm @ W.T + b)).max() < 1e-9
    # oracle removed the subject axes: linear subject decode falls a lot
    from sklearn.linear_model import LogisticRegression
    def dec(X):
        i = rng.permutation(len(X)); tr, te = i[:int(.7*len(i))], i[int(.7*len(i)):]
        return (LogisticRegression(max_iter=300).fit(X[tr], dsub[tr]).predict(X[te]) == dsub[te]).mean()
    assert dec(Zrm) < dec(Z) - 0.15


def test_random_null_also_preserves_softmax():
    rng = np.random.default_rng(5)
    W = rng.standard_normal((4, 20)); b = rng.standard_normal(4); Z = rng.standard_normal((100, 20))
    P, rank = EO.random_head_null_projector(W, 4, seed=0)
    Zrm = Z @ (np.eye(20) - P).T
    assert np.abs(_softmax(Z @ W.T + b) - _softmax(Zrm @ W.T + b)).max() < 1e-10
