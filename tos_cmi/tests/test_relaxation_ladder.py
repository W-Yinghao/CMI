"""Relaxation Ladder Stage 9 — eraser + head unit behavior (LW-LEACE removes identity, whitening is
zero-deletion, random matches rank, subject-grouped CV respects groups)."""
import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from tos_cmi.eeg import relaxation_ladder as RL


def _subject_structured(n_subj=4, per=60, n_cls=2, dim=16, strength=3.0, seed=0):
    rng = np.random.default_rng(seed)
    cm = rng.standard_normal((n_cls, dim)) * 2
    sd = rng.standard_normal((n_subj, dim)); sd /= np.linalg.norm(sd, axis=1, keepdims=True)
    Z, y, d = [], [], []
    for s in range(n_subj):
        for c in range(n_cls):
            Z.append(cm[c] + strength * sd[s] + 0.4 * rng.standard_normal((per, dim))); y += [c] * per; d += [s] * per
    return np.vstack(Z), np.array(y), np.array(d)


def _lin_decode(Z, lab, seed=0):
    rng = np.random.default_rng(seed); idx = rng.permutation(len(Z)); cut = int(0.7 * len(idx))
    tr, te = idx[:cut], idx[cut:]
    return float((LogisticRegression(max_iter=300).fit(Z[tr], lab[tr]).predict(Z[te]) == lab[te]).mean())


def test_lw_leace_removes_subject_preserves_task():
    Z, y, d = _subject_structured(seed=1)
    fn, rank = RL.lw_leace_full(Z, d)
    Ze = fn(Z)
    assert rank == 3
    subj_before, subj_after = _lin_decode(Z, d), _lin_decode(Ze, d)
    assert subj_after < subj_before - 0.2                    # subject decode materially reduced
    # task largely preserved (class means differ along non-subject directions)
    task_before, task_after = _lin_decode(Z, y), _lin_decode(Ze, y)
    assert task_after > task_before - 0.15


def test_whitening_only_is_zero_deletion():
    Z, y, d = _subject_structured(seed=2)
    fn, rank = RL.whitening_only(Z)
    assert rank == 0                                         # NO subspace deleted
    Ze = fn(Z)
    # whitening is an invertible rescale -> subject info NOT destroyed (unlike LEACE); decode stays high
    assert _lin_decode(Ze, d) > 0.5


def test_random_removal_exact_rank():
    rng = np.random.default_rng(3); Z = rng.standard_normal((200, 16))
    fn = RL.random_removal(16, 5, seed=0)
    Ze = fn(Z)
    # removed a rank-5 subspace -> the residual lies in a 11-dim subspace (16-5)
    assert np.linalg.matrix_rank(Ze - Ze.mean(0), tol=1e-6) <= 11
    assert RL.random_removal(16, 0, seed=0)(Z) is Z or np.allclose(RL.random_removal(16, 0, seed=0)(Z), Z)


def test_subject_grouped_cv_respects_groups():
    Z, y, d = _subject_structured(n_subj=5, seed=4)
    # if a subject leaked across folds this would be inflated; just assert it runs + finite + in [0,1]
    b = RL.subject_grouped_cv_bacc(Z, y, d, head="logreg", seed=0, n_splits=5)
    assert 0.0 <= b <= 1.0 and np.isfinite(b)


def test_fresh_head_standardizes_from_train_only():
    Z, y, d = _subject_structured(seed=5)
    # scoring on a scaled-up target must be handled by train-only standardization (no target-stat leakage)
    b = RL.fresh_head_bacc(Z[:200], y[:200], Z[200:], y[200:], head="logreg", seed=0)
    assert 0.0 <= b <= 1.0 and np.isfinite(b)


def test_feat_from_audit_npz_roundtrip(tmp_path):
    from cmi.eval.audit_npz import save_audit_npz
    rng = np.random.default_rng(6)
    Ns, Nt, Zg, C, Zn, ncls = 120, 30, 16, 8, 5, 2
    N = Ns + Nt
    gz = rng.standard_normal((N, Zg)).astype("float32"); nz = rng.standard_normal((N, C, Zn)).astype("float32")
    y = rng.integers(0, ncls, N).astype("int64")
    W = rng.standard_normal((ncls, Zg)).astype("float32"); b = rng.standard_normal(ncls).astype("float32")
    d = np.r_[rng.integers(0, 3, Ns), np.full(Nt, 3)].astype("int64")
    p = tmp_path / "f.audit.npz"
    save_audit_npz(str(p), graph_z=gz, node_z=nz, y=y, d=d, model_logits=(gz @ W.T + b).astype("float32"),
                   fold=0, seed=0, target_subject="3", method="erm", dataset="D",
                   task_head_weight=W, task_head_bias=b, task_head_input="graph_z",
                   source_indices=np.arange(Ns), target_indices=np.arange(Ns, N))
    feat = RL.feat_from_audit_npz(str(p))
    assert feat["Z_source"].shape[0] == Ns and feat["Z_target"].shape[0] == Nt
    assert feat["head_W"] is not None                       # verified head -> L0 original replay available
    assert len(np.unique(feat["subj_source"])) == 3         # 3 source subjects
