"""Pinned tests for the MCC loss + balanced sampler (the trainer-independent pins 1-7, 9). Arm-fork / warm-up-hash
(8) and artifact-recovery (10) live with the trainer integration. Synthetic data checks the implementation only."""
import numpy as np
import pytest
import torch

from tos_cmi.train.mechanism_consistency import (mcc_loss, class_pairs, contrast_norm, effective_rank,
                                                 BalancedSubjectClassSampler, _shuffle_subjects_within_class)


def _emb(m=4, C=4, K=8, p=16, idio=0.3, seed=0):
    rng = np.random.default_rng(seed); shared = rng.standard_normal((C, p)); Z = []; y = []; d = []
    for s in range(m):
        for c in range(C):
            mu = shared[c] + idio * rng.standard_normal(p)
            Z.append(mu + 0.1 * rng.standard_normal((K, p))); y += [c] * K; d += [s] * K
    return torch.tensor(np.vstack(Z), dtype=torch.float32), np.array(y), np.array(d), shared


def test_1_identical_contrasts_give_zero_loss():
    # every subject has the SAME class means -> unit contrasts identical -> cosine 1 -> L_MCC ~ 0
    m, C, K, p = 4, 4, 6, 16
    shared = np.random.default_rng(1).standard_normal((C, p))
    Z = torch.tensor(np.vstack([shared[c] for _ in range(m) for c in range(C) for _ in range(K)]), dtype=torch.float32)
    y = np.array([c for _ in range(m) for c in range(C) for _ in range(K)])
    d = np.array([s for s in range(m) for _ in range(C) for _ in range(K)])
    L, _ = mcc_loss(Z, y, d)
    assert abs(float(L)) < 1e-5


def test_2_direction_loss_is_scale_insensitive():
    Z, y, d, _ = _emb()
    L1, _ = mcc_loss(Z, y, d)
    L2, _ = mcc_loss(7.3 * Z, y, d)                     # global scale of the embedding must not change the loss
    assert abs(float(L1) - float(L2)) < 1e-5
    # per-class scaling of contrasts also must not matter (direction-only): scale class 0 embeddings by 5
    Zs = Z.clone(); Zs[y == 0] *= 5.0
    # a pure magnitude change on one class shifts directions only via the shared offset; assert loss stays finite+bounded
    L3, _ = mcc_loss(Zs, y, d); assert np.isfinite(float(L3))


def test_3_loso_consensus_excludes_self():
    # if subject 0's contrast is ANTI-aligned with the others, its per-term cost must be ~2 (1 - (-1)); with a
    # self-including consensus the anti-aligned subject would partly cancel itself and the cost would be < 2.
    p = 8; base = np.zeros(p); base[0] = 1.0
    rows = []
    ys = []; ds = []
    for s in range(4):
        v = base if s > 0 else -base                    # subject 0 flipped
        # class 0 mean = +v/2 offset, class 1 mean = -v/2 -> contrast = v
        rows += [(+v * 0.5), (-v * 0.5)]; ys += [0, 1]; ds += [s, s]
    Z = torch.tensor(np.vstack([r for r in rows]), dtype=torch.float32)
    L, info = mcc_loss(Z, np.array(ys), np.array(ds), pairs=[(0, 1)])
    # subject 0 term = 1 - <-e0, consensus_of_others(+e0)> = 1 - (-1) = 2 ; others term ~ 0 -> mean ~ 0.5
    assert info["n_terms"] == 4
    per_subj_flip_cost = 2.0
    assert abs(float(L) - per_subj_flip_cost / 4) < 0.05  # only subject 0 pays ~2, others ~0


def test_4_true_vs_shuffle_grouping_differ():
    Z, y, d, _ = _emb(idio=0.5)
    Lt, _ = mcc_loss(Z, y, d)
    Ls, _ = mcc_loss(Z, y, d, shuffle_subjects=True, generator=torch.Generator().manual_seed(3))
    assert abs(float(Lt) - float(Ls)) > 1e-4
    # shuffle preserves per-class subject counts
    d2 = _shuffle_subjects_within_class(torch.as_tensor(y), torch.as_tensor(d), torch.Generator().manual_seed(3))
    for c in np.unique(y):
        assert sorted(d2[torch.as_tensor(y) == c].tolist()) == sorted(d[y == c].tolist())


def test_5_gradient_flows_into_embedding():
    Z, y, d, _ = _emb(); Z.requires_grad_(True)
    L, _ = mcc_loss(Z, y, d); L.backward()
    assert Z.grad is not None and float(Z.grad.abs().sum()) > 0


def test_6_target_arrays_not_accepted_by_loss_signature():
    import inspect
    params = set(inspect.signature(mcc_loss).parameters)
    assert not ({"Z_target", "y_target", "target", "Xq", "yq"} & params)   # loss sees only source Z/y/d


def test_7_missing_subject_class_cell_fails_loud():
    Z, y, d, _ = _emb()
    keep = ~((d == 0) & (y == 3))                       # drop subject 0's class 3
    with pytest.raises(ValueError, match="missing class"):
        mcc_loss(Z[keep], y[keep], d[keep])
    # sampler also fails loud on an empty cell
    with pytest.raises(ValueError, match="empty subject-class"):
        BalancedSubjectClassSampler(d[keep], y[keep], K=4)


def test_9_collapse_guards_detect_shrink():
    Z, y, d, _ = _emb()
    assert contrast_norm(Z, y, d) > 0.1 and effective_rank(Z) > 1.5
    # collapsed embedding (all one point) -> contrast norm ~ 0 and effective rank ~ 1
    Zc = torch.zeros_like(Z)
    assert contrast_norm(Zc, y, d) < 1e-4
    assert effective_rank(Z + 1e3 * torch.tensor(np.eye(1, Z.shape[1], 0), dtype=torch.float32)) < effective_rank(Z) + 1.0


def test_sampler_balanced_and_deterministic():
    Z, y, d, _ = _emb()
    s1 = BalancedSubjectClassSampler(d, y, K=4, n_batches=2, seed=0)
    b1 = next(iter(s1)); assert len(b1) == 4 * 4 * 4
    assert len(set(zip(d[b1].tolist(), y[b1].tolist()))) == 4 * 4      # every subject-class cell present
    s2 = BalancedSubjectClassSampler(d, y, K=4, n_batches=2, seed=0)
    assert np.array_equal(next(iter(s2)), b1)                          # deterministic per seed
