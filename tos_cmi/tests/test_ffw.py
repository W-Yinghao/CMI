"""Tests for FFW-EEG (Finding Fantastic Weights) — gate, mask ordering, frontier, and a disentangled case
where a fantastic (task-safe) neuron sub-network SHOULD exist."""
import numpy as np
import torch
import pytest
from tos_cmi.eeg import ffw as FFW


def test_gate_limits():
    m = torch.tensor([5.0, -5.0, 0.0])
    g_warm = FFW.gate(m, 1.0)
    assert g_warm[0] > 0.9 and g_warm[1] < 0.1
    g_cold = FFW.gate(m, 0.01)                              # tau->0 : {0,1}
    assert g_cold[0] > 0.99 and g_cold[1] < 0.01


def test_mask_from_scores_prunes_lowest():
    scores = np.array([3.0, -1.0, 2.0, -2.0, 0.5])
    mask = FFW.mask_from_scores(scores, 2)                  # prune the 2 lowest (idx 3, 1)
    assert mask[3] == 0 and mask[1] == 0 and mask.sum() == 3


def test_random_mask_count():
    m = FFW.random_mask(10, 4, seed=0)
    assert m.sum() == 6 and set(np.unique(m)) <= {0.0, 1.0}


def _disentangled(n_subj=4, per=80, n_cls=2, seed=0):
    """graph_z where task lives in dims 0-1 (correlate with y) and subject in dims 2-3 (correlate with d, NOT
    y); a frozen head using ONLY the task dims. Fantastic weights (prune dims 2-3) exist."""
    rng = np.random.default_rng(seed); d = 8
    Z = 0.3 * rng.standard_normal((n_subj * per * n_cls, d))
    y = np.tile(np.repeat(np.arange(n_cls), per), n_subj)
    dsub = np.repeat(np.arange(n_subj), per * n_cls)
    Z[:, 0] += (y * 3.0)                                    # task dims
    Z[:, 2] += (dsub * 3.0)                                 # subject dims (independent of y)
    W = np.zeros((n_cls, d)); W[1, 0] = 2.0; b = np.zeros(n_cls)   # head uses only task dim 0
    return Z, y, dsub, W, b, n_cls, n_subj


def _cm_subj(Zx, y, d):
    from sklearn.linear_model import LogisticRegression
    accs = []; rng = np.random.default_rng(0)
    for c in np.unique(y):
        mm = y == c; zz, dd = Zx[mm], d[mm]
        if len(np.unique(dd)) < 2: continue
        idx = rng.permutation(len(zz)); cut = int(.7 * len(idx))
        accs.append((LogisticRegression(max_iter=200).fit(zz[idx[:cut]], dd[idx[:cut]]).predict(zz[idx[cut:]]) == dd[idx[cut:]]).mean())
    return float(np.mean(accs))


def test_frontier_finds_fantastic_weights_when_disentangled():
    # frontier LOGIC: with a correct ordering (subject dims 2,3 scored lowest), a task-safe prune exists that
    # cuts subject decodability without hurting task (head uses only dim 0).
    Z, y, d, W, b, ncls, ndom = _disentangled()
    from sklearn.metrics import balanced_accuracy_score
    tb = lambda Zx: balanced_accuracy_score(y, (Zx @ W.T + b).argmax(1))
    cm = lambda Zx: _cm_subj(Zx, y, d)
    scores = np.array([5.0, 4.0, -5.0, -4.0, 3.0, 2.0, 1.0, 0.5])   # subject dims 2,3 lowest
    fr = FFW.prune_frontier(Z, y, d, W, b, scores, cm, tb, ks=[1, 2, 3], seed=0)
    assert fr["task_safe_exists"]
    assert fr["full_cmi"] - fr["task_safe_best"]["ffw_cmi"] > 0.05


def test_ffw_keeps_task_ranks_subject_prunable():
    # FFW must score the TASK dim (0, used by the frozen head) HIGHER than the SUBJECT dim (2), i.e. it keeps
    # the task neuron and marks the subject neuron as prunable. (It prunes pure-noise dims first, which is
    # correct -- they are the most task-free to remove.) A task-safe deep prune then reduces subject CMI.
    Z, y, d, W, b, ncls, ndom = _disentangled()
    from sklearn.metrics import balanced_accuracy_score
    tb = lambda Zx: balanced_accuracy_score(y, (Zx @ W.T + b).argmax(1))
    cm = lambda Zx: _cm_subj(Zx, y, d)
    scores, mask, diag = FFW.find_fantastic_weights(Z, y, d, W, b, ncls, ndom, gamma=5.0, n_temps=3,
                                                    inner_epochs=80, p_epochs=40, seed=0)
    assert scores[0] > scores[2]                            # task dim kept above subject dim
    fr = FFW.prune_frontier(Z, y, d, W, b, scores, cm, tb, ks=[2, 4, 6], seed=0)
    assert fr["task_safe_exists"] and (fr["full_cmi"] - fr["task_safe_best"]["ffw_cmi"]) > 0.05


def test_find_fantastic_weights_returns_valid_scores():
    Z, y, d, W, b, ncls, ndom = _disentangled()
    scores, mask, diag = FFW.find_fantastic_weights(Z, y, d, W, b, ncls, ndom, gamma=1.0, n_temps=2,
                                                    inner_epochs=20, p_epochs=10, seed=0)
    assert scores.shape == (Z.shape[1],) and np.isfinite(scores).all()
    assert set(np.unique(mask)) <= {0.0, 1.0}
