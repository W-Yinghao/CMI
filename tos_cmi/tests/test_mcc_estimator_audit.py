"""Pinned tests for the MCC estimator audit. The load-bearing one is test_1: the two-pass EXACT population gradient
must equal a single-graph full-batch reference (numerically, eval/BN-frozen)."""
import numpy as np
import pytest
import torch

from tos_cmi.eval import mcc_estimator_audit as MEA
from tos_cmi.train.mechanism_consistency import class_pairs


class _Fixture(torch.nn.Module):
    """Tiny backbone mimicking HookedBackbone: forward(x) -> (logits, z), with a BatchNorm so the eval/frozen-BN
    requirement is actually exercised."""
    def __init__(self, p_in=8, p=6, C=4):
        super().__init__()
        self.lin = torch.nn.Linear(p_in, p); self.bn = torch.nn.BatchNorm1d(p); self.head = torch.nn.Linear(p, C)
        torch.manual_seed(0)
        with torch.no_grad():                       # give BN non-trivial running stats
            self.bn.running_mean.normal_(); self.bn.running_var.uniform_(0.5, 1.5)

    def forward(self, x):
        h = x.reshape(x.shape[0], -1); z = self.bn(self.lin(h)); return self.head(z), z


def _data(N=64, C=4, m=4, p_in=8, seed=0):
    rng = np.random.default_rng(seed)
    y = np.tile(np.arange(C), N // C); d = np.repeat(np.arange(m), N // m)
    # ensure every (subject,class) cell is populated
    y = np.array([(i % C) for i in range(N)]); d = np.array([(i // C) % m for i in range(N)])
    X = rng.standard_normal((N, 2, p_in // 2)).astype("float32")
    return X, y, d


def _ref_full_batch_gradient(bb, X, y, d):
    """Single-graph reference: forward ALL in one batch (eval), prototypes (non-leaf), L_MCC, autograd to params."""
    bb.eval()
    z = bb(torch.tensor(X, dtype=torch.float32))[1]
    classes = sorted(np.unique(y).tolist()); subs = sorted(np.unique(d).tolist()); pairs = class_pairs(classes)
    means = {(s, c): z[(d == s) & (y == c)].mean(0) for s in subs for c in classes}
    L = MEA.mcc_loss_from_means(means, subs, classes, pairs)
    params = list(bb.parameters())
    g = torch.autograd.grad(L, params, allow_unused=True)          # post-z head params legitimately unused -> 0
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten() for gi, p in zip(g, params)]).detach().numpy()


def test_1_two_pass_equals_full_batch_gradient():
    bb = _Fixture(); X, y, d = _data()
    g_ref = _ref_full_batch_gradient(bb, X, y, d)
    g_two, means, gmu, L = MEA.exact_population_gradient(bb, X, y, d, "cpu", bs=16)   # bs<N forces micro-batching
    assert g_two.shape == g_ref.shape
    rel = np.linalg.norm(g_two - g_ref) / (np.linalg.norm(g_ref) + 1e-12)
    assert rel < 1e-5, f"two-pass != full-batch (rel {rel:.2e})"


def test_2_k_all_prototype_gradient_equals_population():
    # prototypes from ALL data == the population prototypes used inside the two-pass Pass 1
    bb = _Fixture(); X, y, d = _data()
    _, means, gmu, _ = MEA.exact_population_gradient(bb, X, y, d, "cpu", bs=16)
    bb.eval(); z = bb(torch.tensor(X, dtype=torch.float32))[1].detach().numpy()
    for s in np.unique(d):
        for c in np.unique(y):
            m = (d == s) & (y == c)
            assert np.allclose(means[(int(s), int(c))], z[m].mean(0), atol=1e-5)


def test_3_batchnorm_running_stats_unchanged():
    bb = _Fixture(); X, y, d = _data()
    rm0 = bb.bn.running_mean.clone(); rv0 = bb.bn.running_var.clone()
    MEA.exact_population_gradient(bb, X, y, d, "cpu", bs=16)
    MEA.episodic_theta_gradients(bb, X, y, d, "cpu", K=4, R=3, seed=0)
    assert torch.equal(bb.bn.running_mean, rm0) and torch.equal(bb.bn.running_var, rv0)


def test_4_no_target_arrays_in_audit_signatures():
    import inspect
    for fn in (MEA.exact_population_gradient, MEA.episodic_theta_gradients):
        p = set(inspect.signature(fn).parameters)
        assert not ({"Xte", "yte", "y_target", "Z_target", "target"} & p)


def test_5_episodic_seeds_are_cell_specific_and_deterministic():
    bb = _Fixture(); X, y, d = _data()
    g_a = MEA.episodic_theta_gradients(bb, X, y, d, "cpu", K=4, R=2, seed=1)
    g_b = MEA.episodic_theta_gradients(bb, X, y, d, "cpu", K=4, R=2, seed=1)
    g_c = MEA.episodic_theta_gradients(bb, X, y, d, "cpu", K=4, R=2, seed=2)
    assert np.allclose(g_a, g_b) and not np.allclose(g_a, g_c)


def test_6_true_and_shuffle_both_computable():
    bb = _Fixture(); X, y, d = _data()
    gt, _, _, _ = MEA.exact_population_gradient(bb, X, y, d, "cpu", shuffle=False, bs=16)
    gs, _, _, _ = MEA.exact_population_gradient(bb, X, y, d, "cpu", shuffle=True, rng=np.random.default_rng(0), bs=16)
    assert gt.shape == gs.shape and not np.allclose(gt, gs)     # shuffle is not a no-op


def test_7_empty_cell_fails_loud():
    bb = _Fixture(); X, y, d = _data()
    keep = ~((d == 0) & (y == 3))
    with pytest.raises(ValueError, match="empty subject-class cell"):
        MEA.exact_population_gradient(bb, X[keep], y[keep], d[keep], "cpu", bs=16)


def test_8_diagnostics_and_one_step_wsci_shapes():
    bb = _Fixture(); X, y, d = _data()
    g_full, means, gmu, _ = MEA.exact_population_gradient(bb, X, y, d, "cpu", bs=16)
    g_K = MEA.episodic_theta_gradients(bb, X, y, d, "cpu", K=4, R=8, seed=0)
    diag = MEA.gradient_diagnostics(g_full, g_K)
    assert -1.001 <= diag["A_K"] <= 1.001 and diag["B_K"] >= 0 and diag["SNR_K"] >= 0
    classes = sorted(np.unique(y).tolist()); subs = sorted(np.unique(d).tolist()); pairs = class_pairs(classes)
    dw = MEA.one_step_prototype_wsci(means, gmu, subs, classes, pairs, alpha=0.1)
    assert np.isfinite(dw)
