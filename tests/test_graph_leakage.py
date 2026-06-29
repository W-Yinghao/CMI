"""CIGL Phase-2 tests: probe-only label-conditional domain-leakage audit (CPU, synthetic).

Validates cmi/eval/graph_leakage.py on controlled DGPs where the answer is known:
  - the label-domain prior is a proper [K,M] distribution that tolerates missing cells;
  - the conditional KL is finite/non-negative;
  - the conditional probe DETECTS injected conditional leakage above the label-prior baseline;
  - the RETRAINED within-label permutation null sits below observed leakage when leakage exists,
    and observed ≈ null when it does not (the central reviewer correctness check);
  - audit_graph_objects returns graph/node/edge summaries + a length-C node map + a [C,C] edge map;
  - edge_binned_cmi_map localizes an injected spurious edge and keeps the diagonal zero.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.eval.graph_leakage import (  # noqa: E402
    compute_label_domain_prior, conditional_kl_to_prior, bootstrap_mean_ci,
    within_label_permutation, fit_conditional_domain_probe, audit_graph_objects,
    edge_binned_cmi_map, _permutation_null)

K, M = 2, 3       # classes, domains
N = 300


# ----------------------------------------------------------------- synthetic data generators
def _leaked_features(seed=0, n=N, fdim=4, strength=2.5):
    """y, d independent (so π_y≈uniform); feature dim 0 encodes d -> conditional leakage exists."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n)
    d = rng.integers(0, M, n)
    y[:K] = np.arange(K); d[:M] = np.arange(M)
    f = rng.standard_normal((n, fdim)).astype(np.float32)
    f[:, 0] += strength * d                          # domain signal, present within every label
    return f, y, d


def _noise_features(seed=1, n=N, fdim=4):
    """Features independent of d -> no conditional leakage; observed KL should match the null."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n); d = rng.integers(0, M, n)
    y[:K] = np.arange(K); d[:M] = np.arange(M)
    return rng.standard_normal((n, fdim)).astype(np.float32), y, d


# ----------------------------------------------------------------- prior + KL primitives
def test_label_domain_prior_is_distribution():
    _, y, d = _leaked_features()
    pi = compute_label_domain_prior(y, d, K, M)
    assert pi.shape == (K, M)
    assert torch.allclose(pi.sum(1), torch.ones(K), atol=1e-5)
    assert (pi > 0).all(), "smoothing must keep every cell strictly positive"


def test_label_domain_prior_handles_missing_cells():
    # class 1 only ever appears with domain 0 -> cells (1,1),(1,2) are empty but must stay finite
    y = np.array([0, 0, 0, 1, 1, 1])
    d = np.array([0, 1, 2, 0, 0, 0])
    pi = compute_label_domain_prior(y, d, K, M, smoothing=1e-3)
    assert torch.isfinite(pi.log()).all()
    assert torch.allclose(pi.sum(1), torch.ones(K), atol=1e-5)
    assert pi[1, 0] > pi[1, 1] and pi[1, 0] > pi[1, 2]


def test_conditional_kl_finite_nonnegative():
    _, y, d = _leaked_features()
    pi = compute_label_domain_prior(y, d, K, M)
    probs = torch.softmax(torch.randn(N, M), dim=1)
    kl = conditional_kl_to_prior(probs, y, pi)
    assert kl.shape == (N,)
    assert torch.isfinite(kl).all() and (kl >= 0).all()
    # KL of the prior against itself is ~0
    pi_rows = pi[torch.as_tensor(y, dtype=torch.long)]
    assert conditional_kl_to_prior(pi_rows, y, pi).abs().max() < 1e-5


def test_bootstrap_mean_ci_orders():
    ci = bootstrap_mean_ci(np.arange(100, dtype=float), n_boot=200, seed=0)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]
    assert abs(ci["mean"] - 49.5) < 5


def test_within_label_permutation_preserves_prior():
    _, y, d = _leaked_features()
    d_perm = within_label_permutation(y, d, seed=3)
    pi0 = compute_label_domain_prior(y, d, K, M)
    pi1 = compute_label_domain_prior(y, d_perm, K, M)
    assert torch.allclose(pi0, pi1, atol=1e-6), "within-label permutation must preserve p(D|Y)"
    assert not np.array_equal(d, d_perm), "permutation should actually move some domains"


# ----------------------------------------------------------------- the probe + null
def test_probe_detects_conditional_leakage():
    f, y, d = _leaked_features()
    res = fit_conditional_domain_probe(f, y, d, K, M, epochs=120, seed=0)
    assert np.isfinite(res["kl_mean"]) and res["kl_mean"] > 0
    assert res["domain_acc"] > res["prior_acc"] + 0.1, "probe must beat the label-only prior baseline"
    assert res["leakage_advantage"] > 0.1


def test_permutation_null_below_observed_when_leaked():
    f, y, d = _leaked_features()
    obs = fit_conditional_domain_probe(f, y, d, K, M, epochs=120, seed=0)["kl_mean"]

    def fit(d_arr):
        return fit_conditional_domain_probe(f, y, d_arr, K, M, epochs=120, seed=0)
    null = _permutation_null(fit, y, d, n_perm=8, seed=0)
    assert obs > null.mean() + 0.1, f"leaked observed KL {obs:.3f} not above null mean {null.mean():.3f}"


def test_no_leakage_matches_null():
    f, y, d = _noise_features()
    obs = fit_conditional_domain_probe(f, y, d, K, M, epochs=120, seed=0)["kl_mean"]

    def fit(d_arr):
        return fit_conditional_domain_probe(f, y, d_arr, K, M, epochs=120, seed=0)
    null = _permutation_null(fit, y, d, n_perm=8, seed=0)
    # observed must not substantially exceed the retrained null when there is no real leakage
    assert obs <= null.mean() + 0.10, f"noise observed KL {obs:.3f} >> null mean {null.mean():.3f}"


# ----------------------------------------------------------------- full audit + edge map
def _graph_tensors(seed=0, n=120, C=5, Dg=6, Dn=4):
    """graph_z encodes d (graph leakage); node channel 1 encodes d (node leakage);
    edge (0,2) encodes d (edge leakage); everything else is noise. y ⟂ d."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, K, n); d = rng.integers(0, M, n)
    y[:K] = np.arange(K); d[:M] = np.arange(M)
    graph_z = rng.standard_normal((n, Dg)).astype(np.float32); graph_z[:, 0] += 2.5 * d
    node_z = rng.standard_normal((n, C, Dn)).astype(np.float32); node_z[:, 1, 0] += 2.5 * d
    base = rng.standard_normal((n, C, C)).astype(np.float32)
    edge = 0.5 * (base + base.transpose(0, 2, 1))
    edge[:, 0, 2] += 2.5 * d; edge[:, 2, 0] = edge[:, 0, 2]
    for k in range(C):
        edge[:, k, k] = 0.0
    return graph_z, node_z, edge, y, d


def test_audit_graph_objects_shapes_and_keys():
    gz, nz, el, y, d = _graph_tensors()
    C = el.shape[1]
    out = audit_graph_objects(gz, nz, el, y, d, K, M, n_perm=4, epochs=60, seed=0)
    for obj in ("graph", "node", "edge"):
        blk = out[obj]
        for key in ("kl_mean", "permutation_mean", "permutation_p", "domain_acc", "prior_acc"):
            assert key in blk and np.isfinite(blk[key]), f"{obj}.{key} missing/non-finite"
    assert len(out["node"]["node_leakage_map"]) == C
    em = np.asarray(out["edge"]["edge_leakage_map"])
    assert em.shape == (C, C)
    # injected leakage should clear the retrained null in at least graph and edge (the strongest signals)
    assert out["graph"]["kl_mean"] > out["graph"]["permutation_mean"]
    assert out["edge"]["kl_mean"] > out["edge"]["permutation_mean"]


def test_edge_binned_cmi_localizes_spurious_edge():
    gz, nz, el, y, d = _graph_tensors(seed=2)
    C = el.shape[1]
    m = edge_binned_cmi_map(el, y, d, K, M, n_bins=4).numpy()
    assert m.shape == (C, C)
    assert np.allclose(np.diag(m), 0.0), "diagonal must be zero"
    assert np.allclose(m, m.T, atol=1e-6), "edge CMI map must be symmetric"
    # the injected (0,2) edge must be the strongest off-diagonal entry
    iu = np.triu_indices(C, 1)
    vals = m[iu]
    amax = np.argmax(vals)
    assert (iu[0][amax], iu[1][amax]) == (0, 2), "spurious edge (0,2) should dominate the CMI map"
    assert m[0, 2] > 3 * np.median(vals[vals != m[0, 2]] + 1e-9)
