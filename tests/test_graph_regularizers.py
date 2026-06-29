"""CIGL Phase-1 tests: node- and edge-level conditional-leakage heads (CPU only, tiny synthetic).

These are the trainable posterior-KL leakage PROXIES q(D | R, Y) for the node features Z_v and the
learned adjacency A (NOT unbiased CMI estimators; see cmi/methods/graph_regularizers.py docstring).
Verifies: finite Step-A (domain-classification) loss, finite regularizer that backpropagates to the
backbone object (node_Z / edge_logits), the length-C node leakage map, and that the empirical prior
pi_y(D) of shape [n_cls, n_dom] is consumed correctly.

See docs/CIGL_02_CODEBASE_AUDIT.md §1.4 and docs/CIGL_03_IMPLEMENTATION_PLAN.md Phase 1.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root -> import cmi.*
from cmi.methods.graph_regularizers import NodePosterior, EdgePosterior  # noqa: E402
from cmi.methods.regularizers import empirical_priors  # noqa: E402

N_CLS = 3
N_DOM = 4
N_CHANS = 22
NODE_D = 16
BATCH = 32


def _synthetic_labels(seed=0):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, N_CLS, BATCH)
    d = rng.integers(0, N_DOM, BATCH)
    # guarantee every class is present so pi_y has full support (Laplace smoothing covers the rest)
    y[:N_CLS] = np.arange(N_CLS)
    return y, d


def _priors(y, d):
    """empirical_priors -> (pi_y[n_cls,n_dom], p_d[n_dom], p_dy[n_cls,n_dom]); heads use priors[0]."""
    return empirical_priors(y, d, N_DOM, N_CLS, alpha=1.0)


# --------------------------------------------------------------------------- prior shape handling
def test_prior_shape_is_consumed():
    y, d = _synthetic_labels()
    priors = _priors(y, d)
    pi_y = priors[0]
    assert pi_y.shape == (N_CLS, N_DOM), "empirical pi_y must be [n_cls, n_dom]"
    assert np.allclose(pi_y.sum(axis=1), 1.0), "pi_y rows (p(D|Y=y)) must be a distribution"
    node = NodePosterior(NODE_D, N_DOM, N_CLS, priors)
    edge = EdgePosterior(N_CHANS, N_DOM, N_CLS, priors)
    # both heads cache log pi_y(D) as a [n_cls, n_dom] buffer
    assert node.log_pi.shape == (N_CLS, N_DOM)
    assert edge.log_pi.shape == (N_CLS, N_DOM)
    assert torch.isfinite(node.log_pi).all() and torch.isfinite(edge.log_pi).all()


# ------------------------------------------------------------------------------------ NodePosterior
def test_node_step_a_loss_finite():
    y, d = _synthetic_labels()
    node = NodePosterior(NODE_D, N_DOM, N_CLS, _priors(y, d))
    node_Z = torch.randn(BATCH, N_CHANS, NODE_D)
    loss = node.step_a_loss(node_Z, torch.tensor(y), torch.tensor(d))
    assert loss.ndim == 0 and torch.isfinite(loss), "Step-A node domain-CE must be a finite scalar"
    assert loss.item() > 0


def test_node_reg_finite_and_backprops_to_node_Z():
    y, d = _synthetic_labels()
    node = NodePosterior(NODE_D, N_DOM, N_CLS, _priors(y, d))
    node_Z = torch.randn(BATCH, N_CHANS, NODE_D, requires_grad=True)
    r = node.reg(node_Z, torch.tensor(y))
    assert r.ndim == 0 and torch.isfinite(r) and r.item() >= 0, "reg = E KL(q||pi_y) is a finite scalar >= 0"
    r.backward()
    assert node_Z.grad is not None, "node-CMI regularizer must produce gradient on node_Z"
    assert torch.isfinite(node_Z.grad).all()
    assert node_Z.grad.abs().sum() > 0, "node_Z gradient is identically zero"


def test_node_leakage_map_length_C():
    y, d = _synthetic_labels()
    node = NodePosterior(NODE_D, N_DOM, N_CLS, _priors(y, d))
    node_Z = torch.randn(BATCH, N_CHANS, NODE_D)
    lmap = node.leakage_map(node_Z, torch.tensor(y))
    assert lmap.shape == (N_CHANS,), "per-channel leakage map must have length C"
    assert torch.isfinite(lmap).all() and (lmap >= 0).all()


# ------------------------------------------------------------------------------------ EdgePosterior
def test_edge_step_a_loss_finite():
    y, d = _synthetic_labels()
    edge = EdgePosterior(N_CHANS, N_DOM, N_CLS, _priors(y, d))
    edge_logits = torch.randn(BATCH, N_CHANS, N_CHANS)
    loss = edge.step_a_loss(edge_logits, torch.tensor(y), torch.tensor(d))
    assert loss.ndim == 0 and torch.isfinite(loss), "Step-A edge domain-CE must be a finite scalar"
    assert loss.item() > 0


def test_edge_reg_finite_and_backprops_to_edge_logits():
    y, d = _synthetic_labels()
    edge = EdgePosterior(N_CHANS, N_DOM, N_CLS, _priors(y, d))
    edge_logits = torch.randn(BATCH, N_CHANS, N_CHANS, requires_grad=True)
    r = edge.reg(edge_logits, torch.tensor(y))
    assert r.ndim == 0 and torch.isfinite(r) and r.item() >= 0, "reg = E KL(q||pi_y) is a finite scalar >= 0"
    r.backward()
    assert edge_logits.grad is not None, "edge-CMI regularizer must produce gradient on edge_logits"
    assert torch.isfinite(edge_logits.grad).all()
    assert edge_logits.grad.abs().sum() > 0, "edge_logits gradient is identically zero"


def test_edge_uses_only_upper_triangle():
    """EdgePosterior reads the strict upper triangle; the diagonal/lower entries get no gradient."""
    y, _ = _synthetic_labels()
    edge = EdgePosterior(N_CHANS, N_DOM, N_CLS, _priors(y, _))
    edge_logits = torch.randn(BATCH, N_CHANS, N_CHANS, requires_grad=True)
    edge.reg(edge_logits, torch.tensor(y)).backward()
    g = edge_logits.grad
    # diagonal must be untouched by the strict-upper-triangle (offset=1) summary
    assert torch.allclose(torch.diagonal(g, dim1=1, dim2=2), torch.zeros(BATCH, N_CHANS), atol=0)


@pytest.mark.parametrize("n_chans", [19, 32, 62])
def test_edge_posterior_builds_for_channel_counts(n_chans):
    y, d = _synthetic_labels()
    edge = EdgePosterior(n_chans, N_DOM, N_CLS, _priors(y, d))
    expect_edges = n_chans * (n_chans - 1) // 2
    assert edge.compress.in_features == expect_edges, "compress must read C*(C-1)/2 upper-triangle edges"
    r = edge.reg(torch.randn(BATCH, n_chans, n_chans), torch.tensor(y))
    assert torch.isfinite(r)
