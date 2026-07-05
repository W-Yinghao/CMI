"""CIGL R2a — baseline registry + same-backbone contract + firewall."""
import numpy as np
import torch

from cmi.eval.baseline_registry import (R2A_METHODS, DEFERRED_METHODS, SAME_BACKBONE_CONTRACT,
                                        build_contract_backbone, validate_registry, BACKBONE)
from cmi.train.trainer import train_model, predict, ALL_METHODS
from cmi.eval.graph_leakage import within_label_permutation


def _synth(n_cls=2, C=8, T=64, n_per=8, n_dom=3, seed=0):
    rng = np.random.default_rng(seed)
    X, y, d = [], [], []
    for yi in range(n_cls):
        for di in range(n_dom):
            X.append(rng.standard_normal((n_per, C, T)).astype("float32")); y += [yi] * n_per; d += [di] * n_per
    return np.concatenate(X), np.array(y, "int64"), np.array(d, "int64")


def test_registry_valid_and_scoped():
    assert validate_registry() == []                                   # all active methods known + parse
    assert set(R2A_METHODS) == {"erm", "cigl_graph", "cigl_node", "cigl_graph_node",
                                "dann", "cond_dann", "cdan"}
    assert not (set(R2A_METHODS) & set(DEFERRED_METHODS))              # deferred zoo not front-loaded
    assert SAME_BACKBONE_CONTRACT["source_only"] and SAME_BACKBONE_CONTRACT["target_firewall"] == "eval_only"
    assert SAME_BACKBONE_CONTRACT["backbone"] == BACKBONE
    for m in ("dann", "cdann", "cdan", "graphcmi"):
        assert m in ALL_METHODS                                        # trainer knows them (incl. new cdan)


def _kw(spec):
    if spec["method"] == "graphcmi":
        lg = 0.01 if "cigl_graph" in spec["config"] or spec["config"].startswith("graphcmi:0.010") else 0.0
        ln = 0.01 if spec["config"].split(":")[2] != "0.000" else 0.0
        return dict(method="graphcmi", lam=lg, gamma=ln, lam_edge=0.0)
    return dict(method=spec["method"], lam=(0.0 if spec["method"] == "erm" else 1.0))


def test_all_r2a_methods_run_on_same_backbone():
    X, y, d = _synth()
    for label, spec in R2A_METHODS.items():
        torch.manual_seed(0)
        bb = build_contract_backbone(8, 64, 2)                        # the ONE shared backbone
        assert type(bb).__name__ == "DGCNNForwardGraphAdapter" and hasattr(bb, "forward_graph")
        bb, _, _ = train_model(bb, X, y, d, 2, epochs=2, bs=16, warmup=1, device="cpu", seed=0, **_kw(spec))
        assert np.isfinite(predict(bb, X[:8], "cpu")).all(), f"{label} produced non-finite preds"


def test_within_label_permutation_firewall():
    # the null permutes D ONLY within each label -> Y unchanged, D multiset preserved per label, D reassigned
    rng = np.random.default_rng(0)
    y = rng.integers(0, 3, 200); d = rng.integers(0, 4, 200)
    d_perm = within_label_permutation(y, d, seed=1)
    assert len(d_perm) == len(d)
    for c in np.unique(y):
        m = y == c
        assert sorted(d_perm[m]) == sorted(d[m])                      # within-label D multiset preserved
    assert not np.array_equal(d_perm, d)                              # actually permuted
