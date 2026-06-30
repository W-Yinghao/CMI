"""Tests for CIGL Phase 3A-S backbone sanity (scientific firewalls + pipeline).

Covers: CPU dry-run works; the candidate list contains the known-good decoders; corrupting target labels
changes NOTHING about source_probe metrics or selected_successful_models (source-only selection); target
metrics are evaluation-only; non-graph CNNs emit NO graph leakage fields; success uses source_probe only.
"""
import json
import runpy
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_backbone_sanity as R  # noqa: E402


def _args(**kw):
    import argparse
    base = dict(dry_run_synthetic=True, dataset="synthetic", fold=0, device="cpu", seeds=[0, 1], epochs=3,
                bs=64, probe_epochs=5, leak_n_perm=3, train_frac=0.7, enc_train_frac=0.7, min_per_cell=2,
                success_bacc_floor=0.45, candidates=R.CANDIDATES, tmin=0.5, tmax=3.5, resample=128)
    base.update(kw)
    return argparse.Namespace(**base)


def test_candidate_list_contains_known_good_decoders():
    for m in ("graphcmi_current_ref", "eegnet", "shallow_convnet"):
        assert m in R.CANDIDATES
    assert R.GRAPH_BACKBONES == {"graphcmi_current_ref"}


def test_minimal_decoders_build_and_learn_a_learnable_task():
    """The known-good CNNs must actually learn (else the sanity reference is meaningless)."""
    from cmi.models.sanity_backbones import build_sanity_backbone
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    rng = np.random.default_rng(0)
    proto = 2.5 * rng.standard_normal((4, 22, 96)).astype("float32")
    X, y = [], []
    for _ in range(320):
        c = rng.integers(0, 4); X.append(proto[c] + 0.6 * rng.standard_normal((22, 96)).astype("float32")); y.append(c)
    X = np.stack(X); y = np.array(y, "int64"); d = np.zeros(len(y), "int64")
    for name in ("eegnet", "shallow_convnet"):
        net = build_sanity_backbone(name, 22, 96, 4)
        assert hasattr(net, "z_dim") and not hasattr(net, "forward_graph")
        net, _, _ = train_model(net, X, y, d, 4, method="erm", epochs=10, bs=64, warmup=1, device="cpu", seed=0)
        acc = classification_metrics(predict(net, X, "cpu"), y)["balanced_acc"]
        assert acc > 0.45, f"{name} failed to learn a learnable task: {acc:.3f}"


def test_non_graph_cnns_emit_no_graph_leakage_fields():
    fold = R._synthetic_fold(seed=0)
    rec = R._train_eval("eegnet", fold, seed=0, args=_args(), device="cpu")
    assert "leakage" not in rec and rec["is_graph_backbone"] is False
    g = R._train_eval("graphcmi_current_ref", fold, seed=0, args=_args(), device="cpu")
    assert "leakage" in g and set(g["leakage"]) == {"graph", "node", "edge"}


def test_target_eval_is_marked_evaluation_only():
    fold = R._synthetic_fold(seed=0)
    rec = R._train_eval("shallow_convnet", fold, seed=0, args=_args(), device="cpu")
    assert rec["target_eval"]["evaluation_only"] is True


def _run_summary(tmp_candidates, monkeypatch):
    """Run main() with a small candidate set; return the written summary dict."""
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--seeds", "0", "1", "--epochs", "3",
            "--probe_epochs", "5", "--leak_n_perm", "3", "--candidates", *tmp_candidates]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_fold0_backbone_sanity_summary.json"))


def test_target_label_corruption_cannot_change_source_or_selection(monkeypatch):
    """Source-only firewall: corrupting ONLY target labels must not move source_probe metrics or the set
    of selected_successful_models. Patch _load/_synthetic so the second run shuffles target labels only."""
    cands = ["eegnet", "shallow_convnet"]
    base = _run_summary(cands, monkeypatch)
    clean_src = {c: base["candidates"][c]["source_probe_bacc"] for c in cands}
    clean_sel = base["selected_successful_models"]

    orig = R._synthetic_fold

    def corrupted(seed, **kw):
        X, y, d, trm, tem, ncls, tgt = orig(seed, **kw)
        rng = np.random.default_rng(123)
        y = y.copy(); y[tem] = rng.integers(0, ncls, size=int(tem.sum()))   # corrupt TARGET labels only
        return X, y, d, trm, tem, ncls, tgt

    monkeypatch.setattr(R, "_synthetic_fold", corrupted)
    corr = _run_summary(cands, monkeypatch)
    for c in cands:
        assert abs(corr["candidates"][c]["source_probe_bacc"] - clean_src[c]) < 1e-9, c
    assert corr["selected_successful_models"] == clean_sel
    assert corr["success_selection_uses_target_eval"] is False


def test_success_selection_uses_source_probe_only(monkeypatch):
    summary = _run_summary(["eegnet", "graphcmi_current_ref"], monkeypatch)
    floor = summary["meta"]["success_bacc_floor"]
    expected = [c for c in ["eegnet", "graphcmi_current_ref"]
                if (summary["candidates"][c]["source_probe_bacc"] or 0.0) >= floor]
    assert summary["selected_successful_models"] == expected
    assert summary["meta"]["used_target_labels_for_selection"] is False
    assert summary["meta"]["target_eval_is_evaluation_only"] is True


def test_dry_run_synthetic_cpu_end_to_end(monkeypatch):
    summary = _run_summary(["eegnet"], monkeypatch)
    assert summary["meta"]["phase"] == "Phase3A_S_backbone_sanity"
    assert summary["meta"]["setting"] == "strict_source_only_DG"
    assert "eegnet" in summary["candidates"]
