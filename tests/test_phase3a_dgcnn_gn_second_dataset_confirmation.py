"""Tests for CIGL Phase 3A-K fixed-config second-dataset confirmation (BNCI2015_001; binary; source-only).

Covers: CPU dry-run; exactly two FIXED configs (erm_fixed, graph_node_010 @0.010/0.010/0.000); default
dataset BNCI2015_001; binary thresholds (mean 0.60 / seed 0.55) recorded; n_classes/chance_bacc recorded;
a non-binary real dataset fails clearly before training; reductions paired by fold AND seed; corrupting
only target labels changes neither source metrics, leakage reductions, nor the primary decision (target
moves only the reported guardrail); edge skipped/not faked; firewall meta; aggregate uses all folds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_dgcnn_gn_second_dataset_confirmation as R          # noqa: E402


def test_exactly_two_fixed_configs_and_default_dataset():
    assert [c[0] for c in R.FIXED_CONFIGS] == ["erm_fixed", "graph_node_010"]
    assert R.FIXED_CONFIGS[1] == ("graph_node_010", 0.010, 0.010)
    assert R.DEFAULT_DATASET == "BNCI2015_001"


def test_decide_second_dataset_uses_all_folds_no_dev():
    def flag(ok):
        return dict(erm_adequate=ok, erm_leakage_exists=ok, reg_reduces=ok, source_retained=ok,
                    target_guardrail=ok, fold_pass=ok)
    pf = {f: flag(True) for f in range(12)}                              # 12 folds, all pass
    d = R.decide_second_dataset(pf, list(range(12)))
    assert d["n_folds"] == 12 and d["need_strong"] == 8 and d["need_majority"] == 7
    assert d["confirmed"] is True and d["decision"] == "A"
    # erm inadequate in most folds -> D
    pf2 = {f: dict(flag(True), erm_adequate=(f < 3)) for f in range(12)}
    assert R.decide_second_dataset(pf2, list(range(12)))["decision"] == "D"
    # adequate+leakage but reduction in only 3/12 -> not confirmed (C)
    pf3 = {f: dict(flag(True), reg_reduces=(f < 3), source_retained=(f < 3)) for f in range(12)}
    assert R.decide_second_dataset(pf3, list(range(12)))["decision"] == "C"


def _run_summary(monkeypatch, extra=()):
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--seeds", "0", "1", "--epochs", "3",
            "--probe_epochs", "5", "--n_perm", "4", *extra]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_dgcnn_gn_2nd_dataset_summary.json"))


def test_dry_run_binary_thresholds_and_chance_recorded(monkeypatch):
    s = _run_summary(monkeypatch)
    m = s["meta"]
    assert m["phase"] == "Phase3A_K_second_dataset_confirmation"
    assert m["second_dataset_confirmation"] is True
    assert m["n_classes"] == 2 and abs(m["chance_bacc"] - 0.5) < 1e-9
    assert m["source_mean_floor"] == 0.60 and m["source_seed_floor"] == 0.55
    assert s["second_dataset_confirmation"]["n_folds"] == 4        # synthetic default 4 folds, all confirmation


def test_non_binary_dataset_fails_before_training(monkeypatch):
    """If the loaded set is not binary, the runner must STOP before training (no silent threshold mismatch)."""
    def fake_load(args):
        rng = np.random.default_rng(0)
        X = rng.standard_normal((40, 8, 48)).astype("float32"); y = rng.integers(0, 4, 40).astype("int64")
        d = rng.integers(0, 4, 40).astype("int64")
        return {f: (X, y, d, d != f, d == f, 4, str(f)) for f in range(4)}, 4   # n_classes=4
    monkeypatch.setattr(R, "_load_real_all", fake_load)
    argv = ["prog", "--dataset", "BNCI2014_001", "--allow_non_default_dataset", "--device", "cpu",
            "--seeds", "0", "--epochs", "2", "--probe_epochs", "3", "--n_perm", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ei:
        R.main()
    assert "n_classes=4" in str(ei.value) and "reviewer approval" in str(ei.value).lower()


def test_expected_n_classes_bypass_arg_is_removed(monkeypatch):
    """The --expected_n_classes bypass must NOT exist; argparse should reject it (the guard keys on 2)."""
    monkeypatch.setattr(sys, "argv", ["prog", "--dry_run_synthetic", "--device", "cpu",
                                      "--expected_n_classes", "4"])
    with pytest.raises(SystemExit):                                    # argparse rejects the unknown arg
        R.main()


def test_non_binary_guard_keys_on_constant_2(monkeypatch):
    """A 4-class real set fails closed before training; the only escape is explicit --allow_non_binary."""
    def fake_load(args):
        rng = np.random.default_rng(1)
        X = rng.standard_normal((48, 8, 48)).astype("float32"); y = rng.integers(0, 4, 48).astype("int64")
        d = rng.integers(0, 4, 48).astype("int64")
        return {f: (X, y, d, d != f, d == f, 4, str(f)) for f in range(4)}, 4
    monkeypatch.setattr(R, "_load_real_all", fake_load)
    monkeypatch.setattr(sys, "argv", ["prog", "--dataset", "BNCI2014_001", "--allow_non_default_dataset",
                                      "--device", "cpu", "--seeds", "0", "--epochs", "2", "--probe_epochs", "3",
                                      "--n_perm", "2"])
    with pytest.raises(SystemExit) as ei:
        R.main()
    assert "n_classes=4" in str(ei.value) and "--allow_non_binary" in str(ei.value)


def test_non_default_dataset_requires_flag(monkeypatch):
    argv = ["prog", "--dataset", "Cho2017", "--device", "cpu", "--seeds", "0", "--epochs", "2"]
    monkeypatch.setattr(sys, "argv", argv)
    with pytest.raises(SystemExit) as ei:
        R.main()
    assert "allow_non_default_dataset" in str(ei.value)


def test_edge_skipped_not_faked_and_firewall_meta(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["edge_regularization_used"] is False and s["meta"]["edge_audit_skipped"] is True
    assert "static" in s["edge_skip_reason"].lower()
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold0_graph_node_010_seed0.json"))
    assert set(seed0["leakage"].keys()) == {"graph", "node"} and seed0["edge_audit_skipped"] is True
    m = seed0["meta"]
    for k in ("used_target_labels_for_training", "used_target_labels_for_selection", "used_target_covariates",
              "selection_uses_target_eval", "confirmation_label_selection_uses_target_eval",
              "edge_regularization_used"):
        assert m[k] is False, k
    assert m["target_eval_is_evaluation_only"] is True and m["fixed_candidate"] == "graph_node_010"
    erm0 = json.load(open(R.OUT_DIR / "synthetic_fold0_erm_fixed_seed0.json"))
    assert erm0["meta"]["cmi_regularization_used"] is False


def test_target_corruption_changes_neither_source_nor_primary_decision(monkeypatch):
    base = _run_summary(monkeypatch)
    clean = {(f, c): base["per_fold"][f][c]["source_probe_bacc"]
             for f in base["per_fold"] for c in ("erm_fixed", "graph_node_010")}
    clean_primary = base["second_dataset_confirmation"]["decision"]
    clean_red = {f: base["per_fold"][f]["flags"]["graph_reduction"] for f in base["per_fold"]}

    orig = R._synthetic_folds

    def corrupted(folds, **kw):
        out, ncls = orig(folds, **kw)
        rng = np.random.default_rng(444)
        new = {}
        for f, (X, y, d, trm, tem, nc, tgt) in out.items():
            y2 = y.copy(); y2[tem] = rng.integers(0, nc, size=int(tem.sum()))   # corrupt TARGET labels only
            new[f] = (X, y2, d, trm, tem, nc, tgt)
        return new, ncls

    monkeypatch.setattr(R, "_synthetic_folds", corrupted)
    corr = _run_summary(monkeypatch)
    for k, v in clean.items():
        f, c = k
        assert abs(corr["per_fold"][f][c]["source_probe_bacc"] - v) < 1e-9, k
    for f, gr in clean_red.items():
        cg = corr["per_fold"][f]["flags"]["graph_reduction"]
        assert (gr is None and cg is None) or abs(cg - gr) < 1e-9, f
    assert corr["second_dataset_confirmation"]["decision"] == clean_primary
    assert any(abs(corr["per_fold"][f][c]["target_eval_bacc"] - base["per_fold"][f][c]["target_eval_bacc"]) > 1e-6
               for f in base["per_fold"] for c in ("erm_fixed", "graph_node_010"))
