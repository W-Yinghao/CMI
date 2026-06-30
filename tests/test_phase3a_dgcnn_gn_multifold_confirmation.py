"""Tests for CIGL Phase 3A-J fixed-config multi-fold confirmation (source-only; fold-0 = dev; no edge).

Covers: CPU dry-run; exactly two FIXED configs (erm_fixed, graph_node_010 at 0.010/0.010/0.000); fold-0 is
marked dev and excluded from the primary (folds 1-8) aggregate; reductions are paired by fold AND seed;
corrupting only target labels changes neither source metrics, leakage reductions, primary pass/fail, nor
any decision (target_eval only moves the reported guardrail); edge audit skipped/not faked; per-seed meta
carries firewall flags; the primary aggregate uses confirmation folds only.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_dgcnn_gn_multifold_confirmation as R          # noqa: E402


def test_exactly_two_fixed_configs_no_edge():
    assert [c[0] for c in R.FIXED_CONFIGS] == ["erm_fixed", "graph_node_010"]
    assert R.FIXED_CONFIGS[1] == ("graph_node_010", 0.010, 0.010)              # lambda_edge implicitly 0
    assert R.DEV_FOLD == 0


def test_decide_multifold_excludes_dev_fold_and_uses_confirmation_only():
    # 8 confirmation folds all passing -> A; dev fold (0) is NOT in the confirmation list
    def flag(ok):
        return dict(erm_adequate=ok, erm_leakage_exists=ok, reg_reduces=ok, source_retained=ok,
                    target_guardrail=ok, fold_pass=ok)
    pf = {f: flag(True) for f in range(1, 9)}
    pf[0] = flag(False)                                                         # dev fold fails; must be ignored
    d = R.decide_multifold(pf, confirmation_folds=[1, 2, 3, 4, 5, 6, 7, 8])
    assert d["n_confirmation_folds"] == 8 and d["need_strong"] == 6 and d["need_majority"] == 5
    assert d["confirmed"] is True and d["decision"] == "A"
    # erm inadequate in most folds -> D regardless of reduction
    pf2 = {f: dict(flag(True), erm_adequate=(f <= 2)) for f in range(1, 9)}
    assert R.decide_multifold(pf2, [1, 2, 3, 4, 5, 6, 7, 8])["decision"] == "D"
    # adequate+leakage but reduction only in 2/8 -> not confirmed (C), reduction below 5/8 majority
    pf3 = {f: dict(flag(True), reg_reduces=(f <= 2), source_retained=(f <= 2)) for f in range(1, 9)}
    assert R.decide_multifold(pf3, [1, 2, 3, 4, 5, 6, 7, 8])["decision"] == "C"


def _run_summary(monkeypatch, folds=("0", "1", "2")):
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--folds", *folds, "--seeds", "0", "1",
            "--epochs", "3", "--probe_epochs", "5", "--n_perm", "4"]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_dgcnn_gn_multifold_summary.json"))


def test_dry_run_cpu_and_dev_fold_excluded_from_primary(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["phase"] == "Phase3A_J_multifold_confirmation"
    assert s["meta"]["fold0_is_dev"] is True
    assert s["meta"]["primary_confirmation_folds"] == [1, 2, 3, 4, 5, 6, 7, 8]
    # primary aggregate counts confirmation folds only (here folds 1,2 -> n=2), excluding dev fold 0
    assert s["folds1_8_confirmation"]["n_confirmation_folds"] == 2
    assert s["all_folds_descriptive"]["n_confirmation_folds"] == 3        # descriptive includes fold 0
    assert s["per_fold"]["0"]["is_dev_fold"] is True
    assert s["fold0_dev"] is not None


def test_edge_skipped_not_faked_and_firewall_meta(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["edge_regularization_used"] is False and s["meta"]["edge_audit_skipped"] is True
    assert "static" in s["edge_skip_reason"].lower()
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold1_graph_node_010_seed0.json"))
    assert set(seed0["leakage"].keys()) == {"graph", "node"} and seed0["edge_audit_skipped"] is True
    m = seed0["meta"]
    for k in ("used_target_labels_for_training", "used_target_labels_for_selection", "used_target_covariates",
              "selection_uses_target_eval", "confirmation_label_selection_uses_target_eval",
              "edge_regularization_used"):
        assert m[k] is False, k
    assert m["target_eval_is_evaluation_only"] is True and m["fixed_candidate"] == "graph_node_010"
    assert m["cmi_regularization_used"] is True                            # graph_node_010 uses CMI reg
    # per-config provenance: erm_fixed records must carry cmi_regularization_used=False
    erm0 = json.load(open(R.OUT_DIR / "synthetic_fold1_erm_fixed_seed0.json"))
    assert erm0["meta"]["cmi_regularization_used"] is False


def test_target_corruption_changes_neither_source_nor_primary_decision(monkeypatch):
    base = _run_summary(monkeypatch)
    clean = {(f, c): base["per_fold"][f][c]["source_probe_bacc"]
             for f in base["per_fold"] for c in ("erm_fixed", "graph_node_010")}
    clean_primary = base["folds1_8_confirmation"]["decision"]
    clean_reductions = {f: base["per_fold"][f]["flags"]["graph_reduction"] for f in base["per_fold"]}

    orig = R._synthetic_folds

    def corrupted(folds, **kw):
        out = orig(folds, **kw)
        rng = np.random.default_rng(555)
        new = {}
        for f, (X, y, d, trm, tem, ncls, tgt) in out.items():
            y2 = y.copy(); y2[tem] = rng.integers(0, ncls, size=int(tem.sum()))   # corrupt TARGET labels only
            new[f] = (X, y2, d, trm, tem, ncls, tgt)
        return new

    monkeypatch.setattr(R, "_synthetic_folds", corrupted)
    corr = _run_summary(monkeypatch)
    for k, v in clean.items():
        f, c = k
        assert abs(corr["per_fold"][f][c]["source_probe_bacc"] - v) < 1e-9, k
    for f, gr in clean_reductions.items():
        cg = corr["per_fold"][f]["flags"]["graph_reduction"]
        assert (gr is None and cg is None) or abs(cg - gr) < 1e-9, f       # source-side leakage unchanged
    assert corr["folds1_8_confirmation"]["decision"] == clean_primary       # primary decision invariant
    # sanity: the corruption DID take effect (target_eval responds), so the invariance above is meaningful
    assert any(abs(corr["per_fold"][f][c]["target_eval_bacc"] - base["per_fold"][f][c]["target_eval_bacc"]) > 1e-6
               for f in base["per_fold"] for c in ("erm_fixed", "graph_node_010"))
