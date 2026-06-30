"""Tests for CIGL Phase 3A-I DGCNN graph/node CMI regularizer pilot (source-only firewall; no edge).

Covers: CPU dry-run; the 8-config list is exact and edge-free; the edge audit is skipped/not faked;
corrupting only target labels changes neither source_probe, the source-only Pareto selection, the
confirmation labels, nor the audited objects; target_eval enters only the reported retention verdict;
confirmation labels are chosen source-only; ERM_fixed reproduction is recorded; graph-usage deltas are
finite; per-seed meta carries all firewall flags.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_dgcnn_gn_regularizer_pilot as R          # noqa: E402


def test_config_list_is_exact_eight_and_edge_free():
    names = [c[0] for c in R.CONFIGS]
    assert names == ["erm_fixed", "graph_001", "node_001", "graph_node_001",
                     "graph_003", "node_003", "graph_node_003", "graph_node_010"]
    # graph/node only: every config is (lambda_g, lambda_node); there is no edge lambda anywhere
    assert all(len(c) == 3 for c in R.CONFIGS)
    assert R.CONFIGS[0] == ("erm_fixed", 0.0, 0.0)


def test_decide_pilot_selection_is_source_only_firewall():
    """Corrupting ONLY target_eval must not change source_only_reducers / confirmation_labels /
    best_pareto, but MAY change the reported final_target_retaining_reducers."""
    def agg_entry(src, gkl, nkl, tgt, n_pass):
        return dict(source_probe_bacc=src, source_probe_per_seed=[src] * 3, n_seeds_source_pass=n_pass,
                    target_eval_bacc=tgt, graph_kl_per_seed=[gkl] * 3, node_kl_per_seed=[nkl] * 3)
    agg = {
        "erm_fixed": agg_entry(0.46, 1.0, 0.5, 0.44, 3),
        "graph_003": agg_entry(0.45, 0.5, 0.5, 0.43, 3),       # 50% graph reduction, source drop 0.01 -> reducer
        "graph_node_010": agg_entry(0.40, 0.2, 0.1, 0.41, 0),  # big reduction but source 0.40 -> NOT a reducer
    }
    base = R.decide_pilot_selection(agg)
    assert "graph_003" in base["source_only_reducers"]
    assert "graph_node_010" not in base["source_only_reducers"]      # source loss too large
    assert base["best_pareto"] == "graph_003"
    assert base["confirmation_labels"] == sorted({"erm_fixed", "graph_003"})

    # corrupt ONLY target_eval -> selection/confirmation invariant
    agg2 = {k: dict(v) for k, v in agg.items()}
    agg2["graph_003"] = dict(agg2["graph_003"], target_eval_bacc=0.10)   # tank target only
    corr = R.decide_pilot_selection(agg2)
    assert corr["source_only_reducers"] == base["source_only_reducers"]
    assert corr["confirmation_labels"] == base["confirmation_labels"]
    assert corr["best_pareto"] == base["best_pareto"]
    # the REPORTED target-retention verdict MAY change
    assert "graph_003" in base["final_target_retaining_reducers"]
    assert "graph_003" not in corr["final_target_retaining_reducers"]


def _run_summary(monkeypatch, configs=("erm_fixed", "graph_003")):
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--seeds", "0", "1", "--epochs", "3",
            "--probe_epochs", "5", "--n_perm", "4", "--n_perm_confirm", "4", "--configs", *configs]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_fold0_dgcnn_gn_pilot_summary.json"))


def test_dry_run_cpu_and_no_edge_term(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["phase"] == "Phase3A_I_dgcnn_gn_regularizer_pilot"
    assert s["meta"]["edge_regularization_used"] is False and s["meta"]["edge_audit_skipped"] is True
    assert "static" in s["edge_skip_reason"].lower()
    # no edge block in any per-config leakage; per-seed json carries the skip reason
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold0_graph_003_seed0.json"))
    assert set(seed0["leakage"].keys()) == {"graph", "node"} and seed0["edge_audit_skipped"] is True
    assert seed0["method"] == "graphcmi" and seed0["lambda_g"] == 0.003 and seed0["lambda_node"] == 0.0
    erm0 = json.load(open(R.OUT_DIR / "synthetic_fold0_erm_fixed_seed0.json"))
    assert erm0["method"] == "erm"


def test_graph_usage_and_reduction_fields_present(monkeypatch):
    s = _run_summary(monkeypatch)
    a = s["per_config"]["graph_003"]
    assert isinstance(a["zero_graph_drop"], float) and isinstance(a["permute_nodes_drop"], float)
    red = s["selection"]["reductions"]
    assert "graph_reduction_vs_erm" in red["graph_003"] and "node_reduction_vs_erm" in red["graph_003"]
    assert "erm_reproduces" in s["selection"]


def test_confirmation_labels_are_source_only_and_meta_flags(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["selection_uses_target_eval"] is False
    assert s["meta"]["confirmation_label_selection_uses_target_eval"] is False
    # confirmation block keys == the source-only confirmation labels
    assert set(s["confirmation"].keys()) == set(s["selection"]["confirmation_labels"])
    for label in ("erm_fixed", "graph_003"):
        seed0 = json.load(open(R.OUT_DIR / f"synthetic_fold0_{label}_seed0.json"))
        m = seed0["meta"]
        for k in ("used_target_labels_for_training", "used_target_labels_for_selection",
                  "used_target_covariates", "selection_uses_target_eval",
                  "confirmation_label_selection_uses_target_eval", "edge_regularization_used"):
            assert m[k] is False, k
        assert m["target_eval_is_evaluation_only"] is True


def test_target_corruption_changes_neither_source_nor_selection(monkeypatch):
    base = _run_summary(monkeypatch)
    clean_src = {c: base["per_config"][c]["source_probe_bacc"] for c in ("erm_fixed", "graph_003")}
    clean_sel = base["selection"]["confirmation_labels"]
    clean_red = base["selection"]["source_only_reducers"]

    orig = R._synthetic_fold

    def corrupted(seed, **kw):
        X, y, d, trm, tem, ncls, tgt = orig(seed, **kw)
        rng = np.random.default_rng(777)
        y = y.copy(); y[tem] = rng.integers(0, ncls, size=int(tem.sum()))    # corrupt TARGET labels only
        return X, y, d, trm, tem, ncls, tgt

    monkeypatch.setattr(R, "_synthetic_fold", corrupted)
    corr = _run_summary(monkeypatch)
    for c in ("erm_fixed", "graph_003"):
        assert abs(corr["per_config"][c]["source_probe_bacc"] - clean_src[c]) < 1e-9, c
    assert corr["selection"]["confirmation_labels"] == clean_sel
    assert corr["selection"]["source_only_reducers"] == clean_red
