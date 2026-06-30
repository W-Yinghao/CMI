"""Tests for CIGL Phase 3A-H DGCNN graph/node leakage audit (diagnostic; edge skipped, source-only).

Covers: CPU dry-run; the EDGE audit is skipped and not faked (no edge block; edge_logits_dynamic=false);
the summary has graph + node blocks but no edge leakage block; corrupting only target labels changes
neither source_probe nor the graph/node leakage selection; target_eval is evaluation-only; graph-usage
deltas are finite; n_perm is recorded; node-map stability summary exists for multiple seeds.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_dgcnn_leakage_audit as R          # noqa: E402


def _run_summary(monkeypatch, seeds=("0", "1")):
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--seeds", *seeds, "--epochs", "3",
            "--probe_epochs", "5", "--n_perm", "5"]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_fold0_dgcnn_leakage_audit_summary.json"))


def test_only_candidate_is_static_dgcnn_adapter():
    assert R.CANDIDATE == "dgcnn_forward_graph_adapter"


def test_dry_run_synthetic_cpu_end_to_end(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["phase"] == "Phase3A_H_dgcnn_leakage_audit"
    assert s["meta"]["setting"] == "strict_source_only_DG"
    assert s["meta"]["n_perm"] == 5                                    # n_perm recorded
    assert isinstance(s["graph_usage"]["zero_graph_drop"], float)      # finite graph-usage deltas
    assert isinstance(s["graph_usage"]["permute_nodes_drop"], float)


def test_edge_audit_is_skipped_and_not_faked(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["edge_audit_skipped"] is True
    assert s["meta"]["edge_logits_dynamic"] is False
    assert "edge" not in s["leakage"]                                 # graph + node only, no edge block
    assert set(s["leakage"].keys()) == {"graph", "node"}
    assert "edge_skip_reason" in s and "static" in s["edge_skip_reason"].lower()
    # per-seed record also carries no edge block and marks the skip
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold0_dgcnn_forward_graph_adapter_seed0.json"))
    assert seed0["edge_audit_skipped"] is True and "edge" not in seed0["leakage"]
    assert seed0["meta_arch"]["edge_logits_dynamic"] is False


def test_summary_has_graph_and_node_blocks(monkeypatch):
    s = _run_summary(monkeypatch)
    for o in ("graph", "node"):
        assert "kl_mean" in s["leakage"][o] and "clears_null_seeds" in s["leakage"][o]
    assert "graph_leakage_exists" in s and "node_leakage_claimed" in s and "node_leakage_signal" in s
    # permutation null is non-trivial: per-seed permutation_mean is recorded and finite
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold0_dgcnn_forward_graph_adapter_seed0.json"))
    import math
    assert math.isfinite(seed0["leakage"]["graph"]["permutation_mean"])
    assert math.isfinite(seed0["leakage"]["node"]["permutation_mean"])


def test_decide_leakage_degenerate_node_is_not_a_claim():
    """A partially-degenerate node map that scores above_random must NOT clear the node claim
    (CIGL_24 criterion 5), but a graph claim is independent of node degeneracy."""
    degen_above = dict(above_random=True, degenerate=True)
    d = R.decide_leakage(graph_clears=0, node_clears=3, node_stab=degen_above,
                         task_ok=True, graph_path_used=True, min_seeds_pass=2)
    assert d["node_leakage_signal"] is True            # per-seed null clearance present
    assert d["node_map_stable"] is False               # degenerate -> not stable
    assert d["node_leakage_claimed"] is False          # so NO node claim (false-positive prevented)
    assert d["leakage_exists"] is False and d["audit_passes"] is False
    # clean non-degenerate, above-random node map -> node claim holds
    clean = dict(above_random=True, degenerate=False)
    d2 = R.decide_leakage(0, 3, clean, True, True, 2)
    assert d2["node_leakage_claimed"] is True and d2["leakage_exists"] is True and d2["audit_passes"] is True
    # graph clearing is independent of node degeneracy
    d3 = R.decide_leakage(2, 0, degen_above, True, True, 2)
    assert d3["graph_leakage_exists"] is True and d3["leakage_exists"] is True
    # graph path NOT used -> audit fails even if leakage exists
    d4 = R.decide_leakage(2, 0, clean, True, False, 2)
    assert d4["leakage_exists"] is True and d4["audit_passes"] is False


def test_node_map_stability_exists_for_multiple_seeds(monkeypatch):
    import math
    s = _run_summary(monkeypatch, seeds=("0", "1"))
    st = s["node_map_stability"]
    assert "mean_corr" in st and "null_q95" in st and "above_random" in st and "degenerate" in st
    assert st["mean_corr"] is not None and math.isfinite(st["mean_corr"])   # >=2 seeds -> finite corr
    assert isinstance(st["degenerate"], bool) and isinstance(st["above_random"], bool)


def test_target_eval_is_evaluation_only(monkeypatch):
    s = _run_summary(monkeypatch)
    assert s["meta"]["target_eval_is_evaluation_only"] is True
    assert s["meta"]["used_target_labels_for_training"] is False
    assert s["meta"]["used_target_labels_for_selection"] is False
    assert s["meta"]["used_target_covariates"] is False
    assert s["meta"]["cmi_regularization_used"] is False


def test_target_label_corruption_changes_nothing_source_or_leakage(monkeypatch):
    """Source-only firewall: corrupting ONLY target labels must not move source_probe, graph/node
    leakage kl, or the leakage-exists selection."""
    base = _run_summary(monkeypatch)
    clean = (base["task"]["source_probe_bacc"], base["leakage"]["graph"]["kl_mean"],
             base["leakage"]["node"]["kl_mean"], base["leakage_exists"])

    orig = R._synthetic_fold

    def corrupted(seed, **kw):
        X, y, d, trm, tem, ncls, tgt = orig(seed, **kw)
        rng = np.random.default_rng(999)
        y = y.copy(); y[tem] = rng.integers(0, ncls, size=int(tem.sum()))    # corrupt TARGET labels only
        return X, y, d, trm, tem, ncls, tgt

    monkeypatch.setattr(R, "_synthetic_fold", corrupted)
    corr = _run_summary(monkeypatch)
    assert abs(corr["task"]["source_probe_bacc"] - clean[0]) < 1e-9
    assert abs(corr["leakage"]["graph"]["kl_mean"] - clean[1]) < 1e-9
    assert abs(corr["leakage"]["node"]["kl_mean"] - clean[2]) < 1e-9
    assert corr["leakage_exists"] == clean[3]


def test_audit_graph_node_objects_skips_edge():
    """The graph/node-only audit wrapper returns graph+node and explicitly skips edge."""
    from cmi.eval.graph_leakage import audit_graph_node_objects
    rng = np.random.default_rng(0)
    N, C, Dg, Dn, K, M = 40, 6, 8, 4, 3, 2
    gz = rng.standard_normal((N, Dg)).astype("float32")
    nz = rng.standard_normal((N, C, Dn)).astype("float32")
    y = rng.integers(0, K, N); d = rng.integers(0, M, N)
    out = audit_graph_node_objects(gz, nz, y, d, K, M, n_perm=3, seed=0, epochs=5)
    assert set(out.keys()) >= {"graph", "node"} and "edge" not in out
    assert out["edge_audit_skipped"] is True
