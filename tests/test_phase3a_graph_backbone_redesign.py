"""Tests for CIGL Phase 3A-G graph-backbone redesign (scientific firewalls + graph-usage).

Covers: CPU dry-run; the candidate list is exactly the small named set; each candidate exposes
forward_graph with valid graph_z/node_z dims; the static adapter is marked edge_logits_dynamic=false;
corrupting only target labels changes neither source_probe nor the selected graph backbones (source-only
firewall); target_eval is evaluation-only; selection never uses target_eval; the graph-usage check yields
a finite source_probe delta.
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
torch.set_num_threads(1)

import scripts.run_cigl_phase3a_graph_backbone_redesign as R          # noqa: E402
from cmi.models.graph_task_backbones import build_graph_task_backbone, GRAPH_TASK_BACKBONES  # noqa: E402


def _args(**kw):
    import argparse
    base = dict(dry_run_synthetic=True, dataset="synthetic", fold=0, device="cpu", seeds=[0, 1], epochs=3,
                bs=64, probe_epochs=5, leak_n_perm=5, train_frac=0.7, enc_train_frac=0.7, min_per_cell=2,
                success_bacc_floor=0.45, min_seeds_pass=2, train_above_chance_margin=0.10,
                graph_usage_min_drop=0.10, nondegen_tol=1e-4, candidates=GRAPH_TASK_BACKBONES,
                tmin=0.5, tmax=3.5, resample=128)
    base.update(kw)
    return argparse.Namespace(**base)


def test_candidate_list_is_exactly_the_small_named_set():
    assert GRAPH_TASK_BACKBONES == ["shallow_graph_stem", "eegnet_graph_stem", "dgcnn_forward_graph_adapter"]
    assert R.GRAPH_TASK_BACKBONES == GRAPH_TASK_BACKBONES


def test_each_candidate_exposes_forward_graph_with_expected_dims():
    C, T, K = 22, 96, 4
    for name in GRAPH_TASK_BACKBONES:
        net = build_graph_task_backbone(name, C, T, K)
        assert hasattr(net, "forward_graph") and hasattr(net, "ablate") and hasattr(net, "meta")
        logits, gz, nz, el = net.forward_graph(torch.zeros(3, C, T))
        assert logits.shape == (3, K)
        assert gz.ndim == 2 and gz.shape[0] == 3
        assert nz.ndim == 3 and nz.shape[0] == 3 and nz.shape[1] == C       # node identity preserved
        assert net.meta["graph_compatible"] is True and net.meta["node_identity_preserved"] is True


def test_static_adapter_marked_non_dynamic_and_emits_no_edge():
    net = build_graph_task_backbone("dgcnn_forward_graph_adapter", 22, 96, 4)
    assert net.meta["edge_logits_dynamic"] is False
    _, _, _, el = net.forward_graph(torch.zeros(2, 22, 96))
    assert el is None                                                       # no faked dynamic edge object
    for name in ("shallow_graph_stem", "eegnet_graph_stem"):
        dyn = build_graph_task_backbone(name, 22, 96, 4)
        assert dyn.meta["edge_logits_dynamic"] is True
        _, _, _, e = dyn.forward_graph(torch.zeros(2, 22, 96))
        assert e is not None and e.shape == (2, 22, 22)


def test_graph_usage_check_zeroing_graph_collapses_task():
    """A task-capable model must lose its task accuracy when the graph readout is zeroed (no bypass)."""
    from cmi.train.trainer import train_model, predict
    from cmi.eval.metrics import classification_metrics
    rng = np.random.default_rng(0)
    proto = 2.5 * rng.standard_normal((4, 22, 96)).astype("float32")
    X, y = [], []
    for _ in range(320):
        c = rng.integers(0, 4); X.append(proto[c] + 0.6 * rng.standard_normal((22, 96)).astype("float32")); y.append(c)
    X = np.stack(X); y = np.array(y, "int64"); d = np.zeros(len(y), "int64")
    net = build_graph_task_backbone("shallow_graph_stem", 22, 96, 4)
    net, _, _ = train_model(net, X, y, d, 4, method="erm", epochs=12, bs=64, warmup=1, device="cpu", seed=0)
    full = classification_metrics(predict(net, X, "cpu"), y)["balanced_acc"]
    zero = R._ablation_bacc(net, X, y, "zero_graph", "cpu")
    perm = R._ablation_bacc(net, X, y, "permute_nodes", "cpu")
    assert full > 0.45 and (full - zero) >= 0.10                            # zeroing graph_z collapses task
    assert (full - perm) >= 0.05                                            # permuting node content also hurts


def _run_summary(cands, monkeypatch, epochs=3):
    argv = ["prog", "--dry_run_synthetic", "--device", "cpu", "--seeds", "0", "1", "--epochs", str(epochs),
            "--probe_epochs", "5", "--leak_n_perm", "0", "--candidates", *cands]
    monkeypatch.setattr(sys, "argv", argv)
    R.main()
    return json.load(open(R.OUT_DIR / "synthetic_fold0_graph_backbone_redesign_summary.json"))


def test_fold_is_reused_from_phase3a_s():
    """The source-only fold/split must be the SAME object imported from Phase 3A-S, not reimplemented."""
    import scripts.run_cigl_phase3a_backbone_sanity as S
    assert R._synthetic_fold is S._synthetic_fold
    assert R._load_real_fold is S._load_real_fold


def test_pass_criteria_fields_on_a_passing_dynamic_backbone(monkeypatch):
    """Run the dynamic-edge candidate that learns the dry-run task and assert the aggregate PASS-criteria
    fields are all enforced (source per-seed, train-above-chance, non-degenerate, graph path used)."""
    s = _run_summary(["eegnet_graph_stem"], monkeypatch, epochs=6)
    a = s["candidates"]["eegnet_graph_stem"]
    n_cls = s["meta"]["n_classes"]; chance = 1.0 / n_cls
    assert a["passes"] is True
    assert a["n_seeds_source_pass"] >= 2 and all(v >= s["meta"]["success_bacc_floor"] for v in a["source_probe_per_seed"])
    assert a["train_bacc"] >= chance + 0.10
    assert a["forward_graph_nondegenerate"] is True and a["forward_graph_valid"] is True
    assert a["graph_path_used"] is True
    assert s["dynamic_edge_backbone_succeeds"] is True


def test_target_eval_evaluation_only_and_selection_excludes_target(monkeypatch):
    s = _run_summary(["dgcnn_forward_graph_adapter"], monkeypatch)
    assert s["meta"]["target_eval_is_evaluation_only"] is True
    assert s["meta"]["graph_backbone_selection_uses_target_eval"] is False
    assert s["meta"]["used_target_labels_for_training"] is False and s["meta"]["used_target_labels_for_selection"] is False
    assert s["meta"]["used_target_covariates"] is False
    assert s["meta"]["cmi_regularization_used"] is False
    for c, a in s["candidates"].items():
        assert a["target_eval_is_evaluation_only"] is True


def test_target_label_corruption_changes_neither_source_nor_selection(monkeypatch):
    """Source-only firewall: corrupting ONLY target labels must not move source_probe or the set of
    selected_successful_graph_backbones."""
    cands = ["dgcnn_forward_graph_adapter"]
    base = _run_summary(cands, monkeypatch)
    clean_src = {c: base["candidates"][c]["source_probe_bacc"] for c in cands}
    clean_sel = base["selected_successful_graph_backbones"]

    orig = R._synthetic_fold

    def corrupted(seed, **kw):
        X, y, d, trm, tem, ncls, tgt = orig(seed, **kw)
        rng = np.random.default_rng(321)
        y = y.copy(); y[tem] = rng.integers(0, ncls, size=int(tem.sum()))    # corrupt TARGET labels only
        return X, y, d, trm, tem, ncls, tgt

    monkeypatch.setattr(R, "_synthetic_fold", corrupted)
    corr = _run_summary(cands, monkeypatch)
    for c in cands:
        assert abs(corr["candidates"][c]["source_probe_bacc"] - clean_src[c]) < 1e-9, c
    assert corr["selected_successful_graph_backbones"] == clean_sel


def test_dry_run_synthetic_cpu_end_to_end(monkeypatch):
    s = _run_summary(["dgcnn_forward_graph_adapter"], monkeypatch)
    assert s["meta"]["phase"] == "Phase3A_G_graph_backbone_redesign"
    a = s["candidates"]["dgcnn_forward_graph_adapter"]
    assert isinstance(a["zero_graph_drop"], float) and isinstance(a["permute_nodes_drop"], float)  # finite deltas
    assert a["forward_graph_valid"] is True and a["forward_graph_nondegenerate"] is True
    assert len(a["source_probe_per_seed"]) == 2 and isinstance(a["n_seeds_source_pass"], int)
    assert "leakage_kl" not in a                                            # static adapter: no edge leakage audit
    # the per-seed record documents WHY the audit was skipped for the static adapter
    seed0 = json.load(open(R.OUT_DIR / "synthetic_fold0_dgcnn_forward_graph_adapter_seed0.json"))
    assert seed0["leakage"] is None and "leakage_skipped_reason" in seed0
