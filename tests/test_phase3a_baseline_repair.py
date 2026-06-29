"""CIGL Phase 3A-R tests: baseline candidate list, gentle micro-ladder, strict target-label rule,
and the baseline-gate -> conditional-Part-B control flow (CPU, synthetic, no EEG data)."""
from __future__ import annotations
import importlib.util
import json
import subprocess
import sys
import types
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "run_cigl_phase3a_baseline_repair.py"
OUT = REPO / "results" / "cigl" / "phase3a_baseline_repair"
SUMMARY = OUT / "synthetic_fold0_baseline_repair_summary.json"
DRY = ["--dry_run_synthetic", "--device", "cpu", "--seeds", "0", "--epochs", "2",
       "--probe_epochs", "3", "--n_perm", "3", "--baseline_n_perm", "3", "--overfit_epochs", "6"]


def _mod():
    spec = importlib.util.spec_from_file_location("p3ar", SCRIPT)
    m = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(REPO)); spec.loader.exec_module(m)
    return m


# ----------------------------------------------------------------- pure-python config checks
def test_baseline_candidates_small_named_not_cartesian():
    m = _mod()
    cands = m.BASELINE_CANDIDATES
    names = [c["name"] for c in cands]
    assert names == ["current_default", "source_channel_zscore", "stronger_graphcmi_backbone",
                     "lower_lr_longer", "no_classbal_sampler", "ce_balance_check"]
    assert len(names) == len(set(names)) == 6, "candidate list must be a small named set (no duplicates)"
    # each candidate carries explicit named knobs (not a Cartesian product of axes)
    for c in cands:
        assert {"feat", "hidden", "hops", "lr", "epochs", "sampler", "balance", "chan_zscore"} <= set(c)


def test_gentle_configs_match_microladder():
    m = _mod()
    got = {lbl: (lg, ln, le) for (lbl, lg, ln, le) in m.GENTLE_CONFIGS}
    assert got == {
        "erm_fixed": (0.0, 0.0, 0.0), "graph_node_003": (0.003, 0.003, 0.0),
        "graph_node_01": (0.01, 0.01, 0.0), "graph_node_03": (0.03, 0.03, 0.0),
        "graph_only_01": (0.01, 0.0, 0.0), "node_only_01": (0.0, 0.01, 0.0),
        "edge_only_03": (0.0, 0.0, 0.03), "edge_only_10": (0.0, 0.0, 0.10),
        "full_01": (0.01, 0.01, 0.003), "full_03": (0.03, 0.03, 0.01)}


# ----------------------------------------------------------------- strict target-label invariance
def test_target_labels_do_not_affect_baseline_metrics_or_selection():
    """Corrupting ONLY target labels must leave train / source_probe / leakage byte-identical (so the
    baseline selection, which uses source_probe, cannot depend on target labels). Only target_eval may
    change."""
    m = _mod()
    X, y, dom, trm, tem, ncls, held = m._synthetic_fold(seed=0)
    args = types.SimpleNamespace(epochs=2, bs=64, enc_train_frac=0.7, train_frac=0.7, min_per_cell=2,
                                 probe_epochs=3, gate_alpha=0.05, dry_run_synthetic=True)
    cand = m.BASELINE_CANDIDATES[0]
    a = m._train_eval(cand, (0., 0., 0.), (X, y, dom, trm, tem, ncls, held), 0, args, "cpu", n_perm=3)
    y2 = y.copy()
    y2[tem] = np.random.default_rng(5).permutation(y2[tem])
    b = m._train_eval(cand, (0., 0., 0.), (X, y2, dom, trm, tem, ncls, held), 0, args, "cpu", n_perm=3)
    assert a["train"] == b["train"] and a["source_probe"] == b["source_probe"]
    for o in ("graph", "node", "edge"):
        assert a[o]["kl_mean"] == b[o]["kl_mean"]
    assert int(held) not in set(int(d) for d in dom[trm])   # target subject excluded from source


# ----------------------------------------------------------------- gate -> conditional Part B
def test_forced_fail_skips_part_b():
    r = subprocess.run([sys.executable, str(SCRIPT)] + DRY + ["--force_baseline_fail"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=900)
    assert r.returncode == 0, r.stderr[-1500:]
    s = json.load(open(SUMMARY))
    assert s["part_a"]["baseline_gate_pass"] is False
    assert s["part_b"] is None, "Part B (gentle regularization) must be skipped when the baseline gate fails"
    assert s["meta"]["used_target_labels_for_selection"] is False
    assert s["meta"]["target_eval_is_evaluation_only"] is True


def test_forced_pass_runs_part_b_gentle_microladder():
    r = subprocess.run([sys.executable, str(SCRIPT)] + DRY + ["--force_baseline_pass"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-1500:]
    s = json.load(open(SUMMARY))
    assert s["part_a"]["baseline_gate_pass"] is True
    pb = s["part_b"]
    assert pb is not None and "gentle" in pb and "gentle_gate_pass" in pb
    assert set(pb["gentle"]) == {lbl for (lbl, *_ ) in _mod().GENTLE_CONFIGS}
    for lbl, a in pb["gentle"].items():
        for o in ("graph", "node", "edge"):
            assert f"{o}_reduction_vs_erm" in a and f"{o}_reduce30_seeds" in a
        assert a["target_eval_is_evaluation_only"] is True
    assert pb["baseline"] in [c["name"] for c in _mod().BASELINE_CANDIDATES]


def test_part_a_candidate_records_are_named_and_evaluation_only():
    # reuse whichever summary the previous subprocess wrote
    if not SUMMARY.exists():
        subprocess.run([sys.executable, str(SCRIPT)] + DRY + ["--force_baseline_fail"],
                       cwd=str(REPO), check=True, timeout=900)
    s = json.load(open(SUMMARY))
    cands = s["part_a"]["candidates"]
    assert set(cands) == {"current_default", "source_channel_zscore", "stronger_graphcmi_backbone",
                          "lower_lr_longer", "no_classbal_sampler", "ce_balance_check"}
    for a in cands.values():
        assert a["target_eval_is_evaluation_only"] is True
        assert "source_probe_bacc" in a and "train_bacc" in a
