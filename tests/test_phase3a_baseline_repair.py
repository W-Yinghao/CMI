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
import pytest  # noqa: E402


@pytest.mark.parametrize("cand_idx", [0, 1])   # 0=current_default (no zscore), 1=source_channel_zscore
def test_target_labels_do_not_affect_baseline_metrics_or_selection(cand_idx):
    """Corrupting ONLY target labels must leave train / source_probe / leakage byte-identical (so the
    baseline selection, which uses source_probe, cannot depend on target labels) — exercised for both the
    no-zscore and the source-channel-zscore candidate. Only target_eval may change."""
    m = _mod()
    X, y, dom, trm, tem, ncls, held = m._synthetic_fold(seed=0)
    args = types.SimpleNamespace(epochs=2, bs=64, enc_train_frac=0.7, train_frac=0.7, min_per_cell=2,
                                 probe_epochs=3, gate_alpha=0.05, dry_run_synthetic=True)
    cand = m.BASELINE_CANDIDATES[cand_idx]
    a = m._train_eval(cand, (0., 0., 0.), (X, y, dom, trm, tem, ncls, held), 0, args, "cpu", n_perm=3)
    y2 = y.copy()
    y2[tem] = np.random.default_rng(5).permutation(y2[tem])
    assert not np.array_equal(y2[tem], y[tem]), "potency: target labels must actually be corrupted"
    b = m._train_eval(cand, (0., 0., 0.), (X, y2, dom, trm, tem, ncls, held), 0, args, "cpu", n_perm=3)
    assert a["train"] == b["train"] and a["source_probe"] == b["source_probe"]
    for o in ("graph", "node", "edge"):
        assert a[o]["kl_mean"] == b[o]["kl_mean"]
    assert int(held) not in set(int(d) for d in dom[trm])   # target subject excluded from source


def test_channel_zscore_is_source_ref_only():
    """Per-channel z-score must be fitted on the SOURCE reference array ONLY; target covariates cannot
    enter the fit. Standardization uses X_ref's per-channel mean/std broadcast over [1,C,1]."""
    m = _mod()
    rng = np.random.default_rng(0)
    Xref = rng.standard_normal((10, 6, 32)).astype("float32")
    Xt = rng.standard_normal((5, 6, 32)).astype("float32")
    zr, zt = m._channel_zscore(Xref, Xref, Xt)
    mu = Xref.mean(axis=(0, 2)); sd = Xref.std(axis=(0, 2)) + 1e-7
    assert np.allclose(zt, (Xt - mu[None, :, None]) / sd[None, :, None], atol=1e-5)
    # perturbing ONLY the non-reference (target) array must not change the reference's normalized output
    zr2, _ = m._channel_zscore(Xref, Xref, Xt + 99.0)
    assert np.allclose(zr, zr2), "target stats must not enter the source-fitted z-score"


def test_baseline_gate_decision_logic():
    """The REAL gate logic (no force flags): a candidate passes on source_probe floor OR
    current_default+gain; the gate needs a passing candidate AND controls_ok; best is highest-source."""
    m = _mod()
    def C(src): return {"source_probe_bacc": src}
    floor, gain = 0.45, 0.10
    # all degenerate (near chance) -> no candidate passes -> gate FAILS even with controls ok
    degen = {"current_default": C(0.33), "a": C(0.34), "b": C(0.30)}
    passing, best, gate = m.decide_baseline_gate(degen, True, floor, gain)
    assert passing == [] and best is None and gate is False
    # one clears the floor -> passes; gate needs controls too
    ok = {"current_default": C(0.33), "a": C(0.50), "b": C(0.34)}
    passing, best, gate = m.decide_baseline_gate(ok, True, floor, gain)
    assert "a" in passing and best == "a" and gate is True
    passing, best, gate = m.decide_baseline_gate(ok, False, floor, gain)   # controls fail -> gate fails
    assert best == "a" and gate is False
    # gain path: current_default low, candidate beats it by >= gain (but below floor)
    gainc = {"current_default": C(0.33), "a": C(0.44)}
    passing, best, gate = m.decide_baseline_gate(gainc, True, floor, gain)
    assert "a" in passing and gate is True
    # controls_ok one-sided shuffle: at/below chance both pass; above chance fails
    assert m.controls_ok(0.9, 0.0, 0.333) is True and m.controls_ok(0.9, 0.333, 0.333) is True
    assert m.controls_ok(0.9, 0.6, 0.333) is False and m.controls_ok(0.4, 0.0, 0.333) is False


def test_gentle_selection_firewall_target_cannot_change_confirmation():
    """SELECTION FIREWALL: target_eval (target_drop_vs_erm) must NOT change source_only_reducers,
    best_reducer, or confirmation_labels; it may change ONLY the final reported verdict."""
    m = _mod()

    def cfg(g30, n30, s_drop, t_drop, red=0.9):
        return dict(graph_reduce30_seeds=g30, node_reduce30_seeds=n30, source_drop_vs_erm=s_drop,
                    target_drop_vs_erm=t_drop, graph_reduction_vs_erm=red, node_reduction_vs_erm=red)
    base = {"erm_fixed": cfg(0, 0, 0.0, 0.0, 0.0),
            "graph_node_01": cfg(3, 3, 0.02, 0.02),     # source-ok AND target-ok
            "graph_node_03": cfg(3, 3, 0.02, 0.20),     # source-ok but target-BAD
            "node_only_01": cfg(0, 3, 0.10, 0.0),       # graph/node-capable but source-task drop too big
            "edge_only_10": cfg(0, 0, 0.01, 0.01)}      # not graph/node-capable
    s1 = m.decide_gentle_selection(base)
    # corrupt ONLY target_drop of every config (simulating arbitrary target-label corruption)
    corrupt = {k: {**v, "target_drop_vs_erm": v["target_drop_vs_erm"] + 0.50} for k, v in base.items()}
    s2 = m.decide_gentle_selection(corrupt)
    # source-only selection is INVARIANT to target corruption
    assert s1["source_only_reducers"] == s2["source_only_reducers"] == ["graph_node_01", "graph_node_03"]
    assert s1["confirmation_labels"] == s2["confirmation_labels"] == ["erm_fixed", "graph_node_01", "graph_node_03"]
    assert s1["best_reducer"] == s2["best_reducer"]
    assert s1["gentle_gate_pass_source_only"] == s2["gentle_gate_pass_source_only"] is True
    # the FINAL target-retention verdict CAN change under target corruption (verdict only)
    assert s1["final_task_preserving_reducers"] == ["graph_node_01"]
    assert s2["final_task_preserving_reducers"] == []
    assert s1["gentle_gate_pass_with_target_retention"] is True and s2["gentle_gate_pass_with_target_retention"] is False


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
        assert "source_drop_vs_erm" in a and "target_drop_vs_erm" in a
        assert a["target_eval_is_evaluation_only"] is True
    assert pb["baseline"] in [c["name"] for c in _mod().BASELINE_CANDIDATES]
    # SELECTION FIREWALL fields: confirmation chosen source-only; target_eval is verdict-only
    assert pb["confirmation_label_selection_uses_target_eval"] is False
    assert pb["target_eval_used_for_verdict_only"] is True
    assert "source_only_reducers" in pb and "final_task_preserving_reducers" in pb
    assert set(pb["confirmation_labels"]) == set({"erm_fixed"}) | set(pb["source_only_reducers"]) | (
        {pb["best_reducer"]} if pb["best_reducer"] else set())
    # final verdict is a subset of the source-only reducers (target only filters, never adds)
    assert set(pb["final_task_preserving_reducers"]) <= set(pb["source_only_reducers"])
    # confirmation re-audit (n_perm_confirm) covers the source-only-chosen labels, with per-seed records
    assert "erm_fixed" in pb["confirmation"] and "erm_fixed" in pb["confirmation_per_seed"]
    assert set(pb["confirmation_per_seed"]) == set(pb["confirmation_labels"])
    for lbl, by_seed in pb["confirmation_per_seed"].items():
        for seed, blocks in by_seed.items():
            for o in ("graph", "node", "edge"):
                for k in ("kl_mean", "permutation_p", "clears_null"):
                    assert k in blocks[o]


def test_natural_path_gate_passes_on_synthetic():
    """Real gate path (NO force flags): with enough epochs the strong-signal synthetic clears the
    source-probe floor and the controls behave -> baseline_gate_pass True, best_baseline named."""
    r = subprocess.run([sys.executable, str(SCRIPT), "--dry_run_synthetic", "--device", "cpu",
                        "--seeds", "0", "--epochs", "25", "--probe_epochs", "5", "--n_perm", "3",
                        "--baseline_n_perm", "3", "--overfit_epochs", "25", "--skip_part_b"],
                       cwd=str(REPO), capture_output=True, text=True, timeout=1200)
    assert r.returncode == 0, r.stderr[-1500:]
    s = json.load(open(SUMMARY))
    pa = s["part_a"]
    assert pa["baseline_gate_pass"] is True, f"natural gate should pass at 25 epochs; controls={pa['controls']}"
    assert pa["controls"]["controls_ok"] is True
    assert pa["controls"]["overfit_small_source_train_bacc"] > s["meta"]["chance"] + 0.15
    assert pa["controls"]["label_shuffle_control_src_bacc"] < s["meta"]["chance"] + 0.10
    assert pa["best_baseline"] in [c["name"] for c in _mod().BASELINE_CANDIDATES]
    assert s["part_b"] is None   # --skip_part_b


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
