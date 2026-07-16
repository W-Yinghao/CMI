"""Relaxation Ladder Stage 9 — aggregation: specificity pairing, cluster unit, verdict determinism, loud fail."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("agg_rl", REPO / "scripts" / "aggregate_cmi_trace_relaxation_ladder.py")
AGG = importlib.util.module_from_spec(spec); sys.modules["agg_rl"] = AGG; spec.loader.exec_module(AGG)

L1 = "L1_STRICT_SOURCE_FRESH_HEAD"; L2 = "L2_TARGET_X_UNLABELED_FRESH_HEAD"; L3 = "L3_ORACLE_GLOBAL_GEOMETRY_FRESH_HEAD"


def _row(level, eraser, subj, seed, delta, draw=-1):
    return {"dataset": "D", "backbone": "bb", "feature_object": "z", "training_method": "erm",
            "heldout_subject": subj, "seed": seed, "fit_regime": level, "eraser": eraser,
            "random_draw_id": draw, "delta_bacc": delta}


# ---- 8: specificity pairs lw_leace and random_k per (fold, seed)
def test_specific_gain_pairs_per_cell():
    rows = []
    for subj in ("s1", "s2", "s3"):
        for seed in (0, 1):
            rows.append(_row(L1, "lw_leace_full", subj, seed, 0.05))
            for d in range(4):
                rows.append(_row(L1, "random_k", subj, seed, 0.01, draw=d))
    gains = AGG._specific_gain_clusters(rows, L1)             # per fold/subject
    # gain = mean_seed[ delta(lw) - mean_draw delta(random) ] = 0.05 - 0.01 = 0.04
    assert all(abs(v - 0.04) < 1e-9 for v in gains.values())
    assert set(gains) == {("D", "bb", "z", "erm", s) for s in ("s1", "s2", "s3")}


# ---- 9: cluster unit is outer subject/fold (not window / not draw)
def test_cluster_unit_is_fold_subject():
    rows = [_row(L1, "lw_leace_full", s, sd, 0.05) for s in ("s1", "s2") for sd in (0, 1)]
    cl = AGG._cell_delta(rows, L1, "lw_leace_full")
    assert set(cl) == {("D", "bb", "z", "erm", "s1"), ("D", "bb", "z", "erm", "s2")}  # 2 clusters, seeds merged


# ---- 10: verdict semantics are deterministic (each branch)
def _summ(mean, lo, hi):
    return {"mean": mean, "ci_lo": lo, "ci_hi": hi, "n_clusters": 9, "state": "x"}


def test_verdict_strict_fresh_positive():
    ls = {L1: {"lw_leace_full": _summ(0.03, 0.01, 0.05), "random_k": _summ(0.0, -0.01, 0.01)}}
    lg = {L1: _summ(0.03, 0.01, 0.05)}
    assert AGG.verdict(ls, lg) == "STRICT_FRESH_POSITIVE"


def test_verdict_transductive_positive():
    ls = {L1: {"lw_leace_full": _summ(0.0, -0.02, 0.02)}, L2: {"lw_leace_full": _summ(0.03, 0.01, 0.05)}}
    lg = {L1: _summ(0.0, -0.02, 0.02), L2: _summ(0.03, 0.01, 0.05)}
    assert AGG.verdict(ls, lg) == "TRANSDUCTIVE_POSITIVE"


def test_verdict_oracle_only_positive():
    ls = {L1: {"lw_leace_full": _summ(0.0, -0.02, 0.02)}, L2: {"lw_leace_full": _summ(0.0, -0.02, 0.02)},
          L3: {"lw_leace_full": _summ(0.03, 0.01, 0.05)}}
    lg = {L1: _summ(0.0, -0.02, 0.02), L2: _summ(0.0, -0.02, 0.02), L3: _summ(0.03, 0.01, 0.05)}
    assert AGG.verdict(ls, lg) == "ORACLE_ONLY_POSITIVE"


def test_verdict_generic_dimensionality():
    # LEACE and random both improve; specific gain CI includes 0
    ls = {L1: {"lw_leace_full": _summ(0.03, 0.005, 0.05), "random_k": _summ(0.028, 0.004, 0.05)}}
    lg = {L1: _summ(0.002, -0.01, 0.014)}
    assert AGG.verdict(ls, lg) == "GENERIC_DIMENSIONALITY_EFFECT"


def test_verdict_no_positive_regime():
    ls = {lv: {"lw_leace_full": _summ(-0.03, -0.05, -0.01), "random_k": _summ(-0.02, -0.04, 0.0)}
          for lv in (L1, L2, L3)}
    lg = {lv: _summ(-0.01, -0.03, -0.005) for lv in (L1, L2, L3)}      # gain never > 0
    assert AGG.verdict(ls, lg) == "NO_POSITIVE_REGIME"


def test_verdict_selective_positive_when_gate():
    ls = {L1: {"lw_leace_full": _summ(0.0, -0.02, 0.02)}}
    lg = {L1: _summ(0.0, -0.02, 0.02)}
    assert AGG.verdict(ls, lg, gate_positive=True) == "SELECTIVE_STRICT_POSITIVE"


def test_verdict_deterministic_repeat():
    ls = {L1: {"lw_leace_full": _summ(0.03, 0.01, 0.05)}}; lg = {L1: _summ(0.03, 0.01, 0.05)}
    assert AGG.verdict(ls, lg) == AGG.verdict(ls, lg) == "STRICT_FRESH_POSITIVE"


# ---- 11: missing results cause a loud failure (no fabricated table)
def test_aggregator_loud_fail_on_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["agg", "--raw", str(tmp_path / "nope_*.jsonl"), "--out", str(tmp_path)])
    with pytest.raises(SystemExit):
        AGG.main()


def test_figure_scripts_loud_fail_on_missing(tmp_path):
    # both figure/table generators must sys.exit (not fabricate) when the aggregation CSV is absent
    import subprocess
    for script in ("paper/cmi_trace/figures/make_fig3_cmi_reliance_bacc.py",):
        r = subprocess.run([sys.executable, str(REPO / script), "--paired", str(tmp_path / "absent.csv")],
                           capture_output=True, text=True)
        assert r.returncode != 0
