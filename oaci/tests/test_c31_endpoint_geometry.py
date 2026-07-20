"""C31 Endpoint-Axis / Accuracy-Calibration Geometry. Frozen C19 locked; endpoint labels vs the per-trajectory ERM
reference; overlap/conflict + epoch-confound control; source-rank endpoint specificity; gauge endpoint specificity
(general vs accuracy-specific); Pareto trajectory geometry; deterministic E1-E9 taxonomy that reports imbalance
first; report forbids selector / joint-deployable language. Synthetic rows encode a KNOWN geometry so the modules'
verdicts are checkable, plus a real-artifact smoke test that the audit runs read-only."""
from __future__ import annotations

import numpy as np

from oaci.endpoint_geometry import (artifact_loader, endpoint_labels, gauge_endpoint, overlap_conflict,
                                     pareto_geometry, report, schema, source_rank_endpoint, taxonomy)


def _synth(n_targets=8, n_per=16, seed=0, tradeoff=False):
    """Known geometry: a LARGE per-target GAUGE offset on every raw metric (so between-target variance dominates =>
    general gauge), a within-target competence latent `comp` INDEPENDENT of the epoch index (so the epoch control
    cannot strip the coupling), and — when tradeoff=False — comp improves bAcc AND calibration together. The ERM
    reference is the per-target min-competence candidate; deltas cancel the offset and track comp. score = offset +
    comp (source-visible within-target rank + gauge)."""
    rng = np.random.RandomState(seed); rows = []
    for t in range(1, n_targets + 1):
        oA, oN, oE = (t - n_targets / 2) * 0.15, (t - n_targets / 2) * 0.15, (t - n_targets / 2) * 0.05
        comps = rng.rand(n_per)                                     # competence latent, INDEPENDENT of k/epoch
        cmin = comps.min()
        erm_b = 0.5 + oA + cmin * 0.10; erm_n = 1.0 - oN - cmin * 0.20; erm_e = 0.2 - oE - cmin * 0.05
        sign = -1.0 if tradeoff else 1.0                           # tradeoff => calibration worsens as accuracy improves
        for k in range(n_per):
            comp = comps[k]
            bacc = 0.5 + oA + comp * 0.10 + rng.randn() * 0.01
            nll = 1.0 - oN - sign * comp * 0.20 + rng.randn() * 0.01
            ece = 0.2 - oE - sign * comp * 0.05 + rng.randn() * 0.005
            rows.append({"mode": "in_regime", "seed": 0, "target": t, "level": 0, "model_hash": f"{t:02d}{k:03d}",
                         "score": oA + comp + rng.randn() * 0.05, "R_src": -comp + rng.randn() * 0.05,
                         "label": int(comp > cmin), "epoch": 10 + k, "order": k,
                         "bacc": bacc, "nll": nll, "ece": ece, "erm_bacc": erm_b, "erm_nll": erm_n, "erm_ece": erm_e})
    return rows


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_endpoint_labels_and_base_rates():
    rows = endpoint_labels.attach_labels(_synth(), margin=schema.IMPROVE_MARGIN)
    br = endpoint_labels.base_rates(rows)
    assert br["n_candidates"] > 0
    for lab in ("accuracy_good", "calibration_good", "joint_good", "pareto_good"):
        assert 0.0 <= br[lab]["rate"] <= 1.0
    # in the no-tradeoff world accuracy-good checkpoints are mostly calibration-good too
    assert br["frac_accuracy_good_also_calibration_good"] > 0.6


def test_no_tradeoff_when_metrics_coimprove():
    rows = endpoint_labels.attach_labels(_synth(tradeoff=False), margin=schema.IMPROVE_MARGIN)
    o = overlap_conflict.overlap_conflict(rows)
    assert o["tradeoff_confirmed"] is False
    assert o["mean_bacc_vs_calib_improve_corr"] > 0                 # positive coupling
    assert o["coupling_survives_epoch_control"] is True            # survives epoch residualization


def test_tradeoff_detected_when_constructed():
    rows = endpoint_labels.attach_labels(_synth(tradeoff=True), margin=schema.IMPROVE_MARGIN)
    o = overlap_conflict.overlap_conflict(rows)
    assert o["tradeoff_confirmed"] is True                          # negative coupling detected


def test_source_rank_endpoint_specificity_defined():
    rows = endpoint_labels.attach_labels(_synth(), margin=schema.IMPROVE_MARGIN)
    s = source_rank_endpoint.source_rank_endpoint(rows)
    assert not (s["source_rank_accuracy_specific"] and s["source_rank_calibration_biased"])
    assert s["per_factor"]["score"]["joint_good"]["within_target_auc"] is not None


def test_source_rank_downgrade_machinery():
    # RED-TEAM: accuracy-specific must be gated on a cluster-bootstrap CI that excludes 0, and a by-construction check
    rows = endpoint_labels.attach_labels(_synth(), margin=schema.IMPROVE_MARGIN)
    s = source_rank_endpoint.source_rank_endpoint(rows)
    for k in ("accuracy_vs_calibration_gap_ci", "accuracy_vs_ece_gap_ci", "accuracy_aligned_by_construction",
              "label_accuracy_good_mismatches"):
        assert k in s
    gc = s["accuracy_vs_calibration_gap_ci"]
    assert {"gap", "ci_lo", "ci_hi", "excludes_zero", "frac_positive"} <= set(gc)
    # accuracy_specific is TRUE only if the gap CI excludes 0 (consistency of the gate)
    assert s["source_rank_accuracy_specific"] == bool(gc["excludes_zero"] and (gc["gap"] or 0) > 0)


def test_e1_robustness_shared_erm_inert():
    rows = endpoint_labels.attach_labels(_synth(tradeoff=False), margin=schema.IMPROVE_MARGIN)
    rob = overlap_conflict.overlap_conflict(rows)["e1_robustness"]
    # ERM reference is a per-trajectory constant => provably inert under residualization
    assert rob["erm_constant_within_trajectory"] is True
    assert rob["n_targets_positive"] == rob["n_targets"]           # all targets co-improve


def test_gauge_general_when_offset_on_all_metrics():
    rows = endpoint_labels.attach_labels(_synth(), margin=schema.IMPROVE_MARGIN)
    g = gauge_endpoint.gauge_endpoint(rows)
    # gauge intercept was put on all three metrics => general per-target offset, not accuracy-specific
    assert g["gauge_general_endpoint_offset"] is True and g["gauge_accuracy_specific"] is False


def test_pareto_geometry_no_wall_when_coimprove():
    rows = endpoint_labels.attach_labels(_synth(tradeoff=False), margin=schema.IMPROVE_MARGIN)
    p = pareto_geometry.pareto_geometry(rows)
    assert p["n_trajectories"] > 0
    assert p["accuracy_oracle_calibration_bad_fraction"] < 0.5      # accuracy oracle is calibration-good


def test_taxonomy_reports_imbalance_and_is_deterministic():
    rows = endpoint_labels.attach_labels(_synth(), margin=schema.IMPROVE_MARGIN)
    r = report._analyze(rows)
    tax = r["taxonomy"]
    assert "imbalance_flags" in tax and isinstance(tax["cases"], list) and tax["cases"]
    assert schema.E1 not in tax["cases"]                           # no tradeoff in the co-improve world


def test_report_forbids_selector_and_joint_deployable_language():
    for bad in ("we deploy an endpoint selector", "a pareto selector that picks joint", "joint deployable improvement here"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("NOT a pareto selector; no joint deployable improvement is claimed; diagnostic-only.")
