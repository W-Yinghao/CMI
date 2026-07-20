"""C30 Rank-Gauge Separation. Frozen C19 locked; the within-target RANK axis is orthogonal to the per-target
GAUGE axis; source families' direction-agnostic rank strength attributes the rank; residualization checks the
rank survives gauge control; C19 is attributed to the rank axis (not a target-free detector); taxonomy
deterministic; report forbids selector language. Synthetic score rows only."""
from __future__ import annotations

import numpy as np

from oaci.rank_gauge import (artifact_loader, c19_signal_attribution, rank_gauge_decomposition, report, residualization,
                            schema, source_error_alignment, source_rank_family, taxonomy)


def _synth(n_targets=9, n_per=14, seed=0):
    rng = np.random.RandomState(seed); rows = []
    allfeat = [f for feats in schema.SOURCE_FAMILIES.values() for f in feats]
    for t in range(1, n_targets + 1):
        gauge = (t - n_targets / 2) * 0.6                    # per-target intercept (gauge axis)
        for k in range(n_per):
            good = k % 2 == 0
            rank = (0.5 if good else -0.5)                   # within-target competence rank
            score = gauge + rank + rng.randn() * 0.15
            row = {"mode": "in_regime", "regime": "S0", "seed": 0, "target": t, "level": 0, "model_hash": f"{t:02d}{k:03d}",
                   "score": score, "label": 1 if good else 0, "R_src": -rank + rng.randn() * 0.05,
                   "train_surrogate": rng.randn(), "epoch": 10 + k, "order": k}
            for f in allfeat:
                if not f.startswith("feat__"):
                    continue                                  # R_src / train_surrogate already set above
                if f == "feat__source_guard_ece":
                    row[f] = -rank * 0.6 + rng.randn() * 0.3  # calibration weakly tracks rank
                elif "leakage" in f:
                    row[f] = rng.randn()                       # leakage = noise (not a rank carrier)
                else:
                    row[f] = rng.randn() * 0.5
            rows.append(row)
    return rows


def test_config_hash_unchanged():
    assert report._lock_config() == schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"


def test_rank_gauge_two_axis_separation():
    rg = rank_gauge_decomposition.rank_gauge_decomposition(_synth(), "in_regime")
    assert rg["score_within_target_auc"] > rg["score_pooled_auc"]      # rank works, pooled broken by gauge
    assert abs(rg["rank_gauge_orthogonality"]) < 0.2                   # axes orthogonal
    assert rg["two_axis_separation"] is True


def test_source_risk_carries_rank():
    srf = source_rank_family.source_rank_family(_synth(), "in_regime")
    assert srf["families"]["source_risk"]["carries_rank"] is True       # R_src encodes -rank
    assert srf["families"]["source_leakage"]["best_rank_strength"] < srf["families"]["source_risk"]["best_rank_strength"]


def test_within_target_auc_direction_agnostic_strength():
    rows = _synth()
    wt = artifact_loader.within_target_auc(rows, "R_src", "in_regime")
    assert artifact_loader.rank_strength(wt) == abs(wt - 0.5) and 0 <= artifact_loader.rank_strength(wt) <= 0.5


def test_residualization_rank_survives_or_not():
    rd = residualization.residualization(_synth(), "in_regime")
    assert "rank_survives_R_src_control" in rd and "gauge_contaminates_rank" in rd


def test_c19_attributed_to_rank_not_selector():
    rg = rank_gauge_decomposition.rank_gauge_decomposition(_synth(), "in_regime")
    c19 = c19_signal_attribution.c19_signal_attribution(rg)
    assert c19["within_target_ranking_supported"] is True
    assert c19["cross_target_gauge_supported"] is False and c19["deployment_selector_established"] is False


def test_taxonomy_g1_when_separated():
    rg = {"two_axis_separation": True}
    srf = {"families": {"source_risk": {"carries_rank": True}, "source_calibration": {"carries_rank": True},
                        "source_leakage": {"carries_rank": False}}, "distributed_residual": True, "top_family": "source_risk",
           "score_rank_transfers": True}
    resid = {"gauge_contaminates_rank": False}
    err = {"rank_tracks_source_error_only": True, "R_src_rank_transfers": False}
    t = taxonomy.gauge_taxonomy(rg, srf, resid, err)
    assert t["primary_case"] == schema.G1
    for g in (schema.G2, schema.G3, schema.G4, schema.G5, schema.G7):
        assert g in t["established"]


def test_report_forbids_selector_language():
    for bad in ("a competence score selector", "rank gauge selector deployed"):
        try:
            report._guard_forbidden(bad); raise AssertionError("guard failed to fire")
        except ValueError:
            pass
    report._guard_forbidden("NOT a competence score selector; diagnostic-only.")
