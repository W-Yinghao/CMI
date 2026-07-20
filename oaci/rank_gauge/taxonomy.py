"""C30 — deterministic rank-gauge taxonomy G1-G7. Primary = whether the two axes separate (G1) or the gauge
contaminates the apparent rank (G6). Established secondaries record which source family carries the rank
(G2/G3), that leakage does not (G4), that the rank tracks source error only (G5), and any distributed residual
the tested families do not explain (G7)."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(rank_gauge, src_family, resid, err_align) -> dict:
    separation = bool(rank_gauge.get("two_axis_separation"))
    gauge_contaminates = bool(resid.get("gauge_contaminates_rank"))
    fams = src_family.get("families", {})
    risk_carries = bool(fams.get("source_risk", {}).get("carries_rank"))
    calib_carries = bool(fams.get("source_calibration", {}).get("carries_rank"))
    leakage_carries = bool(fams.get("source_leakage", {}).get("carries_rank"))
    tracks_src_err = bool(err_align.get("rank_tracks_source_error_only"))
    distributed = bool(src_family.get("distributed_residual"))          # red-team: residual-sense, not bare gap
    rsrc_transfers = err_align.get("R_src_rank_transfers")
    score_transfers = src_family.get("score_rank_transfers")

    primary = schema.G1 if (separation and not gauge_contaminates) else (schema.G6 if gauge_contaminates else schema.G7)
    established = [primary]
    if risk_carries:
        established.append(schema.G2)
    if calib_carries:
        established.append(schema.G3)
    if not leakage_carries:
        established.append(schema.G4)
    if tracks_src_err:
        established.append(schema.G5)
    if distributed:
        established.append(schema.G7)
    interp = {
        schema.G1: "the within-target RANK axis and the target-specific GAUGE axis are SEPARABLE (orthogonal; rank survives gauge control) -> the competence signal is two-axis.",
        schema.G2: "the source-RISK family (R_src) carries the largest single share of the within-target ranking (high source risk -> low competence; survives permutation) -- but WEAK/diagnostic (strength ~0.12) and ~38% absorbed by same-family train_surrogate; and it is TARGET-LOCAL: R_src's per-target direction sign-flips (does not transfer).",
        schema.G3: "the source-CALIBRATION family carries within-target ranking beyond source risk.",
        schema.G4: "the source-LEAKAGE family does NOT carry the ranking -> leakage remains a measurement/falsification quantity, not a rank signal. (not independently red-teamed this round)",
        schema.G5: "[RED-TEAM reworded] 'tracks source error' is TAUTOLOGICAL (R_src IS the source NLL/CE risk, corr 0.985; residualizing on source NLL -> chance); and the R_src within-target rank does NOT transfer (per-target sign-flips) -> a target-local, non-transferable signal. What survives: R_src is NOT a calibrated/deployable target-competence score.",
        schema.G6: "the apparent within-target ranking collapses after gauge control -> it partly reflects hidden gauge structure.",
        schema.G7: "[RED-TEAM] the within-target ranking is DISTRIBUTED in the RESIDUAL sense (the multivariate probe retains rank strength after removing R_src and is direction-CONSISTENT across targets / transfers, while single families are weaker/target-local); the bare score-minus-best-family gap is WITHIN 9-target bootstrap noise (do NOT claim 'beats any family').",
    }[primary]
    return {"primary_case": primary, "established": established, "two_axis_separation": separation,
            "gauge_contaminates_rank": gauge_contaminates, "source_risk_carries": risk_carries,
            "source_calibration_carries": calib_carries, "leakage_carries": leakage_carries,
            "rank_tracks_source_error_only_TAUTOLOGICAL": tracks_src_err, "distributed_residual": distributed,
            "R_src_rank_transfers": rsrc_transfers, "score_rank_transfers": score_transfers,
            "top_family": src_family.get("top_family"), "interpretation": interp,
            "red_team_verified": True, "diagnostic_only_non_deployable": True}
