"""C22 — deterministic transport-failure taxonomy. Q2 (epoch confound) is the GATING override: if the within-
target advantage is explained by epoch/trajectory-position, the case is T2 regardless of the rest (the C19
positive downgrades to a trajectory-position diagnostic). Otherwise the case is decided by whether a within-
target signal exists and whether post-hoc target normalization recovers the pooled estimand."""
from __future__ import annotations

from . import schema


def transport_taxonomy(epoch, decomp_summary, normalization, offset, feature) -> dict:
    margin = schema.SIGNAL_MARGIN
    within = decomp_summary.get("mean_within_target_auc")
    within_present = bool(within is not None and within >= 0.5 + margin)
    cross = normalization.get("per_mode", {}).get("cross_regime", {})
    norm_recovers = bool(cross.get("target_normalization_recovers"))
    offset_frac = feature.get("offset_dominated_fraction")
    offset_dominated = bool(offset_frac is not None and offset_frac >= schema.OFFSET_DOMINATED_FRACTION)

    if epoch.get("epoch_confounded"):
        primary = schema.T2
    elif not within_present and not norm_recovers:
        primary = schema.T5
    elif within_present and norm_recovers:
        primary = schema.T1
    elif within_present and not norm_recovers:
        primary = schema.T3
    else:
        primary = schema.T1
    secondary = [schema.T4] if (offset_dominated and primary in (schema.T1, schema.T3)) else []

    interp = {
        schema.T1: "within-target ranking signal survives; the pooled estimand fails because score offsets/scales differ by target/regime (rank-like signal, score NOT calibrated). Post-hoc target normalization recovers pooled AUC (diagnostic only).",
        schema.T2: "the probe advantage is largely explained by epoch / candidate trajectory-position; the C19 within-target signal is a trajectory-position diagnostic, NOT source competence. Downgrade the C19 claim.",
        schema.T3: "target normalization does NOT recover the pooled estimand -> the source->target relationship itself shifts by support regime (not merely a score offset).",
        schema.T5: "neither a within-target residual signal nor a normalized pooled signal is reliable -> transport is genuinely absent.",
    }.get(primary, "")
    nxt = {
        schema.T1: "future work (diagnostic-only): a target-free score-calibration ESTIMAND. NOT a selector; needs the offset removed WITHOUT target identity, which is unsolved.",
        schema.T2: "retract the competence framing for the in-regime positive; report as trajectory-position diagnostic; re-examine C19.",
        schema.T3: "characterize the regime-specific relationship shift; external validation remains not established.",
        schema.T5: "package the boundary as final; no transportable estimand.",
    }.get(primary, "")
    return {"primary_case": primary, "secondary": secondary, "interpretation": interp, "next_science": nxt,
            "epoch_confounded": bool(epoch.get("epoch_confounded")), "within_target_present": within_present,
            "target_normalization_recovers_pooled": norm_recovers, "feature_offset_dominated": offset_dominated,
            "diagnostic_only_non_deployable": True}
