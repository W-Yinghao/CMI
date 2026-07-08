"""C28 — deterministic source-target homology taxonomy H1-H7. The decisive axis is Q3 (does the source factor
predict the target offset?): if yes, the C23 source registry missed the carrier (H6, reopens the gauge); if no,
source-unobservability is confirmed at the logit-factor level (H7), and the misalignment (H2) / source-error-only
(H4) / target-residual (H5) findings explain WHY."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(homology, offset_pred, error_geo, residual) -> dict:
    source_predicts = bool(offset_pred.get("source_predicts_offset"))
    aligned = bool(homology.get("aligned"))                 # informative (centered) alignment, not raw cosine
    misaligned = bool(homology.get("misaligned"))
    # H4: the source factor tracks SOURCE error geometry but does NOT predict the target offset
    tracks_src_err = bool((error_geo.get("source_factor_vs_source_recall") or 0) >= 0.30 and not source_predicts)
    residual_carries = bool(residual.get("residual_carries_offset"))
    residual_weak = bool(residual.get("residual_over_source_explained") and not residual_carries)

    established = []
    if source_predicts:
        primary = schema.H6                                 # source omission reopens the gauge
        established.append(schema.H3)
        if aligned:
            established.append(schema.H1)
    else:
        primary = schema.H7                                 # source-unobservability confirmed at logit-factor level
        if misaligned:
            established.append(schema.H2)
        if tracks_src_err:
            established.append(schema.H4)
        if residual_carries or residual_weak:
            established.append(schema.H5)
    established = [primary] + established
    interp = {
        schema.H1: "the source class-conditioned confidence factor aligns with the target factor.",
        schema.H2: "source and target class-conditioned confidence are weakly/negatively aligned -> the carrier lives in target decision occupancy, not source logit geometry.",
        schema.H3: "the source factor recovers the target offset under LOTO (survives permutation).",
        schema.H4: "the source factor tracks SOURCE error geometry but not the target factor/offset -> explains the source->target decoupling.",
        schema.H5: "the TARGET RESIDUAL (target factor minus its source-explained component) carries MORE of the offset than the source-explained part (which hurts) -> the missing gauge is target-specific decision occupancy, not the source-visible component.",
        schema.H6: "the source-only class-conditioned confidence gauge RECOVERS the target offset -> C23's source registry OMITTED this carrier (reopens the source gauge as a diagnostic, NOT a selector).",
        schema.H7: "a source analog of the carrier exists but CANNOT identify the target offset -> source-unobservability is confirmed at the logit-factor level (C23 explained, not overturned).",
    }[primary]
    return {"primary_case": primary, "established": established, "source_predicts_offset": source_predicts,
            "source_target_informatively_aligned": aligned, "source_target_misaligned": misaligned,
            "raw_cosine_mean_dominated": bool(homology.get("raw_cosine_mean_dominated")),
            "source_tracks_source_error_only": tracks_src_err,
            "target_residual_carries_offset": residual_carries, "target_residual_weakly_carries": residual_weak,
            "interpretation": interp, "diagnostic_only_non_deployable": True}
