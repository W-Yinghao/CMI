"""C30 Q4 — is the within-target rank separable from the gauge, or does the gauge contaminate it? The within-
target AUC is gauge-free by construction (the per-target intercept is removed within each target); we further
residualize the source rank against R_src (a gauge/source-risk proxy) and report the rank+gauge diagnostic upper
bound. If the rank collapses after control, the apparent ranking reflects hidden gauge structure (G6)."""
from __future__ import annotations

from . import artifact_loader, schema


def residualization(rows, mode="in_regime") -> dict:
    score_within = artifact_loader.within_target_auc(rows, schema.SCORE_KEY, mode)
    # source rank (probe score) within-target after regressing out R_src per target (control the source-risk axis)
    score_ctrl_rsrc = artifact_loader.within_target_auc_residualized(rows, schema.SCORE_KEY, "R_src", mode)
    # R_src within-target after regressing out the probe score (does R_src add beyond the probe?)
    rsrc_ctrl_score = artifact_loader.within_target_auc_residualized(rows, "R_src", schema.SCORE_KEY, mode)
    ss = artifact_loader.rank_strength(score_within)
    ss_ctrl = artifact_loader.rank_strength(score_ctrl_rsrc)
    rank_survives_control = bool(ss_ctrl is not None and ss is not None and ss_ctrl >= 0.5 * ss)
    gauge_contaminates = bool(not rank_survives_control)
    return {"score_within_target_auc": score_within, "score_within_target_auc_ctrl_R_src": score_ctrl_rsrc,
            "R_src_within_target_auc_ctrl_score": rsrc_ctrl_score, "rank_strength": ss, "rank_strength_ctrl_R_src": ss_ctrl,
            "rank_survives_R_src_control": rank_survives_control, "gauge_contaminates_rank": gauge_contaminates,
            "note": ("the within-target rank SURVIVES controlling for R_src (rank strength %.3f -> %.3f) -> the rank "
                     "axis is separable, not a gauge/source-risk artifact"
                     % (ss or 0, ss_ctrl or 0) if rank_survives_control else
                     "the within-target rank collapses after R_src control -> the apparent ranking partly reflects "
                     "source-risk/gauge structure")}
