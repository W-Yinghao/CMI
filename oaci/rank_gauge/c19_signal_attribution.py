"""C30 Q5 — attribute the C19 in-regime positive to an axis. C19's robust-core source-only probe passed in-
regime but C20 pooled cross-regime failed. C30 re-reads this as: the C19 positive is the WITHIN-TARGET RANK axis
(source-visible), NOT the cross-target GAUGE axis (source-unobservable, C23-C29). This prevents misreading the
C19 positive as a target-free detector / deployment selector."""
from __future__ import annotations

from . import schema


def c19_signal_attribution(rank_gauge) -> dict:
    within = rank_gauge.get("score_within_target_auc"); pooled = rank_gauge.get("score_pooled_auc")
    gauge_centered = rank_gauge.get("gauge_centered_pooled_auc")
    within_supported = bool(within is not None and within >= schema.RANK_SIGNAL_MIN)
    gauge_supported = False                                  # pooled fails AND the gauge is source-unobservable (C23-C29)
    return {"within_target_ranking_supported": within_supported, "cross_target_gauge_supported": gauge_supported,
            "deployment_selector_established": False, "within_target_auc": within, "pooled_auc": pooled,
            "gauge_centered_pooled_auc": gauge_centered,
            "attribution": ("C19's in-regime positive is the WITHIN-TARGET RANK axis (AUC %.3f); the cross-target "
                            "GAUGE axis is NOT supported (pooled %.3f fails; the gauge is source-unobservable per "
                            "C23-C29). C19 is a weak within-target competence ranking signal, NOT a target-free "
                            "detector and NOT a deployment selector." % (within or 0, pooled or 0))}
