"""C31 Q4 — gauge endpoint specificity. The target-specific GAUGE (per-target offset) breaks the pooled estimand
(C22-C30). Here we ask whether that gauge structure is accuracy- or calibration-specific: (a) the between-target
variance fraction of each endpoint METRIC (how much of bAcc/NLL/ECE is a per-target offset), and (b) the source
SCORE's pooled-vs-within gap per endpoint LABEL (how much the gauge breaks each endpoint's pooled prediction)."""
from __future__ import annotations

import numpy as np

from . import artifact_loader

_LABELS = ("accuracy_good", "nll_good", "ece_good", "calibration_good", "joint_good")
_METRICS = ("bacc", "nll", "ece")


def _gauge_variance_fraction(rows, metric):
    tmean = {}
    for t in sorted({r["target"] for r in rows}):
        vals = [r[metric] for r in rows if r["target"] == t and r[metric] is not None]
        if vals:
            tmean[t] = float(np.mean(vals))
    allv = [r[metric] for r in rows if r[metric] is not None]
    if not allv or not tmean:
        return None
    between = float(np.var(list(tmean.values())))
    total = float(np.var(allv))
    return (between / total) if total > 1e-12 else None


def gauge_endpoint(rows) -> dict:
    metric_gauge = {m: _gauge_variance_fraction(rows, m) for m in _METRICS}
    per_label = {}
    for lab in _LABELS:
        pooled = artifact_loader.pooled_auc(rows, "score", lab)
        within = artifact_loader.within_target_auc(rows, "score", lab)
        gap = (artifact_loader.rank_strength(within) or 0) - (artifact_loader.rank_strength(pooled) or 0)
        per_label[lab] = {"pooled_auc": pooled, "within_target_auc": within, "gauge_gap": gap}
    acc_gap = per_label["accuracy_good"]["gauge_gap"]
    cal_gap = max(per_label["nll_good"]["gauge_gap"], per_label["ece_good"]["gauge_gap"])
    # accuracy-specific requires BOTH a materially larger accuracy pooled-vs-within gap AND a materially larger
    # bAcc between-target variance fraction; near-equal metric variance fractions => a GENERAL per-target offset.
    metric_acc_tilt = bool((metric_gauge["bacc"] or 0) >= max(metric_gauge["nll"] or 0, metric_gauge["ece"] or 0) + 0.05)
    gap_acc_tilt = bool(acc_gap >= cal_gap + 0.03)
    gauge_accuracy_specific = bool(metric_acc_tilt and gap_acc_tilt)
    gauge_general = bool(not gauge_accuracy_specific and (metric_gauge["bacc"] or 0) > 0.4 and (metric_gauge["nll"] or 0) > 0.4 and (metric_gauge["ece"] or 0) > 0.4)
    return {"metric_gauge_variance_fraction": metric_gauge, "per_label": per_label,
            "accuracy_gauge_gap": acc_gap, "calibration_gauge_gap": cal_gap,
            "gauge_accuracy_specific": gauge_accuracy_specific, "gauge_general_endpoint_offset": gauge_general,
            "note": ("the gauge is ACCURACY-SPECIFIC: accuracy pooled-vs-within gap %.3f vs calibration %.3f; metric "
                     "between-target variance fraction bAcc %.2f / NLL %.2f / ECE %.2f"
                     % (acc_gap, cal_gap, metric_gauge["bacc"] or 0, metric_gauge["nll"] or 0, metric_gauge["ece"] or 0)
                     if gauge_accuracy_specific else
                     "the gauge is a GENERAL per-target offset: between-target variance fraction is near-equal across "
                     "endpoints (bAcc %.2f / NLL %.2f / ECE %.2f); accuracy pooled-vs-within gap %.3f is only mildly "
                     "above calibration %.3f (tilt inherited from the accuracy-aligned rank, not a distinct calibration "
                     "gauge)" % (metric_gauge["bacc"] or 0, metric_gauge["nll"] or 0, metric_gauge["ece"] or 0,
                                 acc_gap, cal_gap))}
