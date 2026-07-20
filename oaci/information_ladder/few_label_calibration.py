"""C24 R5 — few-label target calibration diagnostic. NON-DG, supervised: for a held-out target, reveal the
competence labels of a FEW of its own candidates (k per class, deterministic by model_hash) and estimate that
target's score OFFSET as the midpoint of the revealed good/bad score means; subtract it and re-pool. This is a
LABEL-BUDGET identifiability diagnostic (how fast does the offset become recoverable once target labels are
allowed?), NOT deployment, NOT a selector. k=0 falls back to the label-free transductive target mean."""
from __future__ import annotations

import numpy as np

from ..score_gauge.ceiling_ladder import _pooled_auc
from . import schema


def _target_offset_hat(cands, k):
    """Deterministic reveal of k good + k bad candidates (sorted by model_hash); offset = decision midpoint.
    Returns (offset_hat, n_good_revealed, n_bad_revealed). k=0 -> label-free target mean score."""
    ordered = sorted(cands, key=lambda c: c["model_hash"])
    if k == 0:
        return float(np.mean([c["score"] for c in ordered])), 0, 0
    good = [c for c in ordered if c["label"] == 1][:k]
    bad = [c for c in ordered if c["label"] == 0][:k]
    if not good or not bad:                                 # cannot form a labeled decision midpoint
        return None, len(good), len(bad)
    return 0.5 * (float(np.mean([c["score"] for c in good])) + float(np.mean([c["score"] for c in bad]))), len(good), len(bad)


def few_label_curve(rows, mode="in_regime") -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    by_t = {}
    for r in mr:
        by_t.setdefault(r["target"], []).append(r)
    tgt_mean = {t: float(np.mean([c["score"] for c in cs])) for t, cs in by_t.items()}
    raw = _pooled_auc(mr)
    oracle = _pooled_auc(mr, subtract=lambda r: tgt_mean[r["target"]])   # target-centered ceiling (R6)
    curve = []; per_target = []
    for k in schema.FEW_LABEL_BUDGETS:
        offhat = {}; cov = 0
        for t, cs in by_t.items():
            oh, ng, nb = _target_offset_hat(cs, k)
            if oh is None:                                  # fall back to label-free mean when class missing
                oh = tgt_mean[t]
            else:
                cov += 1
            offhat[t] = oh
            if k == schema.FEW_LABEL_BUDGETS[1]:            # record per-target detail at the smallest labeled budget
                per_target.append({"target": t, "k_per_class": k, "offset_hat": round(oh, 4),
                                   "n_good_revealed": ng, "n_bad_revealed": nb})
        auc = _pooled_auc(mr, subtract=lambda r: offhat[r["target"]])
        gap = ((auc - raw) / (oracle - raw)) if (auc is not None and oracle is not None and (oracle - raw) > 1e-6) else None
        curve.append({"k_per_class": k, "pooled_auc": auc, "auc_improve": (auc - raw) if auc is not None else None,
                      "gap_closed": gap, "n_targets_with_both_classes": cov, "n_targets": len(by_t)})
    small = [c for c in curve if c["k_per_class"] <= schema.FEW_LABEL_RECOVERS_MAX_K and c["k_per_class"] > 0]
    few_recovers = bool(any(c["gap_closed"] is not None and c["gap_closed"] >= schema.SUCCESS_GAP_CLOSED for c in small))
    max_gap = max((c["gap_closed"] for c in curve if c["gap_closed"] is not None), default=None)
    # k=0 IS the label-free transductive target-mean centering == the target-centered oracle (by construction).
    # Disclose this so "few labels recover" is not over-claimed: the offset is a target-GROUPING quantity first.
    k0 = next(c for c in curve if c["k_per_class"] == 0)
    zero_label_gap = k0["gap_closed"]
    kmax = curve[-1]["gap_closed"]
    label_gain_over_grouping = ((kmax - zero_label_gap) if (kmax is not None and zero_label_gap is not None) else None)
    return {"raw_pooled": raw, "target_centered_oracle": oracle, "curve": curve, "per_target_small_budget": per_target,
            "few_labels_recover": few_recovers, "max_gap_closed": max_gap,
            "zero_label_transductive_gap": zero_label_gap,
            "zero_label_grouping_equals_oracle": bool(zero_label_gap is not None and abs(zero_label_gap - 1.0) < 1e-6),
            "label_gain_over_grouping": label_gain_over_grouping,
            "note": schema.FEW_LABEL_NOTE + " NB: k=0 is the LABEL-FREE transductive target-mean centering (== the "
                    "target-centered oracle); positive labels only refine it. The offset is a target-GROUPING "
                    "quantity, recoverable transductively with 0 labels once target candidates can be pooled."}
