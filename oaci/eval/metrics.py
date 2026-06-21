"""Accuracy estimands — three DISTINCT, never-conflated quantities + the paired endpoints.

* ``pooled_bacc``        — class-balanced recall pooled over all rows (supplementary; a big
                           domain can dominate).
* ``mean_domain_bacc``   — equal-domain mean of per-domain bAcc (a PRIMARY DG metric).
* ``worst_domain_bacc``  — ``min_d`` per-domain bAcc (a PRIMARY DG metric).
* ``worst_paired_delta_bacc`` — ``min_d [bAcc^A_d − bAcc^B_d]`` — a stricter paired endpoint that
  catches a domain HARMED even when another domain's gain hides it (≠ difference of the two
  minima). It (and worst_domain_bacc) are recomputed INSIDE each bootstrap replicate.

Per-domain bAcc has two flavours: ``reference`` over the pre-registered class set (non-estimable
for a domain missing a class) and ``observed`` over present classes (with class coverage).
"""
from __future__ import annotations

import numpy as np


def per_class_recall(y, pred, classes) -> dict:
    y, pred = np.asarray(y), np.asarray(pred)
    out = {}
    for c in classes:
        m = y == c
        out[c] = float((pred[m] == c).mean()) if m.any() else np.nan
    return out


def pooled_bacc(y, pred, classes) -> float:
    r = per_class_recall(y, pred, classes)
    vals = [r[c] for c in classes if not np.isnan(r[c])]
    return float(np.mean(vals)) if vals else np.nan


def domain_bacc(y, pred, classes):
    """Return (reference_bacc, observed_bacc, class_coverage) for a single domain. reference is
    NaN (non-estimable) if any pre-registered class is absent."""
    r = per_class_recall(y, pred, classes)
    present = [c for c in classes if not np.isnan(r[c])]
    reference = float(np.mean([r[c] for c in classes])) if len(present) == len(classes) else np.nan
    observed = float(np.mean([r[c] for c in present])) if present else np.nan
    return reference, observed, len(present) / len(classes)


def domain_baccs(y, pred, domain, classes, kind="reference") -> dict:
    y, pred, domain = np.asarray(y), np.asarray(pred), np.asarray(domain)
    out = {}
    for dd in np.unique(domain):
        m = domain == dd
        ref, obs, _ = domain_bacc(y[m], pred[m], classes)
        out[int(dd)] = ref if kind == "reference" else obs
    return out


def mean_domain_bacc(y, pred, domain, classes, kind="reference") -> float:
    v = [x for x in domain_baccs(y, pred, domain, classes, kind).values() if not np.isnan(x)]
    return float(np.mean(v)) if v else np.nan


def worst_domain_bacc(y, pred, domain, classes, kind="reference") -> float:
    vals = domain_baccs(y, pred, domain, classes, kind)
    if kind == "reference" and any(np.isnan(x) for x in vals.values()):
        return np.nan                                  # a non-estimable domain -> worst non-estimable
    v = [x for x in vals.values() if not np.isnan(x)]
    return float(np.min(v)) if v else np.nan


def worst_paired_delta_bacc(y, pred_a, pred_b, domain, classes, kind="reference") -> float:
    """``min_d [ bAcc^A_d − bAcc^B_d ]`` — recomputed (min over domains) on whatever rows it is
    given, so a bootstrap replicate gets its OWN worst domain."""
    ba = domain_baccs(y, pred_a, domain, classes, kind)
    bb = domain_baccs(y, pred_b, domain, classes, kind)
    deltas = [ba[d] - bb[d] for d in ba if not (np.isnan(ba[d]) or np.isnan(bb[d]))]
    return float(np.min(deltas)) if deltas else np.nan
