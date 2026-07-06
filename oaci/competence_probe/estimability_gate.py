"""C19 — endpoint-estimability gate. C18 showed source accuracy ENDPOINT observables (worst-domain reference
bAcc) can become non-estimable (a domain loses a class -> NaN) BEFORE leakage abstention. So estimability is a
first-class OUTPUT here: every candidate gets a score_status BEFORE scoring; non-estimable cases are reason-
coded and reported, never silently dropped or imputed into a score."""
from __future__ import annotations

import math

from . import schema


def _finite(v):
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def score_status(row, robust_cols, endpoint_cols=()) -> str:
    """Per-candidate status. Robust-core needs all robust features finite; endpoint-augmented additionally
    needs the fragile accuracy endpoints finite (else abstained_source_accuracy_endpoint_nonestimable)."""
    if not all(_finite(row.get(c)) for c in robust_cols):
        return "abstained_insufficient_finite_features"
    if endpoint_cols and not all(_finite(row.get(c)) for c in endpoint_cols):
        return "abstained_source_accuracy_endpoint_nonestimable"
    return "scored"


def gate_summary(rows, robust_cols, endpoint_cols=()) -> dict:
    counts = {s: 0 for s in schema.SCORE_STATUS}
    for r in rows:
        counts[score_status(r, robust_cols, endpoint_cols)] += 1
    n = max(len(rows), 1)
    return {"n_candidates": len(rows), "counts": counts,
            "scored_rate": counts["scored"] / n,
            "endpoint_nonestimable_rate": counts["abstained_source_accuracy_endpoint_nonestimable"] / n,
            "insufficient_finite_rate": counts["abstained_insufficient_finite_features"] / n}
