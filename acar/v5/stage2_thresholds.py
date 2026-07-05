"""ACAR V5 Stage-2 FIT-only THRESHOLDS (pure/stdlib). The candidate's operating-point thresholds (harm-veto + family-specific
benefit/conf/lambda quantiles) are computed over the FIT (train∪val) proposed-action records ONLY — never CAL/EVAL/external.
This is a thin, fail-closed wrapper over the frozen scalarization quantile universe so a Stage-2 runner cannot fit a threshold
on a non-FIT split. Zero FIT proposed-action records ⇒ NonEvaluableCandidate (the candidate fails).
"""
from __future__ import annotations
from acar.v5 import scalarization as SCAL

NonEvaluableCandidate = SCAL.NonEvaluableCandidate


class Stage2ThresholdError(RuntimeError):
    """Raised when threshold fitting is asked to use anything other than FIT records (fail-closed)."""


def fit_thresholds(candidate, fit_batches):
    """Compute the FIT-only thresholds for a candidate over its FIT proposed-action records. `fit_batches` MUST be the FIT
    (train∪val) action-record batches only. Reuses the pinned Type-7 quantile universe. Raises NonEvaluableCandidate on zero
    FIT proposed-action records."""
    if fit_batches is None:
        raise Stage2ThresholdError("fit_batches must be the FIT (train∪val) action-record batches, not None")
    return SCAL.fit_quantiles(candidate, list(fit_batches))
