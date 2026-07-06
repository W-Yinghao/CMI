"""Project B router — OACI reason-code contract (Step-2B).

Auditable reason codes for routing decisions. The central design fact: NOT every code is a
blocking refusal. Some are audit-only (prior-shift info, leakage-unavailable), some make a
sub-module unavailable without forcing a full refusal (ACAR calibration degenerate), and only
a defined set blocks by default (support failures, unstable TTA, missing/non-finite diagnostics,
prior-decoupling failure under high prior shift).
"""
from __future__ import annotations

from enum import Enum
from typing import Iterable


class OACIReason(str, Enum):  # str-Enum; value == name for every member
    OACI_OK = "OACI_OK"

    # TOS / support
    OACI_TOS_TOO_FEW_TARGET = "OACI_TOS_TOO_FEW_TARGET"
    OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE = "OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE"
    OACI_TOS_SINGLE_CLASS_IDENTIFIABILITY = "OACI_TOS_SINGLE_CLASS_IDENTIFIABILITY"
    OACI_TOS_DENSITY_OOD = "OACI_TOS_DENSITY_OOD"
    OACI_TOS_SUPPORT_MISMATCH = "OACI_TOS_SUPPORT_MISMATCH"

    # TTA diagnostics
    OACI_TTA_UNSTABLE_TRANSFORM = "OACI_TTA_UNSTABLE_TRANSFORM"
    OACI_TTA_NEGATIVE_EVIDENCE = "OACI_TTA_NEGATIVE_EVIDENCE"
    OACI_TTA_HIGH_PRED_DISAGREEMENT = "OACI_TTA_HIGH_PRED_DISAGREEMENT"
    OACI_TTA_IDENTITY_FALLBACK = "OACI_TTA_IDENTITY_FALLBACK"

    # Prior / support decoupling
    OACI_PRIOR_SHIFT_ONLY_INFO = "OACI_PRIOR_SHIFT_ONLY_INFO"
    OACI_PRIOR_DECOUPLING_UNAVAILABLE = "OACI_PRIOR_DECOUPLING_UNAVAILABLE"
    OACI_PRIOR_DECOUPLING_FAILED = "OACI_PRIOR_DECOUPLING_FAILED"

    # ACAR / calibration
    OACI_ACAR_HIGH_ACTION_RISK = "OACI_ACAR_HIGH_ACTION_RISK"
    OACI_ACAR_INSUFFICIENT_CALIBRATION = "OACI_ACAR_INSUFFICIENT_CALIBRATION"
    OACI_ACAR_HARM_CALIBRATION_DEGENERATE = "OACI_ACAR_HARM_CALIBRATION_DEGENERATE"
    OACI_ACAR_ERROR_CALIBRATION_DEGENERATE = "OACI_ACAR_ERROR_CALIBRATION_DEGENERATE"

    # Conformal / action set
    OACI_CONF_EMPTY_ACTION_SET = "OACI_CONF_EMPTY_ACTION_SET"

    # Legacy gate
    OACI_GATE_HARM_RISK = "OACI_GATE_HARM_RISK"

    # Leakage
    OACI_LEAKAGE_RESIDUAL_HIGH = "OACI_LEAKAGE_RESIDUAL_HIGH"
    OACI_LEAKAGE_RESIDUAL_UNAVAILABLE = "OACI_LEAKAGE_RESIDUAL_UNAVAILABLE"

    # Diagnostics / internal
    OACI_DIAGNOSTIC_MISSING = "OACI_DIAGNOSTIC_MISSING"
    OACI_DIAGNOSTIC_NONFINITE = "OACI_DIAGNOSTIC_NONFINITE"
    OACI_INTERNAL_ERROR = "OACI_INTERNAL_ERROR"


_R = OACIReason

# ------------------------------------------------------------------ classification sets
INFO_CODES = frozenset({_R.OACI_OK, _R.OACI_PRIOR_SHIFT_ONLY_INFO})

# Audit-only: purely informational, never blocking, no negative action implication.
AUDIT_ONLY_CODES = frozenset({
    _R.OACI_OK,
    _R.OACI_PRIOR_SHIFT_ONLY_INFO,
    _R.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE,
    _R.OACI_PRIOR_DECOUPLING_UNAVAILABLE,
    _R.OACI_TTA_IDENTITY_FALLBACK,
})

TOS_CODES = frozenset({
    _R.OACI_TOS_TOO_FEW_TARGET,
    _R.OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE,
    _R.OACI_TOS_SINGLE_CLASS_IDENTIFIABILITY,
    _R.OACI_TOS_DENSITY_OOD,
    _R.OACI_TOS_SUPPORT_MISMATCH,
})

TTA_CODES = frozenset({
    _R.OACI_TTA_UNSTABLE_TRANSFORM,
    _R.OACI_TTA_NEGATIVE_EVIDENCE,
    _R.OACI_TTA_HIGH_PRED_DISAGREEMENT,
    _R.OACI_TTA_IDENTITY_FALLBACK,
})

PRIOR_CODES = frozenset({
    _R.OACI_PRIOR_SHIFT_ONLY_INFO,
    _R.OACI_PRIOR_DECOUPLING_UNAVAILABLE,
    _R.OACI_PRIOR_DECOUPLING_FAILED,
})

ACAR_CODES = frozenset({
    _R.OACI_ACAR_HIGH_ACTION_RISK,
    _R.OACI_ACAR_INSUFFICIENT_CALIBRATION,
    _R.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,
    _R.OACI_ACAR_ERROR_CALIBRATION_DEGENERATE,
})

LEAKAGE_CODES = frozenset({
    _R.OACI_LEAKAGE_RESIDUAL_HIGH,
    _R.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE,
})

DIAGNOSTIC_CODES = frozenset({
    _R.OACI_DIAGNOSTIC_MISSING,
    _R.OACI_DIAGNOSTIC_NONFINITE,
    _R.OACI_INTERNAL_ERROR,
})

# Codes that block a non-refusal action BY DEFAULT. Support/stability/integrity failures block;
# "unavailable"/"info"/"degenerate" codes do NOT (they narrow the toolset, not the whole decision).
BLOCKING_BY_DEFAULT = frozenset({
    # support failures
    _R.OACI_TOS_TOO_FEW_TARGET,
    _R.OACI_TOS_LOW_EFFECTIVE_SAMPLE_SIZE,
    _R.OACI_TOS_SINGLE_CLASS_IDENTIFIABILITY,
    _R.OACI_TOS_DENSITY_OOD,
    _R.OACI_TOS_SUPPORT_MISMATCH,
    # unstable / non-beneficial TTA
    _R.OACI_TTA_UNSTABLE_TRANSFORM,
    _R.OACI_TTA_NEGATIVE_EVIDENCE,
    _R.OACI_TTA_HIGH_PRED_DISAGREEMENT,
    # prior decoupling failed under high prior shift
    _R.OACI_PRIOR_DECOUPLING_FAILED,
    # explicit harm predictions
    _R.OACI_ACAR_HIGH_ACTION_RISK,
    _R.OACI_GATE_HARM_RISK,
    _R.OACI_LEAKAGE_RESIDUAL_HIGH,
    # empty admissible set
    _R.OACI_CONF_EMPTY_ACTION_SET,
    # data integrity
    _R.OACI_DIAGNOSTIC_MISSING,
    _R.OACI_DIAGNOSTIC_NONFINITE,
    _R.OACI_INTERNAL_ERROR,
})


# ------------------------------------------------------------------ helpers
def normalize_reason(reason: "str | OACIReason") -> OACIReason:
    """Coerce a member or the (identical) name/value string to an OACIReason; else ValueError."""
    if isinstance(reason, OACIReason):
        return reason
    if isinstance(reason, str):
        try:
            return OACIReason(reason)
        except ValueError:
            pass
        try:
            return OACIReason[reason]
        except KeyError:
            pass
    raise ValueError(f"unknown OACI reason: {reason!r}")


def normalize_reasons(reasons: "Iterable[str | OACIReason]") -> tuple[OACIReason, ...]:
    """Dedup (order-preserving); empty -> (OACI_OK,); drop OACI_OK if any other code present."""
    seen: list[OACIReason] = []
    seen_set: set[OACIReason] = set()
    for r in reasons:
        nr = normalize_reason(r)
        if nr not in seen_set:
            seen_set.add(nr)
            seen.append(nr)
    if not seen:
        return (OACIReason.OACI_OK,)
    if OACIReason.OACI_OK in seen_set and len(seen) > 1:
        seen = [r for r in seen if r is not OACIReason.OACI_OK]
    return tuple(seen)


def is_audit_only(reason: "str | OACIReason") -> bool:
    return normalize_reason(reason) in AUDIT_ONLY_CODES


def is_blocking_by_default(reason: "str | OACIReason") -> bool:
    return normalize_reason(reason) in BLOCKING_BY_DEFAULT


def has_blocking_reason(reasons: "Iterable[str | OACIReason]") -> bool:
    return any(is_blocking_by_default(r) for r in reasons)


if __name__ == "__main__":
    members = list(OACIReason)
    # value uniqueness + value == name invariant
    assert len({m.value for m in members}) == len(members), "duplicate reason values"
    assert all(m.value == m.name for m in members), "value != name"

    # non-blocking
    for r in (OACIReason.OACI_OK, OACIReason.OACI_PRIOR_SHIFT_ONLY_INFO,
              OACIReason.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE,
              OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE,
              OACIReason.OACI_PRIOR_DECOUPLING_UNAVAILABLE):
        assert not is_blocking_by_default(r), r
    # blocking
    for r in (OACIReason.OACI_DIAGNOSTIC_MISSING, OACIReason.OACI_DIAGNOSTIC_NONFINITE,
              OACIReason.OACI_INTERNAL_ERROR, OACIReason.OACI_PRIOR_DECOUPLING_FAILED):
        assert is_blocking_by_default(r), r

    # audit-only
    assert is_audit_only(OACIReason.OACI_PRIOR_SHIFT_ONLY_INFO)
    assert is_audit_only(OACIReason.OACI_LEAKAGE_RESIDUAL_UNAVAILABLE)
    assert not is_audit_only(OACIReason.OACI_TOS_DENSITY_OOD)

    # disjointness / subset invariants
    assert BLOCKING_BY_DEFAULT.isdisjoint(AUDIT_ONLY_CODES)
    for s in (INFO_CODES, AUDIT_ONLY_CODES, TOS_CODES, TTA_CODES, PRIOR_CODES,
              ACAR_CODES, LEAKAGE_CODES, DIAGNOSTIC_CODES, BLOCKING_BY_DEFAULT):
        assert s <= set(members)

    # normalize_reason
    assert normalize_reason("OACI_OK") is OACIReason.OACI_OK
    assert normalize_reason(OACIReason.OACI_TOS_DENSITY_OOD) is OACIReason.OACI_TOS_DENSITY_OOD
    for bad in ("garbage", "", 5):
        try:
            normalize_reason(bad)  # type: ignore[arg-type]
            raise AssertionError("should have raised")
        except ValueError:
            pass

    # normalize_reasons: empty -> OK; dedup; drop OK when others present; order preserved
    assert normalize_reasons([]) == (OACIReason.OACI_OK,)
    assert normalize_reasons([OACIReason.OACI_OK]) == (OACIReason.OACI_OK,)
    assert normalize_reasons([OACIReason.OACI_OK, OACIReason.OACI_TOS_DENSITY_OOD]) == (
        OACIReason.OACI_TOS_DENSITY_OOD,)
    assert normalize_reasons(["OACI_TOS_DENSITY_OOD", "OACI_TOS_DENSITY_OOD"]) == (
        OACIReason.OACI_TOS_DENSITY_OOD,)
    assert normalize_reasons([OACIReason.OACI_TTA_UNSTABLE_TRANSFORM, OACIReason.OACI_TOS_TOO_FEW_TARGET]) == (
        OACIReason.OACI_TTA_UNSTABLE_TRANSFORM, OACIReason.OACI_TOS_TOO_FEW_TARGET)

    # has_blocking_reason
    assert has_blocking_reason([OACIReason.OACI_OK, OACIReason.OACI_DIAGNOSTIC_MISSING])
    assert not has_blocking_reason([OACIReason.OACI_OK, OACIReason.OACI_PRIOR_SHIFT_ONLY_INFO,
                                    OACIReason.OACI_ACAR_HARM_CALIBRATION_DEGENERATE])
    print("reasons self-test passed")
