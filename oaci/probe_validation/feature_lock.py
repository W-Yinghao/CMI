"""C20 — feature lock. The C20 primary probe MUST use the C19 robust-core feature set byte-for-byte, with the
fragile accuracy endpoints excluded from the primary and the static training-log columns excluded entirely.
A drift here would silently turn a validation into a new feature search."""
from __future__ import annotations

from ..competence_probe import schema as c19
from . import schema


def lock_audit() -> dict:
    return {"robust_core": list(c19.ROBUST_CORE_FEATURES), "n_robust_core": len(c19.ROBUST_CORE_FEATURES),
            "endpoint_secondary": list(c19.ENDPOINT_FEATURES), "static_excluded": list(c19.STATIC_EXCLUDED),
            "robust_core_matches_c19": tuple(schema.robust_core_features()) == c19.ROBUST_CORE_FEATURES,
            "endpoints_excluded_from_primary": not (set(c19.ENDPOINT_FEATURES) & set(c19.ROBUST_CORE_FEATURES)),
            "static_excluded_from_primary": not (set(c19.STATIC_EXCLUDED) & set(c19.ROBUST_CORE_FEATURES))}


def assert_locked() -> None:
    a = lock_audit()
    if not (a["robust_core_matches_c19"] and a["endpoints_excluded_from_primary"] and a["static_excluded_from_primary"]):
        raise ValueError("C20 feature set does not match the frozen C19 robust-core lock")
