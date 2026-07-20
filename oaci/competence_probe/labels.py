"""C19 — diagnostic-only labels. The target-accuracy-good label is used POST HOC only, for LOTO meta-
evaluation; it never enters feature construction and no selector is produced from it."""
from __future__ import annotations

from . import schema


def assert_diagnostic_only(rows) -> None:
    if rows and not all(r.get("diagnostic_only_non_deployable") for r in rows):
        raise ValueError("every C19 row must carry diagnostic_only_non_deployable=True")


def assert_no_target_in_features(feature_cols) -> None:
    bad = [c for c in feature_cols if c.startswith("tgt__") or "target" in c]
    if bad:
        raise ValueError(f"target-derived column in the probe feature set: {bad}")


def label_base_rate(rows) -> float:
    if not rows:
        return 0.0
    good = sum(1 for r in rows if r.get(schema.DIAGNOSTIC_LABEL))
    return good / len(rows)
