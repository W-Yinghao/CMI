"""C18 — source-feature inventory. Classifies every C17 source signal as recomputable_under_mask /
static_training_log_only / not_reconstructable, and enforces that ONLY recomputable features may carry
S1-S7 support-stress claims. Carrying a static training-log scalar (train_surrogate, epoch, lambda) across
masks as if it were masked would recreate the fake-per-regime-probe problem — that is refused here."""
from __future__ import annotations

from ..identifiability.signal_atlas import SOURCE_SIGNALS
from . import schema


def inventory() -> list:
    return [{"feature": s, "class": schema.feature_class(s),
             "usable_for_mask_stress": schema.feature_class(s) == "recomputable_under_mask"}
            for s in SOURCE_SIGNALS]


def recomputable_features() -> tuple:
    return tuple(s for s in SOURCE_SIGNALS if schema.feature_class(s) == "recomputable_under_mask")


def static_features() -> tuple:
    return tuple(s for s in SOURCE_SIGNALS if schema.feature_class(s) == "static_training_log_only")


def assert_inventory_complete() -> None:
    """Every C17 source signal must be classified (no silent omission)."""
    missing = [s for s in SOURCE_SIGNALS if s not in schema.FEATURE_CLASS]
    if missing:
        raise ValueError(f"unclassified source signals (would silently leak into mask claims): {missing}")


def assert_only_recomputable_used(feature_cols) -> None:
    """Guard for the S1-S7 mask-stress probe: reject any static/non-reconstructable column."""
    bad = [c for c in feature_cols if schema.feature_class(c.replace("src__", "")) != "recomputable_under_mask"]
    if bad:
        raise ValueError(f"mask-stress probe may only use recomputable_under_mask features; got static/other: {bad}")
