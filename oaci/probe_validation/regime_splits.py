"""C20 — regime split plan. Development regimes (train the frozen probe) MUST be disjoint from the held-out
validation regimes (evaluate). S1 skew is an implemented-noop negative-control, never a stress."""
from __future__ import annotations

from . import schema


def split_plan() -> dict:
    dev = set(schema.DEVELOPMENT_REGIMES); heldout = set(schema.HELD_OUT_REGIMES)
    if dev & heldout:
        raise ValueError(f"development and held-out regimes overlap: {dev & heldout}")
    return {"development_regimes": list(schema.DEVELOPMENT_REGIMES),
            "held_out_regimes": list(schema.HELD_OUT_REGIMES),
            "noop_negative_control": schema.NOOP_REGIME,
            "deletion_held_out": list(schema.DELETION_HELD_OUT),
            "nondeletion_held_out": list(schema.NONDELETION_HELD_OUT),
            "disjoint": True}


def assert_no_leakage_between_splits() -> None:
    if set(schema.DEVELOPMENT_REGIMES) & set(schema.HELD_OUT_REGIMES):
        raise ValueError("a held-out validation regime is also a development (training) regime")
