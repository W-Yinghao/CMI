"""Typed non-estimability errors. The runner maps ONLY these to a conservative ERM selection /
audit ``nonestimable_*`` status. Hash mismatches, numerical failures and invalid accepted draws are
NOT non-estimability — they propagate as ordinary errors."""
from __future__ import annotations


class LeakageNonEstimableError(ValueError):
    """Structural non-estimability of a leakage quantity. Subclasses ValueError for back-compat (a
    plain ValueError from a hash mismatch / numerical failure is NOT an instance of this, so the
    runner's ``except LeakageNonEstimableError`` lets those propagate)."""


class NoComparableSupport(LeakageNonEstimableError):
    """No comparable class (every class has < 2 eligible domains)."""


class FoldPlanNonEstimable(LeakageNonEstimableError):
    """Cannot form >= 2 grouped cross-fit folds."""


class BootstrapPlanNonEstimable(LeakageNonEstimableError):
    """Cannot draw the requested number of structurally-valid bootstrap replicates."""
