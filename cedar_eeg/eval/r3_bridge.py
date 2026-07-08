"""R3-style functional reliance bridge for CEDAR candidates."""

from __future__ import annotations

from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class R3Result:
    before_drop: float
    after_drop: float
    tolerance: float
    passed: bool

    @property
    def delta(self) -> float:
        return float(self.after_drop - self.before_drop)

    def to_dict(self) -> dict[str, float | bool]:
        out = asdict(self)
        out["delta"] = self.delta
        return out


def task_reliance_drop(full_bacc: float, intervened_bacc: float) -> float:
    """Task drop caused by a representation intervention."""

    return float(full_bacc - intervened_bacc)


def r3_not_increased(before_drop: float, after_drop: float, tolerance: float = 0.0) -> R3Result:
    delta = float(after_drop - before_drop)
    return R3Result(
        before_drop=float(before_drop),
        after_drop=float(after_drop),
        tolerance=float(tolerance),
        passed=bool(delta <= tolerance),
    )
