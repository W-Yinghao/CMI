"""Toy replay determinism checks for TTA-MECH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class ReplayDeterminismFailure(AssertionError):
    pass


@dataclass(frozen=True)
class ReplayDeterminismResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    output_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_replay_determinism(left: dict[str, Any], right: dict[str, Any]) -> ReplayDeterminismResult:
    checks: list[str] = []
    warnings: list[str] = []
    if left.get("toy_output_hash") != right.get("toy_output_hash"):
        raise ReplayDeterminismFailure("toy replay hash changed")
    checks.append("toy_output_hash_identical")
    if left.get("baseline") != right.get("baseline"):
        raise ReplayDeterminismFailure("baseline changed between replay runs")
    checks.append("baseline_identical")
    return ReplayDeterminismResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        output_hash=str(left.get("toy_output_hash")),
    )
