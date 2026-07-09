"""Target-label quarantine checks for future BN audit conditions."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable

from tta_mech_eeg.normalization.bn_schema import (
    FORBIDDEN_CONDITION_API_PARAMETERS,
    REQUIRED_CONDITION_API_PARAMETERS,
)


class BnTargetLabelQuarantineFailure(AssertionError):
    pass


@dataclass(frozen=True)
class BnTargetLabelQuarantineResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    callable_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bn_target_label_quarantine(fn: Callable[..., Any]) -> BnTargetLabelQuarantineResult:
    checks: list[str] = []
    warnings: list[str] = []
    sig = inspect.signature(fn)
    params = tuple(sig.parameters)
    missing = [name for name in REQUIRED_CONDITION_API_PARAMETERS if name not in params]
    if missing:
        raise BnTargetLabelQuarantineFailure(f"missing required BN condition parameters: {missing}")
    checks.append("required_bn_condition_parameters_present")
    forbidden = sorted(set(params) & set(FORBIDDEN_CONDITION_API_PARAMETERS))
    if forbidden:
        raise BnTargetLabelQuarantineFailure(f"forbidden target-label parameters present: {forbidden}")
    checks.append("target_label_and_selection_parameters_absent")
    return BnTargetLabelQuarantineResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        callable_name=getattr(fn, "__name__", "<callable>"),
    )
