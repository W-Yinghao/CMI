"""Target-label quarantine API contract for TTA-MECH replay harnesses."""

from __future__ import annotations

import inspect
from dataclasses import asdict, dataclass
from typing import Any, Callable


class TargetLabelContractFailure(AssertionError):
    pass


FORBIDDEN_REPLAY_PARAMETERS: tuple[str, ...] = (
    "target_y",
    "y_target",
    "target_metric",
    "target_selected_variant",
)


@dataclass(frozen=True)
class TargetLabelContractResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    callable_name: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_target_label_contract(fn: Callable[..., Any]) -> TargetLabelContractResult:
    checks: list[str] = []
    warnings: list[str] = []
    sig = inspect.signature(fn)
    params = tuple(sig.parameters)
    for required in ("source_state", "target_x", "baseline"):
        if required not in params:
            raise TargetLabelContractFailure(f"missing required replay parameter: {required}")
    checks.append("required_parameters_present")
    forbidden = sorted(set(params) & set(FORBIDDEN_REPLAY_PARAMETERS))
    if forbidden:
        raise TargetLabelContractFailure(f"forbidden target-label parameters present: {forbidden}")
    checks.append("target_label_parameters_absent")
    return TargetLabelContractResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        callable_name=getattr(fn, "__name__", "<callable>"),
    )
