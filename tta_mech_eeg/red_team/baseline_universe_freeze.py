"""Baseline-universe freeze checks for TTA-MECH."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tta_mech_eeg.baselines.registry import ALLOWED_BASELINES, stable_hash


FORBIDDEN_BASELINES: tuple[str, ...] = (
    "TALOS",
    "CEDAR",
    "CITA_CMI",
    "CMI_REGULARIZED",
    "LOW_RANK_ADAPTER",
    "NEW_METHOD",
    "SAFETY_GATE",
)


class BaselineUniverseFailure(AssertionError):
    pass


@dataclass(frozen=True)
class BaselineUniverseResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    baseline_registry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_baseline_universe(payload: dict[str, Any]) -> BaselineUniverseResult:
    checks: list[str] = []
    warnings: list[str] = []
    allowed = tuple(payload.get("allowed_baselines", ()))
    if allowed != ALLOWED_BASELINES:
        raise BaselineUniverseFailure(f"allowed baseline universe changed: {allowed}")
    checks.append("allowed_baselines_exact")
    names = tuple(entry.get("name") for entry in payload.get("entries", []))
    if names != ALLOWED_BASELINES:
        raise BaselineUniverseFailure(f"registry entry order changed: {names}")
    checks.append("entry_names_exact")
    forbidden_present = sorted(set(names) & set(FORBIDDEN_BASELINES))
    if forbidden_present:
        raise BaselineUniverseFailure(f"forbidden baselines present: {forbidden_present}")
    checks.append("forbidden_baselines_absent")
    for entry in payload.get("entries", []):
        if entry.get("type") != "existing_baseline":
            raise BaselineUniverseFailure(f"{entry.get('name')} is not an existing baseline")
        if entry.get("requires_target_y") is not False:
            raise BaselineUniverseFailure(f"{entry.get('name')} requires target labels")
        if entry.get("allowed_in_tta_mech") is not True:
            raise BaselineUniverseFailure(f"{entry.get('name')} not allowed in TTA-MECH")
    checks.append("entries_existing_label_free_allowed")
    if payload.get("runtime_addition_allowed") is not False:
        raise BaselineUniverseFailure("runtime baseline addition must be disabled")
    checks.append("runtime_addition_disabled")
    expected = dict(payload)
    observed_hash = expected.pop("baseline_registry_hash", None)
    if observed_hash != stable_hash(expected):
        raise BaselineUniverseFailure("baseline_registry_hash mismatch")
    checks.append("baseline_registry_hash_recomputable")
    return BaselineUniverseResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        baseline_registry_hash=str(observed_hash),
    )
