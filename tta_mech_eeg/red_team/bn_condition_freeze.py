"""Condition-universe freeze checks for TTA_MECH_02B0."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash
from tta_mech_eeg.normalization.condition_registry import ALLOWED_CONDITIONS, FORBIDDEN_CONDITIONS


class BnConditionFreezeFailure(AssertionError):
    pass


@dataclass(frozen=True)
class BnConditionFreezeResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    condition_registry_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_bn_condition_freeze(payload: dict[str, Any]) -> BnConditionFreezeResult:
    checks: list[str] = []
    warnings: list[str] = []
    allowed = tuple(payload.get("allowed_conditions", ()))
    if allowed != ALLOWED_CONDITIONS:
        raise BnConditionFreezeFailure(f"allowed condition universe changed: {allowed}")
    checks.append("allowed_conditions_exact")
    names = tuple(entry.get("name") for entry in payload.get("entries", []))
    if names != ALLOWED_CONDITIONS:
        raise BnConditionFreezeFailure(f"condition entry order changed: {names}")
    checks.append("entry_names_exact")
    forbidden_present = sorted(set(names) & set(FORBIDDEN_CONDITIONS))
    if forbidden_present:
        raise BnConditionFreezeFailure(f"forbidden conditions present: {forbidden_present}")
    checks.append("forbidden_conditions_absent")
    for entry in payload.get("entries", []):
        if entry.get("target_labels_allowed") is not False:
            raise BnConditionFreezeFailure(f"{entry.get('name')} allows target labels")
        if entry.get("deployment_selection_allowed") is not False:
            raise BnConditionFreezeFailure(f"{entry.get('name')} allows deployment selection")
        if entry.get("mutates_original_model_allowed") is not False:
            raise BnConditionFreezeFailure(f"{entry.get('name')} mutates original model")
        if entry.get("mutates_weights_allowed") is not False:
            raise BnConditionFreezeFailure(f"{entry.get('name')} mutates weights")
    checks.append("entries_label_free_no_deployment_no_weight_mutation")
    if payload.get("runtime_addition_allowed") is not False:
        raise BnConditionFreezeFailure("runtime condition addition must be disabled")
    checks.append("runtime_addition_disabled")
    expected = dict(payload)
    observed_hash = expected.pop("condition_registry_hash", None)
    if observed_hash != stable_hash(expected):
        raise BnConditionFreezeFailure("condition_registry_hash mismatch")
    checks.append("condition_registry_hash_recomputable")
    return BnConditionFreezeResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        condition_registry_hash=str(observed_hash),
    )
