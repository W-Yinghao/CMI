"""Target-label quarantine checks for TALOS_00A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class TargetLabelQuarantineFailure(AssertionError):
    pass


@dataclass(frozen=True)
class TargetLabelQuarantineResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    scenarios: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _invariant_signature(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "variant_ranking": payload.get("variant_ranking"),
        "reported_variant": payload.get("reported_variant"),
        "variants": [
            {
                "variant": rec.get("variant"),
                "adapter_state_hash": rec.get("adapter_state_hash"),
                "predictions_hash": rec.get("predictions_hash"),
                "proba_hash": rec.get("proba_hash"),
                "metrics_before_final_y_hash": rec.get("metrics_before_final_y_hash"),
            }
            for rec in payload.get("variants", [])
        ],
    }


def validate_target_label_quarantine(scenarios: dict[str, dict[str, Any]]) -> TargetLabelQuarantineResult:
    required = ("true_y_final_only", "target_y_removed", "target_y_permuted")
    missing = [name for name in required if name not in scenarios]
    if missing:
        raise TargetLabelQuarantineFailure(f"missing scenarios: {missing}")
    checks: list[str] = []
    warnings: list[str] = []
    reference = _invariant_signature(scenarios[required[0]])
    for name in required[1:]:
        if _invariant_signature(scenarios[name]) != reference:
            raise TargetLabelQuarantineFailure(f"target labels changed adapter path for {name}")
    checks.append("adapter_parameters_identical")
    checks.append("adapter_predictions_identical")
    checks.append("adapter_probability_hashes_identical")
    checks.append("variant_ranking_identical")
    checks.append("reported_variant_identical")
    checks.append("pre_final_metric_hashes_identical")

    true_metrics = [rec.get("final_metrics_hash") for rec in scenarios["true_y_final_only"].get("variants", [])]
    removed_metrics = [rec.get("final_metrics_hash") for rec in scenarios["target_y_removed"].get("variants", [])]
    permuted_metrics = [rec.get("final_metrics_hash") for rec in scenarios["target_y_permuted"].get("variants", [])]
    if any(x is not None for x in removed_metrics):
        raise TargetLabelQuarantineFailure("removed-y scenario produced final metric hashes")
    checks.append("removed_y_has_no_final_metrics")
    if true_metrics == permuted_metrics:
        warnings.append("permuted labels did not change final metric hashes")
    else:
        checks.append("only_final_metrics_change_under_permutation")
    return TargetLabelQuarantineResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        scenarios=required,
    )
