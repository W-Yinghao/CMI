"""Adapter determinism checks for TALOS_00A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class AdapterDeterminismFailure(AssertionError):
    pass


@dataclass(frozen=True)
class AdapterDeterminismResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _variant_fingerprints(payload: dict[str, Any]) -> dict[str, dict[str, str]]:
    out = {}
    for rec in payload.get("variants", []):
        out[str(rec["variant"])] = {
            "adapter_state_hash": str(rec["adapter_state_hash"]),
            "predictions_hash": str(rec["predictions_hash"]),
            "proba_hash": str(rec["proba_hash"]),
            "metrics_before_final_y_hash": str(rec["metrics_before_final_y_hash"]),
        }
    return out


def validate_adapter_determinism(left: dict[str, Any], right: dict[str, Any]) -> AdapterDeterminismResult:
    checks: list[str] = []
    warnings: list[str] = []
    if left.get("scenario") != right.get("scenario"):
        raise AdapterDeterminismFailure("scenario mismatch")
    checks.append("scenario_match")
    if left.get("variant_ranking") != right.get("variant_ranking"):
        raise AdapterDeterminismFailure("variant ranking changed")
    checks.append("variant_ranking_identical")
    if left.get("reported_variant") != right.get("reported_variant"):
        raise AdapterDeterminismFailure("reported variant changed")
    checks.append("reported_variant_identical")
    if _variant_fingerprints(left) != _variant_fingerprints(right):
        raise AdapterDeterminismFailure("adapter or prediction hashes changed")
    checks.append("adapter_prediction_probability_hashes_identical")
    return AdapterDeterminismResult(passed=True, checks=tuple(checks), warnings=tuple(warnings))
