"""Variant-universe freeze checks for TALOS_00A."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from talos_eeg.adapters.trust_region import TrustRegionBounds, stable_payload_hash


ALLOWED_VARIANTS: tuple[str, ...] = (
    "ERM",
    "TTA_CONTROL_REPLAY",
    "TALOS_L",
    "TALOS_D",
    "TALOS_LD",
)


class VariantFreezeFailure(AssertionError):
    pass


@dataclass(frozen=True)
class VariantFreezeResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    variant_universe_hash: str
    allowed_variants: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_variant_freeze_config(
    *,
    seed: int = 0,
    bounds: TrustRegionBounds | None = None,
    steps: int = 1,
) -> dict[str, Any]:
    if bounds is None:
        bounds = TrustRegionBounds()
    payload = {
        "phase": "TALOS_00A_adapter_implementation_red_team_preflight",
        "allowed_variants": list(ALLOWED_VARIANTS),
        "forbidden_variants": ["TALOS_LR", "TALOS_full", "CMI", "CEDAR_mask", "safety_gate"],
        "trust_region_bounds": bounds.to_dict(),
        "optimization_steps": int(steps),
        "seed": int(seed),
        "target_labels_allowed_for": ["final_metrics_only"],
        "runtime_variant_addition_allowed": False,
    }
    payload["variant_universe_hash"] = stable_payload_hash(payload)
    return payload


def validate_variant_freeze(payload: dict[str, Any]) -> VariantFreezeResult:
    checks: list[str] = []
    warnings: list[str] = []
    variants = tuple(payload.get("allowed_variants", ()))
    if variants != ALLOWED_VARIANTS:
        raise VariantFreezeFailure(f"allowed_variants changed: {variants}")
    checks.append("allowed_variants_exact")
    if payload.get("runtime_variant_addition_allowed") is not False:
        raise VariantFreezeFailure("runtime variant addition must be disabled")
    checks.append("runtime_addition_disabled")
    forbidden = set(payload.get("forbidden_variants", ()))
    required_forbidden = {"TALOS_LR", "TALOS_full", "CMI", "CEDAR_mask", "safety_gate"}
    missing = sorted(required_forbidden - forbidden)
    if missing:
        raise VariantFreezeFailure(f"missing forbidden variants: {missing}")
    checks.append("forbidden_variants_recorded")
    if payload.get("target_labels_allowed_for") != ["final_metrics_only"]:
        raise VariantFreezeFailure("target label use must be final_metrics_only")
    checks.append("target_labels_final_metrics_only")
    expected = dict(payload)
    observed_hash = expected.pop("variant_universe_hash", None)
    recomputed = stable_payload_hash(expected)
    if observed_hash != recomputed:
        raise VariantFreezeFailure("variant_universe_hash mismatch")
    checks.append("variant_universe_hash_recomputable")
    return VariantFreezeResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        variant_universe_hash=str(observed_hash),
        allowed_variants=variants,
    )
