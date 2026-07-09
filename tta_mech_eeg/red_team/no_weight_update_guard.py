"""No-weight-update guard for future normalization / BN audits."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class NoWeightUpdateFailure(AssertionError):
    pass


FORBIDDEN_UPDATE_TOKENS: tuple[str, ...] = (
    "optimizer.step",
    "loss.backward",
    "backward",
    "encoder weight update",
    "classifier weight update",
    "checkpoint overwrite",
    "requires_grad=true",
    "mutate original model",
)


@dataclass(frozen=True)
class NoWeightUpdateResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    forbidden_update_tokens: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _walk_strings(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield from _walk_strings(val, path)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for idx, val in enumerate(obj):
            yield from _walk_strings(val, f"{prefix}[{idx}]")
    elif isinstance(obj, str):
        yield prefix, obj


def validate_no_weight_update_guard(payload: dict[str, Any]) -> NoWeightUpdateResult:
    checks: list[str] = []
    warnings: list[str] = []
    hits = []
    for path, text in _walk_strings(payload):
        low = text.lower()
        for token in FORBIDDEN_UPDATE_TOKENS:
            if token in low:
                hits.append(f"{path}:{token}")
    if hits:
        raise NoWeightUpdateFailure(f"forbidden weight-update terms: {hits}")
    checks.append("no_forbidden_update_terms")

    for entry in payload.get("entries", []):
        name = entry.get("name")
        if entry.get("mutates_original_model_allowed") is not False:
            raise NoWeightUpdateFailure(f"{name} may mutate original model")
        if entry.get("mutates_weights_allowed") is not False:
            raise NoWeightUpdateFailure(f"{name} may mutate weights")
        if entry.get("target_labels_allowed") is not False:
            raise NoWeightUpdateFailure(f"{name} allows target labels")
        if entry.get("deployment_selection_allowed") is not False:
            raise NoWeightUpdateFailure(f"{name} allows deployment selection")
        if entry.get("train_mode_allowed") is True:
            if entry.get("copy_only_mutation_required") is not True:
                raise NoWeightUpdateFailure(f"{name} train mode lacks copy-only requirement")
            if entry.get("mutates_bn_buffers_allowed") is not True:
                raise NoWeightUpdateFailure(f"{name} train mode does not restrict mutation to BN buffers")
            if entry.get("dropout_disabled_required") is not True:
                raise NoWeightUpdateFailure(f"{name} train mode lacks dropout-disabled requirement")
            if entry.get("parameters_require_grad_false_required") is not True:
                raise NoWeightUpdateFailure(f"{name} train mode lacks parameter freeze requirement")
    checks.append("entries_forbid_original_model_and_weight_mutation")
    checks.append("train_mode_requires_copy_only_bn_buffers_dropout_disabled_and_frozen_params")

    mutation_rules = payload.get("mutation_rules", {})
    for key in (
        "optimizer_step_allowed",
        "loss_backward_allowed",
        "encoder_weight_update_allowed",
        "classifier_weight_update_allowed",
        "checkpoint_overwrite_allowed",
    ):
        if mutation_rules.get(key) is not False:
            raise NoWeightUpdateFailure(f"{key} must be false")
    checks.append("mutation_rules_forbid_optimizer_backward_weight_update_checkpoint_overwrite")
    return NoWeightUpdateResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        forbidden_update_tokens=FORBIDDEN_UPDATE_TOKENS,
    )
