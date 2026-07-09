"""No-new-method guard for active TTA-MECH configs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


class NoNewMethodFailure(AssertionError):
    pass


FORBIDDEN_ACTIVE_TERMS: tuple[str, ...] = (
    "adapter",
    "learned_operator",
    "mask",
    "prune",
    "surgery",
    "cmi_control",
    "safety_gate",
    "router",
)


@dataclass(frozen=True)
class NoNewMethodResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]
    forbidden_terms: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _walk_strings(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, str(key)
            yield from _walk_strings(val, path)
    elif isinstance(obj, list) or isinstance(obj, tuple):
        for idx, val in enumerate(obj):
            yield from _walk_strings(val, f"{prefix}[{idx}]")
    elif isinstance(obj, str):
        yield prefix, obj


def validate_no_new_method(active_payload: dict[str, Any]) -> NoNewMethodResult:
    checks: list[str] = []
    warnings: list[str] = []
    hits = []
    for path, text in _walk_strings(active_payload):
        low = text.lower()
        for term in FORBIDDEN_ACTIVE_TERMS:
            if term in low:
                hits.append(f"{path}:{term}")
    if hits:
        raise NoNewMethodFailure(f"forbidden active method terms: {hits}")
    checks.append("no_forbidden_active_method_terms")
    if active_payload.get("new_method_claim") is not False:
        raise NoNewMethodFailure("new_method_claim must be false")
    checks.append("new_method_claim_false")
    return NoNewMethodResult(
        passed=True,
        checks=tuple(checks),
        warnings=tuple(warnings),
        forbidden_terms=FORBIDDEN_ACTIVE_TERMS,
    )
