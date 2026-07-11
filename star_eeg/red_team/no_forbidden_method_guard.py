"""Guard the active method registry and import path, excluding prose."""

import re
from typing import Dict, Iterable, Mapping

from star_eeg.config import ACTIVE_IMPORT_PATHS, ACTIVE_METHOD_REGISTRY


FORBIDDEN_ACTIVE_IDENTIFIERS = (
    "cmi",
    "adversary",
    "pruning",
    "mask",
    "surgery",
    "tta",
    "target entropy",
    "low rank",
    "lora",
    "safety gate",
    "router",
    "csp init",
)


def _normalize(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value).lower()).strip()


def _matches(values: Iterable[object]) -> Dict[str, list]:
    violations = {}
    normalized_forbidden = [(_normalize(token), token) for token in FORBIDDEN_ACTIVE_IDENTIFIERS]
    for value in values:
        normalized = _normalize(value)
        found = [original for token, original in normalized_forbidden if token in normalized]
        if found:
            violations[str(value)] = found
    return violations


def evaluate_no_forbidden_method_guard(
    registry: Mapping[str, object] = None,
    import_paths: Iterable[str] = None,
) -> Dict[str, object]:
    active_registry = dict(registry or ACTIVE_METHOD_REGISTRY)
    active_imports = list(import_paths or ACTIVE_IMPORT_PATHS)
    registry_violations = _matches(active_registry.values())
    import_violations = _matches(active_imports)
    checks = {
        "active_registry_clear": not registry_violations,
        "active_import_path_clear": not import_violations,
        "target_data_access_disabled": active_registry.get("target_data_access") == "none",
    }
    return {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "scope": "active_registry_config_and_import_path_only",
        "active_registry": active_registry,
        "active_import_paths": active_imports,
        "registry_violations": registry_violations,
        "import_violations": import_violations,
        "checks": checks,
    }
