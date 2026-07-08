"""Red-team checks for CEDAR P0 JSON reports.

The checks are intentionally conservative. They validate claim hygiene and
selection discipline; they do not certify target generalization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


class RedTeamFailure(AssertionError):
    pass


@dataclass(frozen=True)
class RedTeamResult:
    passed: bool
    checks: tuple[str, ...]
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


FORBIDDEN_ARTIFACT_KEYS = {
    "checkpoint",
    "checkpoint_path",
    "model_path",
    "selector_path",
    "mask_path",
    "artifact_path",
}


def _walk_dict(obj: Any, prefix: str = ""):
    if isinstance(obj, dict):
        for key, val in obj.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            yield path, key, val
            yield from _walk_dict(val, path)
    elif isinstance(obj, list):
        for idx, val in enumerate(obj):
            yield from _walk_dict(val, f"{prefix}[{idx}]")


def _require(cond: bool, msg: str) -> None:
    if not cond:
        raise RedTeamFailure(msg)


def _load(path_or_obj: str | Path | dict[str, Any]) -> dict[str, Any]:
    if isinstance(path_or_obj, dict):
        return path_or_obj
    with Path(path_or_obj).open() as f:
        return json.load(f)


def validate_p0_result(
    path_or_obj: str | Path | dict[str, Any],
    *,
    max_perm_null_adv: float = 0.05,
) -> RedTeamResult:
    """Validate a P0 result JSON against CEDAR claim-hygiene rules."""

    data = _load(path_or_obj)
    checks: list[str] = []
    warnings: list[str] = []

    _require(data.get("project") == "CEDAR-EEG", "unexpected project field")
    _require(data.get("phase") == "P0_frozen_latent", "unexpected phase field")
    checks.append("schema_project_phase")

    claim = str(data.get("claim_boundary", "")).lower()
    _require("evaluation-only" in claim, "claim boundary must mark target metrics evaluation-only")
    _require("not a target-generalization guarantee" in claim, "claim boundary must reject target guarantee")
    checks.append("claim_boundary_no_generalization")

    baseline = data.get("baseline", {})
    _require("permutation_null" in baseline, "missing permutation null")
    perm_adv = float(baseline["permutation_null"].get("advantage_mean", 0.0))
    _require(abs(perm_adv) <= max_perm_null_adv, f"permutation null advantage too large: {perm_adv:.3f}")
    checks.append("permutation_null_low")

    candidates = data.get("candidates", [])
    _require(isinstance(candidates, list) and candidates, "missing candidates")
    for rec in candidates:
        cand = rec.get("candidate", {})
        _require("random_control_drop_frac" in cand, "candidate missing random-control field")
        _require("target_bacc_drop" in cand, "candidate missing target diagnostic field")
        reason_text = " ".join(str(x).lower() for x in rec.get("reasons", []))
        _require("target" not in reason_text, "target label metric appears in source-side decision reasons")
    checks.append("target_labels_quarantined_from_decisions")
    checks.append("random_control_present")

    selected = data.get("selected")
    accepted = [rec for rec in candidates if rec.get("decision") == "ACCEPT"]
    if selected is not None:
        _require(selected.get("decision") == "ACCEPT", "selected candidate is not ACCEPT")
        best_accept = max(accepted, key=lambda r: float(r.get("utility", float("-inf"))))
        _require(
            selected.get("candidate", {}).get("name") == best_accept.get("candidate", {}).get("name"),
            "selected candidate is not the highest-utility ACCEPT candidate",
        )
    else:
        warnings.append("no accepted candidate; report-only or abstention outcome")
    checks.append("source_only_selected_candidate")

    for path, key, val in _walk_dict(data):
        if str(key) in FORBIDDEN_ARTIFACT_KEYS:
            _require(val in ("", None), f"forbidden deployable artifact key populated at {path}")
    checks.append("no_deployable_artifact_emitted")

    if not data.get("groups_present", False):
        warnings.append("groups not present; grouped cross-fit could not be audited from JSON")
    else:
        checks.append("groups_present_for_grouped_crossfit")

    return RedTeamResult(passed=True, checks=tuple(checks), warnings=tuple(warnings))
