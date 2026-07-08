"""Red-team checks for CEDAR P0 JSON reports.

The checks are intentionally conservative. They validate claim hygiene and
selection discipline; they do not certify target generalization.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from cedar_eeg.surgery.selection import (
    score_candidate_dict,
    select_best_accept,
    source_side_rank_components,
)


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

REQUIRED_CANDIDATE_KEYS = {
    "name",
    "dropped_units",
    "leakage_before",
    "leakage_after",
    "leakage_drop",
    "leakage_drop_frac",
    "source_bacc_before",
    "source_bacc_after",
    "source_bacc_drop",
    "target_bacc_drop",
    "r3_delta",
    "stability",
    "random_control_drop_frac",
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


def _candidate_signature(record: dict[str, Any] | None) -> dict[str, Any] | None:
    if record is None:
        return None
    candidate = record.get("candidate", {})
    return {
        "decision": record.get("decision"),
        "name": candidate.get("name"),
        "dropped_units": tuple(candidate.get("dropped_units", ())),
        "rank_key": source_side_rank_components(record),
    }


def _perturb_target_metrics(obj: Any) -> Any:
    """Replace target diagnostic values while preserving structure."""

    if isinstance(obj, dict):
        out = {}
        for key, val in obj.items():
            key_text = str(key).lower()
            if "target" in key_text:
                if isinstance(val, (int, float)) or val is None:
                    out[key] = -999999.0
                elif isinstance(val, list):
                    out[key] = list(reversed(val))
                elif isinstance(val, tuple):
                    out[key] = tuple(reversed(val))
                else:
                    out[key] = "TARGET_PERTURBED"
            else:
                out[key] = _perturb_target_metrics(val)
        return out
    if isinstance(obj, list):
        return [_perturb_target_metrics(x) for x in obj]
    return obj


def _require_candidate_completeness(rec: dict[str, Any]) -> None:
    cand = rec.get("candidate", {})
    _require(isinstance(cand, dict), "candidate record missing candidate dictionary")
    missing = sorted(REQUIRED_CANDIDATE_KEYS - set(cand))
    _require(not missing, f"candidate {cand.get('name', '<unknown>')} missing fields: {missing}")

    _require("decision" in rec, f"candidate {cand.get('name')} missing decision")
    _require("reasons" in rec, f"candidate {cand.get('name')} missing reasons")
    _require("utility" in rec, f"candidate {cand.get('name')} missing utility")
    expected_utility = score_candidate_dict(cand)
    observed_utility = float(rec["utility"])
    _require(
        abs(observed_utility - expected_utility) <= 1e-6 * max(1.0, abs(expected_utility)),
        f"candidate {cand.get('name')} utility is not source-side recomputable",
    )

    random_control = rec.get("random_control")
    _require(isinstance(random_control, dict), f"candidate {cand.get('name')} missing random control metadata")
    for key in ("matched_k", "drop_frac", "repeats"):
        _require(key in random_control, f"candidate {cand.get('name')} random control missing {key}")

    grouped_split = rec.get("grouped_split")
    _require(isinstance(grouped_split, dict), f"candidate {cand.get('name')} missing grouped split metadata")
    _require(grouped_split.get("groups_present") is True, f"candidate {cand.get('name')} lacks grouped split")
    _require(int(grouped_split.get("n_groups", 0)) >= 2, f"candidate {cand.get('name')} has <2 groups")

    perm = rec.get("permutation_null")
    _require(isinstance(perm, dict), f"candidate {cand.get('name')} missing permutation null")
    _require("advantage_mean" in perm, f"candidate {cand.get('name')} permutation null missing advantage_mean")

    source_delta = rec.get("source_utility_delta")
    _require(isinstance(source_delta, dict), f"candidate {cand.get('name')} missing source utility delta")
    for key in ("source_bacc_drop", "r3_delta"):
        _require(key in source_delta, f"candidate {cand.get('name')} source utility delta missing {key}")

    reported_rank = rec.get("rank_key")
    _require(isinstance(reported_rank, dict), f"candidate {cand.get('name')} missing rank key")
    expected_rank = source_side_rank_components(rec)
    _require(reported_rank == expected_rank, f"candidate {cand.get('name')} rank key mismatch")


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
        _require_candidate_completeness(rec)
        cand = rec.get("candidate", {})
        _require("random_control_drop_frac" in cand, "candidate missing random-control field")
        _require("target_bacc_drop" in cand, "candidate missing target diagnostic field")
        reason_text = " ".join(str(x).lower() for x in rec.get("reasons", []))
        _require("target" not in reason_text, "target label metric appears in source-side decision reasons")
    checks.append("target_labels_quarantined_from_decisions")
    checks.append("random_control_present")
    checks.append("candidate_completeness")

    selected = data.get("selected")
    deterministic_selected = select_best_accept(candidates)
    if selected is not None:
        _require(selected.get("decision") == "ACCEPT", "selected candidate is not ACCEPT")
        _require(deterministic_selected is not None, "selected candidate reported but no ACCEPT candidate exists")
        _require(
            _candidate_signature(selected) == _candidate_signature(deterministic_selected),
            "selected candidate does not match deterministic source-side rank key",
        )
    else:
        warnings.append("no accepted candidate; report-only or abstention outcome")
    checks.append("source_only_selected_candidate")
    checks.append("tie_break_determinism")

    perturbed = _perturb_target_metrics(candidates)
    _require(
        _candidate_signature(select_best_accept(perturbed)) == _candidate_signature(deterministic_selected),
        "target perturbation changed selected candidate",
    )
    checks.append("target_perturbation_invariance")

    for path, key, val in _walk_dict(data):
        if str(key) in FORBIDDEN_ARTIFACT_KEYS:
            _require(val in ("", None), f"forbidden deployable artifact key populated at {path}")
    checks.append("no_deployable_artifact_emitted")

    _require(data.get("groups_present", False), "groups not present; grouped cross-fit is mandatory")
    grouped_split = data.get("grouped_split", {})
    _require(isinstance(grouped_split, dict), "missing top-level grouped split metadata")
    _require(grouped_split.get("groups_present") is True, "top-level grouped split not present")
    _require(int(grouped_split.get("n_groups", 0)) >= 2, "top-level grouped split has <2 groups")
    checks.append("groups_present_for_grouped_crossfit")

    return RedTeamResult(passed=True, checks=tuple(checks), warnings=tuple(warnings))
