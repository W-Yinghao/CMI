"""Prospective theorem-specific C85V verdict rules."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any, Mapping

from .c85_decision_experiments import DecisionContractError
from .c85v_stage_a_derivation import replay_stage_a_freeze
from .c85v_stage_b_candidate_audit import replay_stage_b_freeze
from .c85v_statement_registry import THEOREM_IDS, canonical_json_bytes, sha256_file


ADJUDICATION_MANIFEST_SCHEMA = "c85v_adjudication_manifest_v1"
ALLOWED_STATUSES = {
    "T1": {"PROVED", "PROVED_FINITE_MODEL_ONLY", "OPEN", "INVALIDATED"},
    "T2": {"COUNTEREXAMPLE", "OPEN", "INVALIDATED"},
    "T3": {"PROVED", "PROVED_FINITE_MODEL_ONLY", "OPEN", "INVALIDATED"},
    "T4": {"PROVED", "PROVED_FINITE_MODEL_ONLY", "OPEN", "INVALIDATED"},
    "T5": {"PROVED", "PROVED_FINITE_MODEL_ONLY", "OPEN", "INVALIDATED"},
    "T6": {"COUNTEREXAMPLE", "OPEN", "INVALIDATED"},
    "T7": {"PROVED", "PROVED_FINITE_MODEL_ONLY", "OPEN", "INVALIDATED"},
}
NONBLOCKING_GAPS = {
    "NONE",
    "EXPOSITION_ONLY",
    "MISSING_ASSUMPTION_BUT_STATEMENT_SUFFICIENT",
}


def _registered_authorization_valid(receipt: Mapping[str, Any] | None) -> bool:
    return bool(
        receipt
        and receipt.get("schema_version") == "c85v_authorization_consumption_receipt_v1"
        and receipt.get("authorized_stage") == "C85V"
        and receipt.get("direct_explicit_PI_authorization") is True
    )


def adjudicate_theorem(
    *,
    stage_a: Mapping[str, Any],
    comparison: Mapping[str, Any],
    adversarial: Mapping[str, Any],
    review_mode: str,
) -> dict[str, Any]:
    theorem_id = str(stage_a.get("theorem_id"))
    if theorem_id not in THEOREM_IDS:
        raise DecisionContractError("C85V adjudication theorem ID drifted")
    if (
        comparison.get("theorem_id") != theorem_id
        or adversarial.get("theorem_id") != theorem_id
        or stage_a.get("statement_sha256") != comparison.get("statement_sha256")
    ):
        raise DecisionContractError("C85V adjudication input identity drifted")
    if (
        stage_a.get("formal_status_after_stage_A") != "OPEN"
        or comparison.get("formal_status_after_stage_B") != "OPEN"
    ):
        raise DecisionContractError("C85V pre-adjudication status is not OPEN")
    gap = str(comparison.get("candidate_gap_label"))
    scope = str(stage_a.get("derivation_scope"))
    unresolved = list(stage_a.get("unresolved_gaps", []))
    reasons: list[str] = []
    if adversarial.get("statement_false") is True or gap == "FALSE_STATEMENT":
        status = "INVALIDATED"
        reasons.append("The frozen statement failed an exact adversarial check.")
    elif theorem_id in {"T2", "T6"}:
        if adversarial.get("exact_counterexample_satisfied") is True and gap in NONBLOCKING_GAPS:
            status = "COUNTEREXAMPLE"
            reasons.append("The exact frozen construction satisfies the counterexample statement.")
        else:
            status = "OPEN"
            reasons.append("The exact counterexample or its candidate comparison remains incomplete.")
    elif theorem_id == "T5" and (
        adversarial.get("frozen_statement_sufficient_for_transition") is False
        or gap == "INCOMPLETE_OPEN"
        or unresolved
    ):
        status = "OPEN"
        reasons.append("The frozen statement lacks a derivable decoder or complete finite Fano conditions; review cannot repair it.")
    elif gap not in NONBLOCKING_GAPS or not adversarial.get("adversarial_checks_pass"):
        status = "OPEN"
        reasons.append("A substantive candidate or adversarial-review gap remains unresolved.")
    elif scope == "GENERAL" and not unresolved:
        status = "PROVED"
        reasons.append("The independent general derivation and adversarial comparison cover the frozen statement and assumptions.")
    elif scope == "EXACT_FINITE" and not unresolved:
        status = "PROVED_FINITE_MODEL_ONLY"
        reasons.append("Only the registered finite model is established exactly.")
    else:
        status = "OPEN"
        reasons.append("The derivation scope is insufficient for a permitted transition.")
    if status not in ALLOWED_STATUSES[theorem_id]:
        raise DecisionContractError("C85V adjudication produced a forbidden status")
    return {
        "schema_version": "c85v_theorem_verdict_v1",
        "review_role": "ADJUDICATOR",
        "review_mode": review_mode,
        "theorem_id": theorem_id,
        "statement_sha256": stage_a["statement_sha256"],
        "candidate_sha256": comparison["candidate_sha256"],
        "candidate_gap_label": gap,
        "formal_status_entering": "OPEN",
        "formal_status": status,
        "reasons": reasons,
        "reviewer_A_artifact_retained": True,
        "reviewer_B_artifact_retained": True,
        "majority_vote_used": False,
        "monte_carlo_rerun": 0,
    }


def _atomic_stage_directory(output_root: Path) -> tuple[Path, Path]:
    final = output_root.resolve()
    staging = final.with_name(f".{final.name}.staging")
    if final.exists() or staging.exists():
        raise DecisionContractError("C85V adjudication root must be fresh")
    staging.mkdir(parents=True)
    return staging, final


def freeze_adjudication(
    *,
    stage_a_root: Path,
    stage_b_root: Path,
    output_root: Path,
    review_mode: str,
    authorization_receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if review_mode == "REGISTERED_C85V":
        if not _registered_authorization_valid(authorization_receipt):
            raise DecisionContractError("registered C85V adjudication requires consumed authorization")
    elif review_mode == "SHADOW_C85VP":
        if authorization_receipt is not None:
            raise DecisionContractError("shadow C85VP adjudication cannot receive authorization")
    else:
        raise DecisionContractError("C85V adjudication review mode is invalid")
    stage_a_manifest = replay_stage_a_freeze(
        stage_a_root, expected_review_mode=review_mode
    )
    stage_b_manifest = replay_stage_b_freeze(
        stage_b_root,
        stage_a_root=stage_a_root,
        expected_review_mode=review_mode,
    )
    a_rows = {row["theorem_id"]: row for row in stage_a_manifest["derivations"]}
    b_rows = {row["theorem_id"]: row for row in stage_b_manifest["comparisons"]}
    audit_rows = {
        row["theorem_id"]: row for row in stage_b_manifest["adversarial_audits"]
    }
    staging, final = _atomic_stage_directory(output_root)
    verdict_rows: list[dict[str, Any]] = []
    registry_rows: list[dict[str, str]] = []
    for theorem_id in THEOREM_IDS:
        stage_a = json.loads((stage_a_root / a_rows[theorem_id]["path"]).read_text())
        comparison = json.loads((stage_b_root / b_rows[theorem_id]["path"]).read_text())
        adversarial = json.loads(
            (stage_b_root / audit_rows[theorem_id]["path"]).read_text()
        )
        verdict = adjudicate_theorem(
            stage_a=stage_a,
            comparison=comparison,
            adversarial=adversarial,
            review_mode=review_mode,
        )
        path = staging / f"{theorem_id}_final_verdict.json"
        path.write_bytes(canonical_json_bytes(verdict))
        verdict_rows.append({
            "theorem_id": theorem_id,
            "path": path.name,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
        registry_rows.append({
            "theorem_id": theorem_id,
            "historical_status": "OPEN",
            "formal_status": str(verdict["formal_status"]),
            "statement_sha256": str(verdict["statement_sha256"]),
            "candidate_sha256": str(verdict["candidate_sha256"]),
        })
    registry_path = staging / "formal_theorem_status_registry.csv"
    with registry_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=tuple(registry_rows[0]))
        writer.writeheader()
        writer.writerows(registry_rows)
    manifest = {
        "schema_version": ADJUDICATION_MANIFEST_SCHEMA,
        "review_mode": review_mode,
        "review_role": "ADJUDICATOR",
        "stage_a_manifest_sha256": sha256_file(
            stage_a_root / "C85V_STAGE_A_DERIVATION_MANIFEST.json"
        ),
        "stage_b_manifest_sha256": sha256_file(
            stage_b_root / "C85V_STAGE_B_COMPARISON_MANIFEST.json"
        ),
        "verdict_count": len(verdict_rows),
        "majority_vote_used": False,
        "monte_carlo_reruns": 0,
        "verdicts": verdict_rows,
        "status_registry": {
            "path": registry_path.name,
            "size_bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
        },
    }
    manifest_path = staging / "C85V_ADJUDICATION_MANIFEST.json"
    manifest_path.write_bytes(canonical_json_bytes(manifest))
    (staging / "C85V_ADJUDICATION_MANIFEST.sha256").write_text(
        f"{sha256_file(manifest_path)}  {manifest_path.name}\n"
    )
    os.replace(staging, final)
    return replay_adjudication(
        final,
        stage_a_root=stage_a_root,
        stage_b_root=stage_b_root,
        expected_review_mode=review_mode,
    )


def replay_adjudication(
    root: Path,
    *,
    stage_a_root: Path,
    stage_b_root: Path,
    expected_review_mode: str,
) -> dict[str, Any]:
    replay_stage_a_freeze(stage_a_root, expected_review_mode=expected_review_mode)
    replay_stage_b_freeze(
        stage_b_root,
        stage_a_root=stage_a_root,
        expected_review_mode=expected_review_mode,
    )
    path = root / "C85V_ADJUDICATION_MANIFEST.json"
    sidecar = root / "C85V_ADJUDICATION_MANIFEST.sha256"
    if not path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85V adjudication freeze is incomplete")
    if sidecar.read_text().split()[0] != sha256_file(path):
        raise DecisionContractError("C85V adjudication manifest sidecar drifted")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema_version") != ADJUDICATION_MANIFEST_SCHEMA
        or manifest.get("review_mode") != expected_review_mode
        or manifest.get("verdict_count") != 7
        or manifest.get("majority_vote_used") is not False
        or manifest.get("monte_carlo_reruns") != 0
    ):
        raise DecisionContractError("C85V adjudication contract drifted")
    rows = manifest.get("verdicts")
    if not isinstance(rows, list) or {row.get("theorem_id") for row in rows} != set(THEOREM_IDS):
        raise DecisionContractError("C85V adjudication theorem coverage drifted")
    for row in rows:
        artifact = root / str(row["path"])
        if (
            not artifact.is_file()
            or artifact.stat().st_size != row.get("size_bytes")
            or sha256_file(artifact) != row.get("sha256")
        ):
            raise DecisionContractError("C85V adjudication artifact identity drifted")
        verdict = json.loads(artifact.read_text())
        if verdict.get("formal_status") not in ALLOWED_STATUSES[str(row["theorem_id"])]:
            raise DecisionContractError("C85V formal theorem status is invalid")
    return manifest
