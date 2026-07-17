"""Manifest-only C85E frozen-input availability audit.

This module deliberately reads only committed compact reports, schema metadata,
and the two frozen C84S manifests. It never opens a manifest-referenced result
table, Q0 shard, label view, field array, or model artifact.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
C84S_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5")
SELECTION_MANIFEST = (
    C84S_ROOT / "stage_b_selection_freeze/C84S_SELECTION_FREEZE_MANIFEST_V3.json"
)
RESULT_MANIFEST = (
    C84S_ROOT / "stage_c_scientific_result/C84S_RESULT_ARTIFACT_MANIFEST.json"
)
METHOD_CONTEXT_SCHEMA = REPORT_DIR / "c84sr1_tables/result_table_registry.csv"
PROTOCOL = REPORT_DIR / "C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_PROTOCOL.json"

EXPECTED_SHA256 = {
    SELECTION_MANIFEST: "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
    RESULT_MANIFEST: "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
    METHOD_CONTEXT_SCHEMA: "f3d2ae5907f18cc4c4e672aa1f95aa1f7688fc283f55d08cffb830ad1ae50961",
    PROTOCOL: "a42cc71498971ee6eeb75ef53e62744e73e91b92e444ef78c9e4c856d61ac052",
}

FAILURE_GATE = "C85E_FROZEN_CANDIDATE_UTILITY_OR_SELECTION_INPUT_UNAVAILABLE"
EXPECTED_CONTEXTS = 944
EXPECTED_CANDIDATES = 81
EXPECTED_CANDIDATE_CONTEXTS = EXPECTED_CONTEXTS * EXPECTED_CANDIDATES


class C85EInputAvailabilityError(RuntimeError):
    """Raised when a manifest or frozen schema identity drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_exact_json(path: Path) -> dict[str, Any]:
    if not path.is_file() or sha256_file(path) != EXPECTED_SHA256[path]:
        raise C85EInputAvailabilityError(f"frozen metadata identity drifted: {path}")
    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise C85EInputAvailabilityError(f"frozen metadata is not an object: {path}")
    return value


def _load_method_context_schema() -> dict[str, str]:
    path = METHOD_CONTEXT_SCHEMA
    if not path.is_file() or sha256_file(path) != EXPECTED_SHA256[path]:
        raise C85EInputAvailabilityError("method-context schema identity drifted")
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    row = next((item for item in rows if item["table"] == "method_context_decisions.csv"), None)
    if row is None:
        raise C85EInputAvailabilityError("method-context schema row is absent")
    return row


def _artifact_map(manifest: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    artifacts = manifest.get("artifacts")
    if isinstance(artifacts, dict):
        rows = list(artifacts.values())
    elif isinstance(artifacts, list):
        rows = artifacts
    else:
        raise C85EInputAvailabilityError("manifest artifact registry is malformed")
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        if not isinstance(row, Mapping) or not isinstance(row.get("path"), str):
            raise C85EInputAvailabilityError("manifest artifact row is malformed")
        result[str(row["path"])] = row
    if len(result) != len(rows):
        raise C85EInputAvailabilityError("manifest artifact paths are not unique")
    return result


def _has_complete_candidate_utility(
    result_artifacts: Mapping[str, Mapping[str, Any]], method_schema: Mapping[str, str]
) -> tuple[bool, str]:
    explicit_names = {
        "candidate_evaluation_utilities.csv",
        "candidate_held_evaluation_utilities.csv",
        "candidate_utility_vectors.npz",
        "candidate_utility_vectors.json",
    }
    explicit = sorted(
        path for path in result_artifacts if Path(path).name in explicit_names
    )
    count_matches = sorted(
        path
        for path, row in result_artifacts.items()
        if row.get("rows") == EXPECTED_CANDIDATE_CONTEXTS
    )
    fields = set(str(method_schema["fields"]).split("|"))
    vector_fields = fields.intersection(
        {"candidate_utility_vector", "all_candidate_utilities", "evaluation_utility_vector"}
    )
    present = bool(explicit or count_matches or vector_fields)
    evidence = (
        f"explicit_artifacts={explicit}; row_count_{EXPECTED_CANDIDATE_CONTEXTS}={count_matches}; "
        f"method_context_vector_fields={sorted(vector_fields)}"
    )
    return present, evidence


def audit_frozen_input_availability() -> dict[str, Any]:
    selection = _load_exact_json(SELECTION_MANIFEST)
    result = _load_exact_json(RESULT_MANIFEST)
    method_schema = _load_method_context_schema()
    selection_artifacts = _artifact_map(selection)
    result_artifacts = _artifact_map(result)

    utility_present, utility_evidence = _has_complete_candidate_utility(
        result_artifacts, method_schema
    )
    context_rows = selection_artifacts.get("C84S_STAGE_B_CONTEXT_REGISTRY.json", {}).get(
        "rows"
    )
    score_rows = selection_artifacts.get("candidate_scores.csv", {}).get("rows")
    rank_rows = selection_artifacts.get("candidate_ranks.csv", {}).get("rows")
    fixed_rows = selection_artifacts.get("fixed_default_selections.csv", {}).get("rows")
    shard_rows = selection_artifacts.get("q0_selection_shard_index.csv", {}).get("rows")
    inference_rows = {
        "target_level_method_effects.csv": 1416,
        "dataset_Q1_Q2.csv": 18,
        "level_specific_Q1_Q2.csv": 36,
        "panel_seed_stability.csv": 18,
        "leave_one_target_out.csv": 118,
        "label_budget_frontier.csv": 15,
    }
    observed_inference_rows = {
        name: result_artifacts.get(name, {}).get("rows") for name in inference_rows
    }

    rows = [
        {
            "requirement_id": "A1_COMPLETE_CANDIDATE_UTILITY",
            "required": 1,
            "expected": f"{EXPECTED_CONTEXTS} vectors x {EXPECTED_CANDIDATES} candidates ({EXPECTED_CANDIDATE_CONTEXTS} values)",
            "observed": "no explicit artifact, equivalent row registry, or vector field",
            "status": "FAIL_ABSENT_FROM_FROZEN_MANIFEST" if not utility_present else "PASS",
            "evidence": utility_evidence,
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
        {
            "requirement_id": "A2_CANDIDATE_ID_AND_ORDER_METADATA",
            "required": 1,
            "expected": "944 contexts with canonical 81-candidate identity/order metadata",
            "observed": f"context_rows={context_rows}; score_rows={score_rows}; rank_rows={rank_rows}",
            "status": "PASS_BY_MANIFEST_AND_FROZEN_SCHEMA"
            if (context_rows, score_rows, rank_rows) == (944, 535248, 535248)
            else "FAIL_METADATA_COVERAGE",
            "evidence": "selection-freeze manifest only; candidate rows not opened",
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
        {
            "requirement_id": "A3_IMMUTABLE_SELECTION_ACTIONS",
            "required": 1,
            "expected": "4720 fixed rows and 944 Q0 shards / 8750000 records",
            "observed": f"fixed_rows={fixed_rows}; shard_rows={shard_rows}; Q0_records={selection.get('Q0_records')}",
            "status": "PASS_BY_MANIFEST"
            if (fixed_rows, shard_rows, selection.get("Q0_records")) == (4720, 944, 8750000)
            else "FAIL_METADATA_COVERAGE",
            "evidence": "selection-freeze manifest only; Q0 shards not opened",
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
        {
            "requirement_id": "A4_CONTEXT_IDENTITY",
            "required": 1,
            "expected": "944 target/panel/seed/level contexts",
            "observed": f"context_rows={context_rows}; manifest_contexts={selection.get('contexts')}",
            "status": "PASS_BY_MANIFEST"
            if context_rows == selection.get("contexts") == 944
            else "FAIL_METADATA_COVERAGE",
            "evidence": "C84S_STAGE_B_CONTEXT_REGISTRY identity only; registry not opened",
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
        {
            "requirement_id": "A5_FROZEN_TARGET_INFERENCE_COMPONENTS",
            "required": 1,
            "expected": "target effects, Q1/Q2, level, panel/seed, LOTO, frontier",
            "observed": ";".join(
                f"{name}:{observed_inference_rows[name]}" for name in inference_rows
            ),
            "status": "PASS_BY_MANIFEST"
            if observed_inference_rows == inference_rows
            else "FAIL_METADATA_COVERAGE",
            "evidence": "result-artifact manifest only; result tables not opened",
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
        {
            "requirement_id": "A6_NO_LABEL_ROOT_OR_STAGE_C_REOPEN",
            "required": 1,
            "expected": "all C85E inputs available without labels, logits, or Stage-C rerun",
            "observed": "candidate utility unavailable; prohibited reconstruction would be required",
            "status": "FAIL_CANNOT_SATISFY_WITH_FROZEN_OBJECTS"
            if not utility_present
            else "PASS",
            "evidence": "protocol hard gate",
            "full_object_opened": 0,
            "reconstruction_forbidden": 1,
        },
    ]
    failures = [row["requirement_id"] for row in rows if str(row["status"]).startswith("FAIL")]
    return {
        "schema_version": "c85e_frozen_input_availability_audit_v1",
        "status": "BLOCKED" if failures else "AVAILABLE",
        "gate": FAILURE_GATE if failures else "C85E_FROZEN_INPUTS_AVAILABLE",
        "failures": failures,
        "rows": rows,
        "selection_manifest_artifact_count": len(selection_artifacts),
        "result_manifest_artifact_count": len(result_artifacts),
        "candidate_level_objects_opened": 0,
        "chain_level_objects_opened": 0,
        "direct_result_tables_opened": 0,
        "direct_label_or_field_arrays_opened": 0,
        "execution_lock_permitted": not failures,
    }


def frozen_input_identity_rows() -> list[dict[str, Any]]:
    rows = [
        ("C84F_complete_field_manifest", "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/lock_f0c369ee273352b47e36/C84F_COMPLETE_FIELD_MANIFEST.json", "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8", 6334399, "IDENTITY_ONLY_NOT_OPENED"),
        ("C84S_selection_freeze_manifest", str(SELECTION_MANIFEST), EXPECTED_SHA256[SELECTION_MANIFEST], 2161, "MANIFEST_METADATA_OPENED"),
        ("C84S_scientific_result", str(C84S_ROOT / "stage_c_scientific_result/C84S_RESULT.json"), "5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7", 1059, "IDENTITY_ONLY_NOT_OPENED"),
        ("C84S_result_artifact_manifest", str(RESULT_MANIFEST), EXPECTED_SHA256[RESULT_MANIFEST], 2606, "MANIFEST_METADATA_OPENED"),
        ("C84A_compact_report", str(REPORT_DIR / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.json"), "bafdaad0eb18ca17b56aecddc90eee9e9359555d9a5f93e736bfba938fd23d0b", 8358, "COMMITTED_COMPACT_REPORT_OPENED"),
        ("C85T_compact_report", str(REPORT_DIR / "C85T_OVERALL_REPORT.json"), "740552432b838acb8927d83e42431c4897ef238007fee0336ccc1ea3eeb2fd59", 10973, "COMMITTED_COMPACT_REPORT_OPENED"),
        ("C85V_compact_report", str(REPORT_DIR / "C85V_OVERALL_REPORT.json"), "24014f4d5443bb590a1d48788cc0d032bbecc5e285eb9df00799b1620a1f0e51", 3752, "COMMITTED_COMPACT_REPORT_OPENED"),
        ("C84S_method_context_schema", str(METHOD_CONTEXT_SCHEMA), EXPECTED_SHA256[METHOD_CONTEXT_SCHEMA], 2830, "COMMITTED_SCHEMA_METADATA_OPENED"),
        ("C85E_protocol", str(PROTOCOL), EXPECTED_SHA256[PROTOCOL], PROTOCOL.stat().st_size, "COMMITTED_PROTOCOL_OPENED"),
    ]
    result = []
    for object_id, path, expected, size, access_class in rows:
        identity_only = access_class.startswith("IDENTITY_ONLY")
        observed = "NOT_REHASHED_C85EP" if identity_only else sha256_file(Path(path))
        result.append(
            {
                "object_id": object_id,
                "path": path,
                "expected_sha256": expected,
                "observed_sha256": observed,
                "size_bytes": size,
                "access_class": access_class,
                "status": "BOUND_IDENTITY_ONLY_NOT_REHASHED"
                if identity_only
                else ("PASS" if observed == expected else "FAIL"),
            }
        )
    return result


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise C85EInputAvailabilityError(f"refusing to write empty evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = tuple(rows[0])
    if any(tuple(row) != fields for row in rows):
        raise C85EInputAvailabilityError(f"evidence schema drifted: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_availability_evidence(output_dir: Path) -> dict[str, Any]:
    audit = audit_frozen_input_availability()
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "frozen_input_availability_audit.csv", audit["rows"])
    _write_csv(output_dir / "frozen_input_identity_registry.csv", frozen_input_identity_rows())
    return audit


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args(argv)
    audit = write_availability_evidence(args.output_dir)
    print(json.dumps({key: value for key, value in audit.items() if key != "rows"}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
