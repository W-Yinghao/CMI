"""Atomic C85E result-bundle publication and semantic replay."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

from .c85e_runtime_guard import (
    ValidatedC85EExecutionContext, canonical_json_bytes,
    revalidate_execution_context, sha256_file,
)


RESULT_SCHEMA = "c85e_frozen_field_decision_theory_bridge_result_v1"
MANIFEST_SCHEMA = "c85e_result_artifact_manifest_v1"
COMPLETION_SCHEMA = "c85e_completion_receipt_v1"
SUCCESS_GATE = "C85E_FROZEN_FIELD_DECISION_THEORY_BRIDGE_COMPLETE_C86_PROTOCOL_REVIEW_REQUIRED"
REGISTERED_TABLES = (
    "realized_action_divergence.csv", "exact_policy_equivalence_classes.csv",
    "action_entropy_and_regime_distribution.csv", "divergent_context_risk_contribution.csv",
    "q0_action_distribution_use.csv", "candidate_gap_geometry.csv",
    "near_optimal_set_grid.csv", "effective_multiplicity_grid.csv",
    "geometry_by_dataset_level.csv", "rank_topk_regret_geometry_separation.csv",
    "geometry_regret_descriptive_association.csv", "leave_target_geometry_stability.csv",
    "target_robust_risk_profile.csv", "cvar_grid.csv", "cott_mean_tail_profile.csv",
    "mano_policy_collapse_risk_profile.csv", "level_robust_risk_interaction.csv",
    "dataset_level_geometry_matrix.csv", "panel_seed_geometry_stability.csv",
    "support_level_policy_use_profile.csv", "heterogeneity_explanation_limits.csv",
    "theorem_empirical_applicability_matrix.csv", "assumption_identification_ledger.csv",
    "forbidden_theorem_transfer_claims.csv", "future_active_acquisition_requirements.csv",
    "untouched_population_options.csv",
)
LIFECYCLE_STAGES = (
    "PREFLIGHT_COMPLETED", "AUTHORIZATION_CONSUMED", "INPUT_REPLAY_COMPLETED",
    "ANALYSIS_STARTED", "ANALYSIS_COMPLETED", "MANIFEST_COMPLETED",
    "ATOMIC_PUBLISH_COMMIT_READY",
)


class C85EResultError(RuntimeError):
    """Raised when a C85E result cannot be published or replayed exactly."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C85EResultError(message)


def _json_value(value: Any) -> Any:
    if isinstance(value, (dict, list, tuple)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if value is None:
        return ""
    return value


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> int:
    _require(bool(rows), f"refusing empty C85E result table: {path.name}")
    fields = list(rows[0])
    _require(all(list(row) == fields for row in rows), f"C85E table schema drift: {path.name}")
    _require("result_tag" in fields and all(
        row["result_tag"] == "POST_C84S_EXPLORATORY" for row in rows
    ), f"C85E exploratory result tag drift: {path.name}")
    with path.open("x", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _json_value(value) for key, value in row.items()})
        handle.flush()
        os.fsync(handle.fileno())
    return len(rows)


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(dict(value)))
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        _require(not path.is_symlink(), "C85E result bundle cannot contain symlinks")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory in reversed(directories + [root]):
        descriptor = os.open(directory, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)


def replay_result_bundle(root: str | Path) -> dict[str, Any]:
    base = Path(root)
    _require(base.is_dir(), "C85E result bundle absent")
    manifest = json.loads((base / "C85E_RESULT_ARTIFACT_MANIFEST.json").read_text(encoding="utf-8"))
    result = json.loads((base / "C85E_RESULT.json").read_text(encoding="utf-8"))
    completion = json.loads((base / "C85E_COMPLETION_RECEIPT.json").read_text(encoding="utf-8"))
    _require(manifest.get("schema_version") == MANIFEST_SCHEMA and
             result.get("schema_version") == RESULT_SCHEMA and
             completion.get("schema_version") == COMPLETION_SCHEMA,
             "C85E result control schema drift")
    rows = manifest.get("artifacts")
    _require(isinstance(rows, list) and len(rows) == manifest.get("artifact_count"),
             "C85E manifest artifact count drift")
    observed: set[str] = set()
    for row in rows:
        path = base / str(row["path"])
        _require(path.is_file() and row["path"] not in observed and
                 path.stat().st_size == int(row["size_bytes"]) and
                 sha256_file(path) == row["sha256"],
                 f"C85E result artifact identity drift: {row.get('path')}")
        observed.add(str(row["path"]))
        if str(row["path"]).endswith(".csv"):
            with path.open(newline="", encoding="utf-8") as handle:
                table_rows = list(csv.DictReader(handle))
            derived = len(table_rows)
            _require(derived == int(row["rows"]), f"C85E result row count drift: {row['path']}")
            _require(table_rows and all(
                item.get("result_tag") == "POST_C84S_EXPLORATORY" for item in table_rows
            ), f"C85E result table tag drift: {row['path']}")
    expected_artifacts = set(REGISTERED_TABLES) | {
        "C85E_RESTRICTED_POLICY_AND_INFORMATION_VALUE_SYNTHESIS.md",
        "C85E_RESULT.json", "authorization_consumed.json",
    }
    _require(observed == expected_artifacts, "C85E result manifest file-set drift")
    expected_files = expected_artifacts | {
        "C85E_RESULT_ARTIFACT_MANIFEST.json", "C85E_COMPLETION_RECEIPT.json",
        "C85E_LIFECYCLE.jsonl",
    }
    _require({str(path.relative_to(base)) for path in base.rglob("*") if path.is_file()}
             == expected_files, "C85E result bundle contains missing or extra files")
    _require(result.get("gate") == SUCCESS_GATE and
             result.get("result_tag") == "POST_C84S_EXPLORATORY" and
             result.get("C84_primary_gate") == "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous" and
             result.get("C84_label_frontier") == "C84-L4" and
             result.get("theorem_statuses_changed") is False,
             "C85E immutable result state drift")
    _require(all(int(value) == 0 for value in result["protected_counters"].values()),
             "C85E protected counter nonzero")
    authorization = json.loads((base / "authorization_consumed.json").read_text(encoding="utf-8"))
    for field in (
        "authorization_binding_sha256", "execution_lock_sha256", "attempt_id",
    ):
        _require(authorization.get(field) == result.get(field) == completion.get(field),
                 f"C85E authorization/result identity drift: {field}")
    lifecycle = [json.loads(line) for line in (base / "C85E_LIFECYCLE.jsonl").read_text().splitlines()]
    _require(tuple(row["stage"] for row in lifecycle) == LIFECYCLE_STAGES and
             [row["sequence"] for row in lifecycle] == list(range(len(LIFECYCLE_STAGES))),
             "C85E lifecycle drift")
    _require(completion.get("gate") == SUCCESS_GATE and
             completion.get("manifest_sha256") == sha256_file(base / "C85E_RESULT_ARTIFACT_MANIFEST.json") and
             completion.get("result_sha256") == sha256_file(base / "C85E_RESULT.json"),
             "C85E completion linkage drift")
    return {
        "status": "PASS", "artifacts": len(rows),
        "registered_tables": len(REGISTERED_TABLES),
        "gate": SUCCESS_GATE,
    }


def publish_result_bundle(
    *,
    context: ValidatedC85EExecutionContext,
    tables: Mapping[str, Sequence[Mapping[str, Any]]],
    synthesis_markdown: str,
    input_replay_sha256: str,
) -> dict[str, Any]:
    """Write, validate, fsync, and publish with exactly one final rename."""
    revalidate_execution_context(context)
    _require(set(tables) == set(REGISTERED_TABLES), "C85E result-table set drift")
    final = context.output_root
    _require(not final.exists(), "C85E final result root exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging-", dir=final.parent))
    lifecycle: list[dict[str, Any]] = []

    def event(stage: str, artifact_sha256: str | None = None) -> None:
        lifecycle.append({
            "schema_version": "c85e_lifecycle_v1", "sequence": len(lifecycle),
            "stage": stage, "timestamp_unix_ns": time.time_ns(),
            "attempt_id": context.attempt_id,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "execution_lock_sha256": context.execution_lock_sha256,
            "artifact_sha256": artifact_sha256,
        })

    try:
        event("PREFLIGHT_COMPLETED")
        event("AUTHORIZATION_CONSUMED", context.consumption_receipt_sha256)
        event("INPUT_REPLAY_COMPLETED", input_replay_sha256)
        event("ANALYSIS_STARTED")
        artifact_rows: list[dict[str, Any]] = []
        for name in REGISTERED_TABLES:
            rows = tables[name]
            count = _write_csv(staging / name, rows)
            artifact_rows.append({
                "path": name, "size_bytes": (staging / name).stat().st_size,
                "sha256": sha256_file(staging / name), "rows": count,
            })
        synthesis_name = "C85E_RESTRICTED_POLICY_AND_INFORMATION_VALUE_SYNTHESIS.md"
        with (staging / synthesis_name).open("x", encoding="utf-8") as handle:
            handle.write(synthesis_markdown)
            handle.flush()
            os.fsync(handle.fileno())
        artifact_rows.append({
            "path": synthesis_name, "size_bytes": (staging / synthesis_name).stat().st_size,
            "sha256": sha256_file(staging / synthesis_name), "rows": None,
        })
        event("ANALYSIS_COMPLETED")
        result = {
            "schema_version": RESULT_SCHEMA, "gate": SUCCESS_GATE,
            "result_tag": "POST_C84S_EXPLORATORY",
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "execution_lock_sha256": context.execution_lock_sha256,
            "attempt_id": context.attempt_id, "input_replay_sha256": input_replay_sha256,
            "C84_primary_gate": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
            "C84_label_frontier": "C84-L4", "theorem_statuses_changed": False,
            "protected_counters": {
                "direct_label_access": 0, "target_logit_access": 0, "EEG_access": 0,
                "selector_or_Q0_builder_calls": 0, "inference_calls": 0,
                "theorem_status_writes": 0, "active_acquisition": 0,
                "C86": 0, "manuscript_writes": 0,
            },
        }
        result_sha = _write_json(staging / "C85E_RESULT.json", result)
        artifact_rows.append({
            "path": "C85E_RESULT.json", "size_bytes": (staging / "C85E_RESULT.json").stat().st_size,
            "sha256": result_sha, "rows": None,
        })
        (staging / "authorization_consumed.json").write_bytes(
            context.consumption_receipt_path.read_bytes()
        )
        artifact_rows.append({
            "path": "authorization_consumed.json",
            "size_bytes": (staging / "authorization_consumed.json").stat().st_size,
            "sha256": sha256_file(staging / "authorization_consumed.json"), "rows": None,
        })
        manifest = {
            "schema_version": MANIFEST_SCHEMA, "artifact_count": len(artifact_rows),
            "artifacts": artifact_rows, "registered_result_tables": list(REGISTERED_TABLES),
            "result_tag": "POST_C84S_EXPLORATORY",
        }
        manifest_sha = _write_json(staging / "C85E_RESULT_ARTIFACT_MANIFEST.json", manifest)
        event("MANIFEST_COMPLETED", manifest_sha)
        completion = {
            "schema_version": COMPLETION_SCHEMA, "gate": SUCCESS_GATE,
            "manifest_sha256": manifest_sha, "result_sha256": result_sha,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "execution_lock_sha256": context.execution_lock_sha256,
            "attempt_id": context.attempt_id,
        }
        completion_sha = _write_json(staging / "C85E_COMPLETION_RECEIPT.json", completion)
        event("ATOMIC_PUBLISH_COMMIT_READY", completion_sha)
        with (staging / "C85E_LIFECYCLE.jsonl").open("xb") as handle:
            for row in lifecycle:
                handle.write(canonical_json_bytes(row))
            handle.flush()
            os.fsync(handle.fileno())
        # Control files are validated directly. Excluding them from the artifact
        # list avoids a manifest/completion/lifecycle self-reference cycle.
        replay_result_bundle(staging)
        _require(sum(path.stat().st_size for path in staging.rglob("*") if path.is_file())
                 <= 2 * 1024 ** 3, "C85E result output exceeds 2 GiB")
        _fsync_tree(staging)
        os.replace(staging, final)
        return {"status": "PASS", "gate": SUCCESS_GATE, "root": str(final)}
    except BaseException:
        # Preserve the staging tree as failure evidence; never publish a partial final root.
        raise


__all__ = [
    "COMPLETION_SCHEMA", "C85EResultError", "MANIFEST_SCHEMA", "REGISTERED_TABLES",
    "RESULT_SCHEMA", "SUCCESS_GATE", "publish_result_bundle", "replay_result_bundle",
]
