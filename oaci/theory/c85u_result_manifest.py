"""Atomic publication and semantic replay for the C85U utility field."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Iterable, Mapping
import uuid

from oaci.multidataset.c84s_common import (
    canonical_sha256,
    require,
    sha256_file,
)

from .c85u_persistence import (
    candidate_index_rows,
    load_context_artifact,
    write_context_artifact,
)


MANIFEST_SCHEMA = "c85u_complete_utility_manifest_v1"
INDEX_SCHEMA = "c85u_candidate_utility_index_v1"
EXPECTED_CONTEXTS = 944
EXPECTED_CANDIDATE_ROWS = 76_464

INDEX_FIELDS = (
    "context_id", "dataset", "target_subject_id", "panel", "training_seed",
    "level", "candidate_index", "candidate_id", "regime",
    "trajectory_order", "epoch", "evaluation_trial_count",
    "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
    "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
    "composite_utility", "utility_rank_midrank",
    "canonical_utility_order_position", "standardized_regret",
    "is_canonical_best", "is_in_canonical_top5", "is_in_canonical_top10",
    "context_artifact_path", "context_artifact_sha256",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _write_bytes_fsynced(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _context_identity(payload: Mapping[str, Any]) -> dict[str, Any]:
    def scalar(field: str) -> Any:
        return payload[field].item()

    return {
        "context_id": str(scalar("context_id")),
        "dataset": str(scalar("dataset")),
        "target_subject_id": str(scalar("target_subject_id")),
        "panel": str(scalar("panel")),
        "training_seed": int(scalar("training_seed")),
        "level": int(scalar("level")),
    }


def validate_utility_manifest(
    root: str | Path,
    *,
    expected_contexts: int = EXPECTED_CONTEXTS,
    expected_candidate_rows: int = EXPECTED_CANDIDATE_ROWS,
) -> dict[str, Any]:
    """Replay every utility artifact and derive complete-field counts."""
    base = Path(root)
    manifest_path = base / "C85U_CANDIDATE_UTILITY_MANIFEST.json"
    sidecar_path = base / "C85U_CANDIDATE_UTILITY_MANIFEST.sha256"
    require(manifest_path.is_file() and sidecar_path.is_file(),
            "C85U utility manifest or sidecar absent")
    sidecar_sha = sidecar_path.read_text(encoding="ascii").split()[0]
    require(sidecar_sha == sha256_file(manifest_path), "C85U manifest sidecar drift")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    require(manifest["schema_version"] == MANIFEST_SCHEMA,
            "C85U utility manifest schema drift")
    require(manifest["status"] == "COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN",
            "C85U utility field is not complete")
    artifacts = manifest["context_artifacts"]
    require(isinstance(artifacts, list), "C85U context artifact registry malformed")
    require(len(artifacts) == expected_contexts, "C85U context artifact count drift")
    require(len({row["context_id"] for row in artifacts}) == len(artifacts),
            "C85U duplicate context artifact identity")
    derived_rows = 0
    derived_bytes = 0
    context_payloads: dict[str, tuple[dict[str, Any], Mapping[str, Any]]] = {}
    for row in artifacts:
        path = base / str(row["path"])
        payload, replay = load_context_artifact(path, expected_sha256=str(row["sha256"]))
        identity = _context_identity(payload)
        require(identity["context_id"] == str(row["context_id"]),
                "C85U context artifact manifest identity drift")
        require(replay["artifact_bytes"] == int(row["bytes"]),
                "C85U context artifact size drift")
        require(replay["metric_matrix_sha256"] == str(row["metric_matrix_sha256"]) and
                replay["utility_vector_sha256"] == str(row["utility_vector_sha256"]) and
                replay["evaluation_trial_count"] == int(row["evaluation_trial_count"]),
                "C85U context artifact semantic manifest linkage drift")
        context_payloads[identity["context_id"]] = (payload, row)
        derived_rows += int(replay["candidate_rows"])
        derived_bytes += int(replay["artifact_bytes"])
    require(derived_rows == expected_candidate_rows,
            "C85U candidate utility row count drift")

    index = manifest["candidate_utility_index"]
    index_path = base / str(index["path"])
    require(index_path.is_file() and sha256_file(index_path) == str(index["sha256"]),
            "C85U candidate utility index identity drift")
    index_rows = 0
    contexts: set[str] = set()
    candidate_keys: set[tuple[str, int]] = set()
    with index_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        require(tuple(reader.fieldnames or ()) == INDEX_FIELDS,
                "C85U candidate utility index schema drift")
        for row in reader:
            context_id = str(row["context_id"])
            candidate_index = int(row["candidate_index"])
            require(0 <= candidate_index < 81, "C85U index candidate range drift")
            key = (context_id, candidate_index)
            require(key not in candidate_keys, "C85U duplicate index candidate row")
            candidate_keys.add(key)
            contexts.add(context_id)
            require(context_id in context_payloads,
                    "C85U index references unknown context artifact")
            payload, artifact = context_payloads[context_id]
            scalar_fields = {
                "dataset": "dataset", "target_subject_id": "target_subject_id",
                "panel": "panel", "training_seed": "training_seed", "level": "level",
                "evaluation_trial_count": "evaluation_trial_count",
            }
            for index_field, payload_field in scalar_fields.items():
                require(str(row[index_field]) == str(payload[payload_field].item()),
                        f"C85U index scalar identity drift: {index_field}")
            string_vectors = ("candidate_id", "regime")
            integer_vectors = (
                "trajectory_order", "epoch", "canonical_utility_order_position",
                "is_canonical_best", "is_in_canonical_top5", "is_in_canonical_top10",
            )
            float_vectors = (
                "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
                "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
                "composite_utility", "utility_rank_midrank", "standardized_regret",
            )
            require(all(str(row[field]) == str(payload[field][candidate_index])
                        for field in string_vectors),
                    "C85U index string-vector replay drift")
            require(all(int(row[field]) == int(payload[field][candidate_index])
                        for field in integer_vectors),
                    "C85U index integer-vector replay drift")
            require(all(float(row[field]) == float(payload[field][candidate_index])
                        for field in float_vectors),
                    "C85U index float-vector replay drift")
            require(row["context_artifact_path"] == str(artifact["path"]) and
                    row["context_artifact_sha256"] == str(artifact["sha256"]),
                    "C85U index context-artifact linkage drift")
            index_rows += 1
    require(index_rows == expected_candidate_rows and len(contexts) == expected_contexts,
            "C85U index exact coverage drift")
    require(int(index["rows"]) == index_rows, "C85U index manifest row count drift")
    protected = manifest["protected_counters"]
    require(all(int(value) == 0 for value in protected.values()),
            "C85U utility field protected counter is nonzero")
    require(manifest["contexts"] == expected_contexts and
            manifest["candidate_rows"] == expected_candidate_rows and
            manifest["candidates_per_context"] == 81,
            "C85U utility manifest arithmetic drift")
    return {
        "schema_version": MANIFEST_SCHEMA,
        "manifest_sha256": sidecar_sha,
        "contexts": len(contexts),
        "candidate_rows": index_rows,
        "context_artifact_bytes": derived_bytes,
        "protected_counters": dict(protected),
        "status": "PASS",
    }


def publish_utility_field(
    *,
    payloads: Iterable[Mapping[str, Any]],
    final_root: str | Path,
    input_identity: Mapping[str, Any],
    expected_contexts: int = EXPECTED_CONTEXTS,
    expected_candidate_rows: int = EXPECTED_CANDIDATE_ROWS,
    failure_after_context: int | None = None,
) -> dict[str, Any]:
    """Write the complete U1 field in staging and publish with one rename."""
    final = Path(final_root)
    require(not final.exists(), "C85U final utility root already exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{uuid.uuid4().hex}"
    require(not staging.exists(), "C85U staging root collision")
    staging.mkdir()
    artifacts_dir = staging / "context_artifacts"
    artifacts_dir.mkdir()
    index_path = staging / "candidate_utility_index.csv"
    artifact_rows: list[dict[str, Any]] = []
    context_ids: set[str] = set()
    row_count = 0
    try:
        with index_path.open("x", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=INDEX_FIELDS, lineterminator="\n")
            writer.writeheader()
            for position, payload in enumerate(payloads):
                identity = _context_identity(payload)
                context_id = identity["context_id"]
                require(context_id not in context_ids, "C85U duplicate U1 context")
                context_ids.add(context_id)
                relative = Path("context_artifacts") / f"{context_id}.npz"
                replay = write_context_artifact(staging / relative, payload)
                rows = candidate_index_rows(
                    payload, artifact_path=str(relative),
                    artifact_sha256=str(replay["artifact_sha256"]),
                )
                writer.writerows(rows)
                row_count += len(rows)
                artifact_rows.append({
                    **identity,
                    "path": str(relative),
                    "bytes": int(replay["artifact_bytes"]),
                    "sha256": str(replay["artifact_sha256"]),
                    "candidate_rows": int(replay["candidate_rows"]),
                    "metric_matrix_sha256": str(replay["metric_matrix_sha256"]),
                    "utility_vector_sha256": str(replay["utility_vector_sha256"]),
                    "evaluation_trial_count": int(replay["evaluation_trial_count"]),
                })
                if failure_after_context is not None and position == failure_after_context:
                    raise RuntimeError("injected C85U U1 publication failure")
            handle.flush()
            os.fsync(handle.fileno())

        require(len(context_ids) == expected_contexts and row_count == expected_candidate_rows,
                "C85U U1 complete-field arithmetic drift")
        index_sha = sha256_file(index_path)
        manifest = {
            "schema_version": MANIFEST_SCHEMA,
            "index_schema_version": INDEX_SCHEMA,
            "status": "COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN",
            "created_at_unix_ns": time.time_ns(),
            "contexts": len(context_ids),
            "candidates_per_context": 81,
            "candidate_rows": row_count,
            "context_artifact_schema": "c85u_candidate_utility_context_v1",
            "context_artifacts": artifact_rows,
            "candidate_utility_index": {
                "path": index_path.name,
                "bytes": index_path.stat().st_size,
                "sha256": index_sha,
                "rows": row_count,
                "field_order": list(INDEX_FIELDS),
            },
            "input_identity": dict(input_identity),
            "numerical_contract": {
                "metric_and_utility_max_abs_tolerance": 1e-12,
                "midrank_identity": "EXACT",
                "candidate_trial_digest_and_order": "EXACT",
            },
            "protected_counters": {
                "construction_label_rows_accessed": 0,
                "Stage_B_selection_objects_accessed": 0,
                "Q0_records_accessed": 0,
                "scientific_inference_calls": 0,
                "theorem_status_writes": 0,
            },
            "forbidden_output_payloads": {
                "EEG": 0, "logits": 0, "probabilities": 0, "labels": 0,
            },
            "context_registry_sha256": canonical_sha256([
                {
                    "context_id": row["context_id"],
                    "sha256": row["sha256"],
                    "metric_matrix_sha256": row["metric_matrix_sha256"],
                    "utility_vector_sha256": row["utility_vector_sha256"],
                }
                for row in artifact_rows
            ]),
        }
        manifest_path = staging / "C85U_CANDIDATE_UTILITY_MANIFEST.json"
        _write_bytes_fsynced(manifest_path, _canonical_json_bytes(manifest))
        manifest_sha = sha256_file(manifest_path)
        _write_bytes_fsynced(
            staging / "C85U_CANDIDATE_UTILITY_MANIFEST.sha256",
            f"{manifest_sha}  {manifest_path.name}\n".encode("ascii"),
        )
        replay = validate_utility_manifest(
            staging, expected_contexts=expected_contexts,
            expected_candidate_rows=expected_candidate_rows,
        )
        _fsync_directory(artifacts_dir)
        _fsync_directory(staging)
        os.replace(staging, final)
        return {**replay, "root": str(final)}
    except BaseException:
        # A failed staging tree is evidence. The caller decides whether to quarantine it.
        raise


def discard_shadow_staging(parent: str | Path, prefix: str) -> None:
    """Remove only shadow-test staging trees, never production failure evidence."""
    for path in Path(parent).glob(f".{prefix}.staging-*"):
        shutil.rmtree(path)


__all__ = [
    "EXPECTED_CANDIDATE_ROWS", "EXPECTED_CONTEXTS", "INDEX_FIELDS",
    "INDEX_SCHEMA", "MANIFEST_SCHEMA", "discard_shadow_staging",
    "publish_utility_field", "validate_utility_manifest",
]
