"""All-or-none C84S selection artifact freeze.

The public freeze entrypoint rejects evaluation-view descriptors and publishes
an immutable manifest before held-evaluation scoring can start.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c84s_common import (
    C84SContractError, atomic_publish_directory, canonical_sha256, read_json,
    require, sha256_file, write_csv, write_json,
)


CANDIDATES = 81
ZERO_SCORE_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "method_id", "candidate_index", "candidate_id", "raw_score",
)
ZERO_RANK_FIELDS = ZERO_SCORE_FIELDS[:-1] + ("rank",)
Q0_ROW_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "chain", "chain_seed", "budget", "sample_trial_id_sha256", "sample_size",
    "selected_candidate_index", "selected_candidate_id", "top5_candidate_indices",
    "top5_candidate_ids", "top10_candidate_indices", "top10_candidate_ids",
    "candidate_score_vector_sha256", "construction_metrics_sha256",
)
Q0_DIGEST_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "chain", "chain_seed", "budget", "sample_trial_id_sha256", "sample_size",
)
FIXED_FIELDS = (
    "dataset", "target_subject_id", "panel", "training_seed", "level",
    "method_id", "selected_candidate_index", "selected_candidate_id",
)
ACCESS_FIELDS = ("stage", "method_id", "view", "read_allowed", "rows", "labels")
FORBIDDEN_INPUT_TOKENS = (
    "evaluation", "oracle", "held_utility", "target_accuracy", "regret",
)


def _contains_forbidden(value: Any, path: str = "") -> str | None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            if lowered == "same_label_oracle_accessed" and nested is False:
                continue
            if any(token in lowered for token in FORBIDDEN_INPUT_TOKENS):
                return f"{path}/{key}"
            result = _contains_forbidden(nested, f"{path}/{key}")
            if result:
                return result
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            result = _contains_forbidden(nested, f"{path}/{index}")
            if result:
                return result
    return None


def validate_selection_inputs(input_descriptor: Mapping[str, Any]) -> None:
    forbidden = _contains_forbidden(input_descriptor)
    require(forbidden is None, f"evaluation/oracle information reached selection: {forbidden}")
    require(input_descriptor.get("same_label_oracle_accessed", False) is False,
            "same-label oracle reached selection")


def validate_zero_label_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    normalized: list[dict[str, Any]] = []
    keys: set[tuple[Any, ...]] = set()
    for raw in rows:
        require(set(raw) == set(ZERO_SCORE_FIELDS), "zero-label score schema drift")
        row = {field: raw[field] for field in ZERO_SCORE_FIELDS}
        index = int(row["candidate_index"])
        score = float(row["raw_score"])
        require(0 <= index < CANDIDATES and np.isfinite(score), "zero-label candidate score is invalid")
        key = tuple(row[field] for field in ZERO_SCORE_FIELDS[:6]) + (index,)
        require(key not in keys, "duplicate zero-label candidate score")
        keys.add(key)
        row["candidate_index"], row["raw_score"] = index, score
        normalized.append(row)
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in normalized:
        key = tuple(row[field] for field in ZERO_SCORE_FIELDS[:6])
        groups.setdefault(key, []).append(row)
    require(groups and all(len(group) == CANDIDATES for group in groups.values()),
            "zero-label context does not contain 81 candidates")
    for group in groups.values():
        require({row["candidate_index"] for row in group} == set(range(CANDIDATES)),
                "zero-label candidate-index coverage drift")
    return normalized


def build_rank_rows(score_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[Mapping[str, Any]]] = {}
    for row in score_rows:
        key = tuple(row[field] for field in ZERO_SCORE_FIELDS[:6])
        groups.setdefault(key, []).append(row)
    output: list[dict[str, Any]] = []
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: int(row["candidate_index"]))
        scores = np.asarray([float(row["raw_score"]) for row in group])
        order = np.lexsort((np.arange(CANDIDATES), -scores))
        ranks = np.empty(CANDIDATES, dtype=int)
        ranks[order] = np.arange(1, CANDIDATES + 1)
        for row, rank in zip(group, ranks):
            output.append({
                **{field: row[field] for field in ZERO_RANK_FIELDS[:-1]},
                "rank": int(rank),
            })
    return output


def validate_q0_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    identities: set[tuple[Any, ...]] = set()
    for raw in rows:
        require(set(raw) == set(Q0_ROW_FIELDS), "Q0 selection schema drift")
        row = {field: raw[field] for field in Q0_ROW_FIELDS}
        row["chain"] = int(row["chain"])
        row["chain_seed"] = int(row["chain_seed"])
        row["sample_size"] = int(row["sample_size"])
        row["selected_candidate_index"] = int(row["selected_candidate_index"])
        for field, length in (("top5_candidate_indices", 5), ("top10_candidate_indices", 10)):
            values = [int(value) for value in row[field]]
            require(len(values) == length and len(set(values)) == length,
                    f"Q0 {field} shape/uniqueness drift")
            require(all(0 <= value < CANDIDATES for value in values), f"Q0 {field} value drift")
            row[field] = values
        for field, length in (("top5_candidate_ids", 5), ("top10_candidate_ids", 10)):
            values = [str(value) for value in row[field]]
            require(len(values) == length and len(set(values)) == length, f"Q0 {field} drift")
            row[field] = values
        require(0 <= row["selected_candidate_index"] < CANDIDATES, "Q0 selected index drift")
        require(row["selected_candidate_index"] == row["top5_candidate_indices"][0], "Q0 selected/top5 mismatch")
        require(str(row["selected_candidate_id"]) == row["top5_candidate_ids"][0], "Q0 selected/top5 ID mismatch")
        for field in ("sample_trial_id_sha256", "candidate_score_vector_sha256", "construction_metrics_sha256"):
            require(len(str(row[field])) == 64, f"Q0 digest drift: {field}")
        identity = tuple(row[field] for field in Q0_ROW_FIELDS[:5]) + (row["chain"], str(row["budget"]))
        require(identity not in identities, "duplicate Q0 chain selection")
        identities.add(identity)
        output.append(row)
    require(output, "Q0 selection freeze is empty")
    return output


def _write_q0_npz(path: Path, rows: Sequence[Mapping[str, Any]]) -> str:
    arrays: dict[str, np.ndarray] = {}
    for field in Q0_ROW_FIELDS:
        values = [row[field] for row in rows]
        if field in ("top5_candidate_indices", "top10_candidate_indices"):
            arrays[field] = np.asarray(values, dtype=np.int16)
        elif field in ("top5_candidate_ids", "top10_candidate_ids"):
            arrays[field] = np.asarray(values, dtype="<U80")
        elif field in ("chain", "chain_seed", "sample_size", "selected_candidate_index"):
            arrays[field] = np.asarray(values, dtype=np.uint64 if field == "chain_seed" else np.int64)
        else:
            arrays[field] = np.asarray(values, dtype="<U80")
    np.savez_compressed(path, **arrays)
    return sha256_file(path)


def freeze_selection(
    final_root: str | Path,
    *,
    input_descriptor: Mapping[str, Any],
    zero_label_rows: Sequence[Mapping[str, Any]],
    q0_rows: Sequence[Mapping[str, Any]],
    fixed_default_rows: Sequence[Mapping[str, Any]],
    access_rows: Sequence[Mapping[str, Any]],
    failure_injection_after: str | None = None,
) -> Path:
    """Validate every selection object, then publish one immutable root."""
    validate_selection_inputs(input_descriptor)
    scores = validate_zero_label_rows(zero_label_rows)
    ranks = build_rank_rows(scores)
    q0 = validate_q0_rows(q0_rows)
    fixed = [dict(row) for row in fixed_default_rows]
    access = [dict(row) for row in access_rows]
    require(fixed and all(set(row) == set(FIXED_FIELDS) for row in fixed), "fixed-default schema drift")
    require(access and all(set(row) == set(ACCESS_FIELDS) for row in access), "selection-access schema drift")
    require(all("evaluation" not in str(row["view"]).lower() for row in access),
            "evaluation view appears in selection access ledger")

    def writer(staging: Path) -> None:
        artifacts: dict[str, dict[str, Any]] = {}
        for name, rows in (
            ("zero_label_candidate_scores.csv", scores),
            ("zero_label_candidate_ranks.csv", ranks),
            ("fixed_default_selections.csv", fixed),
            ("selection_input_access_ledger.csv", access),
        ):
            digest = write_csv(staging / name, rows)
            artifacts[name] = {"sha256": digest, "rows": len(rows)}
            if failure_injection_after == name:
                raise C84SContractError("injected selection freeze failure")
        q0_sha = _write_q0_npz(staging / "q0_chain_selection.npz", q0)
        artifacts["q0_chain_selection.npz"] = {"sha256": q0_sha, "rows": len(q0)}
        digest_rows = [
            {field: row[field] for field in Q0_DIGEST_FIELDS}
            for row in q0
        ]
        digest_sha = write_csv(staging / "q0_sample_digest_registry.csv", digest_rows)
        artifacts["q0_sample_digest_registry.csv"] = {"sha256": digest_sha, "rows": len(q0)}
        schema = {
            "schema_version": "c84s_q0_chain_selection_schema_v1",
            "fields": list(Q0_ROW_FIELDS), "rows": len(q0),
            "object_dtype": False,
        }
        schema_sha = write_json(staging / "q0_chain_selection_schema.json", schema)
        artifacts["q0_chain_selection_schema.json"] = {"sha256": schema_sha, "rows": 1}
        manifest = {
            "schema_version": "c84s_selection_freeze_manifest_v1",
            "status": "SELECTION_FROZEN_EVALUATION_DESCRIPTOR_NOT_YET_AVAILABLE",
            "input_descriptor_sha256": canonical_sha256(input_descriptor),
            "evaluation_label_descriptor_received": False,
            "same_label_oracle_accessed": False,
            "artifacts": artifacts,
        }
        manifest_sha = write_json(staging / "C84S_SELECTION_FREEZE_MANIFEST.json", manifest)
        (staging / "C84S_SELECTION_FREEZE_MANIFEST.sha256").write_text(
            f"{manifest_sha}  C84S_SELECTION_FREEZE_MANIFEST.json\n", encoding="ascii",
        )
    return atomic_publish_directory(final_root, writer)


def replay_selection_freeze(root: str | Path) -> dict[str, Any]:
    root = Path(root)
    manifest_path = root / "C84S_SELECTION_FREEZE_MANIFEST.json"
    sidecar_path = root / "C84S_SELECTION_FREEZE_MANIFEST.sha256"
    require(manifest_path.is_file() and sidecar_path.is_file(), "selection manifest is absent")
    expected = sidecar_path.read_text(encoding="ascii").split()[0]
    require(sha256_file(manifest_path) == expected, "selection manifest SHA drift")
    manifest = read_json(manifest_path)
    require(manifest["evaluation_label_descriptor_received"] is False, "selection freeze saw evaluation descriptor")
    for name, identity in manifest["artifacts"].items():
        require(sha256_file(root / name) == identity["sha256"], f"selection artifact SHA drift: {name}")
    return manifest
