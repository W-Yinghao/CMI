"""Exact C85U per-context persistence and replay validation."""
from __future__ import annotations

import os
from pathlib import Path
import re
import tempfile
from typing import Any, Mapping

import numpy as np

from oaci.multidataset import c84s_evaluation as historical_evaluation
from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file

from .c85u_utility_builder import FLOAT_TOLERANCE, array_sha256, candidate_order_sha256


CANDIDATES = 81
CONTEXT_SCHEMA = "c85u_candidate_utility_context_v1"

SCALAR_FIELDS = {
    "schema_version", "context_id", "dataset", "target_subject_id", "panel",
    "training_seed", "level", "evaluation_trial_count",
    "candidate_id_order_sha256", "evaluation_trial_id_sha256",
    "evaluation_label_view_manifest_sha256", "target_artifact_input_sha256",
    "metric_matrix_sha256", "utility_vector_sha256", "best_candidate_id",
    "best_candidate_index", "utility_min", "utility_max", "utility_spread",
    "exact_comaximizer_count",
}
VECTOR_FIELDS = {
    "candidate_index", "candidate_id", "regime", "trajectory_order", "epoch",
    "target_artifact_sha256", "balanced_accuracy", "NLL", "ECE",
    "bAcc_midrank_percentile", "negative_NLL_midrank_percentile",
    "negative_ECE_midrank_percentile", "composite_utility",
    "utility_rank_midrank", "canonical_utility_order_position",
    "standardized_regret", "is_canonical_best", "is_in_canonical_top5",
    "is_in_canonical_top10",
}
REQUIRED_FIELDS = SCALAR_FIELDS | VECTOR_FIELDS
FLOAT_VECTOR_FIELDS = {
    "balanced_accuracy", "NLL", "ECE", "bAcc_midrank_percentile",
    "negative_NLL_midrank_percentile", "negative_ECE_midrank_percentile",
    "composite_utility", "utility_rank_midrank", "standardized_regret",
}
FLOAT_SCALAR_FIELDS = {"utility_min", "utility_max", "utility_spread"}
SHA_SCALAR_FIELDS = {
    "candidate_id_order_sha256", "evaluation_trial_id_sha256",
    "evaluation_label_view_manifest_sha256", "target_artifact_input_sha256",
    "metric_matrix_sha256", "utility_vector_sha256",
}
SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")


def _scalar(payload: Mapping[str, np.ndarray], field: str) -> Any:
    value = np.asarray(payload[field])
    require(value.shape == (), f"C85U scalar field shape drift: {field}")
    return value.item()


def validate_context_payload(payload: Mapping[str, np.ndarray]) -> dict[str, Any]:
    require(set(payload) == REQUIRED_FIELDS, "C85U context artifact field-set drift")
    require(str(_scalar(payload, "schema_version")) == CONTEXT_SCHEMA,
            "C85U context artifact schema drift")
    for field, value in payload.items():
        array = np.asarray(value)
        require(not array.dtype.hasobject, f"C85U object dtype forbidden: {field}")
        if np.issubdtype(array.dtype, np.number):
            require(np.all(np.isfinite(array)), f"C85U nonfinite field: {field}")
    for field in VECTOR_FIELDS:
        require(np.asarray(payload[field]).shape == (CANDIDATES,),
                f"C85U vector field shape drift: {field}")
    for field in FLOAT_VECTOR_FIELDS:
        require(np.asarray(payload[field]).dtype == np.dtype("<f8"),
                f"C85U float64 dtype drift: {field}")
    for field in FLOAT_SCALAR_FIELDS:
        require(np.asarray(payload[field]).dtype == np.dtype("<f8"),
                f"C85U float64 scalar dtype drift: {field}")
    expected_dtypes = {
        "training_seed": "<i8", "level": "<i8", "evaluation_trial_count": "<i8",
        "candidate_index": "<i2", "trajectory_order": "<i2", "epoch": "<i2",
        "canonical_utility_order_position": "<i2", "best_candidate_index": "<i2",
        "exact_comaximizer_count": "<i2", "is_canonical_best": "u1",
        "is_in_canonical_top5": "u1", "is_in_canonical_top10": "u1",
    }
    for field, dtype in expected_dtypes.items():
        require(np.asarray(payload[field]).dtype == np.dtype(dtype),
                f"C85U exact dtype drift: {field}")
    require(all(SHA256_PATTERN.fullmatch(str(_scalar(payload, field)))
                for field in SHA_SCALAR_FIELDS),
            "C85U scalar SHA-256 field drift")

    candidate_index = np.asarray(payload["candidate_index"], dtype=np.int64)
    require(np.array_equal(candidate_index, np.arange(CANDIDATES)),
            "C85U candidate index drift")
    candidate_ids = np.asarray(payload["candidate_id"], dtype=str)
    require(len(set(candidate_ids.tolist())) == CANDIDATES,
            "C85U candidate IDs are not unique")
    require(str(_scalar(payload, "candidate_id_order_sha256")) ==
            candidate_order_sha256(candidate_ids.tolist()),
            "C85U candidate order digest drift")
    target_hashes = np.asarray(payload["target_artifact_sha256"], dtype=str)
    require(all(SHA256_PATTERN.fullmatch(value) for value in target_hashes),
            "C85U target artifact digest shape drift")

    metrics = np.column_stack((
        payload["balanced_accuracy"], payload["NLL"], payload["ECE"],
    )).astype("<f8", copy=False)
    oriented = np.column_stack((
        payload["bAcc_midrank_percentile"],
        payload["negative_NLL_midrank_percentile"],
        payload["negative_ECE_midrank_percentile"],
    )).astype("<f8", copy=False)
    expected_oriented = np.column_stack((
        historical_evaluation.midrank_percentile(metrics[:, 0]),
        historical_evaluation.midrank_percentile(-metrics[:, 1]),
        historical_evaluation.midrank_percentile(-metrics[:, 2]),
    )).astype("<f8", copy=False)
    require(np.array_equal(oriented, expected_oriented),
            "C85U oriented midrank replay drift")
    utility = np.asarray(payload["composite_utility"], dtype="<f8")
    require(np.max(np.abs(utility - np.mean(oriented, axis=1))) <= FLOAT_TOLERANCE,
            "C85U utility replay exceeds tolerance")
    require(str(_scalar(payload, "metric_matrix_sha256")) == array_sha256(metrics),
            "C85U metric matrix digest drift")
    require(str(_scalar(payload, "utility_vector_sha256")) == array_sha256(utility),
            "C85U utility vector digest drift")

    order = np.lexsort((np.arange(CANDIDATES), -utility))
    positions = np.empty(CANDIDATES, dtype=np.int16)
    positions[order] = np.arange(1, CANDIDATES + 1, dtype=np.int16)
    require(np.array_equal(payload["canonical_utility_order_position"], positions),
            "C85U canonical utility order drift")
    expected_rank = historical_evaluation.midrank_percentile(utility)
    require(np.array_equal(payload["utility_rank_midrank"], expected_rank),
            "C85U utility midrank replay drift")
    expected_regret = np.asarray([
        historical_evaluation.standardized_regret(utility, index)
        for index in range(CANDIDATES)
    ])
    require(np.max(np.abs(payload["standardized_regret"] - expected_regret)) <= FLOAT_TOLERANCE,
            "C85U standardized regret replay exceeds tolerance")
    best = int(order[0])
    require(int(_scalar(payload, "best_candidate_index")) == best and
            str(_scalar(payload, "best_candidate_id")) == candidate_ids[best],
            "C85U canonical best identity drift")
    require(np.array_equal(payload["is_canonical_best"],
                           (np.arange(CANDIDATES) == best).astype("u1")),
            "C85U canonical-best indicator drift")
    require(np.array_equal(payload["is_in_canonical_top5"],
                           np.isin(np.arange(CANDIDATES), order[:5]).astype("u1")),
            "C85U top5 indicator drift")
    require(np.array_equal(payload["is_in_canonical_top10"],
                           np.isin(np.arange(CANDIDATES), order[:10]).astype("u1")),
            "C85U top10 indicator drift")
    require(abs(float(_scalar(payload, "utility_min")) - float(np.min(utility))) <= FLOAT_TOLERANCE and
            abs(float(_scalar(payload, "utility_max")) - float(np.max(utility))) <= FLOAT_TOLERANCE,
            "C85U utility extrema drift")
    require(abs(float(_scalar(payload, "utility_spread")) -
                float(np.max(utility) - np.min(utility))) <= FLOAT_TOLERANCE,
            "C85U utility spread drift")
    require(int(_scalar(payload, "exact_comaximizer_count")) ==
            int(np.sum(utility == np.max(utility))),
            "C85U exact co-maximizer count drift")
    return {
        "schema_version": CONTEXT_SCHEMA,
        "context_id": str(_scalar(payload, "context_id")),
        "candidate_rows": CANDIDATES,
        "metric_matrix_sha256": array_sha256(metrics),
        "utility_vector_sha256": array_sha256(utility),
        "best_candidate_index": best,
        "evaluation_trial_count": int(_scalar(payload, "evaluation_trial_count")),
    }


def load_context_artifact(
    path: str | Path, *, expected_sha256: str | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    target = Path(path)
    require(target.is_file(), f"C85U context artifact absent: {target}")
    observed = sha256_file(target)
    if expected_sha256 is not None:
        require(observed == expected_sha256, "C85U context artifact SHA drift")
    with np.load(target, allow_pickle=False) as archive:
        payload = {field: np.asarray(archive[field]) for field in archive.files}
    replay = validate_context_payload(payload)
    replay.update({"artifact_sha256": observed, "artifact_bytes": target.stat().st_size})
    return payload, replay


def write_context_artifact(
    path: str | Path, payload: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    validate_context_payload(payload)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{target.name}.", dir=target.parent)
    os.close(descriptor)
    temp_path = Path(temporary)
    try:
        with temp_path.open("wb") as handle:
            np.savez_compressed(handle, **payload)
            handle.flush()
            os.fsync(handle.fileno())
        observed = sha256_file(temp_path)
        loaded, replay = load_context_artifact(temp_path, expected_sha256=observed)
        for field in REQUIRED_FIELDS:
            require(array_sha256(np.asarray(loaded[field])) ==
                    array_sha256(np.asarray(payload[field])),
                    f"C85U persisted array digest drift: {field}")
        os.replace(temp_path, target)
        replay.update({
            "artifact_path": str(target),
            "artifact_sha256": sha256_file(target),
            "artifact_bytes": target.stat().st_size,
        })
        return replay
    finally:
        if temp_path.exists():
            temp_path.unlink()


def candidate_index_rows(
    payload: Mapping[str, np.ndarray], *, artifact_path: str,
    artifact_sha256: str,
) -> list[dict[str, Any]]:
    validate_context_payload(payload)
    identity = {
        "context_id": str(_scalar(payload, "context_id")),
        "dataset": str(_scalar(payload, "dataset")),
        "target_subject_id": str(_scalar(payload, "target_subject_id")),
        "panel": str(_scalar(payload, "panel")),
        "training_seed": int(_scalar(payload, "training_seed")),
        "level": int(_scalar(payload, "level")),
    }
    rows: list[dict[str, Any]] = []
    for index in range(CANDIDATES):
        rows.append({
            **identity,
            "candidate_index": index,
            "candidate_id": str(payload["candidate_id"][index]),
            "regime": str(payload["regime"][index]),
            "trajectory_order": int(payload["trajectory_order"][index]),
            "epoch": int(payload["epoch"][index]),
            "evaluation_trial_count": int(_scalar(payload, "evaluation_trial_count")),
            "balanced_accuracy": float(payload["balanced_accuracy"][index]),
            "NLL": float(payload["NLL"][index]),
            "ECE": float(payload["ECE"][index]),
            "bAcc_midrank_percentile": float(payload["bAcc_midrank_percentile"][index]),
            "negative_NLL_midrank_percentile": float(payload["negative_NLL_midrank_percentile"][index]),
            "negative_ECE_midrank_percentile": float(payload["negative_ECE_midrank_percentile"][index]),
            "composite_utility": float(payload["composite_utility"][index]),
            "utility_rank_midrank": float(payload["utility_rank_midrank"][index]),
            "canonical_utility_order_position": int(payload["canonical_utility_order_position"][index]),
            "standardized_regret": float(payload["standardized_regret"][index]),
            "is_canonical_best": int(payload["is_canonical_best"][index]),
            "is_in_canonical_top5": int(payload["is_in_canonical_top5"][index]),
            "is_in_canonical_top10": int(payload["is_in_canonical_top10"][index]),
            "context_artifact_path": artifact_path,
            "context_artifact_sha256": artifact_sha256,
        })
    return rows


__all__ = [
    "CONTEXT_SCHEMA", "REQUIRED_FIELDS", "candidate_index_rows",
    "load_context_artifact", "validate_context_payload", "write_context_artifact",
]
