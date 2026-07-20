"""C85U Stage-U1 held-evaluation utility construction."""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from oaci.multidataset import c84s_evaluation as historical_evaluation
from oaci.multidataset.c84s_common import canonical_sha256, require
from oaci.multidataset.c84sr1_context_enumerator import ContextDescriptor


CANDIDATES = 81
FLOAT_TOLERANCE = 1e-12


def array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(array.dtype.str.encode("ascii"))
    digest.update(b"\0")
    digest.update(canonical_sha256(list(array.shape)).encode("ascii"))
    digest.update(b"\0")
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def candidate_order_sha256(candidate_ids: Sequence[str]) -> str:
    require(len(candidate_ids) == CANDIDATES, "C85U candidate ID vector length drift")
    return hashlib.sha256("\n".join(map(str, candidate_ids)).encode("ascii")).hexdigest()


def align_evaluation_rows(
    context: ContextDescriptor,
    target_trial_ids: np.ndarray,
    evaluation_rows: Sequence[Mapping[str, Any]],
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    trial_ids = np.asarray(target_trial_ids, dtype=str)
    require(trial_ids.ndim == 1 and len(set(trial_ids.tolist())) == len(trial_ids),
            "C85U target trial identity drift")
    rows = [
        row for row in evaluation_rows
        if str(row["dataset"]) == context.dataset
        and str(row["target_subject_id"]) == context.target_subject_id
    ]
    require(rows, "C85U evaluation rows absent for target")
    label_by_id = {
        str(row["target_trial_id"]): int(row["canonical_class_label"])
        for row in rows
    }
    require(len(label_by_id) == len(rows), "C85U duplicate evaluation trial identity")
    require(set(label_by_id.values()) == {0, 1}, "C85U evaluation target lacks a class")
    index_by_id = {value: index for index, value in enumerate(trial_ids.tolist())}
    require(set(label_by_id).issubset(index_by_id),
            "C85U evaluation trial is outside frozen target artifact")
    ordered_ids = tuple(value for value in trial_ids.tolist() if value in label_by_id)
    require(set(ordered_ids) == set(label_by_id), "C85U evaluation trial alignment drift")
    return (
        np.asarray([index_by_id[value] for value in ordered_ids], dtype=np.int64),
        np.asarray([label_by_id[value] for value in ordered_ids], dtype=np.int64),
        ordered_ids,
    )


def compute_context_utility_payload(
    *,
    context: ContextDescriptor,
    candidate_data: Mapping[str, Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
    evaluation_label_view_manifest_sha256: str,
) -> dict[str, np.ndarray]:
    candidate_ids = tuple(map(str, candidate_data["candidate_ids"]))
    expected_ids = tuple(row.unit_id for row in context.candidates)
    require(candidate_ids == expected_ids, "C85U candidate identity/order drift")
    logits = np.asarray(candidate_data["target_logits"], dtype=np.float64)
    require(logits.ndim == 3 and logits.shape[0] == CANDIDATES and logits.shape[2] == 2,
            "C85U target logits shape drift")
    evaluation_index, labels, evaluation_ids = align_evaluation_rows(
        context,
        np.asarray(candidate_data["target_trial_ids"], dtype=str),
        evaluation_rows,
    )
    evaluation_logits = logits[:, evaluation_index]
    utility, metrics = historical_evaluation.context_candidate_utility(
        evaluation_logits, labels,
    )
    utility = np.asarray(utility, dtype="<f8")
    metrics = np.asarray(metrics, dtype="<f8")
    require(utility.shape == (CANDIDATES,) and metrics.shape == (CANDIDATES, 3),
            "C85U historical utility shape drift")
    oriented = np.column_stack((
        historical_evaluation.midrank_percentile(metrics[:, 0]),
        historical_evaluation.midrank_percentile(-metrics[:, 1]),
        historical_evaluation.midrank_percentile(-metrics[:, 2]),
    )).astype("<f8", copy=False)
    require(np.max(np.abs(np.mean(oriented, axis=1) - utility)) <= FLOAT_TOLERANCE,
            "C85U historical utility/oriented replay drift")

    order = np.lexsort((np.arange(CANDIDATES), -utility)).astype("<i2")
    order_position = np.empty(CANDIDATES, dtype="<i2")
    order_position[order] = np.arange(1, CANDIDATES + 1, dtype="<i2")
    utility_rank = historical_evaluation.midrank_percentile(utility).astype("<f8")
    regret = np.asarray([
        historical_evaluation.standardized_regret(utility, index)
        for index in range(CANDIDATES)
    ], dtype="<f8")
    best = int(order[0])
    target_hashes = np.asarray(
        [row.target_artifact_sha256 for row in context.candidates], dtype="<U64",
    )
    target_input_digest = canonical_sha256([
        {"unit_id": row.unit_id, "sha256": row.target_artifact_sha256}
        for row in context.candidates
    ])
    candidate_id_array = np.asarray(candidate_ids, dtype="<U64")
    regime = np.asarray([row.regime for row in context.candidates], dtype="<U4")
    payload = {
        "schema_version": np.asarray("c85u_candidate_utility_context_v1"),
        "context_id": np.asarray(context.context_id),
        "dataset": np.asarray(context.dataset),
        "target_subject_id": np.asarray(context.target_subject_id),
        "panel": np.asarray(context.panel),
        "training_seed": np.asarray(context.training_seed, dtype="<i8"),
        "level": np.asarray(context.level, dtype="<i8"),
        "candidate_index": np.arange(CANDIDATES, dtype="<i2"),
        "candidate_id": candidate_id_array,
        "regime": regime,
        "trajectory_order": np.asarray(
            [row.trajectory_order for row in context.candidates], dtype="<i2",
        ),
        "epoch": np.asarray([row.epoch for row in context.candidates], dtype="<i2"),
        "target_artifact_sha256": target_hashes,
        "evaluation_trial_count": np.asarray(len(evaluation_ids), dtype="<i8"),
        "balanced_accuracy": metrics[:, 0],
        "NLL": metrics[:, 1],
        "ECE": metrics[:, 2],
        "bAcc_midrank_percentile": oriented[:, 0],
        "negative_NLL_midrank_percentile": oriented[:, 1],
        "negative_ECE_midrank_percentile": oriented[:, 2],
        "composite_utility": utility,
        "utility_rank_midrank": utility_rank,
        "canonical_utility_order_position": order_position,
        "standardized_regret": regret,
        "is_canonical_best": (np.arange(CANDIDATES) == best).astype("u1"),
        "is_in_canonical_top5": np.isin(np.arange(CANDIDATES), order[:5]).astype("u1"),
        "is_in_canonical_top10": np.isin(np.arange(CANDIDATES), order[:10]).astype("u1"),
        "candidate_id_order_sha256": np.asarray(candidate_order_sha256(candidate_ids)),
        "evaluation_trial_id_sha256": np.asarray(canonical_sha256(list(evaluation_ids))),
        "evaluation_label_view_manifest_sha256": np.asarray(
            evaluation_label_view_manifest_sha256,
        ),
        "target_artifact_input_sha256": np.asarray(target_input_digest),
        "metric_matrix_sha256": np.asarray(array_sha256(metrics)),
        "utility_vector_sha256": np.asarray(array_sha256(utility)),
        "best_candidate_id": np.asarray(candidate_ids[best]),
        "best_candidate_index": np.asarray(best, dtype="<i2"),
        "utility_min": np.asarray(np.min(utility), dtype="<f8"),
        "utility_max": np.asarray(np.max(utility), dtype="<f8"),
        "utility_spread": np.asarray(np.max(utility) - np.min(utility), dtype="<f8"),
        "exact_comaximizer_count": np.asarray(np.sum(utility == np.max(utility)), dtype="<i2"),
    }
    require(not any(array.dtype.hasobject for array in payload.values()),
            "C85U payload contains object dtype")
    return payload


class ProtectedTargetZooReader:
    """Read only target identities/logits after the runtime guard authorizes U1."""

    def __init__(self, execution_context: Any) -> None:
        from .c85u_runtime_guard import require_protected_replay
        require_protected_replay(execution_context)
        self._execution_context = execution_context
        self._zoo_key: tuple[str, str, int, int] | None = None
        self._data: dict[str, Any] | None = None
        self.files_opened = 0

    def _load_zoo(self, context: ContextDescriptor) -> None:
        from .c85u_runtime_guard import require_protected_replay
        require_protected_replay(self._execution_context)
        subjects: np.ndarray | None = None
        trial_ids: np.ndarray | None = None
        logits: list[np.ndarray] = []
        for candidate in context.candidates:
            with np.load(candidate.target_artifact_path, allow_pickle=False) as archive:
                self.files_opened += 1
                require(str(archive["unit_id"].item()) == candidate.unit_id,
                        "C85U target artifact unit drift")
                current_subjects = np.asarray(archive["target_subject_id"], dtype=str)
                current_trials = np.asarray(archive["target_trial_id"], dtype=str)
                if subjects is None:
                    subjects, trial_ids = current_subjects, current_trials
                else:
                    require(np.array_equal(current_subjects, subjects),
                            "C85U target subject order differs across candidates")
                    require(np.array_equal(current_trials, trial_ids),
                            "C85U target trial order differs across candidates")
                logits.append(np.asarray(archive["logits"], dtype=np.float64))
        require(subjects is not None and trial_ids is not None and len(logits) == CANDIDATES,
                "C85U target zoo load incomplete")
        self._zoo_key = (
            context.dataset, context.panel, context.training_seed, context.level,
        )
        self._data = {
            "candidate_ids": [row.unit_id for row in context.candidates],
            "all_target_subjects": subjects,
            "all_target_trial_ids": trial_ids,
            "all_target_logits": np.stack(logits),
        }

    def __call__(self, context: ContextDescriptor) -> dict[str, Any]:
        key = (context.dataset, context.panel, context.training_seed, context.level)
        if key != self._zoo_key:
            self._load_zoo(context)
        require(self._data is not None, "C85U target zoo cache absent")
        mask = self._data["all_target_subjects"] == context.target_subject_id
        require(np.any(mask), "C85U target subject absent from zoo")
        return {
            "candidate_ids": list(self._data["candidate_ids"]),
            "target_trial_ids": self._data["all_target_trial_ids"][mask],
            "target_logits": self._data["all_target_logits"][:, mask],
        }


__all__ = [
    "FLOAT_TOLERANCE", "ProtectedTargetZooReader", "align_evaluation_rows",
    "array_sha256", "candidate_order_sha256", "compute_context_utility_payload",
]
