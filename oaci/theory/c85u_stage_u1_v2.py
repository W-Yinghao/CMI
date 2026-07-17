"""Authorized C85U V2 Stage U1 with an isolated protected-input registry."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file

from .c85u_result_manifest_v2 import publish_utility_field_v2
from .c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    C85UExecutionContextV2,
    RuntimeOpenPolicyV2,
    create_stage_receipt_v2,
    load_execution_context_record_v2,
    validate_context_lock_v2,
    validate_protected_replay_receipt_v2,
    validate_stage_receipt_v2,
)
from .c85u_u1_registry_v2 import (
    EVALUATION_LABEL_TABLE,
    EVALUATION_SEAL,
    EVALUATION_VIEW_MANIFEST,
    U1RuntimeRegistry,
    build_u1_runtime_registry,
    u1_allowed_paths,
)
from .c85u_utility_builder import compute_context_utility_payload


def _load_evaluation_rows_v2(
    context: C85UExecutionContextV2,
    registry: U1RuntimeRegistry,
    policy: RuntimeOpenPolicyV2,
) -> list[dict[str, str]]:
    validate_protected_replay_receipt_v2(context, registry)
    seal_path = policy.require_allowed(EVALUATION_SEAL)
    view_path = policy.require_allowed(EVALUATION_VIEW_MANIFEST)
    label_path = policy.require_allowed(EVALUATION_LABEL_TABLE)
    seal = json.loads(seal_path.read_text(encoding="utf-8"))
    receipt_sha = str(seal.pop("receipt_identity_sha256"))
    require(receipt_sha == canonical_sha256(seal), "C85U V2 evaluation seal identity drift")
    seal["receipt_identity_sha256"] = receipt_sha
    descriptor = seal["evaluation_descriptor"]
    require(seal["released_to_Stage_B"] is False and descriptor["kind"] == "evaluation",
            "C85U V2 evaluation seal contract drift")
    view = json.loads(view_path.read_text(encoding="utf-8"))
    require(view["kind"] == "evaluation" and view["candidate_artifacts"] == 0 and
            view["EEG_arrays"] == 0, "C85U V2 evaluation view contains forbidden payload")
    require(sha256_file(label_path) == registry.evaluation_label_table_sha256,
            "C85U V2 evaluation label-table identity drift")
    with label_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == registry.evaluation_label_table_rows == int(descriptor["row_count"]),
            "C85U V2 evaluation label row coverage drift")
    expected_fields = {
        "dataset", "target_subject_id", "target_trial_id", "canonical_class_label",
        "session", "run", "split_identity",
    }
    require(rows and all(set(row) == expected_fields for row in rows),
            "C85U V2 evaluation label-view schema drift")
    return rows


class ProtectedTargetZooReaderV2:
    """Open each candidate target artifact once under the U1 stage receipt."""

    def __init__(
        self, context: C85UExecutionContextV2, policy: RuntimeOpenPolicyV2,
        *, stage_receipt_sha256: str,
    ) -> None:
        require(context.protected_replay_sha256 is not None,
                "C85U V2 U1 reader precedes protected replay")
        validate_stage_receipt_v2(
            context, "U1", prerequisite_sha256=context.protected_replay_sha256,
        )
        self.context = context
        self.policy = policy
        self.stage_receipt_sha256 = stage_receipt_sha256
        self._zoo_key: tuple[str, str, int, int] | None = None
        self._data: dict[str, Any] | None = None
        self.files_opened = 0
        self.bytes_opened = 0

    def _load_zoo(self, descriptor: Any) -> None:
        require(self.context.protected_replay_sha256 is not None,
                "C85U V2 U1 reader lost protected replay")
        _, observed = validate_stage_receipt_v2(
            self.context, "U1",
            prerequisite_sha256=self.context.protected_replay_sha256,
        )
        require(observed == self.stage_receipt_sha256,
                "C85U V2 U1 stage receipt changed during target read")
        subjects: np.ndarray | None = None
        trial_ids: np.ndarray | None = None
        logits: list[np.ndarray] = []
        for candidate in descriptor.candidates:
            path = self.policy.require_allowed(candidate.target_artifact_path)
            with np.load(path, allow_pickle=False) as archive:
                self.files_opened += 1
                self.bytes_opened += path.stat().st_size
                require(str(archive["unit_id"].item()) == candidate.unit_id,
                        "C85U V2 target artifact unit drift")
                current_subjects = np.asarray(archive["target_subject_id"], dtype=str)
                current_trials = np.asarray(archive["target_trial_id"], dtype=str)
                if subjects is None:
                    subjects, trial_ids = current_subjects, current_trials
                else:
                    require(np.array_equal(current_subjects, subjects),
                            "C85U V2 target subject order differs across candidates")
                    require(np.array_equal(current_trials, trial_ids),
                            "C85U V2 target trial order differs across candidates")
                logits.append(np.asarray(archive["logits"], dtype=np.float64))
        require(subjects is not None and trial_ids is not None and len(logits) == 81,
                "C85U V2 target zoo load incomplete")
        self._zoo_key = (
            descriptor.dataset, descriptor.panel,
            descriptor.training_seed, descriptor.level,
        )
        self._data = {
            "candidate_ids": [row.unit_id for row in descriptor.candidates],
            "all_target_subjects": subjects,
            "all_target_trial_ids": trial_ids,
            "all_target_logits": np.stack(logits),
        }

    def __call__(self, descriptor: Any) -> dict[str, Any]:
        key = (
            descriptor.dataset, descriptor.panel,
            descriptor.training_seed, descriptor.level,
        )
        if key != self._zoo_key:
            self._load_zoo(descriptor)
        require(self._data is not None, "C85U V2 target zoo cache absent")
        mask = self._data["all_target_subjects"] == descriptor.target_subject_id
        require(np.any(mask), "C85U V2 target subject absent from zoo")
        return {
            "candidate_ids": list(self._data["candidate_ids"]),
            "target_trial_ids": self._data["all_target_trial_ids"][mask],
            "target_logits": self._data["all_target_logits"][:, mask],
        }


def _context_sort_key(context: Any) -> tuple[str, str, int, int, int]:
    return (
        str(context.dataset), str(context.panel), int(context.training_seed),
        int(context.level), int(context.target_subject_id),
    )


def _payload_stream(
    *, contexts: Iterable[Any], evaluation_rows: Sequence[Mapping[str, Any]],
    registry: U1RuntimeRegistry, reader: ProtectedTargetZooReaderV2,
    counters: dict[str, int],
) -> Iterable[Mapping[str, Any]]:
    for context in contexts:
        yield compute_context_utility_payload(
            context=context,
            candidate_data=reader(context),
            evaluation_rows=evaluation_rows,
            evaluation_label_view_manifest_sha256=registry.evaluation_view_manifest_sha256,
        )
    counters["target_artifact_files_opened"] = reader.files_opened
    counters["target_artifact_bytes_opened"] = reader.bytes_opened


def run_stage_u1_v2(
    *, execution_context_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    context = load_execution_context_record_v2(execution_context_path)
    validate_context_lock_v2(context)
    output = Path(output_root).resolve()
    require(output.parent == context.output_root and
            output.name == "stage_u1_candidate_utility_v2",
            "C85U V2 U1 output-root binding drift")
    require(context.protected_replay_sha256 is not None,
            "C85U V2 U1 protected replay identity absent")
    stage_path, stage_sha = create_stage_receipt_v2(
        context, "U1", prerequisite_sha256=context.protected_replay_sha256,
    )
    registry = build_u1_runtime_registry()
    validate_protected_replay_receipt_v2(context, registry)
    policy = RuntimeOpenPolicyV2(u1_allowed_paths(registry))
    evaluation_rows = _load_evaluation_rows_v2(context, registry, policy)
    reader = ProtectedTargetZooReaderV2(
        context, policy, stage_receipt_sha256=stage_sha,
    )
    allowed = {
        "evaluation_label_rows_read": len(evaluation_rows),
        "evaluation_metadata_files_opened": 2,
        "evaluation_label_files_opened": 1,
        "target_artifact_files_opened": 0,
        "target_artifact_bytes_opened": 0,
        "target_sidecar_payloads_opened_by_U1": 0,
    }
    forbidden = {
        "construction_label_rows": 0,
        "selection_objects": 0,
        "Q0_shards": 0,
        "direct_scientific_result_tables": 0,
        "inference_calls": 0,
        "theorem_status_writes": 0,
        "runtime_file_policy_rejections": 0,
    }
    contexts = sorted(registry.contexts, key=_context_sort_key)
    result = publish_utility_field_v2(
        payloads=_payload_stream(
            contexts=contexts, evaluation_rows=evaluation_rows,
            registry=registry, reader=reader, counters=allowed,
        ),
        final_root=output,
        context=context,
        registry=registry,
        stage_receipt_sha256=stage_sha,
        allowed_access_counters=allowed,
        forbidden_access_counters=forbidden,
    )
    require(reader.files_opened == 1_944 and
            reader.bytes_opened == registry.target_artifact_total_bytes,
            "C85U V2 U1 target read-plan coverage drift")
    require(policy.forbidden_opens == 0,
            "C85U V2 U1 runtime file-open policy rejected a path")
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append(
        "STAGE_U1_COMPLETED", context=context,
        artifact_or_receipt_sha256=result["handoff_sha256"],
        details={
            "manifest_sha256": result["manifest_sha256"],
            "stage_receipt_sha256": stage_sha,
            "stage_receipt_path": str(stage_path),
        },
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--execution-context", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_stage_u1_v2(
        execution_context_path=arguments.execution_context,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["ProtectedTargetZooReaderV2", "run_stage_u1_v2"]
