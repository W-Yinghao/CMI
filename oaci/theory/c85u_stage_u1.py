"""Authorized C85U Stage U1: labels plus persisted target logits to utilities."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file

from .c85u_input_registry import (
    EVALUATION_LABEL_TABLE,
    EVALUATION_SEAL,
    EVALUATION_VIEW_MANIFEST,
    build_frozen_input_registry,
)
from .c85u_result_manifest import publish_utility_field
from .c85u_runtime_guard import (
    C85UExecutionContext,
    load_execution_context_record,
    require_protected_replay,
)
from .c85u_utility_builder import (
    ProtectedTargetZooReader,
    compute_context_utility_payload,
)


def _load_evaluation_rows(context: C85UExecutionContext) -> list[dict[str, str]]:
    require_protected_replay(context)
    seal = json.loads(EVALUATION_SEAL.read_text(encoding="utf-8"))
    receipt_sha = str(seal.pop("receipt_identity_sha256"))
    require(receipt_sha == canonical_sha256(seal), "C85U evaluation seal identity drift")
    seal["receipt_identity_sha256"] = receipt_sha
    descriptor = seal["evaluation_descriptor"]
    require(seal["released_to_Stage_B"] is False and descriptor["kind"] == "evaluation",
            "C85U evaluation seal contract drift")
    manifest = json.loads(EVALUATION_VIEW_MANIFEST.read_text(encoding="utf-8"))
    require(manifest["kind"] == "evaluation" and manifest["candidate_artifacts"] == 0 and
            manifest["EEG_arrays"] == 0, "C85U evaluation view contains forbidden payload")
    require(sha256_file(EVALUATION_LABEL_TABLE) == manifest["table"]["sha256"],
            "C85U evaluation label-table identity drift")
    with EVALUATION_LABEL_TABLE.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(len(rows) == int(descriptor["row_count"]) == 4848,
            "C85U evaluation label row coverage drift")
    expected_fields = {
        "dataset", "target_subject_id", "target_trial_id", "canonical_class_label",
        "session", "run", "split_identity",
    }
    require(rows and set(rows[0]) == expected_fields and
            all(set(row) == expected_fields for row in rows),
            "C85U evaluation label-view schema drift")
    return rows


def _context_sort_key(context: Any) -> tuple[str, str, int, int, int]:
    return (
        str(context.dataset), str(context.panel), int(context.training_seed),
        int(context.level), int(context.target_subject_id),
    )


def _payload_stream(
    *,
    contexts: Iterable[Any],
    evaluation_rows: Sequence[Mapping[str, Any]],
    execution_context: C85UExecutionContext,
) -> Iterable[Mapping[str, Any]]:
    reader = ProtectedTargetZooReader(execution_context)
    for context in contexts:
        yield compute_context_utility_payload(
            context=context,
            candidate_data=reader(context),
            evaluation_rows=evaluation_rows,
            evaluation_label_view_manifest_sha256=sha256_file(EVALUATION_VIEW_MANIFEST),
        )


def run_stage_u1(
    *, execution_context_path: str | Path, output_root: str | Path,
) -> dict[str, Any]:
    context = load_execution_context_record(execution_context_path)
    output = Path(output_root).resolve()
    require(output.parent == context.output_root and output.name == "stage_u1_candidate_utility",
            "C85U U1 output-root binding drift")
    registry = build_frozen_input_registry()
    evaluation_rows = _load_evaluation_rows(context)
    contexts = sorted(registry.contexts, key=_context_sort_key)
    return publish_utility_field(
        payloads=_payload_stream(
            contexts=contexts, evaluation_rows=evaluation_rows,
            execution_context=context,
        ),
        final_root=output,
        input_identity={
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "protected_input_replay_sha256": context.protected_replay_sha256,
            "evaluation_label_view_manifest_sha256": sha256_file(EVALUATION_VIEW_MANIFEST),
            "target_artifacts": 1944,
            "target_artifact_bytes": sum(
                int(row["target_artifact_bytes"]) for row in registry.target_artifact_rows
            ),
        },
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("run-real",))
    parser.add_argument("--execution-context", required=True)
    parser.add_argument("--output-root", required=True)
    arguments = parser.parse_args(argv)
    result = run_stage_u1(
        execution_context_path=arguments.execution_context,
        output_root=arguments.output_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_stage_u1"]
