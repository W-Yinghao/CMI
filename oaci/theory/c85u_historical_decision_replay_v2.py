"""Attempt-bound C85U V2 historical decision replay and handoff."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping
from uuid import uuid4

from oaci.multidataset.c84s_common import require, sha256_file
from oaci.multidataset.c84sr1_common import Q0_CHAINS
from oaci.multidataset.c84sr3_q0_store import read_context_shard

from . import c85u_historical_decision_replay as historical
from .c85u_result_manifest_v2 import validate_u1_manifest_v2
from .c85u_runtime_guard_v2 import (
    C85UExecutionContextV2,
    canonical_json_bytes,
    validate_execution_context_v2,
    validate_stage_receipt_v2,
)
from .c85u_u2_registry_v2 import U2RuntimeRegistry


U2_RESULT_SCHEMA_V2 = "c85u_historical_decision_replay_v2"
U2_HANDOFF_SCHEMA_V2 = "c85u_stage_u2_handoff_v2"
U2_RESULT_NAME_V2 = "C85U_HISTORICAL_DECISION_REPLAY_V2.json"
U2_HANDOFF_NAME = "C85U_STAGE_U2_HANDOFF.json"
EXPECTED_METHOD_ROWS = 18_432
EXPECTED_FINITE_Q0_RECORDS = 8_749_056


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _write_sidecar(path: Path, digest: str, name: str) -> None:
    with path.open("xb") as handle:
        handle.write(f"{digest}  {name}\n".encode("ascii"))
        handle.flush()
        os.fsync(handle.fileno())


def validate_u2_result_v2(
    root: str | Path, *, context: C85UExecutionContextV2 | None = None,
    expected_handoff_sha256: str | None = None,
    expected_contexts: int = 944,
    expected_method_rows: int = EXPECTED_METHOD_ROWS,
    expected_finite_q0_records: int = EXPECTED_FINITE_Q0_RECORDS,
) -> dict[str, Any]:
    base = Path(root).resolve()
    result_path = base / U2_RESULT_NAME_V2
    handoff_path = base / U2_HANDOFF_NAME
    result_sidecar = result_path.with_suffix(".sha256")
    handoff_sidecar = handoff_path.with_suffix(".sha256")
    require(all(path.is_file() for path in (
        result_path, handoff_path, result_sidecar, handoff_sidecar,
    )), "C85U V2 U2 result/handoff absent")
    result_sha = sha256_file(result_path)
    handoff_sha = sha256_file(handoff_path)
    require(result_sidecar.read_text(encoding="ascii").split()
            == [result_sha, result_path.name], "C85U V2 U2 result sidecar drift")
    require(handoff_sidecar.read_text(encoding="ascii").split()
            == [handoff_sha, handoff_path.name], "C85U V2 U2 handoff sidecar drift")
    require(expected_handoff_sha256 is None or handoff_sha == expected_handoff_sha256,
            "C85U V2 U2 handoff SHA drift")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    handoff = json.loads(handoff_path.read_text(encoding="utf-8"))
    require(result.get("schema_version") == U2_RESULT_SCHEMA_V2 and
            result.get("status") == "PASS_PROVISIONAL_U2_NOT_ACCEPTED_FOR_C85E",
            "C85U V2 U2 result state drift")
    require(handoff.get("schema_version") == U2_HANDOFF_SCHEMA_V2 and
            handoff.get("U2_result_sha256") == result_sha,
            "C85U V2 U2 handoff linkage drift")
    require(result.get("contexts") == expected_contexts and
            result.get("method_context_rows") == expected_method_rows and
            result.get("finite_Q0_action_records_replayed") == expected_finite_q0_records,
            "C85U V2 U2 exact arithmetic drift")
    require(result.get("selected_regime_mismatches") == 0 and
            all(float(value) <= 1e-12
                for value in result.get("maximum_absolute_differences", {}).values()),
            "C85U V2 U2 endpoint replay drift")
    require(all(int(value) == 0 for value in result["protected_input_access"].values()),
            "C85U V2 U2 protected input counter nonzero")
    require(result.get("accepted_for_C85E") is False and
            handoff.get("accepted_for_C85E") is False,
            "C85U V2 U2 cannot independently accept C85E")
    if context is not None:
        validate_execution_context_v2(context)
        identity = {
            "execution_lock_sha256": context.execution_lock_sha256,
            "authorization_binding_sha256": context.authorization_binding_sha256,
            "attempt_id": context.attempt_id,
            "parent_output_root": str(context.output_root),
        }
        require(all(result.get(key) == value for key, value in identity.items()),
                "C85U V2 U2 result attempt binding drift")
        require(all(handoff.get(key) == value for key, value in identity.items()),
                "C85U V2 U2 handoff attempt binding drift")
    return {
        "schema_version": U2_RESULT_SCHEMA_V2,
        "result_sha256": result_sha,
        "handoff_sha256": handoff_sha,
        "contexts": int(result["contexts"]),
        "method_context_rows": int(result["method_context_rows"]),
        "finite_Q0_action_records_replayed": int(result["finite_Q0_action_records_replayed"]),
        "status": "PASS_PROVISIONAL_U2_REPLAY",
    }


def run_historical_decision_replay_v2(
    *, utility_root: str | Path, registry: U2RuntimeRegistry,
    final_root: str | Path, context: C85UExecutionContextV2,
    u1_handoff_sha256: str, stage_receipt_sha256: str,
    q0_chains: int = Q0_CHAINS,
    expected_contexts: int = 944,
    expected_rows: int = EXPECTED_METHOD_ROWS,
) -> dict[str, Any]:
    """Replay immutable actions without receiving any label or target-array path."""
    validate_execution_context_v2(context)
    require(q0_chains == Q0_CHAINS, "C85U V2 U2 real Q0 chain reduction forbidden")
    _, observed_stage_sha = validate_stage_receipt_v2(
        context, "U2", prerequisite_sha256=u1_handoff_sha256,
    )
    require(observed_stage_sha == stage_receipt_sha256,
            "C85U V2 U2 stage receipt SHA drift")
    utility_base = Path(utility_root).resolve()
    utility_replay = validate_u1_manifest_v2(
        utility_base, context=context,
        expected_handoff_sha256=u1_handoff_sha256,
        expected_contexts=expected_contexts,
        expected_candidate_rows=expected_contexts * 81,
    )
    historical._verify_stage_b(registry.selection_root)
    contexts, payloads = historical._utility_contexts(utility_base)
    require(len(contexts) == expected_contexts, "C85U V2 U2 utility context coverage drift")
    score_orders, fixed, shards = historical._load_actions(registry.selection_root, contexts)
    historical_identity, historical_sha = historical._historical_table_identity(
        registry.result_manifest_path, registry.method_context_path,
    )
    historical_rows = historical._load_historical_rows(registry.method_context_path)

    maxima = {field: 0.0 for field in historical.REPLAY_FIELDS}
    compared = 0
    q0_shards = 0
    finite_records = 0
    context_digest_rows: list[dict[str, Any]] = []
    for context_id in sorted(contexts):
        payload = payloads[context_id]
        shard = shards[context_id]
        q0_payload, shard_replay = read_context_shard(
            registry.selection_root / str(shard["path"]),
            expected_sha256=str(shard["sha256"]), chains=q0_chains,
        )
        require(shard_replay["context_id"] == context_id,
                "C85U V2 U2 Q0 shard/context mismatch")
        endpoints = historical.replay_context_endpoints(
            payload=payload, score_orders=score_orders[context_id],
            fixed_selected_indices=fixed[context_id], q0_payload=q0_payload,
            q0_chains=q0_chains,
        )
        identity = contexts[context_id]
        expected = {
            method: historical_rows[(
                str(identity["dataset"]), str(identity["target_subject_id"]),
                str(identity["panel"]), int(identity["training_seed"]),
                int(identity["level"]), method,
            )]
            for method in endpoints
        }
        context_maxima = historical.compare_context_endpoints(endpoints, expected)
        for field, value in context_maxima.items():
            maxima[field] = max(maxima[field], value)
        compared += len(endpoints)
        q0_shards += 1
        finite_records += len(q0_payload["finite_budget_code"])
        context_digest_rows.append({
            "context_id": context_id,
            "method_count": len(endpoints),
            "utility_vector_sha256": str(historical._scalar(payload, "utility_vector_sha256")),
            "q0_shard_sha256": str(shard["sha256"]),
        })
    require(compared == expected_rows == len(historical_rows),
            "C85U V2 U2 method-context coverage drift")
    require(finite_records == EXPECTED_FINITE_Q0_RECORDS,
            "C85U V2 U2 finite Q0 action-record count drift")

    final = Path(final_root).resolve()
    require(not final.exists(), "C85U V2 U2 final root exists")
    final.parent.mkdir(parents=True, exist_ok=True)
    staging = final.parent / f".{final.name}.staging-{uuid4().hex}"
    staging.mkdir()
    result = {
        "schema_version": U2_RESULT_SCHEMA_V2,
        "status": "PASS_PROVISIONAL_U2_NOT_ACCEPTED_FOR_C85E",
        "execution_lock_sha256": context.execution_lock_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "attempt_id": context.attempt_id,
        "parent_output_root": str(context.output_root),
        "U1_manifest_sha256": utility_replay["manifest_sha256"],
        "U1_handoff_sha256": u1_handoff_sha256,
        "U2_stage_receipt_sha256": stage_receipt_sha256,
        "selection_manifest_sha256": registry.selection_manifest_sha256,
        "candidate_ranks_sha256": registry.candidate_ranks_sha256,
        "fixed_actions_sha256": registry.fixed_actions_sha256,
        "Q0_shard_index_sha256": registry.q0_index_sha256,
        "historical_result_manifest_sha256": registry.result_manifest_sha256,
        "historical_method_context_table": {
            "path": str(registry.method_context_path),
            "sha256": historical_sha,
            "manifest_identity": historical_identity,
        },
        "contexts": len(contexts),
        "method_context_rows": compared,
        "Q0_shards": q0_shards,
        "finite_Q0_action_records_replayed": finite_records,
        "endpoint_tolerance": 1e-12,
        "maximum_absolute_differences": maxima,
        "selected_regime_mismatches": 0,
        "context_replay_registry_sha256": hashlib.sha256(
            json.dumps(context_digest_rows, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "U2_action_result_files_opened": 6 + q0_shards,
        "forbidden_analysis": {
            "Q1": 0, "Q2": 0, "max_T": 0, "LOTO": 0,
            "label_frontier": 0, "taxonomy": 0, "new_pvalues": 0,
        },
        "protected_input_access": {
            "evaluation_label_rows": 0,
            "target_artifacts": 0,
            "target_logit_arrays": 0,
            "construction_label_rows": 0,
            "inference_calls": 0,
        },
        "accepted_for_C85E": False,
    }
    result_path = staging / U2_RESULT_NAME_V2
    result_sha = _write_json(result_path, result)
    _write_sidecar(result_path.with_suffix(".sha256"), result_sha, result_path.name)
    handoff = {
        "schema_version": U2_HANDOFF_SCHEMA_V2,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "attempt_id": context.attempt_id,
        "parent_output_root": str(context.output_root),
        "U1_manifest_sha256": utility_replay["manifest_sha256"],
        "U1_handoff_sha256": u1_handoff_sha256,
        "U2_stage_receipt_sha256": stage_receipt_sha256,
        "U2_result_sha256": result_sha,
        "U2_output_root": str(final),
        "method_context_rows": compared,
        "finite_Q0_action_records": finite_records,
        "accepted_for_C85E": False,
    }
    handoff_path = staging / U2_HANDOFF_NAME
    handoff_sha = _write_json(handoff_path, handoff)
    _write_sidecar(handoff_path.with_suffix(".sha256"), handoff_sha, handoff_path.name)
    replay = validate_u2_result_v2(
        staging, context=context, expected_handoff_sha256=handoff_sha,
        expected_contexts=expected_contexts, expected_method_rows=expected_rows,
        expected_finite_q0_records=EXPECTED_FINITE_Q0_RECORDS,
    )
    descriptor = os.open(staging, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(staging, final)
    return {**replay, "root": str(final)}


__all__ = [
    "EXPECTED_FINITE_Q0_RECORDS",
    "EXPECTED_METHOD_ROWS",
    "U2_HANDOFF_NAME",
    "U2_HANDOFF_SCHEMA_V2",
    "U2_RESULT_NAME_V2",
    "U2_RESULT_SCHEMA_V2",
    "run_historical_decision_replay_v2",
    "validate_u2_result_v2",
]
