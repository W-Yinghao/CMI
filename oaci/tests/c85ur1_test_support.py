"""Shadow-only C85UR1 attempt and receipt fixtures."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import time

from oaci.multidataset.c84s_common import canonical_sha256, sha256_file
from oaci.theory.c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    C85UExecutionContextV2,
    CONSUMPTION_SCHEMA_V2,
    LIFECYCLE_SCHEMA_V2,
    PROTECTED_REPLAY_SCHEMA_V2,
    canonical_json_bytes,
)
from oaci.theory.c85u_u1_registry_v2 import U1RuntimeRegistry


def _identity(context: C85UExecutionContextV2) -> dict[str, object]:
    return {
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_path": str(context.execution_lock_path),
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.head,
    }


def make_shadow_context(tmp_path: Path, *, attempt_id: str = "shadow-attempt") -> C85UExecutionContextV2:
    tmp_path.mkdir(parents=True, exist_ok=True)
    output = (tmp_path / f"output-{attempt_id}").resolve()
    output.mkdir()
    staging = output / f".final_acceptance_bundle.staging-{attempt_id}"
    staging.mkdir()
    lifecycle = staging / "C85U_LIFECYCLE.jsonl"
    provisional = C85UExecutionContextV2(
        authorization_file_sha256="a" * 64,
        authorization_binding_sha256="b" * 64,
        authorization_id="c" * 64,
        execution_lock_path=(tmp_path / "C85U_EXECUTION_LOCK_V2.json").resolve(),
        execution_lock_sha256="d" * 64,
        execution_lock_commit="e" * 40,
        attempt_id=attempt_id,
        output_root=output,
        receipt_path=(tmp_path / f"receipt-{attempt_id}.json").resolve(),
        receipt_sha256="0" * 64,
        head="f" * 40,
        lifecycle_path=lifecycle,
        acceptance_staging_root=staging,
    )
    rows = []
    for sequence, stage in enumerate((
        "PREFLIGHT_STARTED", "PREFLIGHT_COMPLETED", "AUTHORIZATION_CONSUMED",
    )):
        rows.append({
            "schema_version": LIFECYCLE_SCHEMA_V2,
            "sequence": sequence,
            "timestamp_unix_ns": time.time_ns(),
            "stage": stage,
            "authorization_binding_sha256": provisional.authorization_binding_sha256,
            "execution_lock_sha256": provisional.execution_lock_sha256,
            "attempt_id": provisional.attempt_id,
            "output_root": str(provisional.output_root),
            "artifact_or_receipt_sha256": None,
            "details": {},
        })
    lifecycle.write_bytes(b"".join(canonical_json_bytes(row) for row in rows))
    receipt = {
        "schema_version": CONSUMPTION_SCHEMA_V2,
        "authorized_stage": "C85U",
        **_identity(provisional),
        "consumed_at_utc": "2026-07-17T00:00:00Z",
    }
    provisional.receipt_path.write_bytes(canonical_json_bytes(receipt))
    context = replace(provisional, receipt_sha256=sha256_file(provisional.receipt_path))
    (staging / "authorization_consumed.json").write_bytes(provisional.receipt_path.read_bytes())
    return context


def make_shadow_registry(tmp_path: Path) -> U1RuntimeRegistry:
    label = tmp_path / "shadow-labels.csv"
    label.write_text("dataset,target\nshadow,1\n", encoding="utf-8")
    rows = []
    for index in range(2):
        target = tmp_path / f"target-{index}.bin"
        sidecar = tmp_path / f"sidecar-{index}.json"
        target.write_bytes(bytes([index + 1]) * (index + 3))
        sidecar.write_text(json.dumps({"index": index}), encoding="utf-8")
        rows.append({
            "unit_id": f"unit-{index}",
            "target_artifact_path": str(target.resolve()),
            "target_artifact_bytes": str(target.stat().st_size),
            "target_artifact_sha256": sha256_file(target),
            "target_sidecar_path": str(sidecar.resolve()),
            "target_sidecar_bytes": str(sidecar.stat().st_size),
            "target_sidecar_sha256": sha256_file(sidecar),
        })
    target_digest = canonical_sha256([
        {"unit_id": row["unit_id"], "path": row["target_artifact_path"],
         "bytes": int(row["target_artifact_bytes"]), "sha256": row["target_artifact_sha256"]}
        for row in rows
    ])
    sidecar_digest = canonical_sha256([
        {"unit_id": row["unit_id"], "path": row["target_sidecar_path"],
         "bytes": int(row["target_sidecar_bytes"]), "sha256": row["target_sidecar_sha256"]}
        for row in rows
    ])
    return U1RuntimeRegistry(
        contexts=(), target_artifact_rows=tuple(rows), context_rows=(), candidate_rows=(),
        evaluation_label_table_path=label.resolve(),
        evaluation_label_table_sha256=sha256_file(label),
        evaluation_label_table_rows=1,
        evaluation_view_manifest_sha256="1" * 64,
        target_artifact_registry_sha256=target_digest,
        target_sidecar_registry_sha256=sidecar_digest,
        target_artifact_total_bytes=sum(int(row["target_artifact_bytes"]) for row in rows),
    )


def attach_shadow_protected_replay(
    context: C85UExecutionContextV2, registry: U1RuntimeRegistry,
) -> C85UExecutionContextV2:
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    lifecycle.append("PROTECTED_INPUT_REPLAY_STARTED", context=context)
    receipt = {
        "schema_version": PROTECTED_REPLAY_SCHEMA_V2,
        "authorization_file_sha256": context.authorization_file_sha256,
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_id": context.authorization_id,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "evaluation_label_table_sha256": registry.evaluation_label_table_sha256,
        "evaluation_label_table_rows": registry.evaluation_label_table_rows,
        "evaluation_view_manifest_sha256": registry.evaluation_view_manifest_sha256,
        "target_artifact_rows": len(registry.target_artifact_rows),
        "target_artifact_total_bytes": registry.target_artifact_total_bytes,
        "target_artifact_registry_sha256": registry.target_artifact_registry_sha256,
        "target_sidecar_rows": len(registry.target_artifact_rows),
        "target_sidecar_registry_sha256": registry.target_sidecar_registry_sha256,
        "replay_completed_at_utc": "2026-07-17T00:00:01Z",
        "status": "PASS_PROTECTED_INPUTS_REPLAYED_AFTER_AUTHORIZATION_CONSUMPTION",
    }
    path = context.output_root / "C85U_PROTECTED_INPUT_REPLAY_V2.json"
    path.write_bytes(canonical_json_bytes(receipt))
    updated = replace(
        context, protected_replay_path=path.resolve(),
        protected_replay_sha256=sha256_file(path),
    )
    lifecycle.append(
        "PROTECTED_INPUT_REPLAY_COMPLETED", context=updated,
        artifact_or_receipt_sha256=updated.protected_replay_sha256,
    )
    return updated


__all__ = [
    "attach_shadow_protected_replay", "make_shadow_context", "make_shadow_registry",
]
