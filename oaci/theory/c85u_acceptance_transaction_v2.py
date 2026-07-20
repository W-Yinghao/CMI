"""Single-rename final C85U V2 acceptance transaction and recovery."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

from oaci.multidataset.c84s_common import require, sha256_file

from .c85u_historical_decision_replay_v2 import validate_u2_result_v2
from .c85u_result_manifest_v2 import validate_u1_manifest_v2
from .c85u_runtime_guard_v2 import (
    AppendOnlyLifecycleV2,
    C85UExecutionContextV2,
    SUCCESS_LIFECYCLE_V2,
    canonical_json_bytes,
    validate_execution_context_v2,
    utc_now,
)


ACCEPTANCE_BUNDLE_SCHEMA_V2 = "c85u_atomic_acceptance_bundle_v2"
ACCEPTANCE_MANIFEST_SCHEMA_V2 = "c85u_result_artifact_manifest_v2"
COMPLETION_SCHEMA_V2 = "c85u_completion_receipt_v2"
RESULT_SCHEMA_V2 = "c85u_execution_result_v2"
SUCCESS_GATE = "C85U_COMPLETE_CANDIDATE_UTILITY_FIELD_FROZEN_C85E_REVIEW_REQUIRED"
FINAL_ACCEPTANCE_NAME = "final_acceptance_bundle"
CONTROL_FILES = {
    "C85U_RESULT_ARTIFACT_MANIFEST.json",
    "C85U_COMPLETION_RECEIPT.json",
    "C85U_LIFECYCLE.jsonl",
}
REQUIRED_ARTIFACTS = {
    "C85U_EXECUTION_RESULT.json",
    "authorization_consumed.json",
    "preflight_completed.json",
    "protected_input_replay_receipt.json",
    "U1_ACCEPTANCE_IDENTITY.json",
    "U2_ACCEPTANCE_IDENTITY.json",
}


def _write_json(path: Path, value: Mapping[str, Any]) -> str:
    require(not path.exists(), f"C85U V2 acceptance artifact exists: {path.name}")
    with path.open("xb") as handle:
        handle.write(canonical_json_bytes(value))
        handle.flush()
        os.fsync(handle.fileno())
    return sha256_file(path)


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in CONTROL_FILES:
            continue
        rows.append({
            "path": relative,
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        })
    return rows


def replay_acceptance_manifest_v2(root: str | Path) -> dict[str, Any]:
    base = Path(root).resolve()
    path = base / "C85U_RESULT_ARTIFACT_MANIFEST.json"
    require(path.is_file(), "C85U V2 acceptance manifest absent")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(value.get("schema_version") == ACCEPTANCE_MANIFEST_SCHEMA_V2 and
            value.get("bundle_schema_version") == ACCEPTANCE_BUNDLE_SCHEMA_V2,
            "C85U V2 acceptance manifest schema drift")
    rows = value.get("artifacts")
    require(isinstance(rows, list) and value.get("artifact_count") == len(rows),
            "C85U V2 acceptance manifest count drift")
    observed: set[str] = set()
    for row in rows:
        relative = str(row.get("path", ""))
        require(relative and relative not in observed and relative not in CONTROL_FILES,
                "C85U V2 acceptance manifest path drift")
        observed.add(relative)
        artifact = base / relative
        require(artifact.is_file() and artifact.stat().st_size == int(row["size_bytes"])
                and sha256_file(artifact) == row["sha256"],
                f"C85U V2 acceptance artifact identity drift: {relative}")
    actual = {
        str(path.relative_to(base)) for path in base.rglob("*")
        if path.is_file() and str(path.relative_to(base)) not in CONTROL_FILES
    }
    require(observed == actual and REQUIRED_ARTIFACTS.issubset(observed),
            "C85U V2 acceptance manifest coverage drift")
    return value


def _lifecycle_prefix_sha(path: Path, lines: int) -> str:
    payload = path.read_bytes().splitlines(keepends=True)
    require(len(payload) >= lines, "C85U V2 lifecycle prefix incomplete")
    return hashlib.sha256(b"".join(payload[:lines])).hexdigest()


def _replay_acceptance_chain(
    root: Path, *, external_receipt_path: Path | None = None,
) -> dict[str, Any]:
    lifecycle_path = root / "C85U_LIFECYCLE.jsonl"
    rows = AppendOnlyLifecycleV2(lifecycle_path).replay()
    require([row["stage"] for row in rows] == list(SUCCESS_LIFECYCLE_V2),
            "C85U V2 acceptance lifecycle incomplete")
    manifest_path = root / "C85U_RESULT_ARTIFACT_MANIFEST.json"
    completion_path = root / "C85U_COMPLETION_RECEIPT.json"
    manifest_sha = sha256_file(manifest_path)
    require(rows[-2]["artifact_or_receipt_sha256"] == manifest_sha,
            "C85U V2 acceptance manifest lifecycle hash drift")
    completion = json.loads(completion_path.read_text(encoding="utf-8"))
    require(completion.get("schema_version") == COMPLETION_SCHEMA_V2 and
            completion.get("manifest_sha256") == manifest_sha and
            completion.get("final_gate") == SUCCESS_GATE,
            "C85U V2 completion receipt drift")
    require(completion.get("lifecycle_prefix_sha256")
            == _lifecycle_prefix_sha(lifecycle_path, len(rows) - 1),
            "C85U V2 completion lifecycle-prefix drift")
    require(rows[-1]["artifact_or_receipt_sha256"] == sha256_file(completion_path),
            "C85U V2 terminal completion hash drift")
    if external_receipt_path is not None:
        copied = root / "authorization_consumed.json"
        require(external_receipt_path.is_file() and
                copied.read_bytes() == external_receipt_path.read_bytes() and
                completion.get("external_consumption_receipt_sha256")
                == sha256_file(external_receipt_path),
                "C85U V2 external/copied authorization receipt drift")
    return completion


def validate_complete_acceptance_bundle_v2(
    root: str | Path, *, external_receipt_path: Path | None = None,
) -> dict[str, Any]:
    base = Path(root).resolve()
    manifest = replay_acceptance_manifest_v2(base)
    completion = _replay_acceptance_chain(
        base, external_receipt_path=external_receipt_path,
    )
    result = json.loads((base / "C85U_EXECUTION_RESULT.json").read_text(encoding="utf-8"))
    require(result.get("schema_version") == RESULT_SCHEMA_V2 and
            result.get("gate") == SUCCESS_GATE,
            "C85U V2 final result state drift")
    require(result.get("contexts") == 944 and
            result.get("candidate_utility_rows") == 76_464 and
            result.get("historical_method_context_rows_replayed") == 18_432 and
            result.get("finite_Q0_action_records_replayed") == 8_749_056,
            "C85U V2 final result arithmetic drift")
    require(all(int(value) == 0 for value in result["protected_counters"].values()),
            "C85U V2 final protected counter nonzero")
    require(manifest.get("derived_counts") == {
        "contexts": 944,
        "candidate_utility_rows": 76_464,
        "historical_method_context_rows": 18_432,
        "finite_Q0_action_records": 8_749_056,
    }, "C85U V2 acceptance manifest derived counts drift")
    require(completion.get("result_sha256")
            == sha256_file(base / "C85U_EXECUTION_RESULT.json"),
            "C85U V2 completion result identity drift")
    return completion


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "C85U V2 acceptance bundle cannot contain symlinks")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for path in reversed(directories):
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AtomicC85UAcceptanceTransactionV2:
    def __init__(self, context: C85UExecutionContextV2) -> None:
        validate_execution_context_v2(context)
        self.context = context
        self.staging = context.acceptance_staging_root
        self.final = context.output_root / FINAL_ACCEPTANCE_NAME
        require(self.staging.is_dir() and not self.final.exists(),
                "C85U V2 acceptance staging/final state drift")
        self._prepared: dict[str, Any] | None = None

    def prepare(
        self, *, u1_root: str | Path, u2_root: str | Path,
        u1_handoff_sha256: str, u2_handoff_sha256: str,
    ) -> dict[str, Any]:
        require(self._prepared is None, "C85U V2 acceptance already prepared")
        u1 = validate_u1_manifest_v2(
            u1_root, context=self.context,
            expected_handoff_sha256=u1_handoff_sha256,
        )
        u2 = validate_u2_result_v2(
            u2_root, context=self.context,
            expected_handoff_sha256=u2_handoff_sha256,
        )
        lifecycle = AppendOnlyLifecycleV2(self.context.lifecycle_path)
        lifecycle.append("ACCEPTANCE_MANIFEST_STARTED", context=self.context)
        protected_copy = self.staging / "protected_input_replay_receipt.json"
        require(self.context.protected_replay_path is not None and
                self.context.protected_replay_sha256 is not None,
                "C85U V2 protected replay absent at acceptance")
        protected_copy.write_bytes(self.context.protected_replay_path.read_bytes())
        _write_json(self.staging / "preflight_completed.json", {
            "schema_version": "c85u_preflight_completed_v2",
            "authorization_binding_sha256": self.context.authorization_binding_sha256,
            "execution_lock_sha256": self.context.execution_lock_sha256,
            "attempt_id": self.context.attempt_id,
            "HEAD": self.context.head,
            "status": "PASS",
        })
        _write_json(self.staging / "U1_ACCEPTANCE_IDENTITY.json", {
            "schema_version": "c85u_u1_acceptance_identity_v2",
            "root": str(Path(u1_root).resolve()),
            "manifest_sha256": u1["manifest_sha256"],
            "handoff_sha256": u1["handoff_sha256"],
            "contexts": u1["contexts"],
            "candidate_rows": u1["candidate_rows"],
            "provisional_without_final_acceptance": True,
        })
        _write_json(self.staging / "U2_ACCEPTANCE_IDENTITY.json", {
            "schema_version": "c85u_u2_acceptance_identity_v2",
            "root": str(Path(u2_root).resolve()),
            "result_sha256": u2["result_sha256"],
            "handoff_sha256": u2["handoff_sha256"],
            "method_context_rows": u2["method_context_rows"],
            "finite_Q0_action_records": u2["finite_Q0_action_records_replayed"],
            "provisional_without_final_acceptance": True,
        })
        result = {
            "schema_version": RESULT_SCHEMA_V2,
            "gate": SUCCESS_GATE,
            "execution_lock_sha256": self.context.execution_lock_sha256,
            "execution_lock_commit": self.context.execution_lock_commit,
            "authorization_file_sha256": self.context.authorization_file_sha256,
            "authorization_binding_sha256": self.context.authorization_binding_sha256,
            "authorization_id": self.context.authorization_id,
            "attempt_id": self.context.attempt_id,
            "output_root": str(self.context.output_root),
            "U1_manifest_sha256": u1["manifest_sha256"],
            "U1_handoff_sha256": u1["handoff_sha256"],
            "U2_result_sha256": u2["result_sha256"],
            "U2_handoff_sha256": u2["handoff_sha256"],
            "protected_replay_sha256": self.context.protected_replay_sha256,
            "contexts": 944,
            "candidate_utility_rows": 76_464,
            "historical_method_context_rows_replayed": 18_432,
            "finite_Q0_action_records_replayed": 8_749_056,
            "protected_counters": {
                "construction_label_access": 0,
                "selector_recomputation": 0,
                "Q0_resampling": 0,
                "scientific_inference": 0,
                "theorem_status_writes": 0,
                "C85E": 0,
                "C86": 0,
            },
        }
        result_path = self.staging / "C85U_EXECUTION_RESULT.json"
        result_sha = _write_json(result_path, result)
        artifact_rows = _artifact_rows(self.staging)
        manifest = {
            "schema_version": ACCEPTANCE_MANIFEST_SCHEMA_V2,
            "bundle_schema_version": ACCEPTANCE_BUNDLE_SCHEMA_V2,
            "created_at_utc": utc_now(),
            "artifact_count": len(artifact_rows),
            "derived_counts": {
                "contexts": 944,
                "candidate_utility_rows": 76_464,
                "historical_method_context_rows": 18_432,
                "finite_Q0_action_records": 8_749_056,
            },
            "U1_manifest_sha256": u1["manifest_sha256"],
            "U2_result_sha256": u2["result_sha256"],
            "artifacts": artifact_rows,
        }
        manifest_path = self.staging / "C85U_RESULT_ARTIFACT_MANIFEST.json"
        manifest_sha = _write_json(manifest_path, manifest)
        replay_acceptance_manifest_v2(self.staging)
        lifecycle.append(
            "ACCEPTANCE_MANIFEST_COMPLETED", context=self.context,
            artifact_or_receipt_sha256=manifest_sha,
        )
        prefix_sha = sha256_file(self.context.lifecycle_path)
        completion = {
            "schema_version": COMPLETION_SCHEMA_V2,
            "bundle_schema_version": ACCEPTANCE_BUNDLE_SCHEMA_V2,
            "created_at_utc": utc_now(),
            "final_gate": SUCCESS_GATE,
            "result_sha256": result_sha,
            "manifest_sha256": manifest_sha,
            "lifecycle_prefix_sha256": prefix_sha,
            "external_consumption_receipt_sha256": self.context.receipt_sha256,
            "protected_replay_sha256": self.context.protected_replay_sha256,
            "U1_manifest_sha256": u1["manifest_sha256"],
            "U2_result_sha256": u2["result_sha256"],
            "authorization_binding_sha256": self.context.authorization_binding_sha256,
            "execution_lock_sha256": self.context.execution_lock_sha256,
            "execution_lock_commit": self.context.execution_lock_commit,
            "attempt_id": self.context.attempt_id,
            "output_root": str(self.context.output_root),
            "HEAD": self.context.head,
        }
        completion_path = self.staging / "C85U_COMPLETION_RECEIPT.json"
        completion_sha = _write_json(completion_path, completion)
        lifecycle.append(
            "ATOMIC_ACCEPTANCE_COMMIT_READY", context=self.context,
            artifact_or_receipt_sha256=completion_sha,
        )
        validate_complete_acceptance_bundle_v2(
            self.staging, external_receipt_path=self.context.receipt_path,
        )
        _fsync_tree(self.staging)
        parent = os.open(self.staging.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(parent)
        finally:
            os.close(parent)
        self._prepared = completion
        return dict(completion)

    def commit(self) -> dict[str, Any]:
        require(self._prepared is not None, "C85U V2 acceptance is not commit-ready")
        require(self.staging.is_dir() and not self.final.exists(),
                "C85U V2 acceptance final/staging state drift before rename")
        completion = dict(self._prepared)
        os.replace(self.staging, self.final)
        return completion

    def publish(
        self, *, u1_root: str | Path, u2_root: str | Path,
        u1_handoff_sha256: str, u2_handoff_sha256: str,
    ) -> dict[str, Any]:
        self.prepare(
            u1_root=u1_root, u2_root=u2_root,
            u1_handoff_sha256=u1_handoff_sha256,
            u2_handoff_sha256=u2_handoff_sha256,
        )
        return self.commit()


def recover_post_rename_acceptance_v2(
    final_root: str | Path, *, external_receipt_path: Path,
) -> dict[str, Any]:
    base = Path(final_root).resolve()
    require(base.is_dir(), "C85U V2 recovery final acceptance bundle absent")
    completion = validate_complete_acceptance_bundle_v2(
        base, external_receipt_path=external_receipt_path,
    )
    return {
        "schema_version": "c85u_post_rename_recovery_v2",
        "classification": "RECOVERED_SUCCESS_AFTER_FINAL_ACCEPTANCE_RENAME",
        "final_gate": completion["final_gate"],
        "manifest_sha256": completion["manifest_sha256"],
        "attempt_id": completion["attempt_id"],
        "output_root": completion["output_root"],
    }


def preserve_primary_exception_v2(
    context: C85UExecutionContextV2, primary: BaseException,
) -> dict[str, Any] | None:
    """Never replace the primary failure with lifecycle or reporting errors."""
    secondary: list[str] = []
    final = context.output_root / FINAL_ACCEPTANCE_NAME
    if final.exists():
        try:
            return recover_post_rename_acceptance_v2(
                final, external_receipt_path=context.receipt_path,
            )
        except BaseException as error:
            secondary.append(f"recovery:{type(error).__name__}:{error}")
    lifecycle = AppendOnlyLifecycleV2(context.lifecycle_path)
    try:
        rows = lifecycle.replay()
        terminal = rows[-1]["stage"] in {"ATOMIC_ACCEPTANCE_COMMIT_READY", "FAILED"}
    except BaseException as error:
        secondary.append(f"lifecycle-replay:{type(error).__name__}:{error}")
        terminal = False
    if not terminal:
        try:
            lifecycle.append_failed(
                context=context, primary=primary, secondary_errors=secondary,
            )
        except BaseException as error:
            secondary.append(f"lifecycle-append:{type(error).__name__}:{error}")
    elif not final.exists():
        blocker = context.acceptance_staging_root / "C85U_RECONCILIATION_BLOCKER.json"
        try:
            if not blocker.exists():
                _write_json(blocker, {
                    "schema_version": "c85u_terminal_staging_reconciliation_blocker_v2",
                    "primary_exception_type": type(primary).__name__,
                    "primary_exception_message": str(primary),
                    "secondary_errors": secondary,
                    "attempt_id": context.attempt_id,
                    "automatic_retry": False,
                })
        except BaseException as error:
            secondary.append(f"reconciliation-report:{type(error).__name__}:{error}")
    try:
        failure = context.output_root / "C85U_FAILURE_RECEIPT_V2.json"
        if not failure.exists():
            _write_json(failure, {
                "schema_version": "c85u_failure_receipt_v2",
                "primary_exception_type": type(primary).__name__,
                "primary_exception_message": str(primary),
                "secondary_errors": secondary,
                "authorization_consumed": True,
                "attempt_id": context.attempt_id,
                "U1_provisional": (
                    context.output_root / "stage_u1_candidate_utility_v2"
                ).is_dir(),
                "U2_provisional": (
                    context.output_root / "stage_u2_historical_replay_v2"
                ).is_dir(),
                "final_acceptance_bundle": False,
                "automatic_retry": False,
            })
    except BaseException:
        pass
    return None


__all__ = [
    "ACCEPTANCE_BUNDLE_SCHEMA_V2",
    "ACCEPTANCE_MANIFEST_SCHEMA_V2",
    "AtomicC85UAcceptanceTransactionV2",
    "COMPLETION_SCHEMA_V2",
    "FINAL_ACCEPTANCE_NAME",
    "RESULT_SCHEMA_V2",
    "SUCCESS_GATE",
    "preserve_primary_exception_v2",
    "recover_post_rename_acceptance_v2",
    "replay_acceptance_manifest_v2",
    "validate_complete_acceptance_bundle_v2",
]
