"""Single-rename C85T V3 transaction bundle and recovery semantics."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .c85_decision_experiments import DecisionContractError
from .c85t_execution_context_v3 import (
    SUCCESS_EVENTS_V3,
    ValidatedC85TExecutionContext,
    canonical_json_bytes,
    replay_lifecycle_v3,
    sha256_file,
    utc_now,
    validate_registered_execution_context,
)
from .c85t_proofs import PROOF_FILENAMES
from .c85t_result_manifest import write_deterministic_npz
from .c85t_semantic_replay_v3 import (
    RESULT_SCHEMA_V3,
    SUCCESS_GATE_V3,
    validate_result_semantics_v3,
)


MANIFEST_SCHEMA_V3 = "c85t_atomic_result_manifest_v3"
BUNDLE_SCHEMA_V3 = "c85t_atomic_execution_bundle_v3"
COMPLETION_SCHEMA_V3 = "c85t_execution_completion_receipt_v3"
RECOVERY_SCHEMA_V3 = "c85t_post_rename_recovery_receipt_v3"
CONTROL_FILES = {
    "C85T_RESULT_ARTIFACT_MANIFEST.json",
    "C85T_V3_LIFECYCLE.jsonl",
    "C85T_V3_COMPLETION_RECEIPT.json",
}
REQUIRED_ARTIFACT_FILES = {
    "preflight_completed.json",
    "authorization_consumed.json",
    "exact_scenario_results.json",
    "monte_carlo_summary.json",
    "S6_replicates.npz",
    "S7_replicates.npz",
    "S9_replicates.npz",
    "S9_raw_draw_digest_registry.csv",
    "proof_candidate_dispositions.csv",
    "C85T_RESULT.json",
    "C85T_V3_SEMANTIC_REPLAY_RECEIPT.json",
}
REQUIRED_ARTIFACT_FILES.update(
    f"c85t_proof_candidates/{filename}" for filename in PROOF_FILENAMES.values()
)


def _artifact_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        relative = str(path.relative_to(root))
        if relative in CONTROL_FILES:
            continue
        rows.append(
            {
                "path": relative,
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return rows


def replay_artifact_manifest_v3(root: Path) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
    if not manifest_path.is_file():
        raise DecisionContractError("C85T V3 result manifest is absent")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema_version") != MANIFEST_SCHEMA_V3:
        raise DecisionContractError("C85T V3 result manifest schema drifted")
    rows = manifest.get("artifacts")
    if not isinstance(rows, list) or manifest.get("artifact_count") != len(rows):
        raise DecisionContractError("C85T V3 result manifest count drifted")
    observed: set[str] = set()
    for row in rows:
        relative = row.get("path")
        if not isinstance(relative, str) or relative in observed or relative in CONTROL_FILES:
            raise DecisionContractError("C85T V3 result manifest path drifted")
        observed.add(relative)
        path = root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row.get("size_bytes")
            or sha256_file(path) != row.get("sha256")
        ):
            raise DecisionContractError(f"C85T V3 artifact identity drifted: {relative}")
    actual = {
        str(path.relative_to(root))
        for path in root.rglob("*")
        if path.is_file() and str(path.relative_to(root)) not in CONTROL_FILES
    }
    if observed != actual:
        raise DecisionContractError("C85T V3 result manifest coverage drifted")
    if not REQUIRED_ARTIFACT_FILES.issubset(observed):
        missing = sorted(REQUIRED_ARTIFACT_FILES - observed)
        raise DecisionContractError(f"C85T V3 required artifact is absent: {missing[0]}")
    counts = manifest.get("derived_counts")
    if not isinstance(counts, dict):
        raise DecisionContractError("C85T V3 derived manifest counts are absent")
    return manifest


def _lifecycle_prefix_sha(path: Path, line_count: int) -> str:
    lines = path.read_bytes().splitlines(keepends=True)
    if len(lines) < line_count:
        raise DecisionContractError("C85T V3 lifecycle prefix is incomplete")
    return hashlib.sha256(b"".join(lines[:line_count])).hexdigest()


def _validate_bundle_chain(
    root: Path,
    *,
    expected_external_receipt_path: Path | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    manifest_path = root / "C85T_RESULT_ARTIFACT_MANIFEST.json"
    lifecycle_path = root / "C85T_V3_LIFECYCLE.jsonl"
    completion_path = root / "C85T_V3_COMPLETION_RECEIPT.json"
    events = replay_lifecycle_v3(lifecycle_path)
    if [event["stage"] for event in events] != list(SUCCESS_EVENTS_V3):
        raise DecisionContractError("C85T V3 success lifecycle is incomplete")
    manifest_sha = sha256_file(manifest_path)
    if events[-2]["artifact_or_receipt_sha256"] != manifest_sha:
        raise DecisionContractError("C85T V3 MANIFEST_COMPLETED hash drifted")
    if not completion_path.is_file():
        raise DecisionContractError("C85T V3 completion receipt is absent")
    completion = json.loads(completion_path.read_text())
    if completion.get("schema_version") != COMPLETION_SCHEMA_V3:
        raise DecisionContractError("C85T V3 completion schema drifted")
    if completion.get("manifest_sha256") != manifest_sha:
        raise DecisionContractError("C85T V3 completion manifest hash drifted")
    expected_prefix_sha = _lifecycle_prefix_sha(lifecycle_path, len(events) - 1)
    if completion.get("lifecycle_prefix_sha256") != expected_prefix_sha:
        raise DecisionContractError("C85T V3 completion lifecycle-prefix hash drifted")
    completion_sha = sha256_file(completion_path)
    if events[-1]["artifact_or_receipt_sha256"] != completion_sha:
        raise DecisionContractError("C85T V3 terminal completion hash drifted")
    copied_receipt = root / "authorization_consumed.json"
    if expected_external_receipt_path is not None:
        external = expected_external_receipt_path.resolve()
        if (
            not external.is_file()
            or copied_receipt.read_bytes() != external.read_bytes()
            or completion.get("external_consumption_receipt_sha256")
            != sha256_file(external)
        ):
            raise DecisionContractError("C85T V3 external/copy receipt identity drifted")
    actual_files = {
        str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
    }
    manifest = replay_artifact_manifest_v3(root)
    expected_files = {row["path"] for row in manifest["artifacts"]} | CONTROL_FILES
    if actual_files != expected_files:
        raise DecisionContractError("C85T V3 final bundle file coverage drifted")
    return completion


def validate_complete_bundle_v3(
    root: Path,
    *,
    context: ValidatedC85TExecutionContext,
    contract: Mapping[str, Any],
    statements: Mapping[str, str],
    shadow_expected_exact: Mapping[str, Any] | None = None,
    shadow_expected_s9_arrays: Mapping[str, np.ndarray] | None = None,
    shadow_expected_s9_digest_rows: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    validate_registered_execution_context(context)
    semantic = validate_result_semantics_v3(
        root,
        context=context,
        contract=contract,
        statements=statements,
        shadow_expected_exact=shadow_expected_exact,
        shadow_expected_s9_arrays=shadow_expected_s9_arrays,
        shadow_expected_s9_digest_rows=shadow_expected_s9_digest_rows,
    )
    manifest = replay_artifact_manifest_v3(root)
    if manifest.get("derived_counts") != semantic:
        raise DecisionContractError("C85T V3 manifest semantic counts drifted")
    completion = _validate_bundle_chain(
        root,
        expected_external_receipt_path=context.external_consumption_receipt_path,
    )
    identity = {
        "authorization_binding_sha256": context.authorization_binding_sha256,
        "authorization_file_sha256": context.authorization_file_sha256,
        "execution_lock_sha256": context.execution_lock_sha256,
        "execution_lock_commit": context.execution_lock_commit,
        "attempt_id": context.attempt_id,
        "output_root": str(context.output_root),
        "HEAD": context.head,
    }
    for key, expected in identity.items():
        if completion.get(key) != expected:
            raise DecisionContractError(f"C85T V3 completion identity drifted: {key}")
    return completion


def _fsync_tree(root: Path) -> None:
    directories: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise DecisionContractError("C85T V3 bundle cannot contain symlinks")
        if path.is_file():
            descriptor = os.open(path, os.O_RDONLY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        elif path.is_dir():
            directories.append(path)
    for directory_path in reversed(directories):
        descriptor = os.open(directory_path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    descriptor = os.open(root, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


class AtomicExecutionBundleV3:
    """Build one terminal staging bundle and publish it with one final rename."""

    def __init__(self, context: ValidatedC85TExecutionContext) -> None:
        self.context = validate_registered_execution_context(context)
        self.staging_root = context.staging_bundle_root
        self.output_root = context.output_root
        self._prepared_completion: dict[str, Any] | None = None

    def path(self, relative: str | Path) -> Path:
        path = (self.staging_root / relative).resolve()
        if path != self.staging_root and self.staging_root not in path.parents:
            raise DecisionContractError("C85T V3 artifact path escapes staging")
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, relative: str | Path, value: Any) -> Path:
        path = self.path(relative)
        if path.exists():
            raise DecisionContractError(f"C85T V3 artifact must be fresh: {path.name}")
        path.write_bytes(canonical_json_bytes(value))
        return path

    def write_text(self, relative: str | Path, value: str) -> Path:
        path = self.path(relative)
        if path.exists():
            raise DecisionContractError(f"C85T V3 artifact must be fresh: {path.name}")
        path.write_text(value)
        return path

    def write_npz(self, relative: str | Path, arrays: Mapping[str, np.ndarray]) -> Path:
        path = self.path(relative)
        write_deterministic_npz(path, dict(arrays))
        return path

    def _prepare_commit(
        self,
        result: Mapping[str, Any],
        *,
        contract: Mapping[str, Any],
        statements: Mapping[str, str],
        shadow_expected_exact: Mapping[str, Any] | None = None,
        shadow_expected_s9_arrays: Mapping[str, np.ndarray] | None = None,
        shadow_expected_s9_digest_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        if self._prepared_completion is not None:
            raise DecisionContractError("C85T V3 bundle was already prepared")
        validate_registered_execution_context(self.context)
        if result.get("schema_version") != RESULT_SCHEMA_V3:
            raise DecisionContractError("C85T V3 result schema drifted")
        if result.get("final_gate") != SUCCESS_GATE_V3:
            raise DecisionContractError("C85T V3 success gate drifted")
        self.context._lifecycle.append("MANIFEST_STARTED")
        self.write_json("C85T_RESULT.json", dict(result))
        semantic = validate_result_semantics_v3(
            self.staging_root,
            context=self.context,
            contract=contract,
            statements=statements,
            shadow_expected_exact=shadow_expected_exact,
            shadow_expected_s9_arrays=shadow_expected_s9_arrays,
            shadow_expected_s9_digest_rows=shadow_expected_s9_digest_rows,
        )
        semantic_path = self.write_json(
            "C85T_V3_SEMANTIC_REPLAY_RECEIPT.json", semantic
        )
        rows = _artifact_rows(self.staging_root)
        manifest = {
            "schema_version": MANIFEST_SCHEMA_V3,
            "bundle_schema_version": BUNDLE_SCHEMA_V3,
            "created_at_utc": utc_now(),
            "artifact_count": len(rows),
            "derived_counts": semantic,
            "semantic_replay_receipt_sha256": sha256_file(semantic_path),
            "artifacts": rows,
        }
        manifest_path = self.write_json(
            "C85T_RESULT_ARTIFACT_MANIFEST.json", manifest
        )
        replay_artifact_manifest_v3(self.staging_root)
        manifest_sha = sha256_file(manifest_path)
        self.context._lifecycle.append(
            "MANIFEST_COMPLETED", artifact_or_receipt_sha256=manifest_sha
        )
        lifecycle_prefix_sha = sha256_file(self.context.lifecycle_path)
        completion = {
            "schema_version": COMPLETION_SCHEMA_V3,
            "bundle_schema_version": BUNDLE_SCHEMA_V3,
            "created_at_utc": utc_now(),
            "final_gate": SUCCESS_GATE_V3,
            "result_sha256": sha256_file(self.staging_root / "C85T_RESULT.json"),
            "manifest_sha256": manifest_sha,
            "semantic_replay_receipt_sha256": sha256_file(semantic_path),
            "lifecycle_prefix_sha256": lifecycle_prefix_sha,
            "external_consumption_receipt_sha256": self.context.external_consumption_receipt_sha256,
            "authorization_binding_sha256": self.context.authorization_binding_sha256,
            "authorization_file_sha256": self.context.authorization_file_sha256,
            "execution_lock_sha256": self.context.execution_lock_sha256,
            "execution_lock_commit": self.context.execution_lock_commit,
            "attempt_id": self.context.attempt_id,
            "output_root": str(self.context.output_root),
            "HEAD": self.context.head,
        }
        completion_path = self.write_json(
            "C85T_V3_COMPLETION_RECEIPT.json", completion
        )
        self.context._lifecycle.append(
            "ATOMIC_PUBLISH_COMMIT_READY",
            artifact_or_receipt_sha256=sha256_file(completion_path),
        )
        validate_complete_bundle_v3(
            self.staging_root,
            context=self.context,
            contract=contract,
            statements=statements,
            shadow_expected_exact=shadow_expected_exact,
            shadow_expected_s9_arrays=shadow_expected_s9_arrays,
            shadow_expected_s9_digest_rows=shadow_expected_s9_digest_rows,
        )
        _fsync_tree(self.staging_root)
        parent_descriptor = os.open(
            self.staging_root.parent, os.O_RDONLY | os.O_DIRECTORY
        )
        try:
            os.fsync(parent_descriptor)
        finally:
            os.close(parent_descriptor)
        self.context.close()
        self._prepared_completion = completion
        return dict(completion)

    def _commit_prepared(self) -> dict[str, Any]:
        if self._prepared_completion is None:
            raise DecisionContractError("C85T V3 bundle is not commit-ready")
        if self.output_root.exists() or not self.staging_root.is_dir():
            raise DecisionContractError("C85T V3 final/staging state drifted before rename")
        completion = dict(self._prepared_completion)
        os.replace(self.staging_root, self.output_root)
        return completion

    def publish(
        self,
        result: Mapping[str, Any],
        *,
        contract: Mapping[str, Any],
        statements: Mapping[str, str],
        shadow_expected_exact: Mapping[str, Any] | None = None,
        shadow_expected_s9_arrays: Mapping[str, np.ndarray] | None = None,
        shadow_expected_s9_digest_rows: Sequence[Mapping[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Prepare and immediately commit the official single transaction."""

        self._prepare_commit(
            result,
            contract=contract,
            statements=statements,
            shadow_expected_exact=shadow_expected_exact,
            shadow_expected_s9_arrays=shadow_expected_s9_arrays,
            shadow_expected_s9_digest_rows=shadow_expected_s9_digest_rows,
        )
        return self._commit_prepared()


def recover_post_rename_bundle_v3(
    final_root: Path, *, external_consumption_receipt_path: Path
) -> dict[str, Any]:
    """Classify a fully committed final bundle after a process-return crash."""

    final_root = final_root.resolve()
    if not final_root.is_dir():
        raise DecisionContractError("C85T V3 recovery final bundle is absent")
    manifest = replay_artifact_manifest_v3(final_root)
    completion = _validate_bundle_chain(
        final_root,
        expected_external_receipt_path=external_consumption_receipt_path,
    )
    semantic_path = final_root / "C85T_V3_SEMANTIC_REPLAY_RECEIPT.json"
    semantic = json.loads(semantic_path.read_text())
    if (
        semantic.get("status") != "SEMANTIC_REPLAY_PASS"
        or semantic.get("protected_counters_zero") is not True
        or manifest.get("derived_counts") != semantic
        or manifest.get("semantic_replay_receipt_sha256") != sha256_file(semantic_path)
    ):
        raise DecisionContractError("C85T V3 recovery semantic receipt drifted")
    if completion.get("output_root") != str(final_root):
        raise DecisionContractError("C85T V3 recovery output-root identity drifted")
    return {
        "schema_version": RECOVERY_SCHEMA_V3,
        "classification": "RECOVERED_SUCCESS_AFTER_FINAL_RENAME",
        "final_gate": completion["final_gate"],
        "manifest_sha256": completion["manifest_sha256"],
        "attempt_id": completion["attempt_id"],
        "output_root": str(final_root),
    }


def preserve_primary_exception_v3(
    context: ValidatedC85TExecutionContext, primary: BaseException
) -> dict[str, Any] | None:
    """Record failure evidence without replacing the primary exception."""

    secondary: list[str] = []
    if context.output_root.exists():
        try:
            return recover_post_rename_bundle_v3(
                context.output_root,
                external_consumption_receipt_path=context.external_consumption_receipt_path,
            )
        except BaseException as recovery_error:
            secondary.append(f"post-rename-recovery: {type(recovery_error).__name__}: {recovery_error}")
    try:
        events = replay_lifecycle_v3(context.lifecycle_path)
        terminal = events[-1]["stage"] in {"ATOMIC_PUBLISH_COMMIT_READY", "FAILED"}
    except BaseException as replay_error:
        secondary.append(f"lifecycle-replay: {type(replay_error).__name__}: {replay_error}")
        terminal = context._lifecycle.terminal
    if not terminal:
        try:
            context._lifecycle.append(
                "FAILED",
                failure={
                    "primary_exception_type": type(primary).__name__,
                    "primary_exception_message": str(primary),
                    "secondary_errors": secondary,
                },
            )
        except BaseException as ledger_error:
            secondary.append(f"lifecycle-append: {type(ledger_error).__name__}: {ledger_error}")
    elif not context.output_root.exists():
        blocker = {
            "schema_version": "c85t_v3_terminal_without_final_reconciliation_blocker_v1",
            "primary_exception_type": type(primary).__name__,
            "primary_exception_message": str(primary),
            "secondary_errors": secondary,
            "attempt_id": context.attempt_id,
            "output_root": str(context.output_root),
            "automatic_retry": False,
        }
        try:
            path = context.staging_bundle_root / "C85T_V3_RECONCILIATION_BLOCKER.json"
            if not path.exists():
                path.write_bytes(canonical_json_bytes(blocker))
        except BaseException as report_error:
            secondary.append(f"reconciliation-report: {type(report_error).__name__}: {report_error}")
    try:
        failure_path = context.staging_bundle_root / "C85T_V3_FAILURE_RECEIPT.json"
        if context.staging_bundle_root.exists() and not failure_path.exists():
            failure_path.write_bytes(
                canonical_json_bytes(
                    {
                        "schema_version": "c85t_v3_execution_failure_receipt_v1",
                        "primary_exception_type": type(primary).__name__,
                        "primary_exception_message": str(primary),
                        "secondary_errors": secondary,
                        "authorization_consumed": True,
                        "attempt_id": context.attempt_id,
                        "automatic_retry": False,
                    }
                )
            )
    except BaseException:
        pass
    return None


__all__ = (
    "AtomicExecutionBundleV3",
    "BUNDLE_SCHEMA_V3",
    "COMPLETION_SCHEMA_V3",
    "MANIFEST_SCHEMA_V3",
    "preserve_primary_exception_v3",
    "recover_post_rename_bundle_v3",
    "replay_artifact_manifest_v3",
    "validate_complete_bundle_v3",
)
