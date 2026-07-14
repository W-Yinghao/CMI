"""C84R3 runtime bindings and split float32 instrumentation tolerances."""
from __future__ import annotations

from importlib import metadata
import math
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping, Sequence

from .c84r2_canary_runtime_repair import *  # noqa: F403
from . import c84r2_canary_runtime_repair as base


REPO_ROOT = base.REPO_ROOT
REPORT_DIR = base.REPORT_DIR
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84R3_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.sha256"
CANARY_PROTOCOL_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V4.json"
CANARY_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V4.sha256"
FIELD_PROTOCOL_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V4.json"
FIELD_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V4.sha256"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK_V3.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK_V3.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84C_PI_AUTHORIZATION_RECORD_V3.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v4")
LOCK_READY_STATUS = "LOCKED_READY_FOR_FRESH_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
LINEAR_REPLAY_ABS_TOLERANCE = 1e-5
STRICT_IDENTITY_ABS_TOLERANCE = 1e-6


class C84R3RuntimeError(base.C84R2RuntimeError):
    """Raised when the C84R3 replacement canary cannot prove its lock."""


def validate_instrumentation_errors(
    errors: Mapping[str, Any],
    *,
    linear_tolerance: float = LINEAR_REPLAY_ABS_TOLERANCE,
    strict_tolerance: float = STRICT_IDENTITY_ABS_TOLERANCE,
) -> dict[str, Any]:
    """Apply the split linear-versus-identity tolerance contract."""
    expected = {
        "Wz_plus_b_max_error",
        "softmax_max_error",
        "repeat_logits_max_error",
        "repeat_z_max_error",
    }
    if set(errors) != expected:
        raise C84R3RuntimeError("instrumentation error registry field-set drift")
    observed = {key: float(errors[key]) for key in sorted(expected)}
    if not all(math.isfinite(value) and value >= 0.0 for value in observed.values()):
        raise C84R3RuntimeError("instrumentation error registry contains a non-finite or negative value")
    if observed["Wz_plus_b_max_error"] > linear_tolerance:
        raise C84R3RuntimeError(
            f"float32 linear instrumentation identity failed: {observed}"
        )
    strict_keys = ("softmax_max_error", "repeat_logits_max_error", "repeat_z_max_error")
    if max(observed[key] for key in strict_keys) > strict_tolerance:
        raise C84R3RuntimeError(f"strict instrumentation identity failed: {observed}")
    return {
        **observed,
        "linear_replay_abs_tolerance": float(linear_tolerance),
        "strict_identity_abs_tolerance": float(strict_tolerance),
        "validation_pass": True,
    }


def verify_authorization_record(
    lock: Mapping[str, Any],
    lock_sha256: str,
    lock_commit: str,
    authorization_path: Path,
) -> dict[str, Any]:
    if not authorization_path.is_file():
        raise C84R3RuntimeError("fresh direct C84C replacement authorization record is absent")
    record = read_json(authorization_path)  # noqa: F405
    required = {
        "schema_version": "c84c_direct_pi_authorization_record_v3",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84C",
        "canary_protocol_v4_sha256": lock["canary_protocol"]["sha256"],
        "execution_lock_v3_sha256": lock_sha256,
        "execution_lock_v3_commit": lock_commit,
        "repair_protocol_sha256": lock["repair_protocol"]["sha256"],
        "montage_sha256": lock["interface"]["montage_sha256"],
        "canary_unit_ids_sha256": lock["candidate_identity"]["canary_unit_ids_sha256"],
        "C84F": False,
        "C84S": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
        "failed_authorization_reused": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items() if record.get(key) != wanted}
    if mismatch:
        raise C84R3RuntimeError(f"C84C V4 authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": sha256_file(authorization_path)}  # noqa: F405


def require_authorization_and_lock(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
    lock_path: Path = EXECUTION_LOCK_PATH,
    lock_sha_path: Path = EXECUTION_LOCK_SHA_PATH,
    repo_root: Path = REPO_ROOT,
    version_getter: Callable[[str], str] = metadata.version,
    python_version: str | None = None,
    prefix: str | None = None,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    if not lock_path.is_file():
        raise C84R3RuntimeError("C84C V3 execution lock is absent")
    lock_sha = verify_lock_self(lock_path, lock_sha_path)  # noqa: F405
    lock = read_json(lock_path)  # noqa: F405
    if lock.get("status") != LOCK_READY_STATUS:
        raise C84R3RuntimeError("C84C V3 lock status is not the unique readiness state")
    bound_replay = verify_bound_object_registry(lock, repo_root=repo_root)  # noqa: F405
    protocol_replay = verify_protocol_sidecars(lock, repo_root=repo_root)  # noqa: F405
    montage = verify_montage_binding(lock)  # noqa: F405
    environment = verify_distribution_environment(  # noqa: F405
        lock, version_getter=version_getter, python_version=python_version, prefix=prefix,
    )
    deterministic = verify_deterministic_environment(environ=environ)  # noqa: F405
    candidate = verify_candidate_identity(lock)  # noqa: F405
    lock_commit = commit_for_path(lock_path, repo_root=repo_root)  # noqa: F405
    repository = verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)  # noqa: F405
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = verify_output_root(Path(output_root), lock_sha)  # noqa: F405
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "authorization": authorization,
        "authorization_sha256": authorization["record_sha256"],
        "lock_commit": lock_commit,
        "bound_object_replay": bound_replay,
        "protocol_replay": protocol_replay,
        "montage_replay": montage,
        "environment_replay": environment,
        "deterministic_environment": deterministic,
        "candidate_replay": candidate,
        "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    if path.exists():
        raise C84R3RuntimeError("C84C V4 authorization was already consumed")
    payload = {
        "schema_version": "c84c_authorization_consumption_v3",
        "stage": "C84C",
        "execution_lock_v3_sha256": binding["lock_sha256"],
        "execution_lock_v3_commit": binding["lock_commit"],
        "canary_protocol_v4_sha256": binding["lock"]["canary_protocol"]["sha256"],
        "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "failed_attempt_authorization_reused": False,
        "consumed_at_unix_ns": time.time_ns(),
        "before_package_imports": True,
        "before_CUDA_check": True,
        "before_dataset_loader_import": True,
        "before_download_or_get_data": True,
        "target_scientific_outcomes_authorized": False,
    }
    write_json_atomic(path, payload)  # noqa: F405
    return {**payload, "path": str(path), "sha256": sha256_file(path)}  # noqa: F405


def _softmax_numpy(logits: Any, np: Any) -> Any:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


def replay_target_unlabeled_artifact(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_trial_ids: Sequence[str],
    np: Any,
    linear_tolerance: float = LINEAR_REPLAY_ABS_TOLERANCE,
    strict_tolerance: float = STRICT_IDENTITY_ABS_TOLERANCE,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != TARGET_UNLABELED_FIELDS:  # noqa: F405
            raise C84R3RuntimeError("target-unlabeled artifact field-set drift or target-label field present")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        z = np.asarray(archive["z"])
        weight = np.asarray(archive["classifier_weight"])
        bias = np.asarray(archive["classifier_bias"])
        reconstructed = z @ weight.T + bias
        linear_error = float(np.max(np.abs(reconstructed - logits)))
        if logits.ndim != 2 or logits.shape[1] != 2 or probabilities.shape != logits.shape:
            raise C84R3RuntimeError("target-unlabeled logits/probability shape drift")
        if reconstructed.shape != logits.shape or linear_error > linear_tolerance:
            raise C84R3RuntimeError(
                f"saved z/classifier/logits replay failed: {linear_error} > {linear_tolerance}"
            )
        if np.max(np.abs(_softmax_numpy(logits, np) - probabilities)) > strict_tolerance:
            raise C84R3RuntimeError("saved target-unlabeled softmax replay failed")
        if np.max(np.abs(np.asarray(archive["repeat_logits"]) - logits)) > strict_tolerance:
            raise C84R3RuntimeError("saved repeat-logits replay failed")
        if np.max(np.abs(np.asarray(archive["repeat_z"]) - z)) > strict_tolerance:
            raise C84R3RuntimeError("saved repeat-z replay failed")
        if tuple(archive["target_trial_id"].astype(str)) != tuple(map(str, expected_trial_ids)):
            raise C84R3RuntimeError("target-unlabeled trial-ID replay failed")
        for key in ("dataset", "unit_id"):
            if str(archive[key].item()) != str(expected_identity[key]):
                raise C84R3RuntimeError(f"target-unlabeled identity drift: {key}")
        if int(archive["target_subject_id"].item()) != int(expected_identity["target_subject_id"]):
            raise C84R3RuntimeError("target-unlabeled subject identity drift")
        rows = logits.shape[0]
    return {
        "path": str(path),
        "sha256": sha256_file(path),  # noqa: F405
        "rows": rows,
        "linear_replay_max_abs_error": linear_error,
        "linear_replay_abs_tolerance": linear_tolerance,
        "strict_identity_abs_tolerance": strict_tolerance,
        "replay_pass": True,
    }
