"""Fail-closed runtime guard for the repaired C84L1C canary.

The only numerical change is the tolerance for the 1040-term float32
classifier reconstruction performed by different CPU/GPU kernels. Exact
saved-softmax and repeated-forward tensor checks retain the prior tolerance.
"""
from __future__ import annotations

from importlib import metadata
import math
from pathlib import Path
import time
from typing import Any, Callable, Mapping

from .c84l1_runtime_guard import *  # noqa: F403
from . import c84l1_runtime_guard as base


REPO_ROOT = base.REPO_ROOT
REPORT_DIR = base.REPORT_DIR
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84L1R1_FLOAT32_LINEAR_REPLAY_REPAIR_PROTOCOL.sha256"
CANARY_PROTOCOL_PATH = REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V2.json"
CANARY_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V2.sha256"
FIELD_PROTOCOL_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V6.json"
FIELD_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V6.sha256"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK_V2.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84L1C_EXECUTION_LOCK_V2.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84L1C_PI_AUTHORIZATION_RECORD_V2.json"
FAILED_ATTEMPT_PATH = REPORT_DIR / "C84L1C_FAILED_ATTEMPT_895928.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-level1-canary-v2")
LOCK_READY_STATUS = "LOCKED_READY_FOR_FRESH_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
LINEAR_REPLAY_ABS_TOLERANCE = 2e-5
STRICT_IDENTITY_ABS_TOLERANCE = 1e-6


class C84L1R1RuntimeError(base.C84L1RuntimeError):
    """Raised when the repaired level-1 canary cannot prove its lock."""


def validate_instrumentation_errors(
    errors: Mapping[str, Any],
    *,
    linear_tolerance: float = LINEAR_REPLAY_ABS_TOLERANCE,
    strict_tolerance: float = STRICT_IDENTITY_ABS_TOLERANCE,
) -> dict[str, Any]:
    expected = {
        "Wz_plus_b_max_error",
        "softmax_max_error",
        "repeat_logits_max_error",
        "repeat_z_max_error",
    }
    if set(errors) != expected:
        raise C84L1R1RuntimeError("instrumentation error registry field-set drift")
    observed = {key: float(errors[key]) for key in sorted(expected)}
    if not all(math.isfinite(value) and value >= 0.0 for value in observed.values()):
        raise C84L1R1RuntimeError("instrumentation error registry contains invalid values")
    if observed["Wz_plus_b_max_error"] > linear_tolerance:
        raise C84L1R1RuntimeError(f"float32 linear instrumentation identity failed: {observed}")
    strict_keys = ("softmax_max_error", "repeat_logits_max_error", "repeat_z_max_error")
    if max(observed[key] for key in strict_keys) > strict_tolerance:
        raise C84L1R1RuntimeError(f"strict instrumentation identity failed: {observed}")
    return {
        **observed,
        "linear_replay_abs_tolerance": float(linear_tolerance),
        "strict_identity_abs_tolerance": float(strict_tolerance),
        "validation_pass": True,
    }


def replay_target_unlabeled_artifact(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_trial_ids: list[str] | tuple[str, ...],
    np: Any,
    linear_tolerance: float = LINEAR_REPLAY_ABS_TOLERANCE,
    strict_tolerance: float = STRICT_IDENTITY_ABS_TOLERANCE,
) -> dict[str, Any]:
    return base.prior.replay_target_unlabeled_artifact(
        path,
        expected_identity=expected_identity,
        expected_trial_ids=expected_trial_ids,
        np=np,
        linear_tolerance=linear_tolerance,
        strict_tolerance=strict_tolerance,
    )


def verify_failed_attempt_binding(lock: Mapping[str, Any]) -> dict[str, Any]:
    binding = lock.get("historical_failed_attempt", {})
    if int(binding.get("job_id", -1)) != 895928:
        raise C84L1R1RuntimeError("historical failed-attempt job identity drift")
    if not FAILED_ATTEMPT_PATH.is_file():
        raise C84L1R1RuntimeError("historical failed-attempt report is absent")
    observed = sha256_file(FAILED_ATTEMPT_PATH)  # noqa: F405
    if observed != binding.get("report_sha256"):
        raise C84L1R1RuntimeError("historical failed-attempt report hash drift")
    if binding.get("partial_artifacts_reusable") is not False:
        raise C84L1R1RuntimeError("failed partial artifacts must remain non-reusable")
    failed_root = Path(str(binding.get("external_root", "")))
    if failed_root == DEFAULT_EXTERNAL_ROOT:
        raise C84L1R1RuntimeError("replacement external root aliases the failed root")
    return {
        "job_id": 895928,
        "report_sha256": observed,
        "partial_artifacts_reusable": False,
        "replay_pass": True,
    }


def verify_authorization_record(
    lock: Mapping[str, Any],
    lock_sha256: str,
    lock_commit: str,
    authorization_path: Path,
) -> dict[str, Any]:
    if not authorization_path.is_file():
        raise C84L1R1RuntimeError("fresh repaired C84L1C PI authorization record is absent")
    record = read_json(authorization_path)  # noqa: F405
    required = {
        "schema_version": "c84l1c_direct_pi_authorization_record_v2",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84L1C",
        "canary_protocol_v2_sha256": lock["canary_protocol"]["sha256"],
        "execution_lock_v2_sha256": lock_sha256,
        "execution_lock_v2_commit": lock_commit,
        "repair_protocol_sha256": lock["repair_protocol"]["sha256"],
        "level_intervention_registry_sha256": lock["level_intervention"]["registry_sha256"],
        "canary_unit_ID_digest": lock["candidate_identity"]["canary_unit_ID_digest"],
        "failed_authorization_reused": False,
        "failed_partial_artifacts_reused": False,
        "C84F": False,
        "C84S": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items()
                if record.get(key) != wanted}
    if mismatch:
        raise C84L1R1RuntimeError(f"repaired C84L1C authorization binding mismatch: {mismatch}")
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
        raise C84L1R1RuntimeError("repaired C84L1C execution lock is absent")
    lock_sha = verify_lock_self(lock_path, lock_sha_path)  # noqa: F405
    lock = read_json(lock_path)  # noqa: F405
    if lock.get("status") != LOCK_READY_STATUS:
        raise C84L1R1RuntimeError("repaired C84L1C lock status is not the readiness state")
    bound = base.prior.verify_bound_object_registry(lock, repo_root=repo_root)
    protocols = base.prior.verify_protocol_sidecars(lock, repo_root=repo_root)
    montage = base.prior.verify_montage_binding(lock)
    environment = base.prior.verify_distribution_environment(
        lock, version_getter=version_getter, python_version=python_version, prefix=prefix,
    )
    deterministic = base.prior.verify_deterministic_environment(environ=environ)
    intervention = base.verify_intervention_registry(lock)
    candidate = base.verify_candidate_identity(lock)
    c84c = base.verify_c84c_level0_binding(lock)
    failed = verify_failed_attempt_binding(lock)
    lock_commit = base.prior.commit_for_path(lock_path, repo_root=repo_root)
    repository = base.prior.verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = base.prior.verify_output_root(Path(output_root), lock_sha)
    failed_root = Path(lock["historical_failed_attempt"]["external_root"])
    if Path(run_root).resolve() == failed_root.resolve():
        raise C84L1R1RuntimeError("replacement run root aliases failed attempt root")
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "lock_commit": lock_commit,
        "authorization": authorization,
        "authorization_sha256": authorization["record_sha256"],
        "bound_object_replay": bound,
        "protocol_replay": protocols,
        "montage_replay": montage,
        "environment_replay": environment,
        "deterministic_environment": deterministic,
        "intervention_replay": intervention,
        "candidate_replay": candidate,
        "C84C_level0_replay": c84c,
        "failed_attempt_replay": failed,
        "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    if path.exists():
        raise C84L1R1RuntimeError("repaired C84L1C authorization was already consumed")
    payload = {
        "schema_version": "c84l1c_authorization_consumption_v2",
        "stage": "C84L1C",
        "execution_lock_v2_sha256": binding["lock_sha256"],
        "execution_lock_v2_commit": binding["lock_commit"],
        "canary_protocol_v2_sha256": binding["lock"]["canary_protocol"]["sha256"],
        "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_package_imports": True,
        "before_CUDA_check": True,
        "before_dataset_loader_import": True,
        "before_download_or_get_data": True,
        "failed_authorization_reused": False,
        "failed_partial_artifacts_reused": False,
        "target_scientific_outcomes_authorized": False,
        "C84F_authorized": False,
        "C84S_authorized": False,
    }
    write_json_atomic(path, payload)  # noqa: F405
    return {**payload, "path": str(path), "sha256": sha256_file(path)}  # noqa: F405
