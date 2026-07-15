"""Fail-closed standard-library guard for the C84FR2 target-only runtime."""
from __future__ import annotations

from importlib import metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Mapping

from . import c84f_field_manifest as manifests
from . import c84f_runtime_guard as base
from . import c84fr1_runtime_guard as historical


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84F_TARGET_STAGE_EXECUTION_LOCK_V2.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84F_TARGET_STAGE_NUMERICAL_REPLAY_PI_AUTHORIZATION_RECORD.json"
DEFAULT_EXTERNAL_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2"
)
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_REAUTHORIZATION_NOT_AUTHORIZED"


class C84FR2RuntimeError(RuntimeError):
    """Raised before target access when a C84FR2 binding cannot replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C84FR2RuntimeError(message)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _verify_file(path: Path, expected: str, name: str) -> None:
    _require(path.is_file(), f"frozen C84FR1 object is absent: {name}")
    _require(base.sha256_file(path) == expected, f"frozen C84FR1 object hash drift: {name}")


def verify_frozen_model_and_target_inputs(lock: Mapping[str, Any]) -> dict[str, Any]:
    model = historical.verify_failed_attempt_and_model_field({
        "frozen_failed_attempt": lock["frozen_model_field_source"],
    })
    target = lock.get("frozen_target_input_source", {})
    root = Path(str(target.get("root", "")))
    _require(root.is_dir(), "frozen C84FR1 failed root is absent")
    paths = {
        "failure_evidence": REPORT_DIR / "C84FR1_FAILED_ATTEMPT_896550.json",
        "authorization_consumed": root / "authorization_consumed.json",
        "execution_attempts": root / "execution_attempts.jsonl",
        "partial_manifest": root / "partial_artifact_manifest.json",
        "target_raw_manifest": root / "C84F_TARGET_RAW_INPUT_MANIFEST.json",
        "target_registry": root / manifests.TARGET_REGISTRY_NAME,
        "target_registry_sha": root / manifests.TARGET_REGISTRY_SHA_NAME,
    }
    for name, path in paths.items():
        _verify_file(path, str(target["sha256"][name]), name)

    partial = read_json(paths["partial_manifest"])
    counters = partial.get("counters", {})
    _require(partial.get("status") == "FAILED", "C84FR1 partial manifest is not failed")
    _require(partial.get("error_type") == "C84FManifestError", "C84FR1 failure type drift")
    _require(
        partial.get("error") == "linear persisted replay exceeds its locked tolerance: "
        "2.193450927734375e-05 > 2e-05",
        "C84FR1 failure message drift",
    )
    _require(int(counters.get("target_EEG_arrays", -1)) == 118, "C84FR1 target-X count drift")
    _require(int(counters.get("target_registry_trials", -1)) == 9621, "C84FR1 registry count drift")
    for name in (
        "target_y_accesses", "target_scientific_metrics", "training_phases_started",
        "training_phases_completed",
    ):
        _require(int(counters.get(name, -1)) == 0, f"C84FR1 protected counter is nonzero: {name}")
    _require(int(counters.get("target_unlabeled_artifacts", -1)) == 5,
             "C84FR1 completed target-artifact count drift")
    _require(int(counters.get("target_context_slices", -1)) == 268,
             "C84FR1 partial context-slice count drift")
    _require(not (root / "C84F_COMPLETE_FIELD_MANIFEST.json").exists(),
             "C84FR1 failed root unexpectedly contains a complete manifest")

    partial_objects = target.get("partial_target_objects", ())
    _require(len(partial_objects) == 11, "C84FR1 partial target-object registry is not 6 NPZ + 5 JSON")
    observed_npz = 0
    observed_json = 0
    for row in partial_objects:
        path = Path(str(row["path"]))
        _verify_file(path, str(row["sha256"]), f"partial_target/{path.name}")
        if path.suffix == ".npz":
            observed_npz += 1
        elif path.suffix == ".json":
            observed_json += 1
    _require((observed_npz, observed_json) == (6, 5), "C84FR1 partial target-object arithmetic drift")
    actual_npz = tuple(sorted((root / "complete_target_unlabeled").glob("*.npz")))
    actual_json = tuple(sorted((root / "target_context_index").glob("*.json")))
    _require(len(actual_npz) == 6 and len(actual_json) == 5,
             "C84FR1 partial root file count drift")

    registry = manifests.verify_target_trial_registry(
        paths["target_registry"], paths["target_registry_sha"],
    )
    _require(registry["sha256"] == target["sha256"]["target_registry"],
             "C84FR1 target registry digest drift")
    _require(registry["trial_rows"] == 9621 and registry["target_subjects"] == 118,
             "C84FR1 target registry arithmetic drift")
    return {
        **model,
        "target_input_root": root,
        "target_raw_manifest_path": paths["target_raw_manifest"],
        "target_registry_path": paths["target_registry"],
        "target_registry_sha_path": paths["target_registry_sha"],
        "target_registry_sha256": registry["sha256"],
        "target_registry_rows": registry["trial_rows"],
        "target_subjects": registry["target_subjects"],
        "partial_target_objects_replayed": 11,
        "partial_target_artifacts_reusable": False,
    }


def verify_authorization_record(
    lock: Mapping[str, Any], lock_sha256: str, lock_commit: str, authorization_path: Path,
) -> dict[str, Any]:
    _require(authorization_path.is_file(), "fresh C84FR2 PI authorization record is absent")
    record = read_json(authorization_path)
    required = {
        "schema_version": "c84fr2_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84F_TARGET_STAGE_NUMERICAL_REPLAY_REPAIR",
        "repair_protocol_sha256": lock["protocols"]["c84fr2_repair"]["sha256"],
        "target_instrumentation_protocol_v2_sha256": lock["protocols"]["target_v2"]["sha256"],
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
        "model_field_manifest_sha256": lock["frozen_model_field_source"]["sha256"]["model_manifest"],
        "target_trial_registry_sha256": lock["frozen_target_input_source"]["sha256"]["target_registry"],
        "model_retraining": False,
        "target_labels": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
        "C84S": False,
    }
    mismatch = {
        key: (record.get(key), wanted) for key, wanted in required.items()
        if record.get(key) != wanted
    }
    _require(not mismatch, f"C84FR2 authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": base.sha256_file(authorization_path)}


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
    _require(lock_path.is_file(), "C84FR2 target-stage execution lock is absent")
    lock_sha = base.verify_lock_self(lock_path, lock_sha_path)
    lock = read_json(lock_path)
    _require(lock.get("status") == LOCK_READY_STATUS, "C84FR2 target-stage lock is not ready")
    bound = base.verify_bound_object_registry(lock, repo_root=repo_root)
    protocols = base.verify_protocol_sidecars(lock, repo_root=repo_root)
    interface = base.verify_interface(lock)
    environment = base.verify_distribution_environment(
        lock,
        version_getter=version_getter,
        python_version=python_version or platform.python_version(),
        prefix=prefix or sys.prefix,
    )
    deterministic = base.verify_deterministic_environment(environ)
    candidates = base.verify_candidate_and_wave_registry(lock, repo_root=repo_root)
    reuse = base.verify_dual_canary_reuse(lock, repo_root=repo_root)
    frozen = verify_frozen_model_and_target_inputs(lock)
    lock_commit = base.commit_for_path(lock_path, repo_root=repo_root)
    repository = base.verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = base.verify_output_root(Path(output_root), lock_sha)
    return {
        "lock": lock,
        "lock_sha256": lock_sha,
        "lock_commit": lock_commit,
        "authorization": authorization,
        "authorization_sha256": authorization["record_sha256"],
        "bound_object_replay": bound,
        "protocol_replay": protocols,
        "interface_replay": interface,
        "environment_replay": environment,
        "deterministic_environment": deterministic,
        "candidate_replay": candidates,
        "dual_canary_replay": reuse,
        "frozen_replay": frozen,
        "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    _require(not path.exists(), "C84FR2 authorization was already consumed")
    payload = {
        "schema_version": "c84fr2_authorization_consumption_v1",
        "stage": "C84F_TARGET_STAGE_NUMERICAL_REPLAY_REPAIR",
        "execution_lock_sha256": binding["lock_sha256"],
        "execution_lock_commit": binding["lock_commit"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "repair_protocol_sha256": binding["lock"]["protocols"]["c84fr2_repair"]["sha256"],
        "target_instrumentation_protocol_v2_sha256": binding["lock"]["protocols"]["target_v2"]["sha256"],
        "model_field_manifest_sha256": binding["frozen_replay"]["model_manifest_sha256"],
        "target_trial_registry_sha256": binding["frozen_replay"]["target_registry_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_protected_package_imports": True,
        "before_loader_import": True,
        "before_target_data_access": True,
        "model_retraining_authorized": False,
        "target_labels_authorized": False,
        "scientific_outcomes_authorized": False,
        "C84S_authorized": False,
    }
    digest = manifests.write_json_atomic(path, payload)
    return {**payload, "path": str(path), "sha256": digest}


ExecutionAttemptLedger = base.ExecutionAttemptLedger
verify_loader_source_files = base.verify_loader_source_files
verify_loader_runtime_objects = base.verify_loader_runtime_objects
verify_protected_runtime_versions = base.verify_protected_runtime_versions
