"""Runtime guard for the future authorized C84F full-field execution.

Only standard-library modules are imported before authorization consumption.
Every lock-bound byte, protocol, registry, canary artifact, environment value,
and repository identity is replayed before a real loader can be imported.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Callable, Iterable, Mapping

from . import c84fl2_protocol as protocol
from .c84r_montage_repair import COMMON_CHANNELS


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84F_EXECUTION_LOCK.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84F_EXECUTION_LOCK.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84F_PI_AUTHORIZATION_RECORD.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-v1")
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"


class C84FRuntimeError(RuntimeError):
    """Raised before protected C84F access when any binding cannot replay."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _atomic_json(path: Path, value: Any) -> str:
    from .c84f_field_manifest import write_json_atomic

    return write_json_atomic(path, value)


def _git(repo_root: Path, *arguments: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ("git", *arguments), cwd=repo_root, check=check, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )


def _sidecar_digest(path: Path) -> str:
    if not path.is_file():
        raise C84FRuntimeError(f"missing lock/protocol SHA-256 sidecar: {path}")
    values = path.read_text(encoding="ascii").split()
    if not values or len(values[0]) != 64:
        raise C84FRuntimeError(f"malformed SHA-256 sidecar: {path}")
    return values[0]


def _safe_repo_path(repo_root: Path, relative: str) -> Path:
    try:
        path = (repo_root / relative).resolve()
        path.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise C84FRuntimeError(f"lock-bound path escapes repository: {relative}") from exc
    return path


def verify_lock_self(lock_path: Path, lock_sha_path: Path) -> str:
    expected = _sidecar_digest(lock_sha_path)
    if not lock_path.is_file() or sha256_file(lock_path) != expected:
        raise C84FRuntimeError("C84F execution-lock SHA-256 replay failed")
    return expected


def verify_bound_object_registry(
    lock: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    current_blob_getter: Callable[[Path], str] | None = None,
    head_blob_getter: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    objects = lock.get("runtime_bound_objects")
    if not isinstance(objects, list) or not objects:
        raise C84FRuntimeError("C84F lock has no runtime-bound object registry")
    current_blob_getter = current_blob_getter or (
        lambda path: _git(repo_root, "hash-object", str(path)).stdout.strip()
    )
    head_blob_getter = head_blob_getter or (
        lambda relative: _git(repo_root, "rev-parse", f"HEAD:{relative}").stdout.strip()
    )
    seen: set[str] = set()
    replay = []
    for row in objects:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise C84FRuntimeError(f"duplicate or empty runtime-bound path: {relative!r}")
        seen.add(relative)
        path = _safe_repo_path(repo_root, relative)
        if not path.is_file() or sha256_file(path) != row.get("sha256"):
            raise C84FRuntimeError(f"runtime-bound SHA-256 drift: {relative}")
        if int(row.get("bytes", -1)) != path.stat().st_size:
            raise C84FRuntimeError(f"runtime-bound byte-count drift: {relative}")
        expected_blob = row.get("blob")
        current_blob = current_blob_getter(path)
        head_blob = head_blob_getter(relative)
        if not expected_blob or current_blob != expected_blob or head_blob != expected_blob:
            raise C84FRuntimeError(f"runtime-bound Git blob drift: {relative}")
        replay.append({
            "path": relative, "sha256": row["sha256"], "blob": expected_blob,
            "current_blob": current_blob, "head_blob": head_blob, "replay_pass": True,
        })
    return replay


def verify_protocol_sidecars(lock: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    bindings = lock.get("protocol_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise C84FRuntimeError("C84F lock has no protocol bindings")
    replay = []
    for row in bindings:
        path = _safe_repo_path(repo_root, str(row["path"]))
        sidecar = _safe_repo_path(repo_root, str(row["sha256_path"]))
        observed = sha256_file(path)
        if observed != row["sha256"] or _sidecar_digest(sidecar) != observed:
            raise C84FRuntimeError(f"C84F protocol replay failed: {row['path']}")
        replay.append({"path": row["path"], "sha256": observed, "replay_pass": True})
    return replay


def verify_interface(lock: Mapping[str, Any]) -> dict[str, Any]:
    interface = lock.get("interface", {})
    channels = tuple(interface.get("channels", ()))
    if channels != tuple(COMMON_CHANNELS) or len(channels) != 20:
        raise C84FRuntimeError("C84F channel list/order differs from the exact 20-channel interface")
    digest = hashlib.sha256(canonical_bytes(list(channels))).hexdigest()
    if digest != protocol.HASHES["montage"] or interface.get("montage_sha256") != digest:
        raise C84FRuntimeError("C84F montage digest replay failed")
    expected = {
        "id": protocol.INTERFACE_ID, "sample_rate_hz": 160,
        "sample_count": 480, "linear_replay_abs_tolerance": protocol.LINEAR_TOLERANCE,
        "strict_identity_abs_tolerance": protocol.STRICT_TOLERANCE,
    }
    mismatch = {key: (interface.get(key), value) for key, value in expected.items() if interface.get(key) != value}
    if mismatch:
        raise C84FRuntimeError(f"C84F interface binding drift: {mismatch}")
    if any(bool(interface.get(key)) for key in (
        "Fz_substitution", "FCz_interpolation", "zero_fill", "dataset_specific_mask",
    )):
        raise C84FRuntimeError("C84F interface enables a forbidden montage repair")
    return {"channels": list(channels), "montage_sha256": digest, **expected}


def verify_distribution_environment(
    lock: Mapping[str, Any],
    *,
    version_getter: Callable[[str], str] = metadata.version,
    python_version: str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    expected = lock.get("environment", {})
    observed_python = python_version or platform.python_version()
    observed_prefix = prefix or sys.prefix
    if observed_python != expected.get("python"):
        raise C84FRuntimeError(f"locked Python version mismatch: {observed_python}")
    if os.path.realpath(observed_prefix) != os.path.realpath(str(expected.get("conda_prefix"))):
        raise C84FRuntimeError("locked Conda prefix mismatch")
    versions = {}
    for distribution, wanted in expected.get("distributions", {}).items():
        try:
            observed = version_getter(distribution)
        except metadata.PackageNotFoundError as exc:
            raise C84FRuntimeError(f"locked distribution is absent: {distribution}") from exc
        if observed != wanted:
            raise C84FRuntimeError(
                f"locked distribution mismatch for {distribution}: {observed} != {wanted}"
            )
        versions[distribution] = observed
    return {
        "python": observed_python, "conda_prefix": os.path.realpath(observed_prefix),
        "distributions": versions, "protected_package_imports": 0,
    }


def verify_deterministic_environment(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    values = dict(os.environ if environ is None else environ)
    expected = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
    mismatch = {key: (values.get(key), wanted) for key, wanted in expected.items() if values.get(key) != wanted}
    if mismatch:
        raise C84FRuntimeError(f"deterministic environment mismatch: {mismatch}")
    return expected


def verify_candidate_and_wave_registry(lock: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    binding = lock.get("candidate_identity", {})
    path = _safe_repo_path(repo_root, str(binding.get("registry_path", "")))
    if not path.is_file() or sha256_file(path) != binding.get("registry_sha256"):
        raise C84FRuntimeError("C84F operative candidate registry drift")
    rows = read_csv(path)
    ids = [row["unit_id"] for row in rows]
    if len(rows) != 1944 or len(set(ids)) != 1944:
        raise C84FRuntimeError("C84F operative candidate universe is not 1,944 unique units")
    digest = hashlib.sha256(canonical_bytes(sorted(ids))).hexdigest()
    if digest != binding.get("unit_id_digest"):
        raise C84FRuntimeError("C84F operative unit-ID digest drift")
    if sum(int(row["level"]) == 0 for row in rows) != 972 or sum(int(row["level"]) == 1 for row in rows) != 972:
        raise C84FRuntimeError("C84F operative level arithmetic drift")
    historical = {row["historical_planned_unit_id"] for row in rows if row["identity_status"] == "SUPERSEDED_LEVEL1"}
    operative = {row["unit_id"] for row in rows}
    if historical & operative:
        raise C84FRuntimeError("historical superseded level-1 unit ID is operative")
    return {"units": 1944, "level0": 972, "level1": 972, "unit_id_digest": digest}


def verify_dual_canary_reuse(lock: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    binding = lock.get("dual_canary_reuse", {})
    registry_path = _safe_repo_path(repo_root, str(binding.get("registry_path", "")))
    if not registry_path.is_file() or sha256_file(registry_path) != binding.get("registry_sha256"):
        raise C84FRuntimeError("dual-canary reuse registry drift")
    rows = read_csv(registry_path)
    if len(rows) != 486 or len({row["unit_id"] for row in rows}) != 486:
        raise C84FRuntimeError("dual-canary reuse registry is not 486 unique units")
    counts = {"C84C": 0, "C84L1C": 0}
    artifacts = 0
    failed_roots = tuple(str(value) for value in binding.get("forbidden_failed_roots", ()))
    for row in rows:
        source = row["reuse_source"]
        if source not in counts:
            raise C84FRuntimeError(f"unregistered reuse source: {source}")
        counts[source] += 1
        for path_field, hash_field in (
            ("checkpoint_path", "checkpoint_sha256"),
            ("optimizer_path", "optimizer_sha256"),
            ("sidecar_path", "sidecar_sha256"),
            ("source_audit_path", "source_audit_sha256"),
            ("canary_target_path", "canary_target_sha256"),
        ):
            path = Path(row[path_field])
            if any(str(path).startswith(root) for root in failed_roots):
                raise C84FRuntimeError(f"failed canary root entered reuse registry: {path}")
            if not path.is_file() or sha256_file(path) != row[hash_field]:
                raise C84FRuntimeError(f"dual-canary artifact replay failed: {row['unit_id']}/{path_field}")
            artifacts += 1
        if int(row["failed_artifact_reused"]) or not int(row["byte_hash_manifest_replay_pass"]):
            raise C84FRuntimeError(f"dual-canary reuse gate failed: {row['unit_id']}")
    if counts != {"C84C": 243, "C84L1C": 243} or artifacts != 2430:
        raise C84FRuntimeError(f"dual-canary reuse arithmetic drift: {counts}/{artifacts}")
    return {"units": 486, "artifact_files_replayed": artifacts, "source_counts": counts}


def commit_for_path(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    relative = str(path.resolve().relative_to(repo_root.resolve()))
    value = _git(repo_root, "log", "-1", "--format=%H", "--", relative).stdout.strip()
    if len(value) != 40:
        raise C84FRuntimeError(f"cannot resolve committed identity for {relative}")
    return value


def verify_repository_state(
    lock: Mapping[str, Any], *, lock_commit: str, repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    origin = _git(repo_root, "rev-parse", "origin/oaci").stdout.strip()
    branch = _git(repo_root, "branch", "--show-current").stdout.strip()
    dirty = _git(repo_root, "status", "--porcelain").stdout.strip()
    if head != origin or branch != "oaci" or dirty:
        raise C84FRuntimeError("C84F requires clean HEAD == origin/oaci on branch oaci")
    for ancestor_name, ancestor in (
        ("execution lock", lock_commit), ("implementation", str(lock["implementation"]["commit"])),
        ("protocol", str(lock["chronology"]["protocol_commit"])),
    ):
        if _git(repo_root, "merge-base", "--is-ancestor", ancestor, head, check=False).returncode:
            raise C84FRuntimeError(f"C84F {ancestor_name} is not an ancestor of HEAD")
    return {"HEAD": head, "origin_oaci": origin, "branch": branch, "execution_lock_commit": lock_commit}


def verify_authorization_record(
    lock: Mapping[str, Any], lock_sha256: str, lock_commit: str, authorization_path: Path,
) -> dict[str, Any]:
    if not authorization_path.is_file():
        raise C84FRuntimeError("fresh direct C84F PI authorization record is absent")
    record = read_json(authorization_path)
    required = {
        "schema_version": "c84f_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84F",
        "reconciliation_protocol_sha256": lock["protocols"]["reconciliation"]["sha256"],
        "field_protocol_v7_sha256": lock["protocols"]["field_v7"]["sha256"],
        "full_field_protocol_v2_sha256": lock["protocols"]["full_field_v2"]["sha256"],
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
        "operative_unit_registry_sha256": lock["candidate_identity"]["registry_sha256"],
        "C84S": False,
        "target_labels": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items() if record.get(key) != wanted}
    if mismatch:
        raise C84FRuntimeError(f"C84F authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": sha256_file(authorization_path)}


def verify_output_root(root: Path, lock_sha256: str) -> Path:
    run_root = Path(root) / f"lock_{lock_sha256[:20]}"
    if run_root.exists():
        raise C84FRuntimeError("C84F content-addressed output root already exists")
    run_root.mkdir(parents=True, exist_ok=False)
    return run_root


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
        raise C84FRuntimeError("C84F execution lock is absent")
    lock_sha = verify_lock_self(lock_path, lock_sha_path)
    lock = read_json(lock_path)
    if lock.get("status") != LOCK_READY_STATUS:
        raise C84FRuntimeError("C84F lock is not in the unique not-authorized readiness state")
    bound = verify_bound_object_registry(lock, repo_root=repo_root)
    protocols = verify_protocol_sidecars(lock, repo_root=repo_root)
    interface = verify_interface(lock)
    environment = verify_distribution_environment(
        lock, version_getter=version_getter, python_version=python_version, prefix=prefix,
    )
    deterministic = verify_deterministic_environment(environ)
    candidates = verify_candidate_and_wave_registry(lock, repo_root=repo_root)
    reuse = verify_dual_canary_reuse(lock, repo_root=repo_root)
    lock_commit = commit_for_path(lock_path, repo_root=repo_root)
    repository = verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = verify_output_root(output_root, lock_sha)
    return {
        "lock": lock, "lock_sha256": lock_sha, "lock_commit": lock_commit,
        "authorization": authorization, "authorization_sha256": authorization["record_sha256"],
        "bound_object_replay": bound, "protocol_replay": protocols,
        "interface_replay": interface, "environment_replay": environment,
        "deterministic_environment": deterministic, "candidate_replay": candidates,
        "dual_canary_replay": reuse, "repository_replay": repository,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    if path.exists():
        raise C84FRuntimeError("C84F authorization was already consumed")
    payload = {
        "schema_version": "c84f_authorization_consumption_v1",
        "stage": "C84F",
        "execution_lock_sha256": binding["lock_sha256"],
        "execution_lock_commit": binding["lock_commit"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "field_protocol_v7_sha256": binding["lock"]["protocols"]["field_v7"]["sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_protected_package_imports": True,
        "before_loader_import": True,
        "before_source_data_access": True,
        "before_target_data_access": True,
        "target_labels_authorized": False,
        "scientific_outcomes_authorized": False,
        "C84S_authorized": False,
    }
    digest = _atomic_json(path, payload)
    return {**payload, "path": str(path), "sha256": digest}


def verify_loader_source_files(
    lock: Mapping[str, Any],
    *,
    locate_distribution_file: Callable[[str], Path] | None = None,
) -> list[dict[str, Any]]:
    rows = lock.get("loader_source_identity", {}).get("files", ())
    if not rows:
        raise C84FRuntimeError("C84F lock has no loader source registry")
    if locate_distribution_file is None:
        distribution = metadata.distribution("moabb")
        locate_distribution_file = lambda relative: Path(distribution.locate_file(relative))
    replay = []
    for row in rows:
        path = locate_distribution_file(row["distribution_relative_path"]).resolve()
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise C84FRuntimeError(f"MOABB loader source drift: {row['qualified_object']}")
        replay.append({
            "qualified_object": row["qualified_object"], "path": str(path),
            "sha256": row["sha256"], "before_get_data": True,
        })
    return replay


def verify_loader_runtime_objects(lock: Mapping[str, Any], objects: Mapping[str, Any]) -> list[dict[str, Any]]:
    expected = {row["qualified_object"]: row for row in lock["loader_source_identity"]["files"]}
    if set(objects) != set(expected):
        raise C84FRuntimeError("runtime loader-object set differs from lock")
    replay = []
    for qualified, value in objects.items():
        source = inspect.getsourcefile(value)
        if source is None or sha256_file(source) != expected[qualified]["sha256"]:
            raise C84FRuntimeError(f"runtime loader source drift: {qualified}")
        replay.append({"qualified_object": qualified, "source": source, "sha256": expected[qualified]["sha256"]})
    return replay


def verify_protected_runtime_versions(lock: Mapping[str, Any], *, torch: Any, mne: Any, moabb: Any) -> dict[str, str]:
    expected = lock["environment"]["runtime_versions"]
    observed = {"torch": str(torch.__version__), "mne": str(mne.__version__), "moabb": str(moabb.__version__)}
    if observed != expected:
        raise C84FRuntimeError(f"protected runtime version drift: {observed} != {expected}")
    return observed


@dataclass
class ExecutionAttemptLedger:
    """Persist every post-consumption stage, counter, artifact, and failure."""

    run_root: Path
    consumption: Mapping[str, Any]
    counters: dict[str, int] = field(default_factory=lambda: {
        "package_imports": 0, "CUDA_checks": 0, "loader_source_replays": 0,
        "dataset_loader_imports": 0, "source_get_data_calls": 0,
        "target_get_data_calls": 0, "source_EEG_arrays": 0, "target_EEG_arrays": 0,
        "source_label_arrays_read": 0, "target_y_accesses": 0,
        "training_phases_started": 0, "training_phases_completed": 0,
        "new_training_units": 0, "reused_training_units": 0,
        "source_audit_artifacts": 0, "model_field_units": 0,
        "target_registry_trials": 0, "target_unlabeled_artifacts": 0,
        "target_context_slices": 0, "canary_contexts_replayed": 0,
        "target_scientific_metrics": 0,
    })
    current_stage: str = "authorization_consumed"

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)
        self.path = self.run_root / "execution_attempts.jsonl"
        self.partial_path = self.run_root / "partial_artifact_manifest.json"
        self._append({
            "event": "attempt_started", "stage": self.current_stage,
            "authorization_consumption_sha256": self.consumption["sha256"],
            "started_at_unix_ns": time.time_ns(), "counters": dict(self.counters),
        })
        self.publish_partial_manifest("IN_PROGRESS")

    def _append(self, value: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(value), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def stage(self, name: str) -> None:
        self.current_stage = str(name)
        self._append({
            "event": "stage", "stage": self.current_stage,
            "at_unix_ns": time.time_ns(), "counters": dict(self.counters),
        })

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self.counters:
            raise C84FRuntimeError(f"unregistered C84F attempt counter: {name}")
        self.counters[name] += int(amount)

    def _artifacts(self) -> list[dict[str, Any]]:
        excluded = {self.path.resolve(), self.partial_path.resolve()}
        rows = []
        for path in sorted(self.run_root.rglob("*")):
            if path.is_file() and path.resolve() not in excluded:
                rows.append({
                    "path": str(path.relative_to(self.run_root)), "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        return rows

    def publish_partial_manifest(self, status: str, error: BaseException | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": "c84f_partial_artifact_manifest_v1",
            "status": status, "stage": self.current_stage, "counters": dict(self.counters),
            "artifacts": self._artifacts(),
            "authorization_consumption_sha256": self.consumption["sha256"],
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
            "target_outcome_decision": False,
            "retry_disposition": "PM_REVIEW_OR_ADDITIVE_REPAIR" if error else "NOT_APPLICABLE",
            "published_at_unix_ns": time.time_ns(),
        }
        _atomic_json(self.partial_path, payload)
        return payload

    def fail(self, error: BaseException) -> None:
        partial = self.publish_partial_manifest("FAILED", error)
        self._append({
            "event": "failed", "stage": self.current_stage,
            "error_type": type(error).__name__, "error": str(error),
            "counters": dict(self.counters),
            "partial_manifest_sha256": sha256_file(self.partial_path),
            "retry_disposition": partial["retry_disposition"],
            "failed_at_unix_ns": time.time_ns(),
        })

    def complete(self, complete_manifest_sha256: str) -> None:
        self.publish_partial_manifest("COMPLETE")
        self._append({
            "event": "complete", "stage": self.current_stage,
            "complete_manifest_sha256": complete_manifest_sha256,
            "counters": dict(self.counters), "completed_at_unix_ns": time.time_ns(),
        })
