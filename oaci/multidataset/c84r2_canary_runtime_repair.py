"""Fail-closed runtime binding and persisted replay helpers for C84C V3.

This module intentionally imports only the Python standard library. Package
metadata is read before authorization consumption without importing torch,
MOABB, MNE, NumPy, or any dataset loader.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
from importlib import metadata
import importlib
import inspect
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
import time
from typing import Any, Callable, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
REPAIR_PROTOCOL_PATH = REPORT_DIR / "C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.json"
REPAIR_PROTOCOL_SHA_PATH = REPORT_DIR / "C84R2_CANARY_RUNTIME_AND_REPLAY_REPAIR_PROTOCOL.sha256"
CANARY_PROTOCOL_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V3.json"
CANARY_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V3.sha256"
FIELD_PROTOCOL_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V3.json"
FIELD_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V3.sha256"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK_V2.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK_V2.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84C_PI_AUTHORIZATION_RECORD_V2.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v3")
LOCK_READY_STATUS = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
EXPECTED_MONTAGE_SHA256 = "988e8f89c3001a5144172a10f3a8b30eb50c28d485b900210b91ed1a0cf04f04"
EXPECTED_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)


class C84R2RuntimeError(RuntimeError):
    """Raised whenever the V3 canary cannot prove a locked runtime identity."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sidecar_digest(path: str | Path) -> str:
    path = Path(path)
    if not path.is_file():
        raise C84R2RuntimeError(f"missing SHA-256 sidecar: {path}")
    fields = path.read_text(encoding="ascii").split()
    if not fields or len(fields[0]) != 64:
        raise C84R2RuntimeError(f"malformed SHA-256 sidecar: {path}")
    return fields[0]


def _git(repo_root: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=repo_root, text=True, capture_output=True, check=check,
    )


def _safe_bound_path(repo_root: Path, relative: str) -> Path:
    candidate = (repo_root / relative).resolve()
    try:
        candidate.relative_to(repo_root.resolve())
    except ValueError as exc:
        raise C84R2RuntimeError(f"bound object escapes repository: {relative}") from exc
    return candidate


def verify_lock_self(lock_path: Path, lock_sha_path: Path) -> str:
    digest = _sidecar_digest(lock_sha_path)
    if sha256_file(lock_path) != digest:
        raise C84R2RuntimeError("C84C V2 execution-lock SHA-256 replay failed")
    return digest


def verify_bound_object_registry(
    lock: Mapping[str, Any],
    *,
    repo_root: Path = REPO_ROOT,
    current_blob_getter: Callable[[Path], str] | None = None,
    head_blob_getter: Callable[[str], str] | None = None,
) -> list[dict[str, Any]]:
    """Replay every lock-bound object by bytes and, when bound, Git blob."""
    objects = lock.get("runtime_bound_objects")
    if not isinstance(objects, list) or not objects:
        raise C84R2RuntimeError("C84C V2 lock has no runtime-bound object registry")

    if current_blob_getter is None:
        current_blob_getter = lambda path: _git(repo_root, "hash-object", str(path)).stdout.strip()
    if head_blob_getter is None:
        head_blob_getter = lambda relative: _git(repo_root, "rev-parse", f"HEAD:{relative}").stdout.strip()

    replay = []
    seen: set[str] = set()
    for row in objects:
        relative = str(row.get("path", ""))
        if not relative or relative in seen:
            raise C84R2RuntimeError(f"duplicate or empty runtime-bound path: {relative!r}")
        seen.add(relative)
        path = _safe_bound_path(repo_root, relative)
        if not path.is_file():
            raise C84R2RuntimeError(f"runtime-bound object is missing: {relative}")
        observed_sha = sha256_file(path)
        expected_sha = row.get("sha256")
        if observed_sha != expected_sha:
            raise C84R2RuntimeError(f"runtime-bound SHA-256 drift: {relative}")
        if "bytes" in row and path.stat().st_size != int(row["bytes"]):
            raise C84R2RuntimeError(f"runtime-bound byte-count drift: {relative}")
        expected_blob = row.get("blob")
        current_blob = None
        head_blob = None
        if expected_blob:
            current_blob = current_blob_getter(path)
            head_blob = head_blob_getter(relative)
            if current_blob != expected_blob or head_blob != expected_blob:
                raise C84R2RuntimeError(f"runtime-bound Git blob drift: {relative}")
        replay.append({
            "path": relative,
            "sha256": observed_sha,
            "blob": expected_blob or "SHA_ONLY",
            "current_blob": current_blob or "SHA_ONLY",
            "head_blob": head_blob or "SHA_ONLY",
            "replay_pass": True,
        })
    return replay


def verify_protocol_sidecars(lock: Mapping[str, Any], *, repo_root: Path = REPO_ROOT) -> list[dict[str, Any]]:
    replay = []
    bindings = lock.get("protocol_bindings")
    if not isinstance(bindings, list) or not bindings:
        raise C84R2RuntimeError("C84C V2 lock has no protocol bindings")
    for binding in bindings:
        path = _safe_bound_path(repo_root, binding["path"])
        sidecar = _safe_bound_path(repo_root, binding["sha256_path"])
        observed = sha256_file(path)
        sidecar_digest = _sidecar_digest(sidecar)
        if observed != binding["sha256"] or sidecar_digest != binding["sha256"]:
            raise C84R2RuntimeError(f"protocol binding replay failed: {binding['path']}")
        replay.append({"path": binding["path"], "sha256": observed, "replay_pass": True})
    return replay


def verify_montage_binding(lock: Mapping[str, Any]) -> dict[str, Any]:
    interface = lock.get("interface", {})
    channels = tuple(interface.get("channels", ()))
    observed_digest = sha256_bytes(canonical_bytes(list(channels)))
    if channels != EXPECTED_CHANNELS:
        raise C84R2RuntimeError("C84C V3 channel list/order differs from the exact 20-channel interface")
    if observed_digest != EXPECTED_MONTAGE_SHA256 or interface.get("montage_sha256") != observed_digest:
        raise C84R2RuntimeError("C84C V3 montage digest replay failed")
    forbidden = ("Fz_substitution", "FCz_interpolation", "zero_fill", "dataset_specific_mask")
    if any(bool(interface.get(key)) for key in forbidden):
        raise C84R2RuntimeError("C84C V3 interface enables a forbidden channel repair")
    return {"channels": list(channels), "channel_count": len(channels), "montage_sha256": observed_digest}


def verify_distribution_environment(
    lock: Mapping[str, Any],
    *,
    version_getter: Callable[[str], str] = metadata.version,
    python_version: str | None = None,
    prefix: str | None = None,
) -> dict[str, Any]:
    """Verify package metadata without importing protected distributions."""
    expected = lock.get("environment", {})
    observed_python = python_version or platform.python_version()
    observed_prefix = prefix or sys.prefix
    if observed_python != expected.get("python"):
        raise C84R2RuntimeError(
            f"locked Python version mismatch: observed={observed_python} expected={expected.get('python')}"
        )
    if os.path.realpath(observed_prefix) != os.path.realpath(str(expected.get("conda_prefix"))):
        raise C84R2RuntimeError("locked Conda prefix mismatch")
    observed_distributions = {}
    for distribution, wanted in expected.get("distributions", {}).items():
        try:
            observed = version_getter(distribution)
        except metadata.PackageNotFoundError as exc:
            raise C84R2RuntimeError(f"locked distribution is absent: {distribution}") from exc
        if observed != wanted:
            raise C84R2RuntimeError(
                f"locked distribution mismatch for {distribution}: observed={observed} expected={wanted}"
            )
        observed_distributions[distribution] = observed
    return {
        "python": observed_python,
        "conda_prefix": os.path.realpath(observed_prefix),
        "distributions": observed_distributions,
        "protected_package_imports": 0,
    }


def verify_deterministic_environment(
    *,
    environ: Mapping[str, str] | None = None,
) -> dict[str, str]:
    values = dict(os.environ if environ is None else environ)
    expected = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
    mismatch = {key: (values.get(key), wanted) for key, wanted in expected.items() if values.get(key) != wanted}
    if mismatch:
        raise C84R2RuntimeError(f"deterministic runtime environment mismatch: {mismatch}")
    return expected


def canary_unit_digest(unit_ids: Iterable[str]) -> str:
    values = sorted(str(value) for value in unit_ids)
    if len(values) != 243 or len(set(values)) != 243:
        raise C84R2RuntimeError("C84C V3 canary unit universe is not 243 unique IDs")
    return sha256_bytes(canonical_bytes(values))


def verify_candidate_identity(lock: Mapping[str, Any]) -> dict[str, Any]:
    """Import the already byte-verified pure protocol module and replay IDs."""
    module = importlib.import_module("oaci.multidataset.c84r_v2_protocols")
    units = [row for row in module.candidate_units() if row["canary_subset"]]
    digest = canary_unit_digest(row["unit_id"] for row in units)
    expected = lock.get("candidate_identity", {})
    if digest != expected.get("canary_unit_ids_sha256"):
        raise C84R2RuntimeError("C84C V3 canary unit-ID digest drift")
    if int(expected.get("canary_unit_count", -1)) != 243:
        raise C84R2RuntimeError("C84C V3 lock does not bind 243 canary units")
    return {"canary_unit_count": len(units), "canary_unit_ids_sha256": digest}


def commit_for_path(path: Path, *, repo_root: Path = REPO_ROOT) -> str:
    relative = str(path.resolve().relative_to(repo_root.resolve()))
    value = _git(repo_root, "log", "-1", "--format=%H", "--", relative).stdout.strip()
    if len(value) != 40:
        raise C84R2RuntimeError(f"cannot resolve committed identity for {relative}")
    return value


def verify_repository_state(
    lock: Mapping[str, Any],
    *,
    lock_commit: str,
    repo_root: Path = REPO_ROOT,
) -> dict[str, str]:
    head = _git(repo_root, "rev-parse", "HEAD").stdout.strip()
    origin = _git(repo_root, "rev-parse", "origin/oaci").stdout.strip()
    branch = _git(repo_root, "branch", "--show-current").stdout.strip()
    dirty = _git(repo_root, "status", "--porcelain").stdout.strip()
    if head != origin or branch != "oaci" or dirty:
        raise C84R2RuntimeError("C84C V3 requires clean HEAD == origin/oaci on branch oaci")
    if _git(repo_root, "merge-base", "--is-ancestor", lock_commit, head, check=False).returncode:
        raise C84R2RuntimeError("C84C V2 execution lock is not an ancestor of HEAD")
    return {"HEAD": head, "origin_oaci": origin, "branch": branch, "execution_lock_commit": lock_commit}


def verify_authorization_record(
    lock: Mapping[str, Any],
    lock_sha256: str,
    lock_commit: str,
    authorization_path: Path,
) -> dict[str, Any]:
    if not authorization_path.is_file():
        raise C84R2RuntimeError("fresh direct C84C PI authorization record is absent")
    record = read_json(authorization_path)
    required = {
        "schema_version": "c84c_direct_pi_authorization_record_v2",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84C",
        "canary_protocol_v3_sha256": lock["canary_protocol"]["sha256"],
        "execution_lock_v2_sha256": lock_sha256,
        "execution_lock_v2_commit": lock_commit,
        "repair_protocol_sha256": lock["repair_protocol"]["sha256"],
        "montage_sha256": lock["interface"]["montage_sha256"],
        "canary_unit_ids_sha256": lock["candidate_identity"]["canary_unit_ids_sha256"],
        "C84F": False,
        "C84S": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
    }
    mismatch = {key: (record.get(key), wanted) for key, wanted in required.items() if record.get(key) != wanted}
    if mismatch:
        raise C84R2RuntimeError(f"C84C V3 authorization binding mismatch: {mismatch}")
    return {**record, "record_sha256": sha256_file(authorization_path)}


def verify_output_root(root: Path, lock_sha256: str) -> Path:
    run_root = root / f"lock_{lock_sha256[:20]}"
    if run_root.exists():
        allowed = {"authorization_consumed.json", "execution_attempts.jsonl", "partial_artifact_manifest.json"}
        unexpected = [path.name for path in run_root.iterdir() if path.name not in allowed]
        if unexpected:
            raise C84R2RuntimeError(
                f"C84C V3 output root is not empty or authorization-only: {sorted(unexpected)}"
            )
    else:
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
    """Replay every pre-access binding before creating the external run root."""
    if not lock_path.is_file():
        raise C84R2RuntimeError("C84C V2 execution lock is absent")
    lock_sha = verify_lock_self(lock_path, lock_sha_path)
    lock = read_json(lock_path)
    if lock.get("status") != LOCK_READY_STATUS:
        raise C84R2RuntimeError("C84C V2 lock status is not the unique readiness state")
    bound_replay = verify_bound_object_registry(lock, repo_root=repo_root)
    protocol_replay = verify_protocol_sidecars(lock, repo_root=repo_root)
    montage = verify_montage_binding(lock)
    environment = verify_distribution_environment(
        lock, version_getter=version_getter, python_version=python_version, prefix=prefix,
    )
    deterministic = verify_deterministic_environment(environ=environ)
    candidate = verify_candidate_identity(lock)
    lock_commit = commit_for_path(lock_path, repo_root=repo_root)
    repository = verify_repository_state(lock, lock_commit=lock_commit, repo_root=repo_root)
    authorization = verify_authorization_record(lock, lock_sha, lock_commit, authorization_path)
    run_root = verify_output_root(Path(output_root), lock_sha)
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
        raise C84R2RuntimeError("C84C V3 authorization was already consumed")
    payload = {
        "schema_version": "c84c_authorization_consumption_v2",
        "stage": "C84C",
        "execution_lock_v2_sha256": binding["lock_sha256"],
        "execution_lock_v2_commit": binding["lock_commit"],
        "canary_protocol_v3_sha256": binding["lock"]["canary_protocol"]["sha256"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_package_imports": True,
        "before_CUDA_check": True,
        "before_dataset_loader_import": True,
        "before_download_or_get_data": True,
        "target_scientific_outcomes_authorized": False,
    }
    write_json_atomic(path, payload)
    return {**payload, "path": str(path), "sha256": sha256_file(path)}


def verify_loader_source_files(
    lock: Mapping[str, Any],
    *,
    locate_distribution_file: Callable[[str], Path] | None = None,
) -> list[dict[str, Any]]:
    """Verify loader source bytes after consumption and before loader import/data."""
    entries = lock.get("loader_source_identity", {}).get("files", [])
    if not entries:
        raise C84R2RuntimeError("C84C V3 lock has no loader source identities")
    if locate_distribution_file is None:
        distribution = metadata.distribution("moabb")
        locate_distribution_file = lambda relative: Path(distribution.locate_file(relative))
    replay = []
    for entry in entries:
        path = locate_distribution_file(entry["distribution_relative_path"]).resolve()
        if not path.is_file() or sha256_file(path) != entry["sha256"]:
            raise C84R2RuntimeError(f"MOABB loader source identity drift: {entry['qualified_object']}")
        replay.append({
            "qualified_object": entry["qualified_object"],
            "path": str(path),
            "sha256": entry["sha256"],
            "before_get_data": True,
        })
    return replay


def verify_loader_runtime_objects(
    lock: Mapping[str, Any],
    objects: Mapping[str, Any],
) -> list[dict[str, Any]]:
    expected = {row["qualified_object"]: row for row in lock["loader_source_identity"]["files"]}
    if set(objects) != set(expected):
        raise C84R2RuntimeError("runtime loader-object set differs from the lock")
    replay = []
    for qualified, value in objects.items():
        source = inspect.getsourcefile(value)
        if source is None or sha256_file(source) != expected[qualified]["sha256"]:
            raise C84R2RuntimeError(f"runtime loader class source drift: {qualified}")
        replay.append({"qualified_object": qualified, "source": source, "sha256": expected[qualified]["sha256"]})
    return replay


def verify_protected_runtime_versions(lock: Mapping[str, Any], *, torch: Any, mne: Any, moabb: Any) -> dict[str, str]:
    expected = lock["environment"]
    observed = {
        "torch_runtime": str(torch.__version__),
        "mne_runtime": str(mne.__version__),
        "moabb_runtime": str(moabb.__version__),
    }
    wanted = {
        "torch_runtime": expected["runtime_versions"]["torch"],
        "mne_runtime": expected["runtime_versions"]["mne"],
        "moabb_runtime": expected["runtime_versions"]["moabb"],
    }
    if observed != wanted:
        raise C84R2RuntimeError(f"protected runtime package identity drift: {observed} != {wanted}")
    return observed


@dataclass
class ExecutionAttemptLedger:
    """Persist every post-consumption stage and failure with access counters."""

    run_root: Path
    consumption: Mapping[str, Any]
    counters: dict[str, int] = field(default_factory=lambda: {
        "package_imports": 0,
        "CUDA_checks": 0,
        "loader_source_replays": 0,
        "dataset_loader_imports": 0,
        "get_data_calls_started": 0,
        "get_data_calls_completed": 0,
        "real_EEG_arrays_materialized": 0,
        "source_label_arrays_read": 0,
        "target_y_accesses": 0,
        "training_phases_started": 0,
        "training_phases_completed": 0,
        "source_audit_artifacts": 0,
        "target_unlabeled_artifacts": 0,
        "complete_units": 0,
        "target_scientific_metrics": 0,
    })
    current_stage: str = "authorization_consumed"

    def __post_init__(self) -> None:
        self.run_root = Path(self.run_root)
        self.path = self.run_root / "execution_attempts.jsonl"
        self.partial_path = self.run_root / "partial_artifact_manifest.json"
        self._append({
            "event": "attempt_started",
            "stage": self.current_stage,
            "authorization_consumption_sha256": self.consumption["sha256"],
            "started_at_unix_ns": time.time_ns(),
            "counters": dict(self.counters),
        })
        self.publish_partial_manifest("IN_PROGRESS")

    def _append(self, payload: Mapping[str, Any]) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def stage(self, stage: str) -> None:
        self.current_stage = str(stage)
        self._append({"event": "stage", "stage": self.current_stage, "at_unix_ns": time.time_ns(),
                      "counters": dict(self.counters)})

    def increment(self, name: str, amount: int = 1) -> None:
        if name not in self.counters:
            raise C84R2RuntimeError(f"unregistered C84C attempt counter: {name}")
        self.counters[name] += int(amount)

    def _artifact_rows(self) -> list[dict[str, Any]]:
        rows = []
        excluded = {self.path.resolve(), self.partial_path.resolve()}
        for path in sorted(self.run_root.rglob("*")):
            if path.is_file() and path.resolve() not in excluded:
                rows.append({
                    "path": str(path.relative_to(self.run_root)),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                })
        return rows

    def publish_partial_manifest(self, status: str, *, error: BaseException | None = None) -> dict[str, Any]:
        payload = {
            "schema_version": "c84c_partial_artifact_manifest_v2",
            "status": status,
            "stage": self.current_stage,
            "counters": dict(self.counters),
            "artifacts": self._artifact_rows(),
            "authorization_consumption_sha256": self.consumption["sha256"],
            "error_type": type(error).__name__ if error else None,
            "error": str(error) if error else None,
            "retry_disposition": "NEW_ADDITIVE_REPAIR_AND_LOCK_REQUIRED" if error else "NOT_APPLICABLE",
            "target_outcome_decision": False,
            "published_at_unix_ns": time.time_ns(),
        }
        write_json_atomic(self.partial_path, payload)
        return payload

    def fail(self, error: BaseException) -> None:
        partial = self.publish_partial_manifest("FAILED", error=error)
        self._append({
            "event": "failed",
            "stage": self.current_stage,
            "error_type": type(error).__name__,
            "error": str(error),
            "counters": dict(self.counters),
            "partial_manifest_sha256": sha256_file(self.partial_path),
            "retry_disposition": partial["retry_disposition"],
            "failed_at_unix_ns": time.time_ns(),
        })

    def complete(self, manifest_sha256: str) -> None:
        self.publish_partial_manifest("COMPLETE")
        self._append({
            "event": "completed",
            "stage": self.current_stage,
            "counters": dict(self.counters),
            "complete_manifest_sha256": manifest_sha256,
            "completed_at_unix_ns": time.time_ns(),
        })


def _npz_keys(archive: Any) -> set[str]:
    return set(getattr(archive, "files", tuple(archive.keys())))


def _softmax_numpy(logits: Any, np: Any) -> Any:
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=1, keepdims=True)


SOURCE_AUDIT_FIELDS = frozenset({
    "logits", "probabilities", "source_class_label", "source_domain_id", "source_trial_id",
    "dataset", "panel", "seed", "level", "unit_id",
})
TARGET_UNLABELED_FIELDS = frozenset({
    "logits", "probabilities", "z", "Wz_plus_b", "classifier_weight", "classifier_bias",
    "repeat_logits", "repeat_z", "target_trial_id", "dataset", "target_subject_id", "unit_id",
})


def replay_source_audit_artifact(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_trial_ids: Sequence[str],
    expected_labels: Sequence[int],
    expected_domains: Sequence[int],
    np: Any,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if _npz_keys(archive) != SOURCE_AUDIT_FIELDS:
            raise C84R2RuntimeError("strict-source audit artifact field-set drift")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        if logits.ndim != 2 or logits.shape[1] != 2 or probabilities.shape != logits.shape:
            raise C84R2RuntimeError("strict-source audit logits/probability shape drift")
        if np.max(np.abs(_softmax_numpy(logits, np) - probabilities)) > tolerance:
            raise C84R2RuntimeError("saved strict-source softmax replay failed")
        if tuple(archive["source_trial_id"].astype(str)) != tuple(map(str, expected_trial_ids)):
            raise C84R2RuntimeError("strict-source trial-ID replay failed")
        if tuple(map(int, archive["source_class_label"])) != tuple(map(int, expected_labels)):
            raise C84R2RuntimeError("strict-source label replay failed")
        if tuple(map(int, archive["source_domain_id"])) != tuple(map(int, expected_domains)):
            raise C84R2RuntimeError("strict-source domain-ID replay failed")
        for key in ("dataset", "panel", "unit_id"):
            if str(archive[key].item()) != str(expected_identity[key]):
                raise C84R2RuntimeError(f"strict-source identity drift: {key}")
        for key in ("seed", "level"):
            if int(archive[key].item()) != int(expected_identity[key]):
                raise C84R2RuntimeError(f"strict-source identity drift: {key}")
        rows = logits.shape[0]
    return {"path": str(path), "sha256": sha256_file(path), "rows": rows, "replay_pass": True}


def replay_target_unlabeled_artifact(
    path: Path,
    *,
    expected_identity: Mapping[str, Any],
    expected_trial_ids: Sequence[str],
    np: Any,
    tolerance: float = 1e-6,
) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as archive:
        if _npz_keys(archive) != TARGET_UNLABELED_FIELDS:
            raise C84R2RuntimeError("target-unlabeled artifact field-set drift or target-label field present")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        z = np.asarray(archive["z"])
        weight = np.asarray(archive["classifier_weight"])
        bias = np.asarray(archive["classifier_bias"])
        reconstructed = z @ weight.T + bias
        if logits.ndim != 2 or logits.shape[1] != 2 or probabilities.shape != logits.shape:
            raise C84R2RuntimeError("target-unlabeled logits/probability shape drift")
        if reconstructed.shape != logits.shape or np.max(np.abs(reconstructed - logits)) > tolerance:
            raise C84R2RuntimeError("saved z/classifier/logits replay failed")
        if np.max(np.abs(_softmax_numpy(logits, np) - probabilities)) > tolerance:
            raise C84R2RuntimeError("saved target-unlabeled softmax replay failed")
        if np.max(np.abs(np.asarray(archive["repeat_logits"]) - logits)) > tolerance:
            raise C84R2RuntimeError("saved repeat-logits replay failed")
        if np.max(np.abs(np.asarray(archive["repeat_z"]) - z)) > tolerance:
            raise C84R2RuntimeError("saved repeat-z replay failed")
        if tuple(archive["target_trial_id"].astype(str)) != tuple(map(str, expected_trial_ids)):
            raise C84R2RuntimeError("target-unlabeled trial-ID replay failed")
        for key in ("dataset", "unit_id"):
            if str(archive[key].item()) != str(expected_identity[key]):
                raise C84R2RuntimeError(f"target-unlabeled identity drift: {key}")
        if int(archive["target_subject_id"].item()) != int(expected_identity["target_subject_id"]):
            raise C84R2RuntimeError("target-unlabeled subject identity drift")
        rows = logits.shape[0]
    return {"path": str(path), "sha256": sha256_file(path), "rows": rows, "replay_pass": True}


def replay_checkpoint(
    path: Path,
    *,
    expected_file_sha256: str,
    expected_state_hash: str,
    torch: Any,
    state_hash_fn: Callable[[Mapping[str, Any]], str],
) -> dict[str, Any]:
    if sha256_file(path) != expected_file_sha256:
        raise C84R2RuntimeError("checkpoint file SHA-256 replay failed")
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise C84R2RuntimeError("checkpoint state is unloadable") from exc
    if not isinstance(state, dict) or not state or not all(isinstance(key, str) for key in state):
        raise C84R2RuntimeError("checkpoint state schema is invalid")
    observed_state_hash = state_hash_fn(state)
    if observed_state_hash != expected_state_hash:
        raise C84R2RuntimeError("checkpoint model-state hash replay failed")
    schema = [{"key": key, "shape": list(state[key].shape), "dtype": str(state[key].dtype)} for key in sorted(state)]
    return {"path": str(path), "file_sha256": expected_file_sha256,
            "state_hash": observed_state_hash, "state_schema": schema, "replay_pass": True}


def replay_optimizer_state(
    descriptor: Mapping[str, Any],
    *,
    phase: str,
    trajectory_order: int,
    torch: Any,
) -> dict[str, Any]:
    path = Path(descriptor["path"])
    if sha256_file(path) != descriptor["file_sha256"]:
        raise C84R2RuntimeError("optimizer-state file SHA-256 replay failed")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:
        raise C84R2RuntimeError("optimizer state is unloadable") from exc
    required = {"phase", "trajectory_order", "step_counts", "optimizers"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise C84R2RuntimeError("optimizer-state top-level schema drift")
    if payload["phase"] != phase or int(payload["trajectory_order"]) != int(trajectory_order):
        raise C84R2RuntimeError("optimizer-state phase/order identity drift")
    expected_labels = {"ERM": {"encoder"}, "OACI": {"critic", "encoder"}, "SRC": {"encoder"}}[phase]
    if set(payload["optimizers"]) != expected_labels or set(payload["step_counts"]) != expected_labels:
        raise C84R2RuntimeError("optimizer label-set drift")
    expected_encoder_steps = 200 if phase == "ERM" else int(trajectory_order) * 100
    if int(payload["step_counts"]["encoder"]) != expected_encoder_steps:
        raise C84R2RuntimeError("optimizer encoder-step count drift")
    for label, state in payload["optimizers"].items():
        if not isinstance(state, dict) or set(state) != {"state", "param_groups"}:
            raise C84R2RuntimeError(f"optimizer schema drift for {label}")
    return {"path": str(path), "file_sha256": descriptor["file_sha256"],
            "labels": sorted(expected_labels), "step_counts": payload["step_counts"], "replay_pass": True}


def replay_sidecar(
    path: Path,
    *,
    expected_fields: set[str],
    expected_identity: Mapping[str, Any],
) -> dict[str, Any]:
    payload = read_json(path)
    if set(payload) != expected_fields:
        raise C84R2RuntimeError("candidate sidecar canonical schema drift")
    for key, value in expected_identity.items():
        if payload.get(key) != value:
            raise C84R2RuntimeError(f"candidate sidecar identity drift: {key}")
    return {"path": str(path), "sha256": sha256_file(path), "replay_pass": True, "payload": payload}


def validate_complete_canary_gate(unit_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(unit_rows) != 243 or len({row.get("unit_id") for row in unit_rows}) != 243:
        raise C84R2RuntimeError("complete-canary gate requires 243 unique units")
    requirements = (
        "checkpoint_replay_pass", "optimizer_replay_pass", "sidecar_replay_pass",
        "source_audit_replay_pass", "target_unlabeled_replay_pass",
    )
    missing = [row.get("unit_id") for row in unit_rows if not all(bool(row.get(key)) for key in requirements)]
    if missing:
        raise C84R2RuntimeError(f"complete-canary persisted replay is incomplete for {len(missing)} units")
    return {
        "unit_count": 243,
        "checkpoint_state_sidecar_units": 243,
        "strict_source_audit_artifacts": 243,
        "target_unlabeled_artifacts": 243,
        "persisted_replay_units": 243,
        "complete": True,
    }
