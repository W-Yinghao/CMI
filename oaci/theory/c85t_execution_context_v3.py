"""Receipt-validated execution context and lifecycle controls for C85T V3."""
from __future__ import annotations

import copy
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any, Final, Mapping
from uuid import UUID, uuid4

from .c85_decision_experiments import DecisionContractError


AUTHORIZATION_SCHEMA_V3: Final = "c85t_direct_pi_authorization_record_v3"
CONSUMPTION_SCHEMA_V3: Final = "c85t_authorization_consumption_receipt_v3"
LIFECYCLE_SCHEMA_V3: Final = "c85t_append_only_lifecycle_ledger_v3"
LOCK_SCHEMA_V3: Final = "c85t_execution_lock_v3"
LOCK_STATUS_V3: Final = "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
DIRECT_STATEMENT: Final = "\u6388\u6743 C85T"
PROTECTED_FALSE_FIELDS: Final = (
    "C85V",
    "C85E",
    "active_acquisition",
    "real_data",
    "new_data_or_model_zoo",
    "manuscript",
)
SUCCESS_EVENTS_V3: Final = (
    "PREFLIGHT_STARTED",
    "PREFLIGHT_COMPLETED",
    "AUTHORIZATION_CONSUMED",
    "EXACT_SCENARIOS_STARTED",
    "EXACT_SCENARIOS_COMPLETED",
    "MONTE_CARLO_STARTED",
    "MONTE_CARLO_COMPLETED",
    "PROOF_CANDIDATES_STARTED",
    "PROOF_CANDIDATES_COMPLETED",
    "MANIFEST_STARTED",
    "MANIFEST_COMPLETED",
    "ATOMIC_PUBLISH_COMMIT_READY",
)
TERMINAL_EVENTS_V3: Final = {"ATOMIC_PUBLISH_COMMIT_READY", "FAILED"}
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_AUTHORIZATION_PATH_MARKER: Final = (
    "DERIVED_FROM_NORMALIZED_C85T_V3_AUTHORIZATION_BINDING_SHA256"
)


def utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    )


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
        + "\n"
    ).encode("ascii")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _absolute(path: str | Path, label: str) -> Path:
    candidate = Path(path)
    if not candidate.is_absolute():
        raise DecisionContractError(f"{label} must be absolute")
    return candidate.resolve()


def _valid_authorization_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if _HEX_256.fullmatch(value):
        return True
    try:
        UUID(value)
    except (AttributeError, ValueError):
        return False
    return True


def authorization_binding_sha256(record: Mapping[str, Any]) -> str:
    normalized = dict(record)
    normalized["consumption_ledger_path"] = _AUTHORIZATION_PATH_MARKER
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def expected_output_root(
    output_parent: str | Path, lock_sha256: str, authorization_id: str
) -> Path:
    parent = _absolute(output_parent, "C85T V3 output parent")
    if not _HEX_256.fullmatch(lock_sha256) or not _valid_authorization_id(
        authorization_id
    ):
        raise DecisionContractError("invalid C85T V3 lock SHA or authorization ID")
    compact = authorization_id.replace("-", "")[:16].lower()
    return parent / f"c85t-v3-{lock_sha256[:16]}-{compact}"


def expected_consumption_path(
    consumption_root: str | Path, authorization_binding_sha: str
) -> Path:
    root = _absolute(consumption_root, "C85T V3 consumption root")
    if not _HEX_256.fullmatch(authorization_binding_sha):
        raise DecisionContractError("invalid C85T V3 authorization binding SHA")
    return root / f"{authorization_binding_sha}.json"


def _git(repo_root: Path, *args: str) -> str:
    return subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _git_is_ancestor(repo_root: Path, ancestor: str, descendant: str) -> bool:
    return (
        subprocess.run(
            ["git", "merge-base", "--is-ancestor", ancestor, descendant],
            cwd=repo_root,
            check=False,
            capture_output=True,
        ).returncode
        == 0
    )


def _sidecar_digest(sidecar: Path, expected_name: str) -> str:
    rows = [line.split(maxsplit=1) for line in sidecar.read_text().splitlines() if line]
    if len(rows) != 1 or len(rows[0]) != 2 or rows[0][1].strip() != expected_name:
        raise DecisionContractError("invalid C85T V3 SHA-256 sidecar")
    if not _HEX_256.fullmatch(rows[0][0]):
        raise DecisionContractError("invalid C85T V3 sidecar digest")
    return rows[0][0]


def _repo_root_for_lock(lock_path: Path) -> Path:
    resolved = lock_path.resolve()
    if resolved.parent.name != "reports" or resolved.parent.parent.name != "oaci":
        raise DecisionContractError("C85T V3 lock path is outside oaci/reports")
    root = resolved.parents[2]
    if not (root / ".git").exists():
        raise DecisionContractError("C85T V3 repository root is not a Git worktree")
    return root


def replay_execution_lock_v3(lock_path: Path) -> tuple[dict[str, Any], str, Path]:
    lock_path = lock_path.resolve()
    repo_root = _repo_root_for_lock(lock_path)
    sidecar = lock_path.with_suffix(".sha256")
    if not lock_path.is_file() or not sidecar.is_file():
        raise DecisionContractError("C85T V3 lock or sidecar is absent")
    lock_sha = sha256_file(lock_path)
    if _sidecar_digest(sidecar, lock_path.name) != lock_sha:
        raise DecisionContractError("C85T V3 lock self-hash drifted")
    lock = json.loads(lock_path.read_text())
    if lock.get("schema_version") != LOCK_SCHEMA_V3:
        raise DecisionContractError("C85T V3 lock schema drifted")
    if lock.get("status") != LOCK_STATUS_V3 or lock.get("authorized") is not False:
        raise DecisionContractError("C85T V3 lock is not an unauthorized readiness lock")
    protocol_identities = lock.get("protocol_identities")
    if not isinstance(protocol_identities, list) or not protocol_identities:
        raise DecisionContractError("C85T V3 protocol registry is absent")
    for row in protocol_identities:
        path = repo_root / row["path"]
        if not path.is_file() or sha256_file(path) != row["sha256"]:
            raise DecisionContractError(f"C85T V3 protocol drifted: {row['path']}")
    bound = lock.get("bound_repository_objects")
    if not isinstance(bound, list) or not bound:
        raise DecisionContractError("C85T V3 bound object registry is empty")
    observed: set[str] = set()
    for row in bound:
        relative = row["path"]
        if relative in observed:
            raise DecisionContractError("duplicate C85T V3 bound object")
        observed.add(relative)
        path = repo_root / relative
        if (
            not path.is_file()
            or path.stat().st_size != row["size_bytes"]
            or sha256_file(path) != row["sha256"]
            or _git(repo_root, "hash-object", "--", relative) != row["git_blob"]
        ):
            raise DecisionContractError(f"C85T V3 bound object drifted: {relative}")
    if len(bound) != lock.get("runtime_bound_object_count"):
        raise DecisionContractError("C85T V3 bound object count drifted")
    registry = lock.get("runtime_bound_registry")
    if not isinstance(registry, dict):
        raise DecisionContractError("C85T V3 runtime registry binding is absent")
    registry_path = repo_root / registry["path"]
    if (
        not registry_path.is_file()
        or registry_path.stat().st_size != registry["size_bytes"]
        or sha256_file(registry_path) != registry["sha256"]
        or _git(repo_root, "hash-object", "--", registry["path"])
        != registry["git_blob"]
    ):
        raise DecisionContractError("C85T V3 runtime registry drifted")
    return lock, lock_sha, repo_root


def replay_repository_state_v3(
    repo_root: Path, lock_path: Path, lock: Mapping[str, Any]
) -> dict[str, str]:
    if _git(repo_root, "branch", "--show-current") != "oaci":
        raise DecisionContractError("C85T V3 requires branch oaci")
    if _git(repo_root, "status", "--porcelain"):
        raise DecisionContractError("C85T V3 requires a clean worktree")
    head = _git(repo_root, "rev-parse", "HEAD")
    origin = _git(repo_root, "rev-parse", "origin/oaci")
    if head != origin:
        raise DecisionContractError("C85T V3 requires HEAD == origin/oaci")
    relative = str(lock_path.resolve().relative_to(repo_root))
    lock_commit = _git(repo_root, "log", "-1", "--format=%H", "--", relative)
    if not lock_commit or not _git_is_ancestor(repo_root, lock_commit, head):
        raise DecisionContractError("C85T V3 lock is not committed in current history")
    implementation_commit = str(lock.get("implementation_commit", ""))
    if not implementation_commit or not _git_is_ancestor(
        repo_root, implementation_commit, lock_commit
    ):
        raise DecisionContractError("C85T V3 implementation does not precede lock")
    return {
        "branch": "oaci",
        "HEAD": head,
        "origin_oaci": origin,
        "execution_lock_commit": lock_commit,
    }


def _validate_authorization_record(
    record: Mapping[str, Any],
    *,
    binding_sha: str,
    lock: Mapping[str, Any],
    lock_sha: str,
    lock_commit: str,
    cli_output_root: Path,
) -> dict[str, Any]:
    if authorization_binding_sha256(record) != binding_sha:
        raise DecisionContractError("C85T V3 authorization binding SHA drifted")
    required = {
        "schema_version": AUTHORIZATION_SCHEMA_V3,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": DIRECT_STATEMENT,
        "authorized_stage": "C85T",
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise DecisionContractError(f"C85T V3 authorization drifted: {key}")
    for key in PROTECTED_FALSE_FIELDS:
        if record.get(key) is not False:
            raise DecisionContractError(f"C85T V3 protected field is not false: {key}")
    authorization_id = record.get("authorization_id")
    if not _valid_authorization_id(authorization_id):
        raise DecisionContractError("C85T V3 authorization ID is invalid")
    policy = lock["output_root_policy"]
    expected_output = expected_output_root(policy["parent"], lock_sha, authorization_id)
    observed_output = _absolute(record.get("output_root", ""), "authorization output root")
    if observed_output != expected_output or cli_output_root.resolve() != expected_output:
        raise DecisionContractError("C85T V3 output root binding drifted")
    expected_ledger = expected_consumption_path(
        lock["authorization_consumption_root"], binding_sha
    )
    observed_ledger = _absolute(
        record.get("consumption_ledger_path", ""), "authorization consumption path"
    )
    if observed_ledger != expected_ledger:
        raise DecisionContractError("C85T V3 consumption path drifted")
    return dict(record)


def _write_exclusive_fsynced(path: Path, payload: bytes) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError as error:
        raise DecisionContractError("C85T V3 authorization was already consumed") from error
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            if written <= 0:
                raise DecisionContractError("short C85T V3 receipt write")
            view = view[written:]
        os.fsync(descriptor)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
        os.lseek(descriptor, 0, os.SEEK_SET)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


class AppendOnlyLifecycleLedgerV3:
    """Canonical V3 lifecycle that terminates before the publication rename."""

    def __init__(
        self,
        path: Path,
        *,
        authorization_binding_sha256: str,
        lock_sha256: str,
        attempt_id: str,
        output_root: Path,
    ) -> None:
        self.path = path.resolve()
        self.authorization_binding_sha256 = authorization_binding_sha256
        self.lock_sha256 = lock_sha256
        self.attempt_id = attempt_id
        self.output_root = str(output_root.resolve())
        self._sequence = 0
        self._last_completed = "NONE"
        self._terminal = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError as error:
            raise DecisionContractError("C85T V3 lifecycle already exists") from error
        os.close(descriptor)

    @property
    def terminal(self) -> bool:
        return self._terminal

    @property
    def last_completed_stage(self) -> str:
        return self._last_completed

    def append(
        self,
        stage: str,
        *,
        artifact_or_receipt_sha256: str | None = None,
        failure: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        if self._terminal:
            raise DecisionContractError("C85T V3 lifecycle is already terminal")
        if stage == "FAILED":
            if not failure:
                raise DecisionContractError("FAILED lifecycle requires failure details")
        elif self._sequence >= len(SUCCESS_EVENTS_V3) or stage != SUCCESS_EVENTS_V3[
            self._sequence
        ]:
            raise DecisionContractError("C85T V3 lifecycle stage order drifted")
        event: dict[str, Any] = {
            "schema_version": LIFECYCLE_SCHEMA_V3,
            "sequence_number": self._sequence,
            "timestamp_utc": utc_now(),
            "stage": stage,
            "authorization_binding_sha256": self.authorization_binding_sha256,
            "execution_lock_sha256": self.lock_sha256,
            "attempt_id": self.attempt_id,
            "output_root": self.output_root,
            "artifact_or_receipt_sha256": artifact_or_receipt_sha256,
        }
        if stage == "FAILED":
            event.update(
                {
                    "last_completed_stage": self._last_completed,
                    "primary_exception_type": str(failure["primary_exception_type"]),
                    "primary_exception_message": str(
                        failure["primary_exception_message"]
                    ),
                    "secondary_errors": list(failure.get("secondary_errors", [])),
                }
            )
        payload = canonical_json_bytes(event)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            view = memoryview(payload)
            while view:
                written = os.write(descriptor, view)
                if written <= 0:
                    raise DecisionContractError("short C85T V3 lifecycle write")
                view = view[written:]
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sequence += 1
        if stage.endswith("_COMPLETED") or stage == "AUTHORIZATION_CONSUMED":
            self._last_completed = stage
        if stage in TERMINAL_EVENTS_V3:
            self._terminal = True
        return event


def replay_lifecycle_v3(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise DecisionContractError("C85T V3 lifecycle is absent")
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not events:
        raise DecisionContractError("C85T V3 lifecycle is empty")
    binding: tuple[str, str, str, str] | None = None
    terminal = False
    for sequence, event in enumerate(events):
        if event.get("schema_version") != LIFECYCLE_SCHEMA_V3:
            raise DecisionContractError("C85T V3 lifecycle schema drifted")
        if event.get("sequence_number") != sequence:
            raise DecisionContractError("C85T V3 lifecycle sequence drifted")
        stage = event.get("stage")
        if stage == "FAILED":
            if sequence != len(events) - 1 or not event.get("primary_exception_type"):
                raise DecisionContractError("C85T V3 FAILED event is malformed")
        elif sequence >= len(SUCCESS_EVENTS_V3) or stage != SUCCESS_EVENTS_V3[sequence]:
            raise DecisionContractError("C85T V3 lifecycle order drifted")
        current = (
            event.get("authorization_binding_sha256"),
            event.get("execution_lock_sha256"),
            event.get("attempt_id"),
            event.get("output_root"),
        )
        if binding is None:
            binding = current
        elif current != binding:
            raise DecisionContractError("C85T V3 lifecycle binding drifted")
        if terminal:
            raise DecisionContractError("C85T V3 lifecycle has an event after terminal")
        terminal = stage in TERMINAL_EVENTS_V3
    return events


class ValidatedC85TExecutionContext:
    """Official-result certificate backed by a consumed external receipt."""

    __slots__ = (
        "_authorization_binding_sha256",
        "_authorization_file_sha256",
        "_authorization_id",
        "_execution_lock_sha256",
        "_execution_lock_commit",
        "_receipt_path",
        "_receipt_sha256",
        "_receipt_descriptor",
        "_receipt_device",
        "_receipt_inode",
        "_attempt_id",
        "_output_root",
        "_staging_bundle_root",
        "_lifecycle",
        "_head",
        "_repo_root",
        "_lock",
    )

    def __new__(cls, *args: Any, **kwargs: Any) -> "ValidatedC85TExecutionContext":
        raise DecisionContractError(
            "ValidatedC85TExecutionContext is created only by committed receipt replay"
        )

    def __copy__(self) -> "ValidatedC85TExecutionContext":
        raise DecisionContractError("C85T V3 execution context cannot be copied")

    def __deepcopy__(self, memo: dict[int, Any]) -> "ValidatedC85TExecutionContext":
        raise DecisionContractError("C85T V3 execution context cannot be deep-copied")

    def __reduce__(self) -> Any:
        raise DecisionContractError("C85T V3 execution context cannot be serialized")

    @property
    def authorization_binding_sha256(self) -> str:
        return self._authorization_binding_sha256

    @property
    def authorization_file_sha256(self) -> str:
        return self._authorization_file_sha256

    @property
    def authorization_id(self) -> str:
        return self._authorization_id

    @property
    def execution_lock_sha256(self) -> str:
        return self._execution_lock_sha256

    @property
    def execution_lock_commit(self) -> str:
        return self._execution_lock_commit

    @property
    def external_consumption_receipt_path(self) -> Path:
        return self._receipt_path

    @property
    def external_consumption_receipt_sha256(self) -> str:
        return self._receipt_sha256

    @property
    def attempt_id(self) -> str:
        return self._attempt_id

    @property
    def output_root(self) -> Path:
        return self._output_root

    @property
    def staging_bundle_root(self) -> Path:
        return self._staging_bundle_root

    @property
    def lifecycle_path(self) -> Path:
        return self._lifecycle.path

    @property
    def head(self) -> str:
        return self._head

    @property
    def lock(self) -> Mapping[str, Any]:
        return copy.deepcopy(self._lock)

    def close(self) -> None:
        descriptor = getattr(self, "_receipt_descriptor", -1)
        if descriptor >= 0:
            os.close(descriptor)
            self._receipt_descriptor = -1


def create_validated_c85t_execution_context(
    committed_lock_path: str | Path,
    committed_authorization_record_path: str | Path,
    exact_cli_output_root: str | Path,
) -> ValidatedC85TExecutionContext:
    """Replay, consume, and return the only operative C85T V3 context."""

    lock_path = Path(committed_lock_path).resolve()
    authorization_path = Path(committed_authorization_record_path).resolve()
    output_root = _absolute(exact_cli_output_root, "C85T V3 CLI output root")
    lock, lock_sha, repo_root = replay_execution_lock_v3(lock_path)
    expected_authorization_path = (repo_root / lock["authorization_record_path"]).resolve()
    if authorization_path != expected_authorization_path or not authorization_path.is_file():
        raise DecisionContractError("committed C85T V3 authorization path drifted")
    record = json.loads(authorization_path.read_text())
    binding_sha = authorization_binding_sha256(record)
    attempt_id = uuid4().hex
    staging = output_root.with_name(f".{output_root.name}.staging-{attempt_id}")
    if output_root.exists() or staging.exists():
        raise DecisionContractError("C85T V3 final and staging roots must be fresh")
    staging.mkdir(parents=True, exist_ok=False)
    lifecycle = AppendOnlyLifecycleLedgerV3(
        staging / "C85T_V3_LIFECYCLE.jsonl",
        authorization_binding_sha256=binding_sha,
        lock_sha256=lock_sha,
        attempt_id=attempt_id,
        output_root=output_root,
    )
    descriptor = -1
    try:
        lifecycle.append("PREFLIGHT_STARTED")
        repository = replay_repository_state_v3(repo_root, lock_path, lock)
        relative_authorization = str(authorization_path.relative_to(repo_root))
        try:
            _git(repo_root, "ls-files", "--error-unmatch", "--", relative_authorization)
        except subprocess.CalledProcessError as error:
            raise DecisionContractError(
                "C85T V3 authorization record is not committed"
            ) from error
        if _git(repo_root, "diff", "--name-only", "HEAD", "--", relative_authorization):
            raise DecisionContractError("C85T V3 authorization bytes are not at HEAD")
        authorization_commit = _git(
            repo_root, "log", "-1", "--format=%H", "--", relative_authorization
        )
        if not authorization_commit or not _git_is_ancestor(
            repo_root, repository["execution_lock_commit"], authorization_commit
        ) or not _git_is_ancestor(repo_root, authorization_commit, repository["HEAD"]):
            raise DecisionContractError("C85T V3 authorization chronology drifted")
        authorization = _validate_authorization_record(
            record,
            binding_sha=binding_sha,
            lock=lock,
            lock_sha=lock_sha,
            lock_commit=repository["execution_lock_commit"],
            cli_output_root=output_root,
        )
        environment: dict[str, Any]
        if lock.get("environment", {}).get("enforce_exact", True):
            from .c85t_rng import validate_environment

            environment = validate_environment(strict_prefix=True)
        else:
            if lock.get("execution_scope") != "SHADOW_READINESS_ONLY":
                raise DecisionContractError("only a shadow lock may relax environment replay")
            environment = {"shadow_environment_replay": True}
        preflight = {
            "schema_version": "c85t_v3_preflight_receipt_v1",
            "repository": repository,
            "authorization_commit": authorization_commit,
            "authorization_binding_sha256": binding_sha,
            "authorization_file_sha256": sha256_file(authorization_path),
            "execution_lock_sha256": lock_sha,
            "output_root": str(output_root),
            "environment": environment,
            "bound_repository_objects": lock["runtime_bound_object_count"],
        }
        preflight_path = staging / "preflight_completed.json"
        preflight_path.write_bytes(canonical_json_bytes(preflight))
        lifecycle.append(
            "PREFLIGHT_COMPLETED", artifact_or_receipt_sha256=sha256_file(preflight_path)
        )
        receipt = {
            "schema_version": CONSUMPTION_SCHEMA_V3,
            "authorization_binding_sha256": binding_sha,
            "authorization_file_sha256": sha256_file(authorization_path),
            "authorization_id": authorization["authorization_id"],
            "execution_lock_sha256": lock_sha,
            "execution_lock_commit": repository["execution_lock_commit"],
            "output_root": str(output_root),
            "attempt_id": attempt_id,
            "timestamp_utc": utc_now(),
            "HEAD": repository["HEAD"],
        }
        receipt_path = Path(authorization["consumption_ledger_path"]).resolve()
        descriptor = _write_exclusive_fsynced(
            receipt_path, canonical_json_bytes(receipt)
        )
        receipt_sha = sha256_file(receipt_path)
        copied_receipt = staging / "authorization_consumed.json"
        copied_receipt.write_bytes(canonical_json_bytes(receipt))
        if sha256_file(copied_receipt) != receipt_sha:
            raise DecisionContractError("copied C85T V3 receipt bytes drifted")
        lifecycle.append(
            "AUTHORIZATION_CONSUMED", artifact_or_receipt_sha256=receipt_sha
        )
        stat = os.fstat(descriptor)
        context = object.__new__(ValidatedC85TExecutionContext)
        context._authorization_binding_sha256 = binding_sha
        context._authorization_file_sha256 = sha256_file(authorization_path)
        context._authorization_id = authorization["authorization_id"]
        context._execution_lock_sha256 = lock_sha
        context._execution_lock_commit = repository["execution_lock_commit"]
        context._receipt_path = receipt_path
        context._receipt_sha256 = receipt_sha
        context._receipt_descriptor = descriptor
        context._receipt_device = stat.st_dev
        context._receipt_inode = stat.st_ino
        context._attempt_id = attempt_id
        context._output_root = output_root
        context._staging_bundle_root = staging
        context._lifecycle = lifecycle
        context._head = repository["HEAD"]
        context._repo_root = repo_root
        context._lock = copy.deepcopy(lock)
        validate_registered_execution_context(context)
        return context
    except BaseException as error:
        secondary: list[str] = []
        if descriptor >= 0:
            try:
                os.close(descriptor)
            except OSError as close_error:
                secondary.append(f"receipt-close: {close_error}")
        try:
            if not lifecycle.terminal:
                lifecycle.append(
                    "FAILED",
                    failure={
                        "primary_exception_type": type(error).__name__,
                        "primary_exception_message": str(error),
                        "secondary_errors": secondary,
                    },
                )
        except BaseException as ledger_error:
            secondary.append(f"lifecycle-failure: {ledger_error}")
        raise


def validate_registered_execution_context(
    context: object,
) -> ValidatedC85TExecutionContext:
    if type(context) is not ValidatedC85TExecutionContext:
        raise DecisionContractError("registered C85T V3 dispatch requires validated context")
    descriptor = getattr(context, "_receipt_descriptor", -1)
    if not isinstance(descriptor, int) or descriptor < 0:
        raise DecisionContractError("C85T V3 receipt descriptor is unavailable")
    receipt_path = context._receipt_path
    if not receipt_path.is_file() or sha256_file(receipt_path) != context._receipt_sha256:
        raise DecisionContractError("C85T V3 external receipt is absent or tampered")
    descriptor_stat = os.fstat(descriptor)
    path_stat = receipt_path.stat()
    if (
        descriptor_stat.st_dev,
        descriptor_stat.st_ino,
        path_stat.st_dev,
        path_stat.st_ino,
    ) != (
        context._receipt_device,
        context._receipt_inode,
        context._receipt_device,
        context._receipt_inode,
    ):
        raise DecisionContractError("C85T V3 receipt inode binding drifted")
    receipt = json.loads(receipt_path.read_text())
    expected = {
        "schema_version": CONSUMPTION_SCHEMA_V3,
        "authorization_binding_sha256": context._authorization_binding_sha256,
        "authorization_file_sha256": context._authorization_file_sha256,
        "authorization_id": context._authorization_id,
        "execution_lock_sha256": context._execution_lock_sha256,
        "execution_lock_commit": context._execution_lock_commit,
        "output_root": str(context._output_root),
        "attempt_id": context._attempt_id,
        "HEAD": context._head,
    }
    for key, value in expected.items():
        if receipt.get(key) != value:
            raise DecisionContractError(f"C85T V3 receipt binding drifted: {key}")
    events = replay_lifecycle_v3(context._lifecycle.path)
    consumed = [event for event in events if event["stage"] == "AUTHORIZATION_CONSUMED"]
    if len(consumed) != 1 or consumed[0]["artifact_or_receipt_sha256"] != context._receipt_sha256:
        raise DecisionContractError("C85T V3 lifecycle lacks matching authorization consumption")
    if any(event["stage"] == "FAILED" for event in events):
        raise DecisionContractError("C85T V3 attempt is failed")
    return context


__all__ = (
    "AUTHORIZATION_SCHEMA_V3",
    "CONSUMPTION_SCHEMA_V3",
    "LIFECYCLE_SCHEMA_V3",
    "LOCK_SCHEMA_V3",
    "LOCK_STATUS_V3",
    "SUCCESS_EVENTS_V3",
    "ValidatedC85TExecutionContext",
    "AppendOnlyLifecycleLedgerV3",
    "authorization_binding_sha256",
    "canonical_json_bytes",
    "create_validated_c85t_execution_context",
    "expected_consumption_path",
    "expected_output_root",
    "replay_execution_lock_v3",
    "replay_lifecycle_v3",
    "sha256_file",
    "validate_registered_execution_context",
)
