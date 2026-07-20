"""Single-use authorization, private capability, and lifecycle controls for C85T V2."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any, Final, Mapping
from uuid import UUID

from .c85_decision_experiments import DecisionContractError


AUTHORIZATION_SCHEMA: Final = "c85t_direct_pi_authorization_record_v2"
CONSUMPTION_SCHEMA: Final = "c85t_authorization_consumption_receipt_v2"
LIFECYCLE_SCHEMA: Final = "c85t_append_only_lifecycle_ledger_v2"
DIRECT_STATEMENT: Final = "授权 C85T"
PROTECTED_FALSE_FIELDS: Final = (
    "C85E",
    "active_acquisition",
    "real_data",
    "manuscript",
)
LIFECYCLE_EVENTS: Final = (
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
    "ATOMIC_PUBLISH_COMPLETED",
    "FAILED",
)
_CAPABILITY_SENTINEL: Final = object()
_ISSUED_CAPABILITIES: dict[int, tuple[str, str, str, str]] = {}
_HEX_256 = re.compile(r"^[0-9a-f]{64}$")
_CONSUMPTION_PATH_HASH_MARKER: Final = "DERIVED_FROM_NORMALIZED_AUTHORIZATION_SHA256"


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
        raise DecisionContractError(f"{label} must be an absolute path")
    return candidate.resolve()


def _valid_authorization_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    if _HEX_256.fullmatch(value):
        return True
    try:
        UUID(value)
    except (ValueError, AttributeError):
        return False
    return True


def expected_output_root(
    output_parent: str | Path, lock_sha256: str, authorization_id: str
) -> Path:
    parent = _absolute(output_parent, "C85T output parent")
    if not _HEX_256.fullmatch(lock_sha256) or not _valid_authorization_id(
        authorization_id
    ):
        raise DecisionContractError("invalid lock SHA or authorization ID")
    compact_id = authorization_id.replace("-", "")[:16].lower()
    return parent / f"c85t-v2-{lock_sha256[:16]}-{compact_id}"


def expected_consumption_path(
    consumption_root: str | Path, authorization_sha256: str
) -> Path:
    root = _absolute(consumption_root, "C85T consumption root")
    if not _HEX_256.fullmatch(authorization_sha256):
        raise DecisionContractError("invalid authorization SHA-256")
    return root / f"{authorization_sha256}.json"


def authorization_binding_sha256(record: Mapping[str, Any]) -> str:
    """Hash the record with its self-referential ledger path normalized."""

    normalized = dict(record)
    normalized["consumption_ledger_path"] = _CONSUMPTION_PATH_HASH_MARKER
    return hashlib.sha256(canonical_json_bytes(normalized)).hexdigest()


def validate_authorization_record(
    record: Mapping[str, Any],
    *,
    authorization_sha256: str,
    lock_sha256: str,
    lock_commit: str,
    output_parent: str | Path,
    consumption_root: str | Path,
) -> dict[str, Any]:
    if authorization_binding_sha256(record) != authorization_sha256:
        raise DecisionContractError("C85T V2 normalized authorization SHA drifted")
    required = {
        "schema_version": AUTHORIZATION_SCHEMA,
        "direct_explicit_PI_authorization": True,
        "direct_statement_exact": DIRECT_STATEMENT,
        "authorized_stage": "C85T",
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
    }
    for key, expected in required.items():
        if record.get(key) != expected:
            raise DecisionContractError(f"C85T V2 authorization drifted: {key}")
    for key in PROTECTED_FALSE_FIELDS:
        if record.get(key) is not False:
            raise DecisionContractError(f"C85T V2 protected field is not false: {key}")
    authorization_id = record.get("authorization_id")
    if not _valid_authorization_id(authorization_id):
        raise DecisionContractError("C85T V2 authorization ID is invalid")
    expected_output = expected_output_root(output_parent, lock_sha256, authorization_id)
    observed_output = _absolute(record.get("output_root", ""), "authorization output root")
    if observed_output != expected_output:
        raise DecisionContractError("C85T V2 authorization output root is not content-addressed")
    expected_ledger = expected_consumption_path(
        consumption_root, authorization_sha256
    )
    observed_ledger = _absolute(
        record.get("consumption_ledger_path", ""),
        "authorization consumption ledger",
    )
    if observed_ledger != expected_ledger:
        raise DecisionContractError("C85T V2 consumption ledger path drifted")
    return dict(record)


@dataclass(frozen=True, slots=True, init=False)
class _RegisteredExecutionCapability:
    authorization_sha256: str
    execution_lock_sha256: str
    attempt_id: str
    output_root: str

    def __init__(
        self,
        sentinel: object,
        *,
        authorization_sha256: str,
        execution_lock_sha256: str,
        attempt_id: str,
        output_root: str,
    ) -> None:
        if sentinel is not _CAPABILITY_SENTINEL:
            raise DecisionContractError("registered capability constructor is private")
        object.__setattr__(self, "authorization_sha256", authorization_sha256)
        object.__setattr__(self, "execution_lock_sha256", execution_lock_sha256)
        object.__setattr__(self, "attempt_id", attempt_id)
        object.__setattr__(self, "output_root", output_root)


def _issue_capability(
    *,
    authorization_sha256: str,
    execution_lock_sha256: str,
    attempt_id: str,
    output_root: Path,
) -> _RegisteredExecutionCapability:
    capability = _RegisteredExecutionCapability(
        _CAPABILITY_SENTINEL,
        authorization_sha256=authorization_sha256,
        execution_lock_sha256=execution_lock_sha256,
        attempt_id=attempt_id,
        output_root=str(output_root.resolve()),
    )
    _ISSUED_CAPABILITIES[id(capability)] = (
        authorization_sha256,
        execution_lock_sha256,
        attempt_id,
        str(output_root.resolve()),
    )
    return capability


def require_registered_capability(
    capability: object,
    *,
    authorization_sha256: str | None = None,
    execution_lock_sha256: str | None = None,
    attempt_id: str | None = None,
    output_root: str | Path | None = None,
) -> None:
    if not isinstance(capability, _RegisteredExecutionCapability):
        raise DecisionContractError(
            "registered C85T execution requires a private capability from consumed C85T authorization"
        )
    identity = _ISSUED_CAPABILITIES.get(id(capability))
    observed = (
        capability.authorization_sha256,
        capability.execution_lock_sha256,
        capability.attempt_id,
        capability.output_root,
    )
    if identity != observed:
        raise DecisionContractError("C85T capability identity is not process-issued")
    expected = (
        authorization_sha256,
        execution_lock_sha256,
        attempt_id,
        None if output_root is None else str(_absolute(output_root, "output root")),
    )
    for actual, wanted in zip(observed, expected):
        if wanted is not None and actual != wanted:
            raise DecisionContractError("C85T capability binding drifted")


def consume_authorization_once(
    *,
    record: Mapping[str, Any],
    authorization_sha256: str,
    lock_sha256: str,
    lock_commit: str,
    output_root: Path,
    attempt_id: str,
    head: str,
) -> tuple[dict[str, Any], object]:
    ledger_path = _absolute(
        record["consumption_ledger_path"], "authorization consumption ledger"
    )
    if _absolute(record["output_root"], "authorization output root") != output_root.resolve():
        raise DecisionContractError("authorization is bound to a different output root")
    receipt = {
        "schema_version": CONSUMPTION_SCHEMA,
        "authorization_sha256": authorization_sha256,
        "authorization_id": record["authorization_id"],
        "execution_lock_sha256": lock_sha256,
        "execution_lock_commit": lock_commit,
        "output_root": str(output_root.resolve()),
        "attempt_id": attempt_id,
        "timestamp_utc": utc_now(),
        "HEAD": head,
    }
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        descriptor = os.open(
            ledger_path,
            os.O_WRONLY | os.O_CREAT | os.O_EXCL,
            0o600,
        )
    except FileExistsError as error:
        raise DecisionContractError("C85T authorization was already consumed") from error
    try:
        payload = canonical_json_bytes(receipt)
        written = os.write(descriptor, payload)
        if written != len(payload):
            raise DecisionContractError("short authorization-consumption receipt write")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    capability = _issue_capability(
        authorization_sha256=authorization_sha256,
        execution_lock_sha256=lock_sha256,
        attempt_id=attempt_id,
        output_root=output_root,
    )
    return receipt, capability


class AppendOnlyLifecycleLedger:
    """Canonical JSONL ledger with strictly increasing event sequences."""

    def __init__(
        self,
        path: Path,
        *,
        authorization_sha256: str,
        lock_sha256: str,
        attempt_id: str,
    ) -> None:
        self.path = path.resolve()
        self.authorization_sha256 = authorization_sha256
        self.lock_sha256 = lock_sha256
        self.attempt_id = attempt_id
        self._sequence = 0
        self._last_completed = "NONE"
        self._terminal = False
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as error:
            raise DecisionContractError("C85T lifecycle ledger already exists") from error
        os.close(descriptor)

    @property
    def last_completed_stage(self) -> str:
        return self._last_completed

    def append(
        self,
        stage: str,
        *,
        artifact_or_receipt_sha256: str | None = None,
        failure: Mapping[str, str] | None = None,
    ) -> dict[str, Any]:
        if stage not in LIFECYCLE_EVENTS:
            raise DecisionContractError("unknown C85T lifecycle stage")
        if self._terminal:
            raise DecisionContractError("C85T lifecycle ledger is already terminal")
        if stage == "FAILED" and not failure:
            raise DecisionContractError("FAILED lifecycle event requires failure details")
        if stage != "FAILED" and stage != LIFECYCLE_EVENTS[self._sequence]:
            raise DecisionContractError("C85T lifecycle stage order drifted")
        event: dict[str, Any] = {
            "schema_version": LIFECYCLE_SCHEMA,
            "sequence_number": self._sequence,
            "timestamp_utc": utc_now(),
            "stage": stage,
            "authorization_sha256": self.authorization_sha256,
            "execution_lock_sha256": self.lock_sha256,
            "attempt_id": self.attempt_id,
            "artifact_or_receipt_sha256": artifact_or_receipt_sha256,
        }
        if stage == "FAILED":
            event.update(
                {
                    "last_completed_stage": self._last_completed,
                    "primary_exception_type": failure["primary_exception_type"],
                    "primary_exception_message": failure[
                        "primary_exception_message"
                    ],
                }
            )
        payload = canonical_json_bytes(event)
        descriptor = os.open(self.path, os.O_WRONLY | os.O_APPEND)
        try:
            written = os.write(descriptor, payload)
            if written != len(payload):
                raise DecisionContractError("short lifecycle event write")
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._sequence += 1
        if stage in {"FAILED", "ATOMIC_PUBLISH_COMPLETED"}:
            self._terminal = True
        if stage.endswith("_COMPLETED") or stage == "AUTHORIZATION_CONSUMED":
            self._last_completed = stage
        return event


def replay_lifecycle(path: Path) -> list[dict[str, Any]]:
    events = [json.loads(line) for line in path.read_text().splitlines() if line]
    if not events:
        raise DecisionContractError("C85T lifecycle ledger is empty")
    binding: tuple[str, str, str] | None = None
    for sequence, event in enumerate(events):
        if event.get("schema_version") != LIFECYCLE_SCHEMA:
            raise DecisionContractError("C85T lifecycle schema drifted")
        if event.get("sequence_number") != sequence:
            raise DecisionContractError("C85T lifecycle sequence drifted")
        if event.get("stage") not in LIFECYCLE_EVENTS:
            raise DecisionContractError("C85T lifecycle stage drifted")
        stage = event["stage"]
        if stage != "FAILED" and stage != LIFECYCLE_EVENTS[sequence]:
            raise DecisionContractError("C85T lifecycle event order drifted")
        if stage == "FAILED" and sequence != len(events) - 1:
            raise DecisionContractError("C85T FAILED event must be terminal")
        current = (
            event.get("authorization_sha256"),
            event.get("execution_lock_sha256"),
            event.get("attempt_id"),
        )
        if binding is None:
            binding = current
        elif current != binding:
            raise DecisionContractError("C85T lifecycle binding drifted")
    return events
