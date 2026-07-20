"""Frozen statement and C85T review-input identities for C85V."""
from __future__ import annotations

from dataclasses import dataclass
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

from .c85_decision_experiments import DecisionContractError


THEOREM_IDS = tuple(f"T{index}" for index in range(1, 8))
PROTOCOL_SHA256 = "4b622ee1dd2dda6f681a3cf60b16eda0d873dbbe4f1ee996e565bf037423c586"
PROTOCOL_COMMIT = "436d6ff6a3710cd9a3c75cf2f22d0306a10f2d40"
C85T_BUNDLE_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c85t-synthetic-v3/"
    "c85t-v3-3ee51a994969ebaa-9ec012bedbf24f1f"
)
C85T_CONTROL_IDENTITIES = {
    "C85T_RESULT.json": "ecaff65e942dbb81d93a3bdb61589fa9f1f6590f7188947688e6b30617140cec",
    "C85T_RESULT_ARTIFACT_MANIFEST.json": "a727beebcb45598ea0f92f37bed8ef32369b1c793ecad9efc2f5d9941bd5bb0e",
    "C85T_V3_SEMANTIC_REPLAY_RECEIPT.json": "735edf13a24c074cb3c18e56d168ebd905b3a7bcb29e3c273b3652bb1b7dcc6e",
    "C85T_V3_COMPLETION_RECEIPT.json": "418f74e4c3cf60847b11bf18a890ffebf870ed8adee1a75d304b01075646e65d",
}
STATEMENT_HASHES = {
    "T1": "c6a5a5c30422b8e84d99642e3e61b6fc9b33fb452f1f66cc0a469e313419259d",
    "T2": "34d0a35de8f20c0e0ced7edd4f7e25a54d59883a86f0e323b9c4a34d34720ecc",
    "T3": "d3a1392bd79762ec6b11462af83345dec3b1c7b86134e0d1b6f1c45d902c7654",
    "T4": "ae175775fe57e8c709a8c49f25162ef29756f1d09b92ab892cc453b6d971ebdf",
    "T5": "413217283047286736bfa4e589702b32eddb5316a11d3dcaeae1fc8109f2c9b5",
    "T6": "69756ca0cf40f7dc5801dbe97d9a64aadd80ed15f8dfc71677a1e6d35792abee",
    "T7": "d90311e73d91749cffb6878970366dde75f291f0de12742fbe820853ee61e2df",
}
ALLOWED_CANDIDATE_DISPOSITIONS = {
    "PROPOSED_PROOF",
    "PROPOSED_COUNTEREXAMPLE",
    "INCOMPLETE_OPEN",
    "PROPOSED_INVALIDATION",
}


@dataclass(frozen=True)
class FrozenTheoremStatement:
    theorem_id: str
    text: str
    sha256: str
    formal_status: str = "OPEN"


@dataclass(frozen=True)
class FrozenProofCandidateIdentity:
    theorem_id: str
    relative_path: str
    sha256: str
    known_disposition: str


def canonical_json_bytes(value: Any) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_review_protocol(repo_root: Path) -> dict[str, Any]:
    path = repo_root / "oaci/reports/C85V_INDEPENDENT_PROOF_REVIEW_PROTOCOL.json"
    if not path.is_file() or sha256_file(path) != PROTOCOL_SHA256:
        raise DecisionContractError("C85V review protocol identity drifted")
    protocol = json.loads(path.read_text())
    if protocol.get("schema_version") != "c85v_independent_proof_review_protocol_v1":
        raise DecisionContractError("C85V review protocol schema drifted")
    if protocol.get("chronology", {}).get("proof_candidate_text_opened_for_review_before_protocol") is not False:
        raise DecisionContractError("C85V protocol-first chronology is false")
    return protocol


def load_registered_statements(repo_root: Path) -> dict[str, FrozenTheoremStatement]:
    source = (
        repo_root
        / "oaci/reports/C85T_PROOF_AND_SYNTHETIC_EXECUTION_OPERATIONALIZATION_PROTOCOL.json"
    )
    values = json.loads(source.read_text()).get("proof_statements")
    if not isinstance(values, dict) or set(values) != set(THEOREM_IDS):
        raise DecisionContractError("C85V theorem statement coverage drifted")
    result: dict[str, FrozenTheoremStatement] = {}
    for theorem_id in THEOREM_IDS:
        text = values.get(theorem_id)
        if not isinstance(text, str) or not text.strip():
            raise DecisionContractError(f"C85V statement is empty: {theorem_id}")
        digest = sha256_bytes(text.encode("utf-8"))
        if digest != STATEMENT_HASHES[theorem_id]:
            raise DecisionContractError(f"C85V statement hash drifted: {theorem_id}")
        result[theorem_id] = FrozenTheoremStatement(theorem_id, text, digest)
    return result


def load_candidate_identities(repo_root: Path) -> dict[str, FrozenProofCandidateIdentity]:
    path = repo_root / "oaci/reports/c85vp_tables/proof_candidate_identity_registry.csv"
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if {row.get("theorem_id") for row in rows} != set(THEOREM_IDS):
        raise DecisionContractError("C85V proof-candidate registry coverage drifted")
    result: dict[str, FrozenProofCandidateIdentity] = {}
    for row in rows:
        theorem_id = str(row["theorem_id"])
        disposition = str(row["known_disposition"])
        if disposition not in ALLOWED_CANDIDATE_DISPOSITIONS:
            raise DecisionContractError("C85V candidate disposition drifted")
        if row.get("text_opened_before_protocol") != "0" or row.get("formal_status") != "OPEN":
            raise DecisionContractError("C85V candidate chronology or status drifted")
        result[theorem_id] = FrozenProofCandidateIdentity(
            theorem_id=theorem_id,
            relative_path=str(row["relative_path"]),
            sha256=str(row["sha256"]),
            known_disposition=disposition,
        )
    return result


def validate_c85t_control_identity(bundle_root: Path = C85T_BUNDLE_ROOT) -> dict[str, Any]:
    root = bundle_root.resolve()
    for name, expected in C85T_CONTROL_IDENTITIES.items():
        path = root / name
        if not path.is_file() or sha256_file(path) != expected:
            raise DecisionContractError(f"C85T review control identity drifted: {name}")
    result = json.loads((root / "C85T_RESULT.json").read_text())
    expected_statuses = {theorem_id: "OPEN" for theorem_id in THEOREM_IDS}
    required = {
        "final_gate": "C85T_SYNTHETIC_VALIDATION_AND_PROOF_CANDIDATES_FROZEN_C85V_REVIEW_REQUIRED",
        "scenario_count": 11,
        "S6_S7_logical_replicate_rows": 8192,
        "S9_logical_replicate_design_rows": 8192,
        "S9_raw_draw_digest_rows": 4096,
        "proof_candidate_count": 7,
        "formal_theorem_statuses": expected_statuses,
        "real_project_data_access": 0,
        "active_acquisition": 0,
        "C85V_authorized": False,
        "C85E_authorized": False,
    }
    for key, expected in required.items():
        if result.get(key) != expected:
            raise DecisionContractError(f"C85T review result field drifted: {key}")
    return result


def validate_candidate_file(
    bundle_root: Path,
    identity: FrozenProofCandidateIdentity,
) -> Path:
    path = (bundle_root / identity.relative_path).resolve()
    root = bundle_root.resolve()
    if root not in path.parents or not path.is_file():
        raise DecisionContractError(f"C85V proof candidate is absent: {identity.theorem_id}")
    if sha256_file(path) != identity.sha256:
        raise DecisionContractError(f"C85V proof candidate hash drifted: {identity.theorem_id}")
    return path


def require_open_statuses(statuses: Mapping[str, str]) -> None:
    if dict(statuses) != {theorem_id: "OPEN" for theorem_id in THEOREM_IDS}:
        raise DecisionContractError("C85VP cannot transition a formal theorem status")
