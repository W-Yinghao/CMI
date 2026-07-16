"""Shared constants and fail-closed helpers for C84SR1."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Mapping

from .c84s_common import canonical_sha256, read_json, require, write_json


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
PROTOCOL_PATH = REPORT_DIR / "C84SR1_REAL_EXECUTION_ORCHESTRATION_AND_Q0_INTEGRATION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C84SR1_REAL_EXECUTION_ORCHESTRATION_AND_Q0_INTEGRATION_PROTOCOL.sha256"
LOCK_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V3.json"
LOCK_SHA_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V3.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C84S_V3_PI_AUTHORIZATION_RECORD.json"

COMPLETE_FIELD_MANIFEST_PATH = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/"
    "lock_f0c369ee273352b47e36/C84F_COMPLETE_FIELD_MANIFEST.json"
)
TARGET_TRIAL_REGISTRY_PATH = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c84-full-field-target-replay-v2/"
    "lock_f0c369ee273352b47e36/C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json"
)

DATASET_TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
PANELS = ("A", "B")
SEEDS = (5, 6)
LEVELS = (0, 1)
CANDIDATES = 81
PRIMARY_ZERO_METHODS = ("U5", "U7", "U11", "U13", "U14", "U15")
SCORE_METHODS = ("S1",) + PRIMARY_ZERO_METHODS
FIXED_METHODS = ("B1", "B2", "B3", "B4O", "B4S")
PRIMARY_Q0_METHODS = ("Q0_B1", "Q0_B2", "Q0_B4", "Q0_B8", "Q0_FULL")
SECONDARY_Q0_METHODS = ("Q0_B16", "Q0_B32")
COMMON_METHODS = (
    "B0", "B1", "B2", "B3", "B4O", "B4S", "B5", "S1",
    *PRIMARY_ZERO_METHODS, *PRIMARY_Q0_METHODS,
)
Q0_CHAINS = 2048
Q0_BUDGET_CODES = {1: 1, 2: 2, 4: 4, 8: 8, 16: 16, 32: 32, "FULL": 255}
Q0_CODE_BUDGETS = {value: key for key, value in Q0_BUDGET_CODES.items()}
MAXT_DRAWS = 65536
SUCCESS_GATE = "C84S_REAL_EXECUTION_ORCHESTRATION_Q0_INTEGRATION_REPAIRED_AND_LOCKED_READY_FOR_FRESH_PI_AUTHORIZATION"
FAILURE_GATE = "C84S_REAL_EXECUTION_SELECTION_AGGREGATION_RESOURCE_OR_PROVENANCE_RECONCILIATION_REQUIRED"


def expected_methods(dataset: str) -> tuple[str, ...]:
    require(dataset in DATASET_TARGET_COUNTS, f"unknown C84 dataset: {dataset}")
    return COMMON_METHODS + (SECONDARY_Q0_METHODS if dataset != "PhysionetMI" else ())


def finite_budgets(dataset: str) -> tuple[int, ...]:
    require(dataset in DATASET_TARGET_COUNTS, f"unknown C84 dataset: {dataset}")
    return (1, 2, 4, 8, 16, 32) if dataset != "PhysionetMI" else (1, 2, 4, 8)


def context_identity(
    dataset: str, target_subject_id: str | int, panel: str, training_seed: int, level: int,
) -> dict[str, Any]:
    return {
        "dataset": str(dataset), "target_subject_id": str(target_subject_id),
        "panel": str(panel), "training_seed": int(training_seed), "level": int(level),
    }


def context_id(identity: Mapping[str, Any]) -> str:
    required = {"dataset", "target_subject_id", "panel", "training_seed", "level"}
    require(set(identity) == required, "context identity field drift")
    return canonical_sha256({key: identity[key] for key in (
        "dataset", "target_subject_id", "panel", "training_seed", "level",
    )})[:24]


def normalized_mapping(value: Any) -> dict[str, Any]:
    if is_dataclass(value):
        return asdict(value)
    require(isinstance(value, Mapping), "value is not mapping-like")
    return dict(value)


def reject_evaluation_tokens(value: Any, path: str = "root") -> None:
    forbidden = ("evaluation", "oracle", "held_utility", "target_accuracy", "regret")
    if isinstance(value, Mapping):
        for key, nested in value.items():
            lowered = str(key).lower()
            require(not any(token in lowered for token in forbidden),
                    f"evaluation/oracle token reached Stage B: {path}/{key}")
            reject_evaluation_tokens(nested, f"{path}/{key}")
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value):
            reject_evaluation_tokens(nested, f"{path}/{index}")


def write_stage_receipt(path: str | Path, payload: Mapping[str, Any]) -> str:
    body = dict(payload)
    require("schema_version" in body and "stage" in body, "stage receipt identity missing")
    body["receipt_identity_sha256"] = canonical_sha256(body)
    return write_json(path, body)


def replay_stage_receipt(path: str | Path, *, stage: str) -> dict[str, Any]:
    payload = read_json(path)
    observed = str(payload.pop("receipt_identity_sha256"))
    require(observed == canonical_sha256(payload), "stage receipt identity drift")
    payload["receipt_identity_sha256"] = observed
    require(payload.get("stage") == stage, f"stage receipt mismatch: {stage}")
    return payload
