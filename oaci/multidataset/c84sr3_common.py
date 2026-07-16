"""Shared identities and exact arithmetic for the C84SR3 V5 analysis repair."""
from __future__ import annotations

from pathlib import Path

from .c84s_common import require
from .c84sr1_common import (
    COMMON_METHODS, DATASET_TARGET_COUNTS, SECONDARY_Q0_METHODS,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84sr3_tables"
PROTOCOL_PATH = REPORT_DIR / "C84SR3_Q0_SECONDARY_BUDGET_AVAILABILITY_AND_ATOMIC_FAILURE_REPAIR_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C84SR3_Q0_SECONDARY_BUDGET_AVAILABILITY_AND_ATOMIC_FAILURE_REPAIR_PROTOCOL.sha256"
PROTOCOL_SHA256 = "5c783db9113697b2c710af4c1f1bafd66a3096be7a1b5cbac8aa03ca2a9c3080"
LOCK_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V5.json"
LOCK_SHA_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V5.sha256"
AUTHORIZATION_PATH = REPORT_DIR / "C84S_V5_PI_AUTHORIZATION_RECORD.json"

HISTORICAL_V4_LOCK_PATH = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V4.json"
HISTORICAL_V4_LOCK_SHA256 = "582e5074b4b17d62ff1e5fbfd992f037dd3082b7763b22d707630aa19db81c3d"
HISTORICAL_V4_AUTHORIZATION_PATH = REPORT_DIR / "C84S_V4_PI_AUTHORIZATION_RECORD.json"
HISTORICAL_V4_AUTHORIZATION_SHA256 = "4419303f8282ab132d2f95a5b76993bfb73191b29e7257e8220cefdde408ff5a"
HISTORICAL_V4_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v4")
HISTORICAL_V4_CONSUMPTION_SHA256 = "6dfc058e67ea8fa1ea8ddc0c1d398a4b468c4213a42455b5f864ced800fb0866"
HISTORICAL_V4_LIFECYCLE_SHA256 = "38bf0c660ebe53d9a02248fb297c59cac9686114f54f99272d233e76eda68477"

DEFAULT_OUTPUT_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5")
SYNTHETIC_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84sr3-production-synthetic-v1")
LOCK_READY_STATUS = "LOCKED_READY_FOR_FRESH_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
SUCCESS_GATE = "C84S_SECONDARY_Q0_AVAILABILITY_AND_ATOMIC_FAILURE_REPAIRED_V5_LOCK_READY_FOR_FRESH_PI_AUTHORIZATION"
FAILURE_GATE = "C84S_Q0_AVAILABILITY_ARITHMETIC_ATOMIC_PUBLICATION_OR_V5_LOCK_RECONCILIATION_REQUIRED"

Q0_RECORDS = 8_750_000
Q0_SAMPLE_DIGEST_ROWS = 1_093_750
Q0_STAGE_B_REGIME_ROWS = 15_648
Q0_STAGE_B_COVERAGE_ROWS = 5_216
METHOD_CONTEXT_ROWS = 18_432
Q0_STAGE_C_REGIME_ROWS = 12_816
Q0_STAGE_C_MC_ROWS = 4_272

PRIMARY_BUDGETS = (1, 2, 4, 8, "FULL")
HISTORICAL_SECONDARY_BUDGETS = {
    "Lee2019_MI": (16, 32),
    "Cho2017": (16, 32),
    "PhysionetMI": (),
}
OPERATIVE_FINITE_BUDGETS = {
    "Lee2019_MI": (1, 2, 4, 8, 16),
    "Cho2017": (1, 2, 4, 8, 16, 32),
    "PhysionetMI": (1, 2, 4, 8),
}
EXPECTED_CONSTRUCTION_CLASS_RANGE = {
    "Lee2019_MI": (25, 25),
    "Cho2017": (50, 50),
    "PhysionetMI": (9, 15),
}


def finite_budgets(dataset: str) -> tuple[int, ...]:
    require(dataset in OPERATIVE_FINITE_BUDGETS, f"unknown C84SR3 dataset: {dataset}")
    return OPERATIVE_FINITE_BUDGETS[dataset]


def expected_methods(dataset: str) -> tuple[str, ...]:
    require(dataset in DATASET_TARGET_COUNTS, f"unknown C84SR3 dataset: {dataset}")
    if dataset == "Lee2019_MI":
        return COMMON_METHODS + (SECONDARY_Q0_METHODS[0],)
    if dataset == "Cho2017":
        return COMMON_METHODS + SECONDARY_Q0_METHODS
    return COMMON_METHODS


def expected_method_context_rows() -> int:
    return sum(
        DATASET_TARGET_COUNTS[dataset] * 8 * len(expected_methods(dataset))
        for dataset in DATASET_TARGET_COUNTS
    )


require(expected_method_context_rows() == METHOD_CONTEXT_ROWS,
        "C84SR3 method-context arithmetic is internally inconsistent")
