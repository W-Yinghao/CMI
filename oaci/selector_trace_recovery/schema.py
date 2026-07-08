"""C37 - Exact Local Selector Trace Completion / Leakage-UCL Recovery."""
from __future__ import annotations

from ..selector_mechanics import schema as c36

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c36.LOCKED_C19_CONFIG_HASH

C10_REPLAY_DIR = c36.C10_REPLAY_DIR
C35_TABLE_DIR = c36.C35_TABLE_DIR
C34_TABLE_DIR = c36.C34_TABLE_DIR
C36_TABLE_DIR = "oaci/reports/c36_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"

ROBUST_COMPARISON = c36.ROBUST_COMPARISON
ROBUST_CATEGORY = c36.ROBUST_CATEGORY
UTILITY_GRID_STEP = c36.UTILITY_GRID_STEP
ACTUAL_SELECTOR_SCORE_NAME = c36.ACTUAL_SELECTOR_SCORE_NAME

UCL_IDENTITY_TOL = 1e-9
POINT_IDENTITY_TOL = 1e-9
UCL_CLEAR_EPS = 1e-9
UCL_PLATEAU_EPS = 0.02
P0_SLICE_SIZE = 3
DEFAULT_PARALLEL_N_JOBS = 8

T1 = "T1_exact_ucl_prefers_selected"
T2 = "T2_exact_ucl_prefers_better"
T3 = "T3_point_ucl_disagreement"
T4 = "T4_ucl_plateau_uncertainty"
T5 = "T5_selection_audit_inversion_confirmed"
T6 = "T6_source_pareto_conflict_survives_exact_trace"
T7 = "T7_selector_trace_irrecoverable"
T8 = "T8_actual_selector_misdirection_supported"
T9 = "T9_selector_not_misdirected_trace_incomplete_elsewhere"
ALL_CASES = (T1, T2, T3, T4, T5, T6, T7, T8, T9)

FORBIDDEN_CLAIM_SUBSTRINGS = c36.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "target-free detector",
    "deployable selector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
    "ucl proxy",
)


def frozen_config_hash() -> str:
    return c36.frozen_config_hash()

