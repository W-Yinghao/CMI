"""C38 - Leakage-UCL Objective Geometry / Source-Target Conflict Audit."""
from __future__ import annotations

from ..selector_mechanics import schema as c36

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c36.LOCKED_C19_CONFIG_HASH

C34_TABLE_DIR = "oaci/reports/c34_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C36_TABLE_DIR = "oaci/reports/c36_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"
C27_TABLE_DIR = "oaci/reports/c27_tables"
C29_TABLE_DIR = "oaci/reports/c29_tables"

ROBUST_COMPARISON = c36.ROBUST_COMPARISON
ROBUST_CATEGORY = c36.ROBUST_CATEGORY
ACTUAL_SELECTOR_SCORE_NAME = c36.ACTUAL_SELECTOR_SCORE_NAME
UTILITY_GRID_STEP = c36.UTILITY_GRID_STEP

UCL_CLEAR_EPS = 1e-9
POINT_CLEAR_EPS = 1e-12
WIDTH_CLEAR_EPS = 1e-12
GAUGE_CLEAR_EPS = 1e-12
UCL_PLATEAU_EPS = 0.02
POINT_DOMINANT_FRACTION_GATE = 0.75
INVERSION_RATE_GATE = 0.25
GAUGE_CONFLICT_RATE_GATE = 0.75

L1 = "L1_point_leakage_drives_ucl_preference"
L2 = "L2_uncertainty_width_drives_ucl_preference"
L3 = "L3_cell_concentrated_leakage_advantage"
L4 = "L4_broad_leakage_advantage"
L5 = "L5_selection_audit_leakage_inversion"
L6 = "L6_source_rational_target_wrong"
L7 = "L7_leakage_target_gauge_conflict"
L8 = "L8_leakage_endpoint_decoupled"
L9 = "L9_support_or_estimability_artifact"
L10 = "L10_trace_atom_insufficient"
ALL_CASES = (L1, L2, L3, L4, L5, L6, L7, L8, L9, L10)

FORBIDDEN_CLAIM_SUBSTRINGS = c36.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
    "oaci-v2",
)


def frozen_config_hash() -> str:
    return c36.frozen_config_hash()

