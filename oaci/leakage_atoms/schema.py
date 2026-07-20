"""C39 - Leakage Atom Recovery / Support-Cell Conflict Audit."""
from __future__ import annotations

from ..selector_trace_recovery import schema as c37

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c37.LOCKED_C19_CONFIG_HASH

C34_TABLE_DIR = "oaci/reports/c34_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C36_TABLE_DIR = "oaci/reports/c36_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"
C38_TABLE_DIR = "oaci/reports/c38_tables"
C39_TABLE_DIR = "oaci/reports/c39_tables"

ROBUST_COMPARISON = c37.ROBUST_COMPARISON
ROBUST_CATEGORY = c37.ROBUST_CATEGORY
ACTUAL_SELECTOR_SCORE_NAME = c37.ACTUAL_SELECTOR_SCORE_NAME
UTILITY_GRID_STEP = c37.UTILITY_GRID_STEP

POINT_IDENTITY_TOL = 1e-9
ATOM_ADDITIVE_TOL = 1e-9
ATOM_DELTA_EPS = 1e-12
GAUGE_CLEAR_EPS = 1e-12
CONCENTRATED_TOP3_SHARE_GATE = 0.75
CONCENTRATED_HHI_GATE = 0.25
BROAD_TOP3_SHARE_GATE = 0.50
BROAD_MIN_POSITIVE_ATOMS = 8
BROAD_HHI_GATE = 0.15
CLASS_DOMAIN_SHARE_GATE = 0.50
LOW_MASS_QUANTILE = 0.25
SUPPORT_ARTIFACT_SHARE_GATE = 0.50
ATOM_SIGN_STABILITY_GATE = 0.75
INSTABILITY_RATE_GATE = 0.25
GAUGE_CONFLICT_RATE_GATE = 0.75

A1 = "A1_exact_atom_replay_supported"
A2 = "A2_point_advantage_cell_concentrated"
A3 = "A3_broad_point_leakage_advantage"
A4 = "A4_class_specific_leakage_conflict"
A5 = "A5_domain_or_group_specific_leakage_conflict"
A6 = "A6_support_edge_or_low_mass_driven"
A7 = "A7_selection_audit_atom_instability"
A8 = "A8_atom_target_gauge_conflict"
A9 = "A9_atom_decomposition_irrecoverable"
A10 = "A10_ucl_quantile_atom_limit"
ALL_CASES = (A1, A2, A3, A4, A5, A6, A7, A8, A9, A10)

FORBIDDEN_CLAIM_SUBSTRINGS = c37.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
    "oaci-v2",
    "selected checkpoint method artifact",
)


def frozen_config_hash() -> str:
    return c37.frozen_config_hash()
