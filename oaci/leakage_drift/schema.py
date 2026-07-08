"""C40 - Leakage Point Drift Forensics / Atom-Trace Boundary Closure."""
from __future__ import annotations

from ..leakage_atoms import schema as c39

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c39.LOCKED_C19_CONFIG_HASH

C37_TABLE_DIR = "oaci/reports/c37_tables"
C38_TABLE_DIR = "oaci/reports/c38_tables"
C39_TABLE_DIR = "oaci/reports/c39_tables"
C40_TABLE_DIR = "oaci/reports/c40_tables"

ACTUAL_SELECTOR_SCORE_NAME = c39.ACTUAL_SELECTOR_SCORE_NAME
UTILITY_GRID_STEP = c39.UTILITY_GRID_STEP

POINT_IDENTITY_TOL = c39.POINT_IDENTITY_TOL
ATOM_ADDITIVE_TOL = c39.ATOM_ADDITIVE_TOL
TOLERANCE_LADDER = (1e-9, 1e-8, 1e-6, 1e-4, 1e-3)
BOUNDED_DRIFT_TOL = 1e-3
POINT_SIGN_EPS = 1e-12
STABILITY_FRACTION_GATE = 0.95

D1 = "D1_exact_atom_identity_recovered"
D2 = "D2_numeric_only_drift_bounded"
D3 = "D3_selection_split_semantic_mismatch"
D4 = "D4_aggregate_vs_atom_path_divergence"
D5 = "D5_diagnostic_atom_pattern_stable_but_blocked"
D6 = "D6_atom_trace_irrecoverable_final"
D7 = "D7_future_instrumentation_required"
D8 = "D8_c39_atom_claims_reopenable"
ALL_CASES = (D1, D2, D3, D4, D5, D6, D7, D8)

FORBIDDEN_CLAIM_SUBSTRINGS = c39.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "atom-level leakage mechanism established",
    "small drift therefore acceptable",
)


def frozen_config_hash() -> str:
    return c39.frozen_config_hash()
