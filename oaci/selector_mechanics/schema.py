"""C36 - OACI Selector Mechanics / Feasibility-Regret Audit schema."""
from __future__ import annotations

from ..utility_cone import schema as c35

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c35.LOCKED_C19_CONFIG_HASH

C10_REPLAY_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c10-replay"
C34_TABLE_DIR = "oaci/reports/c34_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"

ROBUST_COMPARISON = "nearest_continuous_better"
ROBUST_CATEGORY = "preference_robust_regret"
UTILITY_GRID_STEP = c35.UTILITY_GRID_STEP

# OACI's real selector score is the selection bootstrap UCL. C10 replay has
# per-candidate point estimates, but not per-candidate UCLs. C36 may use point
# estimates as a leakage component, never as the actual selector score.
ACTUAL_SELECTOR_SCORE_NAME = "selection_bootstrap_ucl"
SELECTION_POINT_COMPONENT = "selection_leakage_point"

POINT_FLAT_EPS = 1e-12
POINT_PLATEAU_EPS = 0.02
SOURCE_PARETO_EPS = 1e-12
SOURCE_ENDPOINT_EPS = 1e-12

SOURCE_PARETO_OBJECTIVES = (
    {"objective": "R_src", "field": "R_src", "orientation": -1.0, "source": "source_risk"},
    {"objective": "selection_leakage_point", "field": "selection_leakage_point",
     "orientation": -1.0, "source": "source_train_point_component"},
    {"objective": "audit_leakage_point", "field": "audit_leakage_point",
     "orientation": -1.0, "source": "source_audit_point_component"},
    {"objective": "source_guard_worst_bacc", "field": "source_guard_worst_bacc",
     "orientation": 1.0, "source": "source_guard_endpoint"},
    {"objective": "source_guard_worst_nll", "field": "source_guard_worst_nll",
     "orientation": -1.0, "source": "source_guard_endpoint"},
    {"objective": "source_guard_worst_ece", "field": "source_guard_worst_ece",
     "orientation": -1.0, "source": "source_guard_endpoint"},
    {"objective": "source_audit_worst_bacc", "field": "source_audit_worst_bacc",
     "orientation": 1.0, "source": "source_audit_endpoint"},
    {"objective": "source_audit_worst_nll", "field": "source_audit_worst_nll",
     "orientation": -1.0, "source": "source_audit_endpoint"},
    {"objective": "source_audit_worst_ece", "field": "source_audit_worst_ece",
     "orientation": -1.0, "source": "source_audit_endpoint"},
)

S1 = "S1_risk_gate_excludes_target_better"
S2 = "S2_leakage_objective_prefers_selected"
S3 = "S3_source_endpoint_prefers_selected"
S4 = "S4_selection_audit_inversion"
S5 = "S5_source_pareto_conflict"
S6 = "S6_selector_plateau_tiebreak"
S7 = "S7_selector_active_misdirection"
S8 = "S8_better_candidate_source_dominates_selected"
S9 = "S9_trace_insufficient"
ALL_CASES = (S1, S2, S3, S4, S5, S6, S7, S8, S9)

FORBIDDEN_CLAIM_SUBSTRINGS = c35.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable selector",
    "target-free selector",
    "oaci-v2",
    "oaci rescue",
    "selected-checkpoint artifact",
    "target-unlabeled dg success",
    "method artifact",
)


def frozen_config_hash() -> str:
    return c35.frozen_config_hash()

