"""C41 - Global Leakage-Target Utility Objective Field Audit."""
from __future__ import annotations

from ..leakage_drift import schema as c40

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c40.LOCKED_C19_CONFIG_HASH

C30_TABLE_DIR = "oaci/reports/c30_tables"
C34_TABLE_DIR = "oaci/reports/c34_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C36_TABLE_DIR = "oaci/reports/c36_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"
C38_TABLE_DIR = "oaci/reports/c38_tables"
C40_TABLE_DIR = "oaci/reports/c40_tables"
C41_TABLE_DIR = "oaci/reports/c41_tables"

ACTUAL_SELECTOR_SCORE_NAME = c40.ACTUAL_SELECTOR_SCORE_NAME
UTILITY_GRID_STEP = c40.UTILITY_GRID_STEP

TARGET_UTILITY_FIELD = "continuous_joint_min_margin"
LOWER_IS_BETTER_FIELDS = {
    "selection_leakage_point",
    "audit_leakage_point",
    "R_src",
    "balanced_err",
    "source_guard_worst_nll",
    "source_guard_worst_ece",
    "source_audit_worst_nll",
    "source_audit_worst_ece",
    "endpoint_vector_norm_regret",
    "pareto_distance",
    "dominated_hypervolume_regret",
}
HIGHER_IS_BETTER_FIELDS = {
    "source_guard_worst_bacc",
    "source_audit_worst_bacc",
    "continuous_joint_min_margin",
    "target_bacc_delta",
    "target_nll_delta",
    "target_ece_delta",
}

ALIGNMENT_AUC_HIGH = 0.55
ALIGNMENT_AUC_LOW = 0.45
LOW_LEAKAGE_ENRICHMENT_GATE = 1.05
SOURCE_RANK_BETTER_MARGIN = 0.05
LOCAL_REPRESENTATIVE_GATE = 0.8

O1 = "O1_global_leakage_target_alignment"
O2 = "O2_global_leakage_target_decoupling"
O3 = "O3_global_leakage_target_anti_alignment"
O4 = "O4_low_leakage_not_good_enriched"
O5 = "O5_source_audit_leakage_no_better"
O6 = "O6_source_rank_better_than_leakage"
O7 = "O7_target_unlabeled_pooled_field_better"
O8 = "O8_local_misdirection_reflects_global_field_conflict"
O9 = "O9_local_misdirection_tail_only"
O10 = "O10_inconclusive_due_to_field_availability"
ALL_CASES = (O1, O2, O3, O4, O5, O6, O7, O8, O9, O10)

FORBIDDEN_CLAIM_SUBSTRINGS = c40.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "atom-level leakage mechanism established",
    "deployable selector",
    "target-free detector",
    "oaci rescue",
)


def frozen_config_hash() -> str:
    return c40.frozen_config_hash()
