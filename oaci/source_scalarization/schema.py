"""C43 - Source-Objective Scalarization Frontier / Escape-Hatch Closure Audit."""
from __future__ import annotations

from ..rank_actionability import schema as c42

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c42.LOCKED_C19_CONFIG_HASH

C30_TABLE_DIR = "oaci/reports/c30_tables"
C32_TABLE_DIR = "oaci/reports/c32_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"
C38_TABLE_DIR = "oaci/reports/c38_tables"
C41_TABLE_DIR = "oaci/reports/c41_tables"
C42_TABLE_DIR = "oaci/reports/c42_tables"
C43_TABLE_DIR = "oaci/reports/c43_tables"

SCALARIZATION_GRID_STEP = 0.10
TOPK_RULES = c42.TOPK_RULES
LABELS = c42.LABELS

PAIRWISE_SIGNAL_GATE = 0.55
RELIABLE_TOP1_JOINT_GATE = 0.70
RELIABLE_TOP1_ENRICHMENT_GATE = 1.50
WEAK_CEILING_TOP1_GATE = 0.65
BEST_GAIN_MIN = 0.05
TARGET_SIGN_CONSISTENCY_GATE = 0.80
LEAKAGE_BLOCKS_GATE = 0.50
TRADEOFF_NEGATIVE_CORR_GATE = -0.20
FRONT_CONTAINS_GOOD_GATE = 0.50
FRONT_REJECTS_GOOD_GATE = 0.50

CORE_SCALAR_OBJECTIVES = ("leakage", "source_rank", "source_risk", "audit_leakage")
PARETO_OBJECTIVES = (
    "selection_leakage_point",
    "audit_leakage_point",
    "R_src",
    "train_surrogate",
    "source_rank_score",
    "balanced_err",
    "source_guard_worst_nll",
    "source_guard_worst_ece",
    "source_audit_worst_nll",
    "source_audit_worst_ece",
)

F1 = "F1_source_objective_frontier_contains_target_good"
F2 = "F2_target_good_source_pareto_rejected"
F3 = "F3_leakage_extreme_blocks_rank_frontier"
F4 = "F4_no_source_scalarization_reliable_topk"
F5 = "F5_hindsight_scalarization_ceiling_weak"
F6 = "F6_source_scalarization_overfit_or_target_heterogeneous"
F7 = "F7_source_rank_leakage_tradeoff_real"
F8 = "F8_source_only_scalarization_escape_hatch_closed"
F9 = "F9_scalarization_escape_hatch_reopened_diagnostic_only"
F10 = "F10_inconclusive_due_to_field_availability"
ALL_CASES = (F1, F2, F3, F4, F5, F6, F7, F8, F9, F10)

FORBIDDEN_CLAIM_SUBSTRINGS = c42.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-only selector",
    "new selector",
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c42.frozen_config_hash()
