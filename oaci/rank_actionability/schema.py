"""C42 - Source-Rank Actionability / Rank-to-Selector Gap Audit."""
from __future__ import annotations

from ..objective_field import schema as c41

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c41.LOCKED_C19_CONFIG_HASH

C30_TABLE_DIR = "oaci/reports/c30_tables"
C32_TABLE_DIR = "oaci/reports/c32_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C37_TABLE_DIR = "oaci/reports/c37_tables"
C38_TABLE_DIR = "oaci/reports/c38_tables"
C40_TABLE_DIR = "oaci/reports/c40_tables"
C41_TABLE_DIR = "oaci/reports/c41_tables"
C42_TABLE_DIR = "oaci/reports/c42_tables"

TOPK_RULES = (("top1", 1), ("top3", 3), ("top5", 5), ("top_decile", 0.10))
LABELS = ("primary_joint_good", "pareto_good", "preference_robust_better_candidate")

SOURCE_RANK_PAIRWISE_SIGNAL_GATE = 0.55
TOP1_RELIABLE_JOINT_GOOD_GATE = 0.70
TOP1_RELIABLE_ENRICHMENT_GATE = 1.50
TOP1_MODEST_GAIN_GATE = 0.10
LEAKAGE_BLOCKS_FRACTION_GATE = 0.50
PLATEAU_EPS = 0.02
PLATEAU_MEAN_SIZE_GATE = 2.0
PLATEAU_LOW_MARGIN_FRACTION_GATE = 0.50
GAUGE_TOP1_GAIN_GATE = 0.05

R1 = "R1_source_rank_pairwise_signal_real"
R2 = "R2_rank_to_topk_gap"
R3 = "R3_source_rank_top1_improves_over_oaci_but_not_reliable"
R4 = "R4_source_rank_top1_reliable_diagnostic"
R5 = "R5_gauge_breaks_source_rank_actionability"
R6 = "R6_dense_base_rate_limits_claim"
R7 = "R7_top_region_plateau_or_instability"
R8 = "R8_leakage_blocks_rank_better_candidates"
R9 = "R9_source_rank_escape_hatch_closed"
R10 = "R10_source_rank_escape_hatch_reopened_diagnostic_only"
ALL_CASES = (R1, R2, R3, R4, R5, R6, R7, R8, R9, R10)

FORBIDDEN_CLAIM_SUBSTRINGS = c41.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-rank selector",
    "source rank selector",
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c41.frozen_config_hash()
