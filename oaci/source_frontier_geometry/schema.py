"""C44 - Source-Pareto Degeneracy / Objective-Geometry Non-Identifiability Audit."""
from __future__ import annotations

from ..source_scalarization import schema as c43

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c43.LOCKED_C19_CONFIG_HASH

C30_TABLE_DIR = "oaci/reports/c30_tables"
C32_TABLE_DIR = "oaci/reports/c32_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C41_TABLE_DIR = "oaci/reports/c41_tables"
C42_TABLE_DIR = "oaci/reports/c42_tables"
C43_TABLE_DIR = "oaci/reports/c43_tables"
C44_TABLE_DIR = "oaci/reports/c44_tables"

NULL_REPS = 50
NULL_SEED = 44044

FRONT_DEGENERATE_FRACTION = 0.85
FRONT_NULL_CLOSE_DELTA = 0.15
FRONT_NONDISCRIM_DELTA = 0.10
CONFLICT_NEGATIVE_FRACTION = 0.35
EFFECTIVE_RANK_MIN = 3.0
REDUCED_FRONT_NARROW_GATE = 0.50
REDUCED_FRONT_COVERAGE_LOSS = 0.20
WEAK_ENRICHMENT_GATE = 1.15
DEPTH_SIGNAL_LOW = 0.55
DEPTH_SIGNAL_HIGH = 0.62

FAMILY_SUBSETS = (
    ("leakage_only", ("leakage",)),
    ("risk_only", ("source_risk",)),
    ("rank_only", ("source_rank",)),
    ("endpoint_only", ("source_endpoint",)),
    ("leakage_risk", ("leakage", "source_risk")),
    ("leakage_rank", ("leakage", "source_rank")),
    ("rank_risk", ("source_rank", "source_risk")),
    ("leakage_rank_risk", ("leakage", "source_rank", "source_risk")),
    ("all_families", ("leakage", "source_risk", "source_rank", "source_endpoint")),
)

PF1 = "PF1_source_pareto_front_degenerate"
PF2 = "PF2_front_membership_non_discriminative"
PF3 = "PF3_objective_conflict_inflates_front"
PF4 = "PF4_objective_redundancy_not_the_issue"
PF5 = "PF5_family_reduced_frontier_narrows_but_loses_target_coverage"
PF6 = "PF6_family_reduced_frontier_has_diagnostic_signal"
PF7 = "PF7_dominance_depth_not_target_informative"
PF8 = "PF8_dominance_depth_weakly_informative_but_non_actionable"
PF9 = "PF9_source_objective_geometry_non_identifiable"
PF10 = "PF10_inconclusive_due_to_objective_availability"
ALL_CASES = (PF1, PF2, PF3, PF4, PF5, PF6, PF7, PF8, PF9, PF10)

FORBIDDEN_CLAIM_SUBSTRINGS = c43.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c43.frozen_config_hash()
