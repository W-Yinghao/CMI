"""C49 - Sparse Local-Bayes Ceiling / Coverage-Actionability Audit."""
from __future__ import annotations

from ..conditioned_local_ceiling import schema as c48

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c48.LOCKED_C19_CONFIG_HASH

C47_TABLE_DIR = "oaci/reports/c47_tables"
C48_TABLE_DIR = "oaci/reports/c48_tables"
C49_TABLE_DIR = "oaci/reports/c49_tables"

GROUP_SCOPES = c48.GROUP_SCOPES
LABELS = c48.LABELS
K_VALUES = (1, 3, 5, 10, 20)
EPSILON_QUANTILES = (0.01, 0.02, 0.05, 0.10, 0.20)
MIN_NEIGHBOR_COUNTS = (1, 2, 3, 5)
COVERAGE_THRESHOLDS = (0.25, 0.50, 0.75)
SOURCE_SPACES = c48.SOURCE_SPACES
STABILITY_GROUPINGS = ("target", "seed", "level", "trajectory", "regime")

PRIMARY_DISTANCE = c48.PRIMARY_DISTANCE
RELIABLE_TOP1_HIT_GATE = c48.RELIABLE_TOP1_HIT_GATE
RELIABLE_ENRICHMENT_GATE = c48.RELIABLE_ENRICHMENT_GATE
SPARSE_COVERAGE_GATE = 0.50
LOW_COVERAGE_GATE = 0.25
UNDERUSE_GAP_GATE = 0.10
MATCH_GAP_GATE = 0.05
STABILITY_WORST_HIT_GATE = 0.50
STABILITY_WORST_COVERAGE_GATE = 0.25
FUTURE_HYPOTHESIS_PERM_GAP_GATE = c48.PERMUTATION_GAP_GATE

SC1 = "SC1_sparse_high_precision_ceiling"
SC2 = "SC2_broad_conditioned_ceiling"
SC3 = "SC3_existing_scores_underuse_available_islands"
SC4 = "SC4_existing_scores_match_local_ceiling"
SC5 = "SC5_ceiling_unstable_across_targets"
SC6 = "SC6_ceiling_stable_but_non_actionable"
SC7 = "SC7_base_rate_or_singleton_artifact"
SC8 = "SC8_conditioned_source_space_future_hypothesis"
SC9 = "SC9_conditioned_actionability_escape_hatch_closed"
SC10 = "SC10_inconclusive_due_to_artifact_availability"
ALL_CASES = (SC1, SC2, SC3, SC4, SC5, SC6, SC7, SC8, SC9, SC10)

FORBIDDEN_CLAIM_SUBSTRINGS = c48.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
    "conditioned local bayes selector",
)


def frozen_config_hash() -> str:
    return c48.frozen_config_hash()
