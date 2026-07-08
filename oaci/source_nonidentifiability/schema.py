"""C45 - Source-Equivalence / Target-Divergence Non-Identifiability Witness Audit."""
from __future__ import annotations

from ..continuous_regret import schema as c34
from ..source_frontier_geometry import schema as c44

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c44.LOCKED_C19_CONFIG_HASH

C29_TABLE_DIR = "oaci/reports/c29_tables"
C30_TABLE_DIR = "oaci/reports/c30_tables"
C32_TABLE_DIR = "oaci/reports/c32_tables"
C35_TABLE_DIR = "oaci/reports/c35_tables"
C41_TABLE_DIR = "oaci/reports/c41_tables"
C42_TABLE_DIR = "oaci/reports/c42_tables"
C43_TABLE_DIR = "oaci/reports/c43_tables"
C44_TABLE_DIR = "oaci/reports/c44_tables"
C45_TABLE_DIR = "oaci/reports/c45_tables"

PRIMARY_DISTANCE = "within_trajectory_z_euclidean"
RANK_DISTANCE = "within_trajectory_rank_l1"
FAMILY_BLOCK_DISTANCE = "family_block_z_euclidean"
DISTANCE_METRICS = (PRIMARY_DISTANCE, RANK_DISTANCE, FAMILY_BLOCK_DISTANCE)

EPSILON_QUANTILES = (0.01, 0.02, 0.05, 0.10)
SOURCE_EQUIVALENT_Q = 0.10
TARGET_UTILITY_LARGE_GAP = 0.50
ENDPOINT_Z_LARGE_GAP = 0.75
GAUGE_JUMP_EPS = c34.GAUGE_JUMP_EPS
PRIMARY_MARGIN = c34.PRIMARY_MARGIN

NEAREST_SCOPES = ("within_trajectory", "within_target", "cross_target", "same_regime")
FAMILY_SPACES = (
    ("leakage_only", ("leakage",)),
    ("risk_only", ("source_risk",)),
    ("rank_only", ("source_rank",)),
    ("endpoint_only", ("source_endpoint",)),
    ("leakage_rank", ("leakage", "source_rank")),
    ("rank_risk", ("source_rank", "source_risk")),
    ("all_source_objectives", ()),
)

N1 = "N1_source_equivalent_target_divergent_witnesses"
N2 = "N2_within_trajectory_nonidentifiability"
N3 = "N3_source_radius_target_variance_persists"
N4 = "N4_source_metric_neighborhood_not_discriminative"
N5 = "N5_target_gauge_residual_drives_divergence"
N6 = "N6_family_reduced_space_not_sufficient"
N7 = "N7_rank_space_reduces_but_does_not_close_ambiguity"
N8 = "N8_empirical_selector_lower_bound_supported"
N9 = "N9_inconclusive_due_to_objective_availability"
ALL_CASES = (N1, N2, N3, N4, N5, N6, N7, N8, N9)

FREQUENT_WITNESS_GATE = 0.25
TRAJECTORY_WITNESS_GATE = 0.50
VARIANCE_PERSIST_RATIO_GATE = 0.50
NONDISCRIMINATIVE_RATIO_GATE = 0.75
GAUGE_WITNESS_RATE_GATE = 0.25
GAUGE_UTILITY_CORR_GATE = 0.30
FAMILY_AMBIGUITY_GATE = 0.20
RANK_REDUCTION_MIN = 0.03
LOWER_BOUND_GATE = 0.10

FORBIDDEN_CLAIM_SUBSTRINGS = c44.FORBIDDEN_CLAIM_SUBSTRINGS + (
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
    return c44.frozen_config_hash()
