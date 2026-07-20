"""C46 - Conditioning Boundary / Grouping-Sensitive Non-Identifiability Audit."""
from __future__ import annotations

from ..source_nonidentifiability import schema as c45

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c45.LOCKED_C19_CONFIG_HASH

C45_TABLE_DIR = "oaci/reports/c45_tables"
C46_TABLE_DIR = "oaci/reports/c46_tables"

PRIMARY_DISTANCE = c45.PRIMARY_DISTANCE
RANK_DISTANCE = c45.RANK_DISTANCE
FAMILY_BLOCK_DISTANCE = c45.FAMILY_BLOCK_DISTANCE
DISTANCE_METRICS = c45.DISTANCE_METRICS
EPSILON_QUANTILES = c45.EPSILON_QUANTILES
SOURCE_EQUIVALENT_Q = c45.SOURCE_EQUIVALENT_Q
TARGET_UTILITY_LARGE_GAP = c45.TARGET_UTILITY_LARGE_GAP
ENDPOINT_Z_LARGE_GAP = c45.ENDPOINT_Z_LARGE_GAP
GAUGE_JUMP_EPS = c45.GAUGE_JUMP_EPS

CONDITIONING_SCOPES = (
    "within_trajectory",
    "within_target",
    "within_seed",
    "within_level",
    "within_regime",
    "cross_target",
    "cross_regime",
)
VARIANCE_GROUPINGS = ("global", "target", "trajectory", "seed", "level", "regime", "target_regime")
OUTCOMES = ("target_utility_score", "target_joint_margin_raw", "endpoint_z_norm")

PAIR_SAMPLE_MAX = 100000
PAIR_SAMPLE_SEED = 46046

WITHIN_HOMOGENEITY_GATE = 0.20
WITHIN_TARGET_STRONG_GATE = 0.05
CROSS_TARGET_DIVERGENCE_GATE = 0.75
SAME_REGIME_INTERMEDIATE_GATE = 0.20
TARGET_COMPONENT_GATE = 0.50
DISTANCE_USEFULNESS_GATE = 0.10

CB1 = "CB1_source_space_informative_after_target_or_trajectory_conditioning"
CB2 = "CB2_cross_target_grouping_breaks_source_equivalence"
CB3 = "CB3_within_trajectory_neighborhoods_relatively_homogeneous"
CB4 = "CB4_source_only_global_comparability_nonidentifiable"
CB5 = "CB5_target_identity_component_explains_divergence"
CB6 = "CB6_regime_conditioning_partial_not_sufficient"
CB7 = "CB7_inconclusive_due_to_artifact_availability"
ALL_CASES = (CB1, CB2, CB3, CB4, CB5, CB6, CB7)

FORBIDDEN_CLAIM_SUBSTRINGS = c45.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "conditioning selector",
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c45.frozen_config_hash()
