"""C47 - Conditioned Source-Space Actionability Audit."""
from __future__ import annotations

from ..source_conditioning_boundary import schema as c46

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c46.LOCKED_C19_CONFIG_HASH

C43_TABLE_DIR = "oaci/reports/c43_tables"
C46_TABLE_DIR = "oaci/reports/c46_tables"
C47_TABLE_DIR = "oaci/reports/c47_tables"

GROUP_SCOPES = (
    "global",
    "within_target",
    "within_trajectory",
    "within_target_seed",
    "within_target_level",
    "within_regime",
)
TOP_KS = (1, 3, 5, 10)
LABELS = ("primary_joint_good", "pareto_good", "preference_robust_better_candidate")

SOURCE_EQUIVALENT_Q = c46.SOURCE_EQUIVALENT_Q
PRIMARY_DISTANCE = c46.PRIMARY_DISTANCE
PAIR_SAMPLE_MAX = 100000
PAIR_SAMPLE_SEED = 47047

RELIABLE_TOP1_HIT_GATE = 0.70
RELIABLE_ENRICHMENT_GATE = 1.50
IMPROVEMENT_GATE = 0.05
REGRET_REDUCTION_GATE = 0.10
BASE_RATE_LIMITED_TOP1_GATE = 0.65
NEIGHBOR_SMOOTHING_GAIN_GATE = 0.03
SIGN_CONSISTENCY_GATE = 0.55
GLOBAL_FAILURE_GAIN_GATE = 0.05

GCA1 = "GCA1_conditioning_restores_source_neighborhood_homogeneity"
GCA2 = "GCA2_conditioning_improves_but_not_reliable_actionability"
GCA3 = "GCA3_trajectory_conditioning_required"
GCA4 = "GCA4_target_conditioning_sufficient_for_diagnostic_localization"
GCA5 = "GCA5_grouped_actionability_still_base_rate_limited"
GCA6 = "GCA6_global_source_only_comparability_fails"
GCA7 = "GCA7_group_conditioning_is_separate_problem_class"
GCA8 = "GCA8_inconclusive_due_to_artifact_availability"
ALL_CASES = (GCA1, GCA2, GCA3, GCA4, GCA5, GCA6, GCA7, GCA8)

FORBIDDEN_CLAIM_SUBSTRINGS = c46.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "conditioned selector",
    "deployable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c46.frozen_config_hash()
