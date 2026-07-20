"""C48 - Conditioned Source-Space Ceiling / Local Bayes Audit."""
from __future__ import annotations

from ..conditioned_actionability import schema as c47

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c47.LOCKED_C19_CONFIG_HASH

C47_TABLE_DIR = "oaci/reports/c47_tables"
C48_TABLE_DIR = "oaci/reports/c48_tables"

GROUP_SCOPES = c47.GROUP_SCOPES
LABELS = c47.LABELS
K_VALUES = (3, 5, 10)
EPSILON_QUANTILES = (0.01, 0.02, 0.05, 0.10)
PERMUTATION_REPS = 64
PERMUTATION_SEED = 48048

SOURCE_SPACES = (
    ("all_source_objectives", ()),
    ("rank_only", ("source_rank",)),
    ("leakage_only", ("leakage",)),
    ("risk_only", ("source_risk",)),
    ("rank_risk", ("source_rank", "source_risk")),
    ("leakage_rank", ("leakage", "source_rank")),
)

PRIMARY_DISTANCE = c47.PRIMARY_DISTANCE
RELIABLE_TOP1_HIT_GATE = c47.RELIABLE_TOP1_HIT_GATE
RELIABLE_ENRICHMENT_GATE = c47.RELIABLE_ENRICHMENT_GATE
MEANINGFUL_CEILING_GAP = 0.10
MEANINGFUL_GAIN_GAP = 0.05
LOW_CEILING_HIT_MARGIN = 0.02
BASE_RATE_GAIN_GATE = 0.05
STABILITY_GAP_GATE = 0.10
PERMUTATION_GAP_GATE = 0.10

LC1 = "LC1_conditioned_source_space_ceiling_high"
LC2 = "LC2_conditioned_source_space_ceiling_low"
LC3 = "LC3_existing_scores_underuse_source_space"
LC4 = "LC4_ceiling_target_local_not_transferable"
LC5 = "LC5_base_rate_explains_apparent_ceiling"
LC6 = "LC6_conditioned_actionability_escape_hatch_closed"
LC7 = "LC7_inconclusive_due_to_artifact_availability"
ALL_CASES = (LC1, LC2, LC3, LC4, LC5, LC6, LC7)

FORBIDDEN_CLAIM_SUBSTRINGS = c47.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-space selector",
    "local bayes selector",
    "usable selector",
    "target-free detector",
    "oaci rescue",
    "external validation success",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c47.frozen_config_hash()
