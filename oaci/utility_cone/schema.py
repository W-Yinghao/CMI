"""C35 - Utility-Cone / Pareto Regret Robustness Audit.

Read-only audit over C34S compact summary JSON, c34_tables CSVs, and the prior
OACI artifact chain. C35 asks whether C34 local continuous regret is robust over
endpoint preferences or depends on fixed scalar/norm summaries.
"""
from __future__ import annotations

from ..continuous_regret import schema as c34

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c34.LOCKED_C19_CONFIG_HASH

C34_COMPACT_JSON = "oaci/reports/C34_CONTINUOUS_LOCAL_REGRET_AUDIT.json"
C34_TABLE_DIR = "oaci/reports/c34_tables"

# C34 stores NLL/ECE as improvements (ERM metric - candidate metric), so all
# three endpoint columns are already higher-is-better in C35.
ENDPOINT_KEYS = ("target_bacc_delta", "target_nll_delta", "target_ece_delta")
ENDPOINT_LABELS = ("bacc", "nll_improve", "ece_improve")

UTILITY_GRID_STEP = 0.05
UTILITY_WIN_EPS = 1e-12
EPSILON_PARETO = (0.0, 0.005, 0.02)

ROBUST_WEIGHT_FRACTION = 0.80
NARROW_WEIGHT_FRACTION = 0.20
SOURCE_FLAT_EPS = c34.SOURCE_FLAT_EPS
TU_FLAT_EPS = c34.SOURCE_FLAT_EPS

U1 = "U1_preference_robust_local_regret"
U2 = "U2_preference_dependent_tradeoff_regret"
U3 = "U3_pareto_dominated_selected_cases_common"
U4 = "U4_scalarization_artifact_substantial"
U5 = "U5_source_active_misranking_preference_robust"
U6 = "U6_source_misranking_scalarization_dependent"
U7 = "U7_target_unlabeled_no_preference_robust_rescue"
U8 = "U8_endpoint_scaling_sensitive"
U9 = "U9_inconclusive_due_to_dense_tradeoff_landscape"
ALL_CASES = (U1, U2, U3, U4, U5, U6, U7, U8, U9)

FORBIDDEN_CLAIM_SUBSTRINGS = c34.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "utility-cone selector",
    "pareto selector",
    "target-free detector",
    "oaci rescue",
    "deployable local-regret selection",
    "target-unlabeled dg success",
    "selected-checkpoint artifact",
)


def frozen_config_hash() -> str:
    return c34.frozen_config_hash()
