"""C34 - Continuous Local Regret / Source-Objective Direction Audit.

Read-only over committed C31/C32R/C33/C30/C29-era artifacts. C34 replaces the
binary joint-good neighborhood question with continuous target endpoint-regret
geometry. It never trains, re-infers, tunes scores, creates a selector, or emits
selected-checkpoint artifacts.
"""
from __future__ import annotations

from ..local_boundary import schema as c33

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c33.LOCKED_C19_CONFIG_HASH
C22_SCORE_SIDECAR = c33.C22_SCORE_SIDECAR
C10_REPLAY_DIR = c33.C10_REPLAY_DIR
C24_TARGET_UNLABELED_SIDECAR = c33.C24_TARGET_UNLABELED_SIDECAR

PRIMARY_MARGIN = c33.PRIMARY_MARGIN
ROBUST_MARGIN = c33.ROBUST_MARGIN

# Frozen local definitions inherited from C33.
ORDER_NEIGHBORHOODS = c33.ORDER_NEIGHBORHOODS
EPOCH_WINDOW = c33.EPOCH_WINDOW
SOURCE_FLAT_EPS = c33.SOURCE_FLAT_EPS
GAUGE_JUMP_EPS = c33.GAUGE_JUMP_EPS

# Frozen continuous-regret gates. These are in standardized endpoint-vector
# units, except COMPONENT_FLAT_EPS/SOURCE_FLAT_EPS which are score-scale gates.
STANDARDIZED_TINY_REGRET = 0.05
STANDARDIZED_MEANINGFUL_REGRET = 0.10
COMPONENT_FLAT_EPS = 0.02

ENDPOINT_RAW_KEYS = ("target_bacc_delta", "target_nll_delta", "target_ece_delta")
ENDPOINT_Z_KEYS = ("target_bacc_z", "target_nll_z", "target_ece_z")

SOURCE_COMPONENTS = (
    {"component": "source_score", "family": "OACI_selector_score", "key": "score", "orientation": 1.0,
     "available": True},
    {"component": "selection_leakage", "family": "leakage_component", "key": "feat__selection_leakage_point",
     "orientation": 1.0, "available": True},
    {"component": "audit_leakage", "family": "leakage_component", "key": "feat__audit_leakage_point",
     "orientation": 1.0, "available": True},
    {"component": "R_src", "family": "source_risk", "key": "R_src", "orientation": -1.0, "available": True},
    {"component": "source_guard_nll", "family": "source_endpoint", "key": "feat__source_guard_nll",
     "orientation": -1.0, "available": True},
    {"component": "source_audit_nll", "family": "source_endpoint", "key": "feat__source_audit_nll",
     "orientation": -1.0, "available": True},
    {"component": "source_guard_ece", "family": "source_endpoint", "key": "feat__source_guard_ece",
     "orientation": -1.0, "available": True},
    {"component": "source_audit_ece", "family": "source_endpoint", "key": "feat__source_audit_ece",
     "orientation": -1.0, "available": True},
    {"component": "source_audit_confidence", "family": "calibration_softness",
     "key": "feat__source_audit_confidence", "orientation": 1.0, "available": True},
    {"component": "robust_core_score", "family": "C19_robust_core", "key": "robust_core_score",
     "orientation": 1.0, "available": True},
    {"component": "c30_source_rank", "family": "C30_source_rank", "key": "c30_source_rank",
     "orientation": 1.0, "available": True},
    {"component": "source_audit_worst_bacc", "family": "source_endpoint", "key": "source_audit_worst_bacc",
     "orientation": 1.0, "available": False},
)

M1 = "M1_binary_margin_artifact"
M2 = "M2_continuous_source_active_misranking"
M3 = "M3_source_leakage_objective_conflict"
M4 = "M4_source_risk_objective_conflict"
M5 = "M5_source_score_too_weak_not_wrong"
M6 = "M6_target_gauge_jump_drives_local_regret"
M7 = "M7_target_unlabeled_pooled_only_reconfirmed"
M8 = "M8_continuous_endpoint_tradeoff_local"
M9 = "M9_unexplained_local_residual"
ALL_CASES = (M1, M2, M3, M4, M5, M6, M7, M8, M9)

FORBIDDEN_CLAIM_SUBSTRINGS = c33.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "continuous-regret selector",
    "source-objective selector",
    "target-free detector",
    "target-unlabeled dg success",
    "deployable joint-good localization",
    "oaci rescue",
    "selected-checkpoint artifact",
)


def frozen_config_hash() -> str:
    return c33.frozen_config_hash()
