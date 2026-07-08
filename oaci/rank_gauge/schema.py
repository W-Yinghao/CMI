"""C30 — Rank-Gauge Separation Audit. The competence signal decomposes into (1) a target-specific GAUGE /
intercept term (the per-target score offset) that breaks pooled cross-target transport and is source-
unobservable (C23-C29), and (2) a within-target RANKING term that is weakly source-visible. C30 tests whether
the ranking axis is separable from the gauge and which source-only factor family carries it.

Read-only over the C22 score sidecar (score + competence label + R_src + 16 robust-core source features) +
optional C18 source / C26 target logits. NO training, NO probe tuning (config hash unchanged), NO feature
selection, NO selector. Target labels diagnostic-only. Factor families FROZEN before analysis.
"""
from __future__ import annotations

from ..rep_head_geometry import schema as c29

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c29.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c29.C22_SCORE_SIDECAR
N_CLASSES = c29.N_CLASSES

# ---- FROZEN source-factor families over the C22 sidecar keys (declared BEFORE analysis) ---------------
SOURCE_RISK = ("R_src", "train_surrogate")
SOURCE_CALIBRATION = ("feat__source_guard_ece", "feat__source_audit_ece", "feat__source_guard_entropy",
                      "feat__source_audit_entropy", "feat__source_guard_conf_on_wrong", "feat__source_audit_conf_on_wrong")
SOURCE_LEAKAGE = ("feat__selection_leakage_point", "feat__audit_leakage_point")
SOURCE_LOGIT_GEOMETRY = ("feat__source_guard_confidence", "feat__source_guard_margin", "feat__source_guard_logit_norm",
                         "feat__source_guard_nll", "feat__source_audit_confidence", "feat__source_audit_margin",
                         "feat__source_audit_logit_norm", "feat__source_audit_nll")
SOURCE_FAMILIES = {"source_risk": SOURCE_RISK, "source_calibration": SOURCE_CALIBRATION,
                   "source_leakage": SOURCE_LEAKAGE, "source_logit_geometry": SOURCE_LOGIT_GEOMETRY}
SCORE_KEY = "score"                                         # frozen source-only probe competence prediction
LABEL_KEY = "label"                                         # tgt__target_bacc_good diagnostic competence label
GAUGE_KEY = "R_src"                                         # C22 source_risk_overlap reference (a gauge-axis proxy)

# ---- thresholds -------------------------------------------------------------------------------------
RANK_SIGNAL_MIN = 0.55                                      # within-target AUC above this => a real ranking signal
CARRIES_RANK_FRACTION = 0.70                                # a family "carries" the rank if its |wtAUC-.5| >= this x score's
N_PERM = c29.N_PERM
PERM_SEED = c29.PERM_SEED

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
G1 = "G1_two_axis_rank_gauge_separation"
G2 = "G2_source_risk_carries_rank"
G3 = "G3_source_calibration_carries_residual_rank"
G4 = "G4_leakage_not_rank_carrier"
G5 = "G5_rank_signal_tracks_source_error_only"
G6 = "G6_target_gauge_contaminates_rank"
G7 = "G7_unexplained_rank_residual"
ALL_CASES = (G1, G2, G3, G4, G5, G6, G7)

FORBIDDEN_CLAIM_SUBSTRINGS = c29.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "rank gauge selector", "competence score selector", "source-visible dg",
)


def frozen_config_hash() -> str:
    return c29.frozen_config_hash()
