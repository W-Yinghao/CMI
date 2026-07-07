"""C25 — Target-Unlabeled Gauge Mechanism + Grouping Boundary Audit. C24 established: source-only HURTS (R1),
target-unlabeled marginal geometry PARTIALLY recovers the per-target score offset (R3, +0.491 gap, permutation-
robust), source+target-unlabeled COLLAPSES (R4), target grouping / target-centered oracle FULLY recovers (R6).
C25 is a read-only MECHANISM audit of WHY:

  Q1  which target-unlabeled feature FAMILY carries the weak R3 recovery? (family-only / leave-one-out / Shapley)
  Q2  is R3 a target-MARGINAL geometry signal or a target-IDENTITY signature? (identity audit + controlled recovery)
  Q3  why do source features DESTROY R3 in R4? (coef-norm domination / condition number / random-dim control)
  Q4  what PROBLEM CLASS is the 0-label target-grouping oracle? (information-assumption ladder)

NOT a selector, NOT DG success, NOT an OACI rescue, NOT deployable. NO feature selection (families are FROZEN
before analysis). NO re-inference (reads the C24 label-free sidecar). Frozen C19 config hash unchanged.
"""
from __future__ import annotations

from ..information_ladder import schema as c24

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c24.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c24.C22_SCORE_SIDECAR
C24_TARGET_UNLABELED_SIDECAR = c24.C24_TARGET_REINFER_SIDECAR

# ---- FROZEN R3 feature families (declared BEFORE analysis; NOT feature selection) ---------------------
# The recovering R3 gauge used exactly these 12 label-free target-unlabeled features (confidence geometry +
# predicted-class proportions). Families partition them. The mechanism audit explains THIS recovery.
FAMILIES = {
    "confidence_entropy": ("target_confidence_mean", "target_confidence_std",
                           "target_entropy_mean", "target_entropy_std"),
    "margin_logitnorm": ("target_margin_mean", "target_margin_std",
                         "target_logit_norm_mean", "target_logit_norm_std"),
    "pred_class_prop": ("target_pred_prop_c0", "target_pred_prop_c1", "target_pred_prop_c2", "target_pred_prop_c3"),
}
ALL_R3_FEATURES = tuple(f for fam in FAMILIES.values() for f in fam)   # 12
# Families named in the C25 spec that the recovering R3 gauge did NOT compute (would require a target-Z / raw-
# feature re-inference); OUT OF SCOPE for auditing the existing recovery. Disclosed, never silently dropped.
NOT_COMPUTED_FAMILIES = ("target_feature_moments", "source_target_distance", "finite_feature_availability")

# ---- offset model (inherit C23/C24: fixed ridge, LOTO, no grid/feature-selection) --------------------
RIDGE_L2 = c24.RIDGE_L2                                      # 1.0
N_PERM = c24.N_PERM                                         # 200
PERM_SEED = c24.PERM_SEED                                   # 707

SUCCESS_GAP_CLOSED = c24.SUCCESS_GAP_CLOSED                 # 0.40
DOMINANT_FAMILY_SHARE = 0.60                                # a family "carries" R3 if it holds >=60% of positive Shapley
IDENTITY_CHANCE = 1.0 / 9
IDENTITY_SIGNATURE_CEILING = 0.35                          # R3-feature target-id acc above this => identity-separable

# ---- R4 interference control ------------------------------------------------------------------------
R4_RANDOM_DIM_TRIALS = 20                                   # add this-many random-noise dims (== #source feats) to R3
R4_SOURCE_COEF_DOMINATION = 0.60                            # source coef-norm share above this => source hijacks ridge

# ---- problem-class information ladder ----------------------------------------------------------------
PROBLEM_CLASSES = (
    {"rung": "source_only_DG", "target_inputs": False, "target_grouping": False, "target_labels": False,
     "recovers": "no (C23: offset source-unobservable)"},
    {"rung": "target_unlabeled_transductive", "target_inputs": True, "target_grouping": True, "target_labels": False,
     "recovers": "partial/weak (C24 R3 +0.491, predictive gauge; no held-out target scores used)"},
    {"rung": "target_grouped_transductive_zero_label", "target_inputs": True, "target_grouping": True, "target_labels": False,
     "recovers": "full (C24 R6 target-centered; uses held-out target's OWN candidate scores' mean)"},
    {"rung": "few_label_target_calibration", "target_inputs": True, "target_grouping": True, "target_labels": True,
     "recovers": "refines beyond grouping (C24 R5)"},
    {"rung": "target_label_oracle", "target_inputs": True, "target_grouping": True, "target_labels": True,
     "recovers": "upper bound"},
)

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
U1 = "U1_confidence_geometry_carries_gauge"
U2 = "U2_predicted_class_mix_carries_gauge"
U3 = "U3_target_feature_geometry_carries_gauge"          # only reachable if families 4-6 were computed
U4 = "U4_target_identity_signature_dominates"
U5 = "U5_weak_multifamily_target_marginal_signal"
U6 = "U6_source_interference_confirmed"
U7 = "U7_grouping_is_separate_problem_class"
ALL_CASES = (U1, U2, U3, U4, U5, U6, U7)

FORBIDDEN_CLAIM_SUBSTRINGS = c24.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "grouping is source-only", "target-centered oracle as method", "target grouping is deployable dg",
)


def frozen_config_hash() -> str:
    return c24.frozen_config_hash()
