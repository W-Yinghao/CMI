"""C29 — Representation-Head Origin of Target Class-Conditioned Confidence. C27 localized the target score-offset
carrier to class-conditioned confidence (a function of the target LOGITS); C28 confirmed it is source-
unobservable at the logit-factor level. C29 asks WHERE inside the model that logit factor originates:

  logit_k = W_k . z + b_k     (ShallowConvNet has a LINEAR head: classifier.weight W (C x D), bias b (C))

so the logit decomposes EXACTLY into a parameter head-bias b_k and a representation projection (W.z)_k. Because
(W.z) = logit - b is computable from the persisted target logits + the extracted head bias, the OFFSET-relevant
representation contribution is available READ-ONLY (no target-z re-persistence): any z-component orthogonal to
W's row space cannot change the logits and is offset-irrelevant. C29 tests whether the carrier is driven by the
parameter head-bias b (R1) or by the representation projection W.z / its target-specific shift (R2/R3/R7).

Read-only: head params from the frozen checkpoint state_dicts (CPU read, NO training / NO GPU / NO inference);
target logits from the C26 re-persistence; source logits from the C18 extract; offsets from the C22 sidecar.
Frozen C19 config hash unchanged. Factor definitions consistent with C27/C28. Target labels post-hoc only.
"""
from __future__ import annotations

from ..source_target_homology import schema as c28

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c28.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c28.C22_SCORE_SIDECAR
C18_EXTRACT_DIR = c28.C18_EXTRACT_DIR
C26_REPERSIST_DIR = c28.C26_REPERSIST_DIR
LOSO_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"
C29_HEAD_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c29-head-params.json"
N_CLASSES = c28.N_CLASSES
HEAD_WEIGHT_KEY = "classifier.weight"                       # (C, D) linear head
HEAD_BIAS_KEY = "classifier.bias"                           # (C,)

# the carrier under study == C27/C28 class-conditioned confidence (identical definition)
CARRIER_FAMILY = c28.CARRIER_FAMILY                         # "class_conditioned_confidence"
CARRIER_NAMES = c28.CARRIER_NAMES                           # conf_c0..c3

# ---- offset model (inherit fixed ridge / LOTO / no grid / no feature-selection) ----------------------
RIDGE_L2 = c28.RIDGE_L2                                     # 1.0
N_PERM = c28.N_PERM                                        # 200
PERM_SEED = c28.PERM_SEED                                  # 707
SUCCESS_GAP_CLOSED = c28.SUCCESS_GAP_CLOSED               # 0.40
DESTROYS_FRACTION = 0.50                                   # counterfactual destroys recovery if gap drops >= 50% of baseline
CARRIES_FRACTION = 0.60                                    # a component "carries" the offset if its gap >= this x full
IDENTITY_CHANCE = 1.0 / 9
IDENTITY_SIGNATURE_CEILING = 0.35

# ---- deterministic logit counterfactuals (frozen W/b/logits; NO retraining) --------------------------
INTERVENTIONS = ("raw", "parameter_bias_removed", "effective_mean_removed", "projection_only",
                 "weight_norm_normalized", "global_scale_removed", "source_mean_centered_projection")

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
R1 = "R1_parameter_head_bias_drives_offset"
R2 = "R2_effective_logit_bias_from_target_representation"
R3 = "R3_target_representation_shift_drives_offset"
R4 = "R4_weight_norm_or_global_scale_not_sufficient"
R5 = "R5_head_representation_interaction_required"
R6 = "R6_source_representation_tracks_source_error_only"
R7 = "R7_target_representation_residual_carries_missing_gauge"
R8 = "R8_no_clean_internal_decomposition"
ALL_CASES = (R1, R2, R3, R4, R5, R6, R7, R8)

FORBIDDEN_CLAIM_SUBSTRINGS = c28.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "representation selector", "deployment-time selector", "head-geometry detector",
)


def frozen_config_hash() -> str:
    return c28.frozen_config_hash()
