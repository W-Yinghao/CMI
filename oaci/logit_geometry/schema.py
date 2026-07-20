"""C27 — Confidence-Occupancy Logit Geometry Counterfactual Audit. C26 established: predicted-class mix is a
stable, error-geometry-aligned DECISION-OCCUPANCY pattern that IS the target identity fingerprint, and the
score-offset recovery is a confidence-mix SYNERGY interaction (P5), not a standalone marginal signal. C27 asks:
WHAT is that confidence-occupancy interaction in LOGIT space?

Read-only logit-space counterfactual mechanism audit over the per-sample target logits persisted by the C26 re-
persistence (oaci-c26-repersist/*.unlabeled.npz). NO re-inference, NO training, NO probe tuning, NO feature
selection, NO selector. Factor families are FROZEN before analysis. Target labels enter ONLY the quarantined
post-hoc label-alignment module. Frozen C19 config hash unchanged.

  C27-A class-conditioned confidence decomposition  -- does occupancy_k x confidence_k explain the interaction?
  C27-B logit-space counterfactuals                 -- which deterministic transform DESTROYS offset recovery?
  C27-C sufficiency / necessity by factor family    -- offset recovery vs target-identity fingerprinting, jointly
  C27-D label alignment under interventions         -- do offset-destroying transforms also destroy error alignment?
"""
from __future__ import annotations

from ..predmix_mechanism import schema as c26

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c26.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c26.C22_SCORE_SIDECAR
C26_REPERSIST_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c26-repersist"   # per-sample logits + splits + quarantined labels
N_CLASSES = 4

# ---- FROZEN logit factor families (declared BEFORE analysis; NOT feature selection) ------------------
# Every family is a fixed function of a candidate's per-sample target logits (no labels).
OCCUPANCY = tuple(f"occ_c{k}" for k in range(N_CLASSES))                 # predicted-class occupancy (== C26 predmix)
GLOBAL_CONF = ("conf_mean", "conf_std", "entropy_mean", "entropy_std",
               "margin_mean", "margin_std", "logit_norm_mean", "logit_norm_std")   # == C26 confidence/margin scaffold
CLASSCOND_CONF = tuple(f"conf_c{k}" for k in range(N_CLASSES))          # mean confidence among samples predicted k
CLASSCOND_MARGIN = tuple(f"margin_c{k}" for k in range(N_CLASSES))
CLASS_BIAS = tuple(f"bias_c{k}" for k in range(N_CLASSES))              # mean logit per class
OCC_X_CONF = tuple(f"occ_x_conf_c{k}" for k in range(N_CLASSES))        # occupancy_k * confidence_k interaction term

FAMILIES = {
    "occupancy": OCCUPANCY,
    "global_confidence": GLOBAL_CONF,
    "class_conditioned_confidence": CLASSCOND_CONF,
    "class_conditioned_margin": CLASSCOND_MARGIN,
    "class_bias": CLASS_BIAS,
    "occ_x_conf_interaction": OCC_X_CONF,
}
# the C24/C26 "full R3" gauge (occupancy + global confidence scaffold) -- the recovery baseline (+0.491)
FULL_R3_FAMILIES = ("occupancy", "global_confidence")

# ---- frozen counterfactual interventions (deterministic logit transforms; NO retraining) -------------
TEMPERATURE = 2.0                                           # soften logits (preserves argmax/occupancy)
INTERVENTIONS = ("raw", "temperature", "class_bias_center", "logit_norm_normalize", "class_uniformize",
                 "confidence_shuffle", "class_shuffle")

# ---- offset model (inherit fixed ridge / LOTO / no grid / no feature-selection) ----------------------
RIDGE_L2 = c26.RIDGE_L2                                     # 1.0
N_PERM = c26.N_PERM                                        # 200
PERM_SEED = c26.PERM_SEED                                  # 707
SUCCESS_GAP_CLOSED = c26.SUCCESS_GAP_CLOSED                # 0.40
DESTROYS_FRACTION = 0.50                                   # an intervention DESTROYS recovery if gap drops >= 50% of baseline
IDENTITY_CHANCE = 1.0 / 9
IDENTITY_SIGNATURE_CEILING = 0.35
DOMINANT_FAMILY_SHARE = 0.60

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
L1 = "L1_class_conditioned_confidence_carries_interaction"
L2 = "L2_class_bias_occupancy_drives_offset"
L3 = "L3_global_logit_scale_drives_offset"
L4 = "L4_sample_level_coupling_required"
L5 = "L5_identity_fingerprint_only"
L6 = "L6_error_geometry_coupled"
L7 = "L7_interaction_real_but_not_factorized"
ALL_CASES = (L1, L2, L3, L4, L5, L6, L7)

FORBIDDEN_CLAIM_SUBSTRINGS = c26.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable calibration", "logit-geometry selector", "confidence-occupancy detector",
)


def frozen_config_hash() -> str:
    return c26.frozen_config_hash()
