"""C26 — Predicted-Class Mix Mechanism / Counterfactual Audit. C25 localized the weak R3 recovery's carrier to
the predicted-class-mix family (U2) but did not explain WHAT that signal is. C26 dissects it:

  Q1 split-stability     is pred-class-mix a stable target decision-occupancy signal or finite-sample noise?
  Q2 signed vs symmetric does the SIGNED class vector matter (class-specific occupancy) or only symmetric
                         concentration (entropy / max-mass / distance-to-uniform / Gini)?
  Q3 identity controls   is it a target-marginal signal or a target-IDENTITY fingerprint?
  Q4 interaction         is pred-class-mix a main effect, or does it work only through a confidence/margin scaffold?
  Q5 label diagnostics   (labels join ONLY here, diagnostic-only) does the mix track target class-error geometry?

STAGED by data availability: Q2/Q3/Q4 are read-only from the C24 aggregate sidecar (per-candidate pred_prop
vector + confidence/margin families). Q1 (needs per-sample target logits to split) and Q5 (needs per-sample
labels) require a scoped re-PERSISTENCE re-inference (the P0-validated forward, persisting per-split mix
summaries + quarantined label diagnostics) -- NOT proxied, NOT permanently deferred. Frozen C19 hash unchanged.
NOT a selector, NOT DG success, NOT deployable. Families FROZEN; no feature selection."""
from __future__ import annotations

from ..unlabeled_gauge import schema as c25

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c25.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c25.C22_SCORE_SIDECAR
C24_TARGET_UNLABELED_SIDECAR = c25.C24_TARGET_UNLABELED_SIDECAR
# Stage-2 re-persistence re-inference writes here (per-split mix summaries + quarantined label diagnostics)
C26_SPLIT_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c26-predmix-splits.json"

N_CLASSES = 4                                                # BNCI2014_001 balanced 4-class MI
PRED_PROP = tuple(f"target_pred_prop_c{c}" for c in range(N_CLASSES))          # SIGNED class-occupancy vector
CONF_MARGIN = ("target_confidence_mean", "target_confidence_std", "target_entropy_mean", "target_entropy_std",
               "target_margin_mean", "target_margin_std", "target_logit_norm_mean", "target_logit_norm_std")
# SYMMETRIC (class-index-invariant) summaries derived per-candidate from the pred_prop vector
SYMMETRIC_SUMMARIES = ("predmix_entropy", "predmix_max_mass", "predmix_dist_uniform", "predmix_gini")

# ---- offset model (inherit fixed ridge / LOTO / no grid / no feature-selection) ----------------------
RIDGE_L2 = c25.RIDGE_L2                                      # 1.0
N_PERM = c25.N_PERM                                         # 200
PERM_SEED = c25.PERM_SEED                                   # 707
SUCCESS_GAP_CLOSED = c25.SUCCESS_GAP_CLOSED                 # 0.40
IDENTITY_CHANCE = 1.0 / 9
IDENTITY_SIGNATURE_CEILING = 0.35

# ---- split-stability (Stage-2) ----------------------------------------------------------------------
SPLITS = ("half", "odd_even", "bootstrap")                  # deterministic target-sample splits
SPLIT_STABLE_CORR = 0.60                                    # per-target pred_prop corr across split halves >= this
SPLIT_STABLE_GAP_RATIO = 0.50                               # split-half recovery gap >= this fraction of full

# ---- counterfactual controls ------------------------------------------------------------------------
CLASS_ROTATIONS = tuple(range(1, N_CLASSES))                # cyclic class-index shifts 1..3
ROTATION_INVARIANT_TOL = 0.05                               # |gap(rotated) - gap(signed)| <= this => rotation-invariant

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
P1 = "P1_decision_occupancy_signal"
P2 = "P2_class_identity_specific_signal"
P3 = "P3_symmetric_collapse_signal"
P4 = "P4_identity_fingerprint_dominant"
P5 = "P5_confidence_mix_interaction"
P6 = "P6_sample_noise_artifact"
P7 = "P7_label_diagnostic_boundary"
ALL_CASES = (P1, P2, P3, P4, P5, P6, P7)

STATUS_OK = "computed"
STATUS_REQUIRES_REINFERENCE = "REQUIRES_REPERSISTENCE_REINFERENCE"

FORBIDDEN_CLAIM_SUBSTRINGS = c25.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "identity-free recovery", "pred-class-mix is deployable", "decision occupancy selector",
)


def frozen_config_hash() -> str:
    return c25.frozen_config_hash()
