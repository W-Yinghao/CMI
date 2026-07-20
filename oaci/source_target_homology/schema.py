"""C28 — Source-Target Logit-Factor Homology Audit. C27 localized the target score-offset carrier to CLASS-
CONDITIONED CONFIDENCE (how confidently the model occupies each predicted class on the target). C28 asks whether
an analogous SOURCE-side logit factor exists and whether it predicts the target factor / target offset / target
competence -- pushing the C23 source-unobservable result down to the logit-factor level.

Read-only: source per-sample logits from the C18 extract (logits-source_{guard,audit}.npy), target logits from
the C26 re-persistence, offsets from the C22 sidecar. NO re-inference, NO training, NO probe tuning, NO feature
selection, NO selector. The source and target factor definitions are IDENTICAL (reuse C27 candidate_features).
Source labels are source-side observable (allowed for source error geometry); TARGET labels are post-hoc only.
Frozen C19 config hash unchanged.
"""
from __future__ import annotations

from ..logit_geometry import schema as c27

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c27.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c27.C22_SCORE_SIDECAR
C18_EXTRACT_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c18-extract"      # per-candidate SOURCE logits + y + domain
C26_REPERSIST_DIR = c27.C26_REPERSIST_DIR                    # per-candidate TARGET logits
N_CLASSES = c27.N_CLASSES

SOURCE_ROLES = ("source_guard", "source_audit")
# the FROZEN factor whose source/target homology is tested (C27's carrier); definition == C27, byte-identical
CARRIER_FAMILY = "class_conditioned_confidence"             # conf_c0..c3
CARRIER_NAMES = tuple(f"conf_c{k}" for k in range(N_CLASSES))
# families reused from C27 (identical) for the source factor
FAMILIES = c27.FAMILIES

# ---- offset model (inherit fixed ridge / LOTO / no grid / no feature-selection) ----------------------
RIDGE_L2 = c27.RIDGE_L2                                     # 1.0
N_PERM = c27.N_PERM                                        # 200
PERM_SEED = c27.PERM_SEED                                  # 707
SUCCESS_GAP_CLOSED = c27.SUCCESS_GAP_CLOSED                # 0.40

# ---- homology thresholds ----------------------------------------------------------------------------
ALIGN_STRONG = 0.50                                         # |cosine| / |corr| >= this => source-target factor aligned
ALIGN_WEAK = 0.20                                           # below this => misaligned
SOURCE_PREDICTS_OFFSET_GAP = 0.40                           # source-factor gauge must close >= this AND survive perm
RESIDUAL_CARRIES_FRACTION = 0.60                            # target residual carries offset if its gap >= this x full

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
H1 = "H1_source_factor_predicts_target_factor"
H2 = "H2_source_factor_misaligned"
H3 = "H3_source_factor_predicts_offset"
H4 = "H4_source_factor_tracks_source_error_only"
H5 = "H5_target_residual_carries_offset"
H6 = "H6_source_feature_omission_reopens_gauge"
H7 = "H7_logit_factor_confirms_source_unobservability"
ALL_CASES = (H1, H2, H3, H4, H5, H6, H7)

FORBIDDEN_CLAIM_SUBSTRINGS = c27.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source factor is a selector", "source-only detector", "source registry rescue",
)


def frozen_config_hash() -> str:
    return c27.frozen_config_hash()
