"""C33 - Local Trajectory Boundary / Checkpoint Neighborhood Audit.

Read-only mechanism audit over committed C32R/C31/C30/C24 artifacts. C33 asks why selected OACI can be close to a
joint-good checkpoint but still hit joint-good at a random-like rate. It analyzes local trajectory boundaries,
selected-vs-nearest-joint pairs, adjacent score gradients, source-score plateaus, and local information rungs.

No training, no re-inference, no score tuning, no feature selection, no selector, no selected-checkpoint artifact.
Target endpoint labels are diagnostic-only.
"""
from __future__ import annotations

from ..joint_good_localization import schema as c32

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c32.LOCKED_C19_CONFIG_HASH
C22_SCORE_SIDECAR = c32.C22_SCORE_SIDECAR
C10_REPLAY_DIR = c32.C10_REPLAY_DIR
C24_TARGET_UNLABELED_SIDECAR = c32.C24_TARGET_UNLABELED_SIDECAR

PRIMARY_MARGIN = c32.IMPROVE_MARGIN
ROBUST_MARGIN = c32.ROBUST_MARGIN

# Frozen local definitions.
ORDER_NEIGHBORHOODS = (1, 2, 3)
EPOCH_WINDOW = 20
AUTOCORR_LAGS = (1, 2, 3)
PLATEAU_EPS = 0.02
SOURCE_FLAT_EPS = 0.02
GAUGE_JUMP_EPS = 0.02
LOCAL_RANDOM_MIN_UNITS = 1

B1 = "B1_dense_boundary_jitter"
B2 = "B2_source_score_flat_near_boundary"
B3 = "B3_source_score_active_misranking"
B4 = "B4_local_gauge_jump_unseen_by_source"
B5 = "B5_rank_signal_local_but_too_weak"
B6 = "B6_target_unlabeled_local_help"
B7 = "B7_target_unlabeled_pooled_only_confirmed"
B8 = "B8_label_margin_instability"
B9 = "B9_selector_plateau_indifference"
B10 = "B10_tail_failure_only"
B11 = "B11_local_boundary_story_inconclusive"
ALL_CASES = (B1, B2, B3, B4, B5, B6, B7, B8, B9, B10, B11)

FORBIDDEN_CLAIM_SUBSTRINGS = c32.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "local-boundary selector",
    "neighborhood selector",
    "target-unlabeled dg success",
    "target-grouped oracle as method",
)


def frozen_config_hash() -> str:
    return c32.frozen_config_hash()
