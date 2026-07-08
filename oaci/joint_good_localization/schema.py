"""C32 - Joint-Good Localization / Selection-Regret Anatomy Audit.

C31 established that joint accuracy+calibration-good checkpoints are common, and that the failure is not a
checkpoint-space accuracy/calibration trade-off. C32 asks the next localization question: if joint-good checkpoints
are common, why do source-side selection/diagnostic scores still fail to localize them?

Read-only over the same frozen C10 replay, C22 score sidecar, and C24 label-free target-unlabeled sidecar. No
training, no new DG penalty, no selector, no selected-checkpoint artifact, no BNCI2014_004, no seeds [3, 4].
All target endpoint labels are diagnostic only; target labels never enter feature construction.
"""
from __future__ import annotations

from ..endpoint_geometry import schema as c31

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c31.LOCKED_C19_CONFIG_HASH
C22_SCORE_SIDECAR = c31.C22_SCORE_SIDECAR
C10_REPLAY_DIR = c31.C10_REPLAY_DIR
C24_TARGET_UNLABELED_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c24-target-unlabeled.json"

IMPROVE_MARGIN = c31.IMPROVE_MARGIN
ROBUST_MARGIN = c31.ROBUST_MARGIN
TOP_KS = (1, 3, 5, 10)
RIDGE_L2 = 1.0

# Gate thresholds. These are deliberately coarse and pre-declared before reading the C32 verdict.
JOINT_COMMON_RATE = 0.25
TRAJECTORY_WITH_JOINT_FRACTION = 0.80
RANDOM_TOP1_NONTRIVIAL = 0.25
WEAK_TOPK_ENRICHMENT_MAX = 1.50
TOP5_NOT_BETTER_THAN_RANDOM = 1.10
SELECTED_RANDOM_TOL = 0.05
NEAR_ORDER_DISTANCE = 1.0
TARGET_UNLABELED_POOLED_AUC_GAIN = 0.03
GROUPED_POOLED_AUC_GAIN = 0.08

# Pre-registered C32 taxonomy.
J8 = "J8_joint_good_landscape_diffuse_or_base_rate_dominated"
J1 = "J1_joint_good_common_not_scarce"
J2 = "J2_source_scores_weak_trajectory_enrichment"
J3 = "J3_source_topk_localization_weak"
J4 = "J4_tail_or_margin_specific_far_region_only"
J5 = "J5_selected_oaci_near_joint_good_margin_sensitive"
J6 = "J6_target_unlabeled_improves_pooled_gauge_not_topk_localization"
J7 = "J7_target_grouped_rank_recovers_pooled_localization_non_deployable"
J9 = "J9_localization_story_inconclusive"
ALL_CASES = (J8, J1, J2, J3, J4, J5, J6, J7, J9)

FORBIDDEN_CLAIM_SUBSTRINGS = c31.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "joint-good selector",
    "localization selector",
    "target-unlabeled selector",
    "selected-checkpoint artifact",
    "deployable localization",
    "target-grouped selector",
)


def frozen_config_hash() -> str:
    return c31.frozen_config_hash()
