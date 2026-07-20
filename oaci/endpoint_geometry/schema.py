"""C31 — Endpoint-Axis / Accuracy-Calibration Geometry Audit. C16 exposed: target-accuracy-good checkpoints
exist, the SELECTED OACI is calibration-improved / accuracy-flat, and joint accuracy+calibration does not
reproduce. C22-C30 explained the RANK/GAUGE structure of the (accuracy) competence signal. C31 asks whether
that mechanism is ENDPOINT-SPECIFIC: are accuracy-good / NLL-good / ECE-good / joint-good checkpoints the same
set; does the source-visible rank predict accuracy or calibration; is the target gauge accuracy- or calibration-
specific; and is the C16 barrier a Pareto trade-off in checkpoint space.

Read-only: per-candidate target bAcc/NLL/ECE + ERM reference from the C10 replay; source rank from the C22
sidecar; target gauge from C26/C27. NO training, NO probe tuning (config hash unchanged), NO feature selection,
NO selector. Endpoint labels/metrics are DIAGNOSTIC-ONLY; any oracle endpoint is explicitly non-deployable.
Endpoint label definitions are FROZEN before analysis.
"""
from __future__ import annotations

from ..rank_gauge import schema as c30

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c30.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = c30.C22_SCORE_SIDECAR
C10_REPLAY_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c10-replay"
N_CLASSES = c30.N_CLASSES

# ---- FROZEN endpoint definitions (deltas vs the per-(seed,target,level) ERM reference) ----------------
# bAcc: higher better (delta = cand - erm); NLL/ECE: lower better (delta = erm - cand, positive = improvement).
# accuracy_good uses the SAME any-improvement margin as the frozen C19/C22 label (tgt__target_bacc_good, 1e-9);
# a 0.02 robustness margin is reported as a sensitivity, never as the primary definition.
IMPROVE_MARGIN = 1e-9                                        # matches the frozen C22 competence label
ROBUST_MARGIN = 0.02                                         # sensitivity variant only
ENDPOINTS = ("accuracy", "nll", "ece")
ENDPOINT_LABELS = ("accuracy_good", "nll_good", "ece_good", "calibration_good", "joint_good", "pareto_good",
                   "dominated", "accuracy_good_calibration_bad", "calibration_good_accuracy_flat")
SCORE_KEY = "score"                                         # C30 source-visible rank factor
LABEL_KEY = "label"                                         # C22 competence label (== accuracy_good, any improvement)

RANK_SIGNAL_MIN = c30.RANK_SIGNAL_MIN                       # 0.55
N_PERM = c30.N_PERM
PERM_SEED = c30.PERM_SEED

# ---- taxonomy (pre-registered) ----------------------------------------------------------------------
E1 = "E1_accuracy_calibration_tradeoff_confirmed"
E2 = "E2_joint_good_checkpoints_absent_or_rare"
E3 = "E3_joint_good_exists_but_source_unobservable"
E4 = "E4_source_rank_accuracy_specific"
E5 = "E5_source_rank_calibration_biased"
E6 = "E6_gauge_accuracy_specific"
E7 = "E7_gauge_general_endpoint_offset"
E8 = "E8_pareto_geometry_explains_c16_barrier"
E9 = "E9_endpoint_story_inconclusive"
ALL_CASES = (E1, E2, E3, E4, E5, E6, E7, E8, E9)

FORBIDDEN_CLAIM_SUBSTRINGS = c30.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "joint deployable improvement", "endpoint selector", "pareto selector", "target oracle as method",
)


def frozen_config_hash() -> str:
    return c30.frozen_config_hash()
