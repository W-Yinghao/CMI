"""C24 — Calibration Information Ladder / Identifiability Boundary Audit. C23 closed the source-only target-
free gauge (G5_offset_source_unobservable). C24 asks the NEXT question: what information, if any, breaks the
per-target score-offset non-identifiability? It builds a fixed, diagnostic-only INFORMATION LADDER:

  R0 raw pooled score
  R1 source-only gauge                       (reproduces C23; read-only)
  R2 source-only + static risk-family gauge  (read-only)
  R3 target-UNLABELED gauge                  (needs frozen-checkpoint target-audit RE-INFERENCE; no labels)
  R4 source + target-unlabeled gauge         (needs the same re-inference)
  R5 few-label target calibration diagnostic (read-only; NON-DG supervised calibration diagnostic)
  R6 target-centered / target-rank ORACLE ceiling (read-only)

STAGED (per PM): read-only rungs (R0/R1/R2/R5/R6 + witnesses + taxonomy shell) build now; R3/R4 are NOT proxied
from method-final target logits (wrong population) and are NOT left permanently PENDING -- they are completed
by a scoped NO-RETRAINING target-audit re-inference behind a P0 replay-identity smoke gate. C24 is NOT
finalized while R3/R4 are absent. NOT a selector / OACI rescue / DG method. Frozen C19 hash unchanged.
"""
from __future__ import annotations

from ..score_gauge import schema as sg

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = sg.LOCKED_C19_CONFIG_HASH          # "664007686afb520f"
C22_SCORE_SIDECAR = sg.C22_SCORE_SIDECAR

# ---- committed artifact roots (READ-ONLY; used for availability probing, never mutated) ----------------
LOSO_ARTIFACT_ROOT = "/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012"   # frozen checkpoints + method-final target_audit.npz
C10_REPLAY_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c10-replay"          # source signals + target SCALAR labels + target_pred_hash
C18_EXTRACT_DIR = "/projects/EEG-foundation-model/yinghao/oaci-c18-extract"        # per-candidate SOURCE logits only
# Stage-3 re-inference writes here (per-candidate target-UNLABELED logits/summaries); Stage-1 only checks presence
C24_TARGET_REINFER_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c24-target-unlabeled.json"

# ---- information ladder (fixed, pre-registered) -------------------------------------------------------
# feasible_readonly=False rungs carry a reason code and a status; they are NOT proxied and NOT skipped.
R0, R1, R2, R3, R4, R5, R6 = "R0", "R1", "R2", "R3", "R4", "R5", "R6"
RUNGS = (
    {"rung": R0, "name": "raw_pooled", "info_class": "none",
     "deployable": True, "needs_target_inputs": False, "needs_target_labels": False, "feasible_readonly": True},
    {"rung": R1, "name": "source_only_gauge", "info_class": "source_only",
     "deployable": True, "needs_target_inputs": False, "needs_target_labels": False, "feasible_readonly": True},
    {"rung": R2, "name": "source_risk_static_gauge", "info_class": "source_only",
     "deployable": True, "needs_target_inputs": False, "needs_target_labels": False, "feasible_readonly": True},
    {"rung": R3, "name": "target_unlabeled_gauge", "info_class": "target_unlabeled_transductive",
     "deployable": False, "needs_target_inputs": True, "needs_target_labels": False, "feasible_readonly": False,
     "reason": "per_candidate_target_logits_not_cached_requires_no_retraining_target_audit_reinference"},
    {"rung": R4, "name": "source_plus_target_unlabeled_gauge", "info_class": "target_unlabeled_transductive",
     "deployable": False, "needs_target_inputs": True, "needs_target_labels": False, "feasible_readonly": False,
     "reason": "per_candidate_target_logits_not_cached_requires_no_retraining_target_audit_reinference"},
    {"rung": R5, "name": "few_label_target_calibration", "info_class": "target_labeled_supervised_calibration",
     "deployable": False, "needs_target_inputs": True, "needs_target_labels": True, "feasible_readonly": True},
    {"rung": R6, "name": "target_centered_rank_oracle", "info_class": "target_identity_oracle",
     "deployable": False, "needs_target_inputs": True, "needs_target_labels": False, "feasible_readonly": True},
)

# re-inference rung status codes
STATUS_OK = "computed"
STATUS_REQUIRES_REINFERENCE = "REQUIRES_REINFERENCE"        # blocked; NOT proxied, NOT finalized
STATUS_BLOCKED_NO_ARTIFACT = "blocked_no_target_unlabeled_artifact"

# ---- offset model (inherit C23: fixed ridge, LOTO, no grid/feature-selection) -------------------------
RIDGE_L2 = sg.RIDGE_L2                                       # 1.0
N_PERM = sg.N_PERM                                           # 200
PERM_SEED = sg.PERM_SEED                                     # 707

# ---- source-only non-identifiability witnesses (C24-A) ------------------------------------------------
WITNESS_UNIT = "target_regime"                              # units = (target, regime) in in_regime mode
WITNESS_NEAR_QUANTILE = 0.15                                # source-summary distance in bottom 15% = "near-identical"
WITNESS_FAR_OFFSET_QUANTILE = 0.85                          # offset difference in top 15% = "divergent offset"
WITNESS_TOP_K = 25                                          # top witness pairs to emit
# Mantel-style identifiability: does source-summary distance PREDICT offset distance across unit pairs? If the
# correlation is weak/insignificant, source summaries are NON-IDENTIFYING for the offset (the C24-A verdict).
MANTEL_IDENTIFY_CORR = 0.30                                 # |corr(source_dist, offset_dist)| >= this AND sig => source identifies offset

# ---- few-label calibration diagnostic (R5) ------------------------------------------------------------
FEW_LABEL_BUDGETS = (0, 1, 2, 4, 8)                         # labeled candidates PER CLASS revealed per target
FEW_LABEL_NOTE = ("NON-DG supervised target-calibration diagnostic; reveals competence labels for a few of the "
                  "held-out target's own candidates to estimate its offset. Not deployment, not a selector.")

# ---- conservative success (inherit C23) ---------------------------------------------------------------
SUCCESS_AUC_IMPROVE = sg.SUCCESS_AUC_IMPROVE                # 0.03
SUCCESS_GAP_CLOSED = sg.SUCCESS_GAP_CLOSED                  # 0.40
IDENTITY_LEAKAGE_CHANCE = sg.IDENTITY_LEAKAGE_CHANCE
IDENTITY_LEAKAGE_CEILING = sg.IDENTITY_LEAKAGE_CEILING

# ---- FORBIDDEN inputs for R3/R4 target-unlabeled features (extends C23 with target-label endpoints) ----
FORBIDDEN_TARGET_UNLABELED_INPUTS = (
    "target_label", "target_bacc", "target_nll", "target_ece", "target_worst", "target_center",
    "target_rank", "target_zscore", "target_id", "target_identity", "source_subject", "loso_complement",
    "score", "label", "y",              # never the answer / never the offset itself
)

# ---- taxonomy (pre-registered information-boundary cases) ----------------------------------------------
I1 = "I1_source_only_nonidentifiable"
I2 = "I2_unlabeled_target_recovers_offset"
I3 = "I3_unlabeled_target_insufficient"
I4 = "I4_few_labels_recover_offset"
I5 = "I5_label_hungry_offset"
I6 = "I6_oracle_only_boundary"
I7 = "I7_identity_leakage_artifact"
ALL_CASES = (I1, I2, I3, I4, I5, I6, I7)

FEW_LABEL_RECOVERS_MAX_K = 4                                # I4 fires if gap closes >= SUCCESS_GAP_CLOSED by k<=4/class

FORBIDDEN_CLAIM_SUBSTRINGS = sg.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "target-unlabeled selector", "dg success", "generalization succeeded",
    "target-unlabeled is deployable", "few-label selector",
)


def frozen_config_hash() -> str:
    return sg.frozen_config_hash()
