"""C18 — Controlled Support-Mismatch x Identifiability Stress Test: shared contract.

C18 asks: when domain x class support is degraded in a controlled way, what happens to the
weak multivariate SOURCE-side competence information found in C17 (Case III)? It is an
identifiability stress test, NOT a new OACI/SRC experiment and NOT a retraining study.

Three layers (all NO-RETRAINING):
  C18-R  artifact-level REAL mask recompute on committed selected-checkpoint per-sample
         predictions + support graphs        -> H3 (calibration-vs-accuracy visibility),
                                                 H4 (class-boundary S6-vs-S7), H5 (leakage abstention)
  C18-P  GPU re-inference of the C17 CANDIDATE checkpoints (forward source splits only,
         extract per-sample logits/features, mask-recompute under S0-S7)  -> H1, H2
  C18-D  observability-dropout SECONDARY appendix (labeled: stresses observability of the
         source signals, NOT the source distribution)

Every target-derived quantity is diagnostic-only and non-deployable. No selector is emitted.
"""
from __future__ import annotations

DIAGNOSTIC_ONLY = True
NON_DEPLOYABLE = True

# ---- stress regimes (deterministic; severity-ordered) --------------------------------------------
# S6 (boundary-aligned) and S7 (random-matched) MUST match in severity — that is the key contrast.
REGIME_ORDER = ("S0_full_support", "S1_label_marginal_skew", "S2_rare_cells", "S3_nonestimable_cells",
                "S4_missing_cells", "S5_block_class_by_domain", "S6_boundary_aligned_mask",
                "S7_random_matched_mask")
REGIME_SEVERITY = {"S0_full_support": 0, "S1_label_marginal_skew": 1, "S2_rare_cells": 2,
                   "S3_nonestimable_cells": 3, "S4_missing_cells": 4, "S5_block_class_by_domain": 4,
                   "S6_boundary_aligned_mask": 3, "S7_random_matched_mask": 3}
REGIME_DESC = {
    "S0_full_support": "original C17 setting; no source cells masked (exact reproduction baseline).",
    "S1_label_marginal_skew": "class marginals skewed within source domains; all domain x class cells remain estimable.",
    "S2_rare_cells": "selected source domain x class cells downweighted to just above the estimability threshold.",
    "S3_nonestimable_cells": "selected cells reduced BELOW the estimability threshold (present but n<m), not deleted.",
    "S4_missing_cells": "selected source domain x class cells removed entirely.",
    "S5_block_class_by_domain": "one class made rare/non-estimable across a subset of source domains.",
    "S6_boundary_aligned_mask": "perturb cells aligned with C17's class-boundary rotation classes.",
    "S7_random_matched_mask": "random support mask matched to S6 in severity (negative/control comparison).",
}
# regimes that DELETE or make cells non-estimable at the SUPPORT-GRAPH level (vs pure reweighting)
REGIME_DELETES = {"S0_full_support": False, "S1_label_marginal_skew": False, "S2_rare_cells": False,
                  "S3_nonestimable_cells": True, "S4_missing_cells": True, "S5_block_class_by_domain": True,
                  "S6_boundary_aligned_mask": True, "S7_random_matched_mask": True}

# ---- C17 source-feature classification for mask stress -------------------------------------------
# Only RECOMPUTABLE_UNDER_MASK features carry S1-S7 support-stress claims. C18's mask is an EVALUATION-
# support mask (no retraining): we recompute source observables on masked source_guard/source_audit eval
# splits + masked support graphs. So the worst-domain EVAL endpoints + leakage points recompute; but
# R_src / balanced_err are TRAINING-REALIZED scalars (rec.R_src, rec.balanced_err) on the actual training
# support and do NOT change under an eval-support mask -> STATIC (with train_surrogate/epoch). They appear
# in the S0 all-column identity probe (to reproduce C17's 0.602) but are EXCLUDED from S1-S7 mask claims;
# carrying them across masks as if masked would recreate the fake-per-regime-probe problem.
RECOMPUTABLE_UNDER_MASK = ("source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece",
                           "source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
                           "selection_leakage_point", "audit_leakage_point")
STATIC_TRAINING_LOG_ONLY = ("R_src", "balanced_err", "train_surrogate", "epoch")   # training-realized / objective / meta
NOT_RECONSTRUCTABLE = ()                                            # (none identified; kept for honesty/audit)
FEATURE_CLASS = ({f: "recomputable_under_mask" for f in RECOMPUTABLE_UNDER_MASK}
                 | {f: "static_training_log_only" for f in STATIC_TRAINING_LOG_ONLY}
                 | {f: "not_reconstructable" for f in NOT_RECONSTRUCTABLE})

# ---- hard gates (predeclared tolerances) ---------------------------------------------------------
IDENTITY_LOGIT_TOL = 1e-9          # selected ckpt re-forward vs stored .npz (argmax parity + max|dlogit|<tol)
S0_BACC_TOL = 5e-3                 # recomputed S0 worst-domain bAcc vs persisted C10 (argmax-stable; cross-node FP)
S0_NLL_TOL = 1e-2                  # recomputed S0 worst-domain NLL / risk vs persisted C10
S0_LEAK_TOL = 5e-2                 # recomputed S0 leakage point vs persisted C10 (probe RNG + cross-node FP)
G2_AUC_TOL = 1e-2                  # S0 LOTO AUC vs C17 0.602
G3_RHO_TOL = 2e-2                  # S0 within-fold rho vs C17 (e.g. source_audit_worst_bacc ~ +0.120)
C17_LOTO_AUC = 0.6023104389834069
C17_ORACLE_SPEARMAN_BACC = 0.119523893328065
C17_CLASS_BOUNDARY_CORR = 0.547

# ---- support-severity response taxonomy (outcomes) ----------------------------------------------
CASE_PRESERVED = "case_III_preserved"
CASE_COLLAPSED_II = "collapsed_to_case_II_calibration_only_signal"      # SIGNAL-level: even cell-present stress fails
CASE_ENDPOINT_NONESTIMABILITY = "collapsed_by_accuracy_endpoint_nonestimability"  # bAcc->NaN under DELETION, not signal
CASE_COLLAPSED_IV = "collapsed_to_case_IV_source_unidentifiable"
CASE_BOUNDARY_DESTROYED = "boundary_visibility_destroyed"
CASE_ABSTENTION_DOMINANT = "support_abstention_dominant"
CASE_INCONCLUSIVE = "support_stress_inconclusive_due_to_feature_loss"
TAXONOMY_INTERPRETATION = {
    CASE_PRESERVED: "weak source-side competence information is not merely a full-support artifact; it survives support degradation that keeps cells present.",
    CASE_COLLAPSED_II: "even support stress that keeps cells present collapses accuracy identifiability while calibration stays source-visible -> a genuine signal-level calibration bias.",
    CASE_ENDPOINT_NONESTIMABILITY: "the weak signal SURVIVES cell-present stress (rare/nonestimable); it collapses only under cell DELETION, and there because the worst-domain accuracy ENDPOINT becomes non-estimable (a domain loses a class -> reference bAcc NaN), not because the model signal vanished. Support deletion destroys accuracy-observable availability before it forces leakage abstention.",
    CASE_COLLAPSED_IV: "support degradation destroys source-side observability of target accuracy entirely (even cell-present stress, with accuracy features still computable).",
    CASE_BOUNDARY_DESTROYED: "source-visible class-boundary structure depends on support coverage of boundary-relevant cells.",
    CASE_ABSTENTION_DOMINANT: "the measurement framework refuses to hallucinate unsupported conditional invariance; it abstains rather than smoothing.",
    CASE_INCONCLUSIVE: "estimability loss removed too many source signals to decide; not a scientific collapse.",
}
ALL_CASES = (CASE_PRESERVED, CASE_COLLAPSED_II, CASE_ENDPOINT_NONESTIMABILITY, CASE_COLLAPSED_IV,
             CASE_BOUNDARY_DESTROYED, CASE_ABSTENTION_DOMINANT, CASE_INCONCLUSIVE)

# ---- per-regime collapse-reason enum (reason-code BEFORE taxonomy; free text is forbidden) -------
COLLAPSE_REASONS = ("none", "implemented_noop", "signal_loss", "endpoint_metric_nonestimability",
                    "leakage_nonestimability", "feature_engineering_bug", "insufficient_rows", "inconclusive")

# ---- allowed / forbidden claim guards (asserted by report + tests) ------------------------------
FORBIDDEN_CLAIM_SUBSTRINGS = (
    "support mismatch caused the original oaci failure", "bnci001 naturally demonstrates support mismatch",
    "target-free competence detector is validated", "oaci is rescued", "target oracle is deployable",
    "a new selector is validated", "all dg fails", "eeg transfer is impossible",
    "support-aware invariance is useless",
)


def regime_is_deleting(regime: str) -> bool:
    return REGIME_DELETES[regime]


def feature_class(name: str) -> str:
    return FEATURE_CLASS.get(name, "not_reconstructable")
