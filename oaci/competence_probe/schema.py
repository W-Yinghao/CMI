"""C19 — Pre-registered Low-Freedom Source-Only Competence Probe with Endpoint-Estimability Gating.

DIAGNOSTIC-ONLY. This is NOT a target-free selector, NOT an OACI rescue, NOT a deployable competence detector.
It tests whether a FIXED, low-capacity, pre-registered probe recovers the weak source-only competence
information found in C17/C18 using DELETION-ROBUST source observables (confidence geometry + calibration +
leakage), while explicitly reason-coding cases where source accuracy ENDPOINTS become non-estimable.

Pre-registration is binding: the feature registry, model class, regularization, validation, permutation
scheme, and success criteria below are FROZEN before any fit and are asserted to match the executed config.
No grid search, no feature selection, no post-hoc tuning. Target labels are post-hoc diagnostic labels only.
"""
from __future__ import annotations

DIAGNOSTIC_ONLY = True
NON_DEPLOYABLE = True

# ---- FROZEN feature registry -----------------------------------------------------------------------
# ROBUST-CORE: source-only observables that stay estimable under cell deletion (aggregate over whatever
# source units remain; never NaN like reference bAcc). Confidence geometry + calibration + leakage.
_CONF_GEOM = ("nll", "ece", "entropy", "confidence", "margin", "logit_norm", "conf_on_wrong")
ROBUST_CORE_FEATURES = tuple(f"source_{role}_{s}" for role in ("guard", "audit") for s in _CONF_GEOM) \
    + ("selection_leakage_point", "audit_leakage_point")           # 7*2 + 2 = 16 robust-core features
# FRAGILE accuracy endpoints: worst-domain reference bAcc — estimable ONLY when no domain misses a class.
# SECONDARY (endpoint-augmented) probe ONLY, and only where estimable; never in the primary probe.
ENDPOINT_FEATURES = ("source_guard_worst_bacc", "source_audit_worst_bacc")
# STATIC training-realized / meta — EXCLUDED entirely (an eval-support mask cannot change them; C18 lesson).
STATIC_EXCLUDED = ("R_src", "balanced_err", "train_surrogate", "epoch")

FEATURE_FAMILY = ({f: "robust_core_confidence_geometry" for f in ROBUST_CORE_FEATURES if not f.endswith("leakage_point")}
                  | {f: "robust_core_leakage" for f in ROBUST_CORE_FEATURES if f.endswith("leakage_point")}
                  | {f: "fragile_accuracy_endpoint" for f in ENDPOINT_FEATURES}
                  | {f: "static_training_log" for f in STATIC_EXCLUDED})

# ---- FROZEN probe config ---------------------------------------------------------------------------
PROBE_MODEL = "l2_logistic"          # fixed model class
PROBE_L2_C = 1.0                     # fixed regularization (NO grid search)
PROBE_STANDARDIZE = True             # per-fold standardization (fit on train only)
PROBE_ITERS = 800                    # fixed optimizer budget
PROBE_LR = 0.3
VALIDATION = "leave_one_target_out"  # LOTO; LOSO as sensitivity
N_PERM = 200                         # fixed within-(seed,target,level) permutation count
PERM_SEED = 707
DIAGNOSTIC_LABEL = "tgt__target_bacc_good"   # post-hoc diagnostic label only

# ---- regimes the primary/robustness probe is scored on (cell-present only; deletion regimes are the
# ---- endpoint-nonestimability domain, reported separately) -----------------------------------------
PRIMARY_REGIME = "S0_full_support"
ROBUSTNESS_REGIMES = ("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells")

# ---- FROZEN success criteria (pre-registered) ------------------------------------------------------
SUCCESS_P = 0.05                     # LOTO beats within-fold permutation at p < 0.05
SUCCESS_AUC_MARGIN = 0.03            # AND loto_auc - permutation_mean >= 0.03
SUCCESS_REGIMES = ("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells")  # must hold on S0 + cell-present
HETEROGENEITY_SPREAD = 0.35          # per-target AUC spread above this -> "weak diagnostic, heterogeneous" (not a detector)

# ---- abstention / score-status enum ----------------------------------------------------------------
SCORE_STATUS = ("scored", "abstained_source_accuracy_endpoint_nonestimable",
                "abstained_insufficient_finite_features", "abstained_constant_or_degenerate_fold")

# ---- C19 case taxonomy -----------------------------------------------------------------------------
CASE_ROBUST_CORE_RECOVERS = "robust_core_recovers_weak_competence"
CASE_ENDPOINT_AUGMENTED_ONLY = "endpoint_augmented_only_recovers"
CASE_NOT_RECOVERABLE = "weak_signal_not_recoverable_under_preregistration"
CASE_HETEROGENEOUS = "weak_diagnostic_but_per_target_heterogeneous"
ALL_CASES = (CASE_ROBUST_CORE_RECOVERS, CASE_ENDPOINT_AUGMENTED_ONLY, CASE_NOT_RECOVERABLE, CASE_HETEROGENEOUS)

# ---- forbidden claim guard -------------------------------------------------------------------------
FORBIDDEN_CLAIM_SUBSTRINGS = (
    "deployable target-free selector", "deployable selector", "validates a selector", "oaci is rescued",
    "target oracle is deployable", "support mismatch caused the original oaci failure", "all dg fails",
    "eeg transfer is impossible", "support-aware invariance is useless", "we built a selector",
    "we found a selector", "competence detector is validated",
)


def feature_family(name: str) -> str:
    return FEATURE_FAMILY.get(name, "unknown")


def frozen_config() -> dict:
    """The pre-registered config, hashed into the report so execution can be asserted to match."""
    return {"robust_core_features": list(ROBUST_CORE_FEATURES), "endpoint_features": list(ENDPOINT_FEATURES),
            "static_excluded": list(STATIC_EXCLUDED), "model": PROBE_MODEL, "l2_C": PROBE_L2_C,
            "standardize": PROBE_STANDARDIZE, "iters": PROBE_ITERS, "lr": PROBE_LR, "validation": VALIDATION,
            "n_perm": N_PERM, "perm_seed": PERM_SEED, "diagnostic_label": DIAGNOSTIC_LABEL,
            "robustness_regimes": list(ROBUSTNESS_REGIMES), "success_p": SUCCESS_P,
            "success_auc_margin": SUCCESS_AUC_MARGIN, "success_regimes": list(SUCCESS_REGIMES)}
