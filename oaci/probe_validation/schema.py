"""C20 — Frozen-probe new-regime validation. Validates the C19 robust-core diagnostic probe (frozen, config
hash 664007686afb520f) by CROSS-REGIME leave-one-target-out: train the FIXED probe on the C19 development
regimes (S0/S2/S3) from non-held-out targets, evaluate on the held-out target in HELD-OUT support-DELETION
regimes (S4/S5/S6/S7). This is external/new-regime validation, NOT a new method: nothing about the probe
(features, model, regularization, normalization, permutation family, thresholds) may change vs C19.

C20-A = this cross-regime validation. C20-B = an external-dataset PROTOCOL DOCUMENT only (no execution, no
second dataset, no BNCI2014_004 unbar). DIAGNOSTIC-ONLY; no selector produced.
"""
from __future__ import annotations

from ..competence_probe import schema as c19

DIAGNOSTIC_ONLY = True
NON_DEPLOYABLE = True

# the C19 config this validation is LOCKED to (asserted byte-for-byte by frozen_config / feature_lock)
LOCKED_C19_CONFIG_HASH = "664007686afb520f"

# development regimes = C19's success regimes (the probe was pre-registered on these)
DEVELOPMENT_REGIMES = ("S0_full_support", "S2_rare_cells", "S3_nonestimable_cells")
# held-out validation regimes = the cell-DELETION regimes C18 flagged for accuracy-endpoint non-estimability
HELD_OUT_REGIMES = ("S4_missing_cells", "S5_block_class_by_domain", "S6_boundary_aligned_mask",
                    "S7_random_matched_mask")
NOOP_REGIME = "S1_label_marginal_skew"          # implemented-noop negative-control only; not a stress

# frozen success criterion (same OR stricter than C19): held-out-regime AUC beats permutation AND margin vs
# STRICT chance 0.5 >= 0.03 (stricter than C19's margin-vs-empirical-null, chosen because C19 disclosed a
# sub-0.5 null; C20 must clear real chance).
SUCCESS_P = c19.SUCCESS_P                         # 0.05
SUCCESS_AUC_MARGIN_VS_CHANCE = 0.03              # vs 0.5, not vs empirical null
HETEROGENEITY_SPREAD = c19.HETEROGENEITY_SPREAD  # 0.35
N_PERM = c19.N_PERM                              # 200
PERM_SEED = c19.PERM_SEED                        # 707

# ---- C20 case taxonomy -----------------------------------------------------------------------------
# Headline label is deliberately CONSERVATIVE: a mixed result where only some held-out regimes marginally
# clear the strict chance bar is NOT "partial success" -- the primary verdict is that broad external
# generalization is NOT established, with the passing regimes reported as marginal secondary exceptions.
CASE_GENERALIZES = "frozen_robust_core_generalizes_to_heldout_regimes"
CASE_LARGELY_REGIME_LOCAL = "largely_regime_local_with_marginal_exceptions"      # the mixed/marginal case
CASE_SURVIVES_NONDELETION = "survives_block_rare_but_not_deletion"
CASE_REGIME_LOCAL = "c19_positive_is_regime_local_no_external_generalization"    # 0 held-out pass
CASE_AVAILABILITY_LIMITED = "validity_limited_by_feature_availability_not_relationship"
ALL_CASES = (CASE_GENERALIZES, CASE_LARGELY_REGIME_LOCAL, CASE_SURVIVES_NONDELETION, CASE_REGIME_LOCAL,
             CASE_AVAILABILITY_LIMITED)
AVAILABILITY_FLOOR = 0.5      # robust-core scored_rate below this on failing regimes -> availability-limited

# layered verdict vocabulary (primary / secondary / failure-mode / availability / strength)
PRIMARY_NOT_ESTABLISHED = "external_new_regime_generalization_not_established"
PRIMARY_ESTABLISHED = "external_new_regime_generalization_established"
FAILURE_RELATIONSHIP = "relationship_level_regime_shift_not_feature_availability"
FAILURE_AVAILABILITY = "feature_availability_collapse"
AVAILABILITY_OK = "robust_core_available_all_heldout_regimes"
CLAIM_WEAK_DIAGNOSTIC = "weak_diagnostic_only"

# a regime is "deletion" (bAcc-endpoint fragile) vs "non-deletion" for the taxonomy split
DELETION_HELD_OUT = ("S4_missing_cells", "S6_boundary_aligned_mask", "S7_random_matched_mask")
NONDELETION_HELD_OUT = ("S5_block_class_by_domain",)

FORBIDDEN_CLAIM_SUBSTRINGS = c19.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "detector is validated", "we deploy", "production selector", "generalizes as a selector",
)


def robust_core_features() -> tuple:
    return c19.ROBUST_CORE_FEATURES


def endpoint_features() -> tuple:
    return c19.ENDPOINT_FEATURES


def diagnostic_label() -> str:
    return c19.DIAGNOSTIC_LABEL
