"""C86LP immutable constants — arithmetic, frozen gate identities, claim boundary.

These are *metadata* facts about the future C86L development-only field, copied
from the accepted C86R2 program (effective manifest V3) and the C84S field.  This
module opens no EEG, no labels, no predictions, no Q0 shards, and no C85U utility
values.  The numbers are checked for internal consistency in the tests so a typo
cannot silently corrupt the field arithmetic.
"""
from __future__ import annotations

# --- Field arithmetic (immutable C84 construction/evaluation scope) -----------
TARGET_SUBJECTS = 118
CONTEXTS = 944
CANDIDATES_PER_CONTEXT = 81
TOTAL_TARGET_TRIALS = 9_621
HELD_EVAL_ROWS = 4_848
CONSTRUCTION_ROWS = 4_773
CONTEXTS_PER_CONSTRUCTION_TRIAL = 8
CONTEXT_TRIAL_ROWS = 38_184           # 4_773 * 8
CANDIDATE_TRIAL_CONTEXT_ROWS = 3_092_904  # 38_184 * 81
BINARY_PROBABILITY_SCALARS = 6_185_808    # 3_092_904 * 2

DATASET_CONSTRUCTION_ROWS = {
    "Lee2019_MI": 1_100,
    "Cho2017": 2_000,
    "PhysionetMI": 1_673,
}  # sum == CONSTRUCTION_ROWS

# --- Total-query availability -------------------------------------------------
BUDGET_GRID: tuple[int | str, ...] = (4, 8, 16, 32, "FULL")
TARGET_BUDGET_CELLS = 590             # 118 * 5
AVAILABLE_CELLS = 514
UNAVAILABLE_CELLS = 76               # PhysionetMI B32 cells
UNSUPPORTED_BUDGET_DISPOSITION = "INPUT_UNAVAILABLE"  # no replacement / no substitution / no deletion

# --- Historical C84S replay counts (bound, not recomputed here) ---------------
Q0_FINITE_ACTION_RECORDS = 8_749_056
Q0_DETERMINISTIC_FULL = 944
Q0_TOTAL = 8_750_000

# --- Frozen upstream identities (opaque; bound only, never rehashed here) ------
FROZEN_INPUT_SHA = {
    "c86_effective_program_v3": "c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e",
    "c84f_complete_field_manifest": "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
    "c84f_target_trial_registry": "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
    "c84s_v5_analysis_lock": "030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846",
    "c84s_selection_freeze": "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
    "c85u_acceptance_manifest": "dfcf84569beb1b34b786cbe72233a22fd3928a4475b7e345f23b40cdb6671620",
}

# --- Frozen scientific gates C86L MUST NOT change -----------------------------
FROZEN_GATES = {
    "C84_primary": "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous",
    "C84_label_frontier": "C84-L4",
    "T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED", "T4": "PROVED",
    "T5": "OPEN", "T6": "COUNTEREXAMPLE", "T7": "PROVED",
}

# --- Claim boundary -----------------------------------------------------------
# Registered LINEAR moments: LURE gives an unbiased estimate ONLY for these,
# under the locked positivity/sampling assumptions.  Everything a policy actually
# cares about downstream is a NONLINEAR plugin with NO unbiasedness claim.
LINEAR_MOMENTS = frozenset({
    "nll_sum",
    "correct_count",
    "class_correct_numerator",
    "signed_calibration_contribution",
    "pairwise_nll_difference",
})
NONLINEAR_PLUGINS = frozenset({
    "balanced_accuracy",
    "ece",
    "candidate_midrank",
    "composite_utility",
    "selected_action",
    "target_regret",
})

# --- Development-only boundary (authorization, deliberately lightweight) -------
# C86LP opens no real protected payload; every test here runs on synthetic shadow
# objects.  Per the PM's granted latitude, the heavyweight C86L execution-lock /
# single-use-authorization-receipt / atomic-final-rename apparatus is intentionally
# NOT reproduced: with no real data opened it gates no real risk.  What survives is
# the isolation contract that actually protects a future result from contamination.
DEVELOPMENT_ONLY_BOUNDARY = (
    "development_only_shadow: opens no real C84 construction labels, target "
    "predictions, Q0 shards, C84S result tables, or C85U utility values"
)

# Isolation is LOGICAL/API only at C86LP: the three stores live in one Python
# object and the server holds it via name mangling.  Real C86L would use separate
# processes and filesystem roots.  Do not call C86LP "physical isolation".
ISOLATION_LEVEL = "logical_api_mock"

# --- Gate (corrected) ---------------------------------------------------------
# The prior gate ("...IMPLEMENTED_AND_LOCKED_READY_FOR_PI_AUTHORIZATION") was
# semantically false once the lock/authorization apparatus was dropped.  What
# actually exists is a shadow instrument whose probe criteria need PM review.
GATE_INSTRUMENT = "C86LP_SHADOW_QUERY_INSTRUMENT_IMPLEMENTED_PROBE_CRITERIA_REVIEW_REQUIRED"
GATE_RECONCILIATION = (
    "C86L_CONSTRUCTION_VIEW_PREDICTION_ALIGNMENT_QUERY_INTERFACE_OR_"
    "PROVENANCE_RECONCILIATION_REQUIRED"
)

# --- Pre-registered probe taxonomy (boundary decision rules) ------------------
# Applied to a FUTURE real-data probe; frozen here BEFORE any real query.  The
# last outcome is explicitly NOT a universal information-impossibility claim.
BOUNDARY_TAXONOMY = (
    "BOUNDARY_OPERATIONALLY_CROSSED",       # same active policy, same total budget, both cohorts:
                                            #   improves mean regret AND tail risk AND near-optimal prob
    "BOUNDARY_WEAKENED_NOT_ROBUST",         # mean/near-opt improve but tail/cohort/composition gate fails
    "POLICY_LIMITED",                       # ceiling shows info is cheaply exploitable, registered policy can't
    "ACQUISITION_VIEW_NONTRANSPORTABLE",    # even FULL construction info -> weak/heterogeneous held actionability
    "NO_REGISTERED_ACTIVE_GAIN",            # no registered active policy beats passive P0 (NOT impossibility)
)

# Operational thresholds for the taxonomy classifier (pre-registered).
TAU_CEILING_REGRET = 0.05     # FULL-budget mean regret at/under this => acquisition view transports
TAU_REGRET_MARGIN = 0.02      # an "improvement" must beat the comparator by at least this

CONFIDENCE_BINS = 15
PROB_FLOOR = 1e-7
