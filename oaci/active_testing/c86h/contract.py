"""C86H executable bindings (content-addressed) — §12 of C86H_CONTRACT.md.

This module holds ONLY the pre-registered identities and locked numeric constants for
the terminal untouched-confirmation. It touches no real EEG/label data. Every method is
the FROZEN C86D production dispatcher (imported by the entrypoint, never copied here).
``verify_bindings`` fail-closes on any content-address or arithmetic drift.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

_REPO = Path(__file__).resolve().parents[3]
_TABLES = _REPO / "oaci" / "reports" / "c86r2_tables"
_DISPATCH = _REPO / "oaci" / "active_testing" / "c86d"

# --- 12.1 authoritative identity ---------------------------------------------------
EFFECTIVE_PROGRAM_V3_SHA256 = (
    "c6b7e490e0f78f74f820428cee138782caff1dc0033422723593a7d8e3c5f77e"
)
C86D_DISPATCHER_COMMIT = "c694315e"
REGISTRY_TABLE_SHA256 = {
    "final_adult_untouched_cohort_registry_v3.csv":
        "82f329a1125a8ffe106c22ad490589aef84c239077105c22e6301d2e39593737",
    "common_field_interface_v3.csv":
        "2e22863fbc447054d196376a48340e09192d99310c01639b67b51879019c99b4",
}
# content sha256 of the frozen dispatcher files at commit c694315e
DISPATCHER_FILE_SHA256 = {
    "policies.py":    "58fb1fc1a5482cc7320f7db13d113156fc9d5fcb7f83cd69f2871c7b4eb1dbc5",
    "core.py":        "a0d06a6648f3d8740a81ab9a3194e4004441b6671cfae853ee9660c0ae1fe649",
    "c85u_config.py": "0793bc6d07694452f3f7bbcfe884e2919d6aef3c7cebe84efdc274f83d92522d",
}
METHOD_REGISTRY = ("P0", "A1", "A2H")   # no add, no delete
ACTIVE_METHODS = ("A1", "A2H")          # realized max-T family = ACTIVE_METHODS x FINITE_BUDGETS

COMMON_INTERFACE_ID = "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3"
INTERFACE_CHANNELS = ("FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4",
                      "CP5", "CP1", "CP2", "CP6")
INTERFACE_SFREQ_HZ = 160
INTERFACE_WINDOW_S = (0.0, 3.0)
INTERFACE_BAND_HZ = (4.0, 38.0)
INTERFACE_EVENTS = ("left_hand", "right_hand")

# --- population (§2 / 12.5) --------------------------------------------------------
COHORTS = {
    "Brandl2020_CANONICAL_ADULT_V1": {
        "native": "Brandl2020", "n_adult": 16,
        "subjects": tuple(str(i) for i in range(1, 17)),
        "min_trials": 504, "license": "CC-BY-NC-ND-4.0",
    },
    "OpenNeuro_ds007221_HYBRID_ADULT_V1": {
        "native": "OpenNeuro_ds007221", "n_adult": 37,
        "subjects": tuple(f"sub-{i}" for i in range(37, 74)),
        "task": "hybrid", "min_trials": 600, "license": "CC0",
    },
}
N_TARGETS = 53

# --- 12.2 candidate field ----------------------------------------------------------
PANELS = ("A", "B")
TRAINING_SEEDS = (5, 6)
LEVELS = (0, 1)
CANDIDATES_PER_CONTEXT = 81               # 1 ERM + 40 OACI + 40 SRC
CANDIDATE_COMPOSITION = {"ERM": 1, "OACI": 40, "SRC": 40}
UNIQUE_TRAINED_MODELS = 648               # 2 * 2 * 2 * 81
CONTEXTS_PER_TARGET = 8                    # 2 panels * 2 seeds * 2 levels
TARGET_CONTEXTS = 424                      # 53 * 8

# --- 12.3 label-blind split --------------------------------------------------------
SPLIT_SALT = "C86_TARGET_SPLIT_V1"
MIN_TRIALS = 80
MIN_POOL = 40
MIN_EVAL = 40
MIN_CLASS_SUPPORT = 8                      # labels / class / view, post-access

# --- 12.4 budgets / endpoints ------------------------------------------------------
BUDGET_GRID = (4, 8, 16, 32, "FULL")
FINITE_BUDGETS = (4, 8, 16, 32)
EPSILON_GRID = (0.005, 0.01, 0.02, 0.05)
PRIMARY_EPSILON = 0.05
CVAR_ALPHA_GRID = (0.50, 0.75, 0.90)
PRIMARY_CVAR_ALPHA = 0.90

# --- 11 confirmatory thresholds (registered; NOT the C86D development TAU=0.02) -----
MATERIALITY_MARGIN = 0.05
FAMILYWISE_ALPHA = 0.05
MAXT_DRAWS = 65_536
FAVORABLE_TARGET_FRACTION = 0.75
WORST_TARGET_EFFECT_FLOOR = -0.10
POSITIVE_CELLS_MIN = 6                     # of 8 panel x seed x level
TAIL_CVAR90_MARGIN = 0.05
LOTO_PRESERVATION_MIN = 0.75
POOLED_DATASET_PVALUE = "FORBIDDEN"

# --- 5 numerical integration -------------------------------------------------------
ACTIVE_CHAINS = 2_048                      # locked confirmation program (NOT the 8 dev chains)

# --- 10 taxonomy -------------------------------------------------------------------
FORMAL_GATE = ("C86-A", "C86-B", "C86-C", "C86-D", "C86-E")
LABEL_FRONTIER = ("C86-L1", "C86-L2", "C86-L3", "C86-L4")
INTERPRETIVE_DESCRIPTOR = (
    "BOUNDARY_OPERATIONALLY_CROSSED", "BOUNDARY_WEAKENED_NOT_ROBUST",
    "POLICY_LIMITED", "ACQUISITION_VIEW_NONTRANSPORTABLE", "NO_REGISTERED_ACTIVE_GAIN",
)
# This contract defines NO oracle-acquisition diagnostic, so POLICY_LIMITED is fixed and
# is never inferred from ordinary P0/A1/A2H outcomes:
ORACLE_ACQUISITION_DIAGNOSTIC = None
POLICY_LIMITED_RESOLUTION = "NOT_IDENTIFIABLE_IN_C86H"


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_bindings() -> dict:
    """Fail-closed content-address + arithmetic check of every §12 binding.

    Reads only the registered tables and the frozen dispatcher source; opens no EEG,
    no labels, no target predictions.
    """
    mismatches = []
    for name, want in REGISTRY_TABLE_SHA256.items():
        got = _sha256_file(_TABLES / name)
        if got != want:
            mismatches.append(f"registry table {name}: {got} != {want}")
    for name, want in DISPATCHER_FILE_SHA256.items():
        got = _sha256_file(_DISPATCH / name)
        if got != want:
            mismatches.append(f"frozen dispatcher {name}: {got} != {want}")
    if len(PANELS) * len(TRAINING_SEEDS) * len(LEVELS) * CANDIDATES_PER_CONTEXT != UNIQUE_TRAINED_MODELS:
        mismatches.append("candidate field model count inconsistent")
    if sum(CANDIDATE_COMPOSITION.values()) != CANDIDATES_PER_CONTEXT:
        mismatches.append("candidate composition inconsistent")
    if N_TARGETS * CONTEXTS_PER_TARGET != TARGET_CONTEXTS:
        mismatches.append("target-context count inconsistent")
    if sum(c["n_adult"] for c in COHORTS.values()) != N_TARGETS:
        mismatches.append("cohort adult count != 53")
    if len(INTERFACE_CHANNELS) != 11:
        mismatches.append("interface channel count != 11")
    if MATERIALITY_MARGIN != 0.05 or ACTIVE_CHAINS != 2_048:
        mismatches.append("confirmatory constant drift")
    return {"ok": not mismatches, "mismatches": mismatches}
