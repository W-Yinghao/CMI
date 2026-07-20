"""C23 — Target-Free Score Calibration / Gauge Audit. C22 showed the C19 signal is a real within-target
ranking signal whose per-target score OFFSET breaks the pooled cross-target/cross-regime estimand. C23 asks a
MECHANISM question only: can that per-target offset be explained / reduced by TARGET-FREE, SOURCE-ONLY,
TARGET-ANONYMOUS gauge summaries -- WITHOUT target identity, target labels, target-wise centering, source
subject IDs, or checkpoint selection? NOT a selector, NOT an OACI rescue, NOT deployable calibration.

HARD GATE: the target-identity-leakage audit runs BEFORE any positive calibration claim -- in LOSO the source
composition nearly identifies the target, so a gauge that merely re-encodes target identity is G3, not G1.
Target labels are used ONLY for diagnostic validation (offset fitting under LOTO), never at score-time.
"""
from __future__ import annotations

from ..competence_probe import schema as c19

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = "664007686afb520f"
C22_SCORE_SIDECAR = "/projects/EEG-foundation-model/yinghao/oaci-c22-scores.json"

# gauge granularity: per-TARGET (the offset is a target-level property); LOTO over the 9 targets is the test
IN_REGIME_ONLY_FOR_GAUGE = True                # fit the gauge on the in-regime scores (where the signal lives)

# ---- gauge feature families (TARGET-ANONYMOUS SOURCE-ONLY summaries; moments over a target's candidates) ----
# The raw per-candidate source observables are the 16 robust-core features (feat__*). The gauge uses per-target
# MOMENTS of the SOURCE observables (NOT the probe score itself -> avoids trivial circularity with the offset).
_SOURCE_OBS = tuple(f"feat__{s}" for s in c19.ROBUST_CORE_FEATURES)
GAUGE_MOMENTS = ("mean", "std")                # low-freedom: 2 moments per source observable
# support/availability summaries (target-anonymous)
GAUGE_EXTRA = ("finite_feature_rate", "n_candidates")

# ---- FORBIDDEN gauge inputs (would leak target identity or use the answer) --------------------------
FORBIDDEN_GAUGE_INPUTS = ("target", "target_id", "seed", "subject", "domain_id", "source_subject",
                          "loso_complement", "score", "label", "target_center", "target_rank",
                          "target_zscore", "regime")   # NB: score/label/target-wise excluded from PRIMARY gauge

# ---- offset model (fixed low-freedom) --------------------------------------------------------------
MODEL = "ridge"
RIDGE_L2 = 1.0                                 # fixed; NO grid search
VALIDATION = "leave_one_target_out"            # + leave-one-regime-out sensitivity
N_PERM = c19.N_PERM                            # 200
PERM_SEED = c19.PERM_SEED

# ---- conservative success criteria -----------------------------------------------------------------
SUCCESS_AUC_IMPROVE = 0.03                      # calibrated pooled AUC over raw pooled
SUCCESS_GAP_CLOSED = 0.40                       # >= 40% of the target-centered oracle gap
IDENTITY_LEAKAGE_CHANCE = 1.0 / 9              # 9 targets
IDENTITY_LEAKAGE_CEILING = 0.35                # gauge target-ID accuracy above this -> identity-laden (block G1)

# ---- taxonomy (pre-registered) ---------------------------------------------------------------------
G1 = "G1_source_gauge_recovers_transport"
G2 = "G2_partial_source_gauge"
G3 = "G3_target_identity_leakage_only"
G4 = "G4_risk_family_gauge_only"
G5 = "G5_offset_source_unobservable"
G6 = "G6_epoch_or_order_residual"
ALL_CASES = (G1, G2, G3, G4, G5, G6)

FORBIDDEN_CLAIM_SUBSTRINGS = (
    "validated selector", "deployable selector", "deployable detector", "deployable calibration",
    "target-free selector", "we built a selector", "oaci is rescued", "external validation succeeded",
    "deployable method", "deployable normalization", "all dg fails", "eeg transfer is impossible",
    "support-aware invariance is useless", "production selector",
)


def frozen_config_hash() -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(c19.frozen_config(), sort_keys=True).encode()).hexdigest()[:16]
