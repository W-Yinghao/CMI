"""C22 — Estimand Transport Mechanism Audit. Explains WHY the C19 in-regime source-only competence signal
does NOT transport as a pooled cross-regime estimand (C20), while within-target AUC persists. NOT a selector,
NOT probe tuning, NOT score calibration for deployment, NOT external validation. Pure read-only mechanism audit
of the FROZEN C19 probe (config hash 664007686afb520f, unchanged).

Central hypothesis: the robust-core probe carries within-target RANKING information, but its absolute score is
distorted by target/regime-specific offsets/scales (and possibly epoch/trajectory-position confounds), so
pooled AUC fails even when within-target AUC persists. HARD GATE: the epoch/order baseline (Q2) MUST be reported
before any 'normalization rescues pooling' interpretation; normalization diagnostics are post-hoc MECHANISM
tests, never deployable procedures.
"""
from __future__ import annotations

from ..competence_probe import schema as c19
from ..probe_validation import schema as c20

DIAGNOSTIC_ONLY = True
LOCKED_C19_CONFIG_HASH = c20.LOCKED_C19_CONFIG_HASH        # 664007686afb520f -- must stay unchanged

# in-regime regimes = where the C19 signal exists (development regimes); cross-regime = C20 held-out
IN_REGIME = c20.DEVELOPMENT_REGIMES                        # S0/S2/S3
CROSS_REGIME = c20.HELD_OUT_REGIMES                        # S4/S5/S6/S7
ALL_REGIMES = tuple(IN_REGIME) + tuple(CROSS_REGIME)
ROBUST_CORE = c19.ROBUST_CORE_FEATURES                     # frozen 16; read-only
DIAGNOSTIC_LABEL = c19.DIAGNOSTIC_LABEL

# ---- Q2 epoch/order baselines (reported BEFORE any transport-rescue claim) --------------------------
EPOCH_BASELINES = ("epoch", "candidate_order", "train_surrogate", "R_src")   # trajectory-position / training-log proxies
EPOCH_CONTROL_BINS = 4                                     # epoch-bin stratification for residual-signal control

# ---- Q3 score normalization diagnostics (post-hoc MECHANISM tests, NOT deployment) -----------------
NORMALIZATIONS = ("none", "target_center", "target_zscore", "target_rank", "regime_center",
                  "target_regime_center", "quantile")
NORMALIZATION_IS_DIAGNOSTIC = True                         # never a deployable procedure

# ---- decomposition / significance ------------------------------------------------------------------
N_PERM = c19.N_PERM                                        # 200
PERM_SEED = c19.PERM_SEED                                  # 707
SIGNAL_MARGIN = 0.03                                       # AUC-above-chance to call a signal "present"
OFFSET_DOMINATED_FRACTION = 0.5                            # between-target score variance fraction -> offset-dominated

# ---- deterministic taxonomy (pre-registered labels) ------------------------------------------------
T1 = "T1_rank_signal_score_not_calibrated"          # within-target signal survives; pooled fails on target/regime offsets
T2 = "T2_epoch_confounded_signal"                   # probe advantage mostly explained by epoch / candidate position
T3 = "T3_regime_specific_relationship_shift"        # target normalization does NOT rescue pooled; relationship itself shifts
T4 = "T4_feature_offset_dominated"                  # robust-core features available but dominated by target/regime location
T5 = "T5_true_transport_absent"                     # neither within-target residual nor normalized pooled signal reliable
ALL_CASES = (T1, T2, T3, T4, T5)

# ---- forbidden claims (guarded) --------------------------------------------------------------------
FORBIDDEN_CLAIM_SUBSTRINGS = (
    "deployable selector", "deployable target-free selector", "target-free selector", "we built a selector",
    "detector is validated", "oaci is rescued", "external validation succeeded", "all dg fails",
    "eeg transfer is impossible", "normalization deploys", "deployable normalization", "calibrated selector",
    "normalization rescues deployment", "production selector", "generalization is established",
)


def frozen_config_hash() -> str:
    import hashlib
    import json
    return hashlib.sha256(json.dumps(c19.frozen_config(), sort_keys=True).encode()).hexdigest()[:16]
