"""Deterministic five-gate engine for the target-X observability audit (amendment 03 C4). Pure functions on
already-aggregated inputs (cluster-CI lower bounds of PAIRED differences). Unit-tested; no I/O, no randomness.

All gates take pre-computed 95% cluster-bootstrap lower bounds (LCB). A gate is a boolean; the verdict combines
them and routes to a pre-committed outcome (amendment 01 A5 / amendment 03).
"""
from __future__ import annotations

RECOVERY_MIN = 0.25
HARM_EPS = 0.005          # "clearly < 0" threshold for cross-dataset safety


def gate1_observability(rank_stratified_spearman_lcb):
    """LCB95 of the median within-subject, rank-stratified Spearman rho(G1, utility) > 0."""
    return bool(rank_stratified_spearman_lcb is not None and rank_stratified_spearman_lcb > 0)


def gate2_actionability(dtx_lcb, dtx_minus_random_lcb, dtx_minus_srcgreedy_lcb,
                        dtx_minus_whitening_lcb, dtx_minus_centering_lcb):
    """G1-selected action helps AND paired-beats random, source-greedy, whitening, mean-centering."""
    vals = [dtx_lcb, dtx_minus_random_lcb, dtx_minus_srcgreedy_lcb, dtx_minus_whitening_lcb, dtx_minus_centering_lcb]
    return bool(all(v is not None and v > 0 for v in vals))


def gate3_oracle_recovery(recovery_ratio_lcb):
    return bool(recovery_ratio_lcb is not None and recovery_ratio_lcb >= RECOVERY_MIN)


def gate4_cross_dataset_safety(per_dataset_dtx_lcb):
    """No dataset clearly harmful (LCB < -HARM_EPS) while another is positive (LCB > 0)."""
    vals = [v for v in per_dataset_dtx_lcb.values() if v is not None]
    if not vals:
        return False
    any_pos = any(v > 0 for v in vals)
    any_harm = any(v < -HARM_EPS for v in vals)
    return bool(not (any_pos and any_harm))


def gate5_specificity(di_specific_lcb):
    """Rule-level cross-fitted CMI: selected-rule leakage removal beyond exact-rank random (LCB95 > 0)."""
    return bool(di_specific_lcb is not None and di_specific_lcb > 0)


def five_gate_verdict(g1, g2, g3, g4, g5):
    """Combine the five booleans into a pre-committed outcome (amendment 01 A5 / amendment 03 routing)."""
    gates = dict(observability=g1, actionability=g2, oracle_recovery=g3, cross_dataset_safety=g4, specificity=g5)
    if g1 and g2 and g3 and g4 and g5:
        outcome = "GO_LIGHT_TARGETX_SELECTOR_PLAN"        # all pass -> propose a light selector plan (PM review)
    elif g2 and g4 and not g5:
        outcome = "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"   # useful alignment but not leakage-specific
    elif not g2:
        outcome = "STOP_ACTIONABILITY_FAILED"             # utility not recoverable -> close selector line
    else:
        outcome = "STOP_INCOMPLETE_OR_UNSAFE"             # observability/recovery/safety not met
    return {"gates": gates, "outcome": outcome, "all_pass": bool(g1 and g2 and g3 and g4 and g5)}


GATE_ORDER = ["observability", "actionability", "oracle_recovery", "cross_dataset_safety", "specificity"]
