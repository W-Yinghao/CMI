"""Deterministic five-gate engine for the target-X observability audit (amendments 03/04). Pure functions on
already-aggregated inputs; unit-tested; no I/O, no randomness.

F2.1b corrections:
 - Gate 3 is a PAIRED inequality against the CONSTRAINED hindsight oracle (no bootstrap of small-denominator
   per-fold ratios): LCB(Δ_hindsight_constrained) > 0 AND LCB(Δ_TX - 0.25·Δ_hindsight_constrained) >= 0.
 - Gate 4 confirms harm with the UPPER confidence bound (UCB < -eps), not the lower bound.
"""
from __future__ import annotations

RECOVERY_FRACTION = 0.25
HARM_EPS = 0.005


def gate1_observability(rank_stratified_spearman_lcb):
    return bool(rank_stratified_spearman_lcb is not None and rank_stratified_spearman_lcb > 0)


def gate2_actionability(dtx_lcb, dtx_minus_random_lcb, dtx_minus_srcgreedy_lcb,
                        dtx_minus_whitening_lcb, dtx_minus_centering_lcb):
    vals = [dtx_lcb, dtx_minus_random_lcb, dtx_minus_srcgreedy_lcb, dtx_minus_whitening_lcb, dtx_minus_centering_lcb]
    return bool(all(v is not None and v > 0 for v in vals))


def gate3_oracle_recovery(hindsight_constrained_lcb, dtx_minus_quarter_hindsight_lcb):
    """Paired: the constrained hindsight ceiling is real (LCB>0) AND Δ_TX recovers >=25% of it (paired LCB>=0)."""
    return bool(hindsight_constrained_lcb is not None and hindsight_constrained_lcb > 0
                and dtx_minus_quarter_hindsight_lcb is not None and dtx_minus_quarter_hindsight_lcb >= 0)


def gate4_cross_dataset_safety(per_dataset_dtx):
    """per_dataset_dtx = {dataset: {'lo':.., 'hi':..}}. Unsafe iff one dataset LCB>0 while another UCB<-eps."""
    los = [v.get("lo") for v in per_dataset_dtx.values() if v.get("lo") is not None]
    his = [v.get("hi") for v in per_dataset_dtx.values() if v.get("hi") is not None]
    if not los or not his:
        return False
    any_pos = any(lo > 0 for lo in los)
    any_confirmed_harm = any(hi < -HARM_EPS for hi in his)
    return bool(not (any_pos and any_confirmed_harm))


def gate5_specificity(di_specific_lcb):
    return bool(di_specific_lcb is not None and di_specific_lcb > 0)


def five_gate_verdict(g1, g2, g3, g4, g5):
    gates = dict(observability=g1, actionability=g2, oracle_recovery=g3, cross_dataset_safety=g4, specificity=g5)
    if g1 and g2 and g3 and g4 and g5:
        outcome = "GO_LIGHT_TARGETX_SELECTOR_PLAN"
    elif g2 and g4 and not g5:
        outcome = "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"
    elif not g2:
        outcome = "STOP_ACTIONABILITY_FAILED"
    else:
        outcome = "STOP_INCOMPLETE_OR_UNSAFE"
    return {"gates": gates, "outcome": outcome, "all_pass": bool(g1 and g2 and g3 and g4 and g5)}


GATE_ORDER = ["observability", "actionability", "oracle_recovery", "cross_dataset_safety", "specificity"]
