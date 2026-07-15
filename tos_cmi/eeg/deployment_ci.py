"""CMI-Trace P0.4 — deployment confidence-interval SEMANTICS.

Separates two decisions that the old aggregation / manuscript wording conflated:

  confirmed_practical_benefit := lower_95_CI(delta_bAcc) > +threshold
  practical_gain_ruled_out    := upper_95_CI(delta_bAcc) < +threshold
  otherwise                   := inconclusive   (the CI straddles +threshold)

The FIX: "practical gain ruled out" must use the UPPER confidence bound, never the lower bound. The old code
supported a "no practical gain" reading from the lower bound, which is invalid — a lower bound below +0.01 is
perfectly consistent with a large true gain. A +0.01 gain is ruled out ONLY when the UPPER bound is below
+0.01. A deployable-benefit CLAIM additionally requires the stricter source-task-safety and random-control
conditions (a confirmed benefit that is not source-safe or does not beat same-rank random removal is NOT
deployable). Pure numpy; no torch.
"""
from __future__ import annotations
import numpy as np

PRACTICAL_THRESHOLD = 0.01

CONFIRMED = "confirmed_practical_benefit"
RULED_OUT = "practical_gain_ruled_out"
INCONCLUSIVE = "inconclusive"


def deployment_ci_state(lo, hi, threshold=PRACTICAL_THRESHOLD):
    """Three-state deployment decision from a 95% CI [lo, hi] of delta balanced accuracy.
    Returns CONFIRMED / RULED_OUT / INCONCLUSIVE. 'ruled out' uses the UPPER bound (hi < threshold)."""
    lo = float(lo); hi = float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return INCONCLUSIVE
    if lo > threshold:
        return CONFIRMED
    if hi < threshold:
        return RULED_OUT
    return INCONCLUSIVE


def practical_gain_ruled_out_everywhere(states):
    """The main multi-dataset statement may say a +threshold gain is ruled out ONLY when EVERY relevant
    cell/method has upper CI < +threshold (i.e. state == RULED_OUT for all)."""
    states = list(states)
    return bool(states) and all(s == RULED_OUT for s in states)


def deployable_benefit(state, source_task_safe, beats_random_control):
    """A deployable-benefit CLAIM requires a CONFIRMED practical benefit AND source-task safety AND beating
    same-rank random removal. Any failure -> not deployable (even if the benefit CI is confirmed)."""
    return bool(state == CONFIRMED and source_task_safe and beats_random_control)
