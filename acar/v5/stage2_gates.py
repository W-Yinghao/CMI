"""ACAR V5 Stage-2 GATES (numpy lazy). Encodes the pinned CAL/EVAL split (user 2026-07-05):

  * CAL certifies: the H1–H3 LTT hypotheses — H1=G3 (UCB[L_harm_all]≤0.10), H2=G4 (UCB[harm_among_adapted]≤0.30), H3=G1
    (LCB[coverage]≥0.15) — are computed on CAL, with raw one-sided p-values by inverting the subject-clustered empirical-Bernstein
    bound (smallest α at which the bound crosses the gate threshold), then Holm-adjusted over the family (candidate×disease×{H1,H2,H3})
    at FWER α=0.05. G4 non-evaluable (no adapting CAL subject) ⇒ H2 raw p = 1 ⇒ never certifiable ⇒ fail.
  * EVAL final: G2 (red>0 AND red−v2_replay_red≥0.02 per disease AND macro) and G5 (red≥0.25·red_upper OR red≥best-eligible-P3;
    red_upper≤0 ⇒ upper non-informative ⇒ P3 fallback) are computed on EVAL. red/red_upper come from policy evaluation.

Holm is reimplemented here (pure numpy) to keep v5 self-contained (no scipy / acar.v4 import). The v2-replay comparator is a
pluggable, fail-closed seam (real internals = acar.features.feature_vector + acar.regressor.ActionRegressor seed 0, wired at real-run time).
"""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import ltt as LTT
from acar.v5 import metrics as M


class Stage2GateError(RuntimeError):
    pass


class V2ReplayNotEvaluable(RuntimeError):
    """Raised when the v2-replay comparator cannot produce v2_replay_red for a disease (⇒ Stage-2 real selection cannot run)."""


# ---------------------------------------------------------------- CAL: empirical-Bernstein p-value inversion + Holm
def _smallest_alpha(is_certifiable, lo=1e-12, hi=1.0, iters=64):
    """Smallest α∈[lo,hi] with is_certifiable(α) True, assuming the predicate flips False→True as α increases (EB radius shrinks
    with α). Returns 1.0 if never certifiable even at α=1."""
    if not is_certifiable(hi):
        return 1.0
    if is_certifiable(lo):
        return float(lo)
    for _ in range(iters):
        mid = 0.5 * (lo + hi)
        if is_certifiable(mid):
            hi = mid
        else:
            lo = mid
    return float(hi)


def eb_pvalue_upper(xs, threshold):
    """Raw one-sided p for an UPPER-bound hypothesis UCB[x] ≤ threshold: smallest α with eb_ucb(xs,α) ≤ threshold."""
    return _smallest_alpha(lambda a: LTT.eb_ucb(xs, a) <= threshold + 1e-15)


def eb_pvalue_lower(xs, threshold):
    """Raw one-sided p for a LOWER-bound hypothesis LCB[x] ≥ threshold: smallest α with eb_lcb(xs,α) ≥ threshold."""
    return _smallest_alpha(lambda a: LTT.eb_lcb(xs, a) >= threshold - 1e-15)


def cal_raw_pvalues(cal_subject_records):
    """Compute the raw H1/H2/H3 p-values (by EB inversion) + the display bounds (at α=ALPHA) for one (candidate, disease) on CAL."""
    if not cal_subject_records:                                                   # no CAL subjects ⇒ not certifiable (fail-closed)
        non_cert = {"n_subjects": 0, "n_adapting": 0, "coverage_lcb": 0.0, "l_harm_all_ucb": 1.0,
                    "harm_among_adapted_ucb": None, "h2_evaluable": False, "G1_coverage": False,
                    "G3_l_harm_all": False, "G4_harm_among_adapted": False, "certification_pass": False}
        return {"H1": 1.0, "H2": 1.0, "H3": 1.0, "h2_evaluable": False, "bounds": non_cert}
    c = M.collect(cal_subject_records)
    h2_evaluable = c["n_adapting"] > 0
    p_h1 = eb_pvalue_upper(c["l_harm_all"], P.L_HARM_ALL_MAX)                      # H1 = G3
    p_h2 = eb_pvalue_upper(c["harm_among_adapted"], P.HARM_AMONG_ADAPTED_MAX) if h2_evaluable else 1.0   # H2 = G4
    p_h3 = eb_pvalue_lower(c["coverage"], P.COVERAGE_MIN)                          # H3 = G1
    bounds = LTT.gate_disease(cal_subject_records)                                 # point bounds at ALPHA (for the report)
    return {"H1": p_h1, "H2": p_h2, "H3": p_h3, "h2_evaluable": bool(h2_evaluable), "bounds": bounds}


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values (FWER), monotone in rank, mapped back to input order. Pure numpy."""
    import numpy as np
    p = np.asarray(pvalues, dtype=float)
    if p.ndim != 1 or p.shape[0] == 0:
        raise Stage2GateError("pvalues must be a non-empty 1-D array")
    if not np.isfinite(p).all():
        raise Stage2GateError("pvalues contains non-finite values")
    L = p.shape[0]
    order = np.argsort(p, kind="stable")
    adj_sorted = np.empty(L, dtype=float)
    running = 0.0
    for i in range(L):
        running = max(running, (L - i) * p[order[i]])
        adj_sorted[i] = min(running, 1.0)
    adj = np.empty(L, dtype=float)
    adj[order] = adj_sorted
    return adj


def cert_pass_from_adjusted(adj_h1, adj_h2, adj_h3, alpha=P.ALPHA):
    """A (candidate, disease) is CAL-certified (G1∧G3∧G4) iff all three Holm-adjusted H1–H3 p-values ≤ family-wise α."""
    return bool(adj_h1 <= alpha and adj_h2 <= alpha and adj_h3 <= alpha)


# ---------------------------------------------------------------- EVAL: G2 utility + G5 benefit retention
def g2_per_disease(red, v2_replay_red, eps=P.UTILITY_EPS):
    """G2 per-disease point-estimate gate: red>0 AND red − v2_replay_red ≥ ε (0.02)."""
    return bool(red > 0.0 and (red - v2_replay_red) >= eps)


def g2_macro(macro_red, macro_v2_replay_red, eps=P.UTILITY_EPS):
    """G2 macro gate: macro(red) − macro(v2_replay_red) ≥ ε."""
    return bool((macro_red - macro_v2_replay_red) >= eps)


def g5_pass(red, red_upper, red_p3_best, frac=P.BENEFIT_RETENTION_FRAC):
    """G5 benefit retention: red ≥ frac·red_upper OR red ≥ red(best-eligible P3). If red_upper ≤ 0 the upper arm is
    NON-INFORMATIVE and G5 falls back to the P3 comparator (which must exist)."""
    if red_upper <= 0.0:
        return bool(red_p3_best is not None and red >= red_p3_best)
    if red >= frac * red_upper:
        return True
    return bool(red_p3_best is not None and red >= red_p3_best)


# ---------------------------------------------------------------- v2-replay comparator (pluggable, fail-closed)
def real_v2_replay_provider(disease, context):
    """Real v2-replay seam: subject-macro v2_replay_red per disease via the bit-for-bit v2 recipe (acar.features.feature_vector +
    acar.regressor.ActionRegressor seed 0 on FIT, one-sided q on CAL, route on EVAL). NOT wired for the synthetic Stage-2B0 run —
    raises so a real run must supply/validate it. If it cannot produce v2_replay_red for a disease, real selection cannot run."""
    raise V2ReplayNotEvaluable(
        f"v2_replay comparator for {disease} must be wired + validated for the real Stage-2B run (acar.regressor seed 0)")
