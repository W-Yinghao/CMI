"""ACAR v4 (CURB) — Direction A: finite-grid risk-control calibration of the executed policy.

NON-BINDING / POST-V3 / synthetic-capable: pure numpy + scipy; reads no real cohort, calls no v3 loader, fits no
predictor, freezes nothing, selects no V4 candidate. The job is NOT to learn a policy but to CALIBRATE one:

    given a FIXED finite λ grid and SUBJECT-level CAL losses of the policy π_λ,
    select the most aggressive λ whose one-sided risk test passes a pre-specified budget.

This is the control-first calibration of Direction A (notes/ACAR_V4_DESIGN_DRAFT.md §4 A2): we calibrate the deployed
policy that is actually executed, not the all-action upper bounds (v3). It borrows the Learn-Then-Test / RCPS idea —
on calibration data, choose a policy parameter so a user-specified loss is risk-controlled — but, because the deployed
risk need NOT be monotone in λ, we make NO monotone conformal-risk-control claim: a FIXED grid is tested with an
FWER-controlling multiple-testing rule (Holm / Bonferroni) and the most aggressive PASSING λ is selected. The selection
is a deterministic function of the CAL losses only — it never refits scores or policies.

Sign convention (frozen): ΔR_a(B) < 0 = reduced risk (good); identity/fallback contribute realized ΔR 0 and stay in
the subject denominator. Subjects are the exchangeable calibration unit; aggregation is subject-equal (subject-macro).

Fail-closed: malformed inputs (non-finite, wrong shape, duplicate/empty λ, bad alpha/budget, unknown options,
out-of-bounds losses for a bounded method) RAISE. A valid calibration with no passing λ is NO_PASS; a calibration too
small to evaluate (< 2 subjects) is NOT_EVALUABLE. Neither reads labels beyond the supplied CAL losses.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Literal, Optional, Sequence

import numpy as np

from acar.v4 import policies as P


@dataclass(frozen=True)
class RiskControlResult:
    selected_index: Optional[int]          # index into lambda_grid of the selected λ (None if no pass / not evaluable)
    selected_lambda: Optional[float]
    status: str                            # "PASS" | "NO_PASS" | "NOT_EVALUABLE"
    p_values: np.ndarray                   # per-λ one-sided p-value for H0: E[L] ≥ budget
    adjusted_p_values: np.ndarray          # after the FWER correction
    empirical_risks: np.ndarray            # per-λ subject-macro mean loss
    upper_confidence: np.ndarray           # per-λ one-sided (1−alpha) UCB on the mean loss (uncorrected, informational)
    passer_mask: np.ndarray                # adjusted_p ≤ alpha
    alpha: float
    budget: float
    correction: str
    selection_rule: str


# ----------------------------------------------------------------------------- fail-closed validators

def _as_loss_matrix(x):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"subject_losses must be 2-D [n_subjects, L], got shape {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"subject_losses must have ≥1 subject and ≥1 λ, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError("subject_losses contains non-finite values (NaN/Inf)")
    return a


def _check_grid(g):
    a = np.asarray(g, dtype=float)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("lambda_grid must be a non-empty 1-D array")
    if not np.all(np.isfinite(a)):
        raise ValueError("lambda_grid contains non-finite values (NaN/Inf)")
    if np.unique(a).shape[0] != a.shape[0]:
        raise ValueError("lambda_grid must contain unique values (duplicate λ rejected)")
    return a


def _check_alpha(alpha):
    a = float(alpha)
    if not (0.0 < a < 1.0):
        raise ValueError("alpha must be in (0, 1)")
    return a


def _check_budget(budget):
    b = float(budget)
    if not math.isfinite(b):
        raise ValueError("budget must be finite")
    return b


def _check_bounds(loss_bounds, sl):
    if loss_bounds is None:
        raise ValueError("method='hoeffding' requires predeclared loss_bounds=(lo, hi)")
    lo, hi = float(loss_bounds[0]), float(loss_bounds[1])
    if not (math.isfinite(lo) and math.isfinite(hi)) or not (hi > lo):
        raise ValueError("loss_bounds must be finite with hi > lo")
    if np.any(sl < lo - 1e-12) or np.any(sl > hi + 1e-12):
        raise ValueError("observed subject_losses fall outside the declared loss_bounds")
    return lo, hi


# ----------------------------------------------------------------------------- subject-level losses from a policy

def subject_losses_from_policy(choices_by_lambda, dr, subject_ids, *, loss):
    """Subject-level CAL loss matrix [n_subjects, L] for the executed policy π_λ across a λ grid.

    choices_by_lambda : int [L, n_batches], -1 identity else action index (e.g. stack of safe_set_policy outputs).
    dr                : true ΔR [n_batches, n_actions].
    subject_ids       : [n_batches]; subjects are aggregated in sorted-unique order.
    loss ∈ {"mean", "positive", "harm_indicator"} — per-batch realized loss, then subject-mean over ALL the subject's
    batches (fallback/identity rows realized 0, kept in the denominator)."""
    dr2 = P._as2d(dr, "dr")
    C = np.asarray(choices_by_lambda)
    if C.ndim != 2:
        raise ValueError("choices_by_lambda must be 2-D [L, n_batches]")
    if not np.issubdtype(C.dtype, np.integer):
        raise ValueError("choices_by_lambda must be integer")
    L, n = C.shape
    if dr2.shape[0] != n:
        raise ValueError(f"dr has {dr2.shape[0]} batches but choices_by_lambda has {n}")
    ids = np.asarray(subject_ids)
    if ids.ndim != 1 or ids.shape[0] != n:
        raise ValueError("subject_ids must be 1-D with length n_batches")
    if loss not in ("mean", "positive", "harm_indicator"):
        raise ValueError("loss must be 'mean', 'positive', or 'harm_indicator'")
    uniq = np.unique(ids)
    index_of = {u: i for i, u in enumerate(uniq.tolist())}
    rows = np.array([index_of[i] for i in ids.tolist()], dtype=int)
    n_subj = uniq.shape[0]
    counts = np.bincount(rows, minlength=n_subj).astype(float)        # batches per subject (incl. fallback)
    out = np.empty((n_subj, L), dtype=float)
    for l in range(L):
        choice = C[l]
        realized = P.realized_dr(choice, dr2)                          # validates choice ∈ {-1,0..A-1}
        if loss == "mean":
            val = realized
        elif loss == "positive":
            val = np.maximum(realized, 0.0)
        else:                                                          # harm_indicator
            val = ((choice != P.IDENTITY) & (realized > 0.0)).astype(float)
        sums = np.bincount(rows, weights=val, minlength=n_subj)
        out[:, l] = sums / counts
    return out


# ----------------------------------------------------------------------------- risk statistics

def empirical_risk(subject_losses):
    """Per-λ subject-macro mean loss (subjects equal-weighted)."""
    return _as_loss_matrix(subject_losses).mean(axis=0)


def _ttest_pvalues(sl, budget):
    """One-sided p-value per λ for H0: E[L] ≥ budget vs H1: E[L] < budget (lower-tail one-sample t)."""
    from scipy import stats
    n = sl.shape[0]
    mean = sl.mean(axis=0)
    sd = sl.std(axis=0, ddof=1)
    p = np.empty(sl.shape[1], dtype=float)
    nz = sd > 0
    se = np.where(nz, sd / math.sqrt(n), 1.0)
    t_obs = (mean - budget) / se
    p[nz] = stats.t.cdf(t_obs[nz], df=n - 1)
    # zero-variance columns are deterministic relative to the budget
    deg = ~nz
    p[deg] = np.where(mean[deg] < budget, 0.0, np.where(mean[deg] > budget, 1.0, 0.5))
    return np.clip(p, 0.0, 1.0)


def _hoeffding_pvalues(sl, budget, lo, hi):
    """Conservative one-sided p-value per λ for H0: E[L] ≥ budget on losses bounded in [lo, hi] (Hoeffding)."""
    n = sl.shape[0]
    mean = sl.mean(axis=0)
    gap = budget - mean                                              # > 0 ⇒ evidence the mean is below budget
    rng2 = (hi - lo) ** 2
    p = np.where(gap > 0.0, np.exp(-2.0 * n * np.maximum(gap, 0.0) ** 2 / rng2), 1.0)
    return np.clip(p, 0.0, 1.0)


def one_sided_mean_risk_pvalue(subject_losses, budget, *, method="ttest", loss_bounds=None):
    """Per-λ one-sided p-value for H0: E[L] ≥ budget. method ∈ {"ttest", "hoeffding"} (hoeffding needs loss_bounds)."""
    sl = _as_loss_matrix(subject_losses)
    b = _check_budget(budget)
    if method == "ttest":
        if sl.shape[0] < 2:
            raise ValueError("ttest p-values require ≥ 2 subjects")
        return _ttest_pvalues(sl, b)
    if method == "hoeffding":
        lo, hi = _check_bounds(loss_bounds, sl)
        return _hoeffding_pvalues(sl, b, lo, hi)
    raise ValueError("method must be 'ttest' or 'hoeffding'")


def _ucb(sl, alpha, method, loss_bounds):
    n = sl.shape[0]
    mean = sl.mean(axis=0)
    if method == "ttest":
        from scipy import stats
        sd = sl.std(axis=0, ddof=1)
        tq = stats.t.ppf(1.0 - alpha, df=n - 1)
        return mean + tq * sd / math.sqrt(n)
    lo, hi = _check_bounds(loss_bounds, sl)
    return mean + (hi - lo) * math.sqrt(math.log(1.0 / alpha) / (2.0 * n))


# ----------------------------------------------------------------------------- multiple-testing corrections

def holm_adjust(p_values):
    """Holm step-down adjusted p-values (FWER control), monotone in rank, mapped back to input order."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.shape[0] == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("p_values contains non-finite values")
    L = p.shape[0]
    order = np.argsort(p, kind="stable")
    ranked = p[order]
    adj_sorted = np.empty(L, dtype=float)
    running = 0.0
    for i in range(L):
        running = max(running, (L - i) * ranked[i])       # 0-based rank i ⇒ factor (L - i)
        adj_sorted[i] = min(running, 1.0)
    adj = np.empty(L, dtype=float)
    adj[order] = adj_sorted
    return adj


def bonferroni_adjust(p_values):
    """Bonferroni adjusted p-values = min(L·p, 1)."""
    p = np.asarray(p_values, dtype=float)
    if p.ndim != 1 or p.shape[0] == 0:
        raise ValueError("p_values must be a non-empty 1-D array")
    if not np.all(np.isfinite(p)):
        raise ValueError("p_values contains non-finite values")
    return np.minimum(p * p.shape[0], 1.0)


def _adjust(p, correction):
    if correction == "holm":
        return holm_adjust(p)
    if correction == "bonferroni":
        return bonferroni_adjust(p)
    if correction == "none":
        return np.asarray(p, dtype=float)
    raise ValueError("correction must be 'holm', 'bonferroni', or 'none'")


# ----------------------------------------------------------------------------- selection

def select_most_aggressive_passer(lambda_grid, passer_mask, *, aggressiveness):
    """Deterministic tie/selection rule: among passers pick the most aggressive λ by VALUE (not position) — the largest
    λ for 'increasing_lambda', the smallest for 'decreasing_lambda'. Returns the grid index, or None if no passer.
    Selecting by λ value makes the choice independent of grid ordering."""
    grid = np.asarray(lambda_grid, dtype=float)
    mask = np.asarray(passer_mask, dtype=bool)
    if mask.shape != grid.shape:
        raise ValueError("passer_mask must match lambda_grid shape")
    idx = np.where(mask)[0]
    if idx.size == 0:
        return None
    if aggressiveness == "increasing_lambda":
        return int(idx[np.argmax(grid[idx])])
    if aggressiveness == "decreasing_lambda":
        return int(idx[np.argmin(grid[idx])])
    raise ValueError("aggressiveness must be 'increasing_lambda' or 'decreasing_lambda'")


def select_ltt_grid(lambda_grid, subject_losses, *, alpha, budget, aggressiveness,
                    correction="holm", method="ttest", loss_bounds=None):
    """Select the most aggressive λ whose one-sided risk test passes the budget under an FWER-controlling correction.

    No monotonicity in λ is assumed: every λ on the FIXED grid is tested, passers = adjusted_p ≤ alpha, and the most
    aggressive passing λ (by value) is chosen. < 2 subjects ⇒ NOT_EVALUABLE; no passer ⇒ NO_PASS; malformed ⇒ raise."""
    grid = _check_grid(lambda_grid)
    sl = _as_loss_matrix(subject_losses)
    L = grid.shape[0]
    if sl.shape[1] != L:
        raise ValueError(f"subject_losses has {sl.shape[1]} columns but lambda_grid has {L}")
    a = _check_alpha(alpha)
    b = _check_budget(budget)
    if correction not in ("holm", "bonferroni", "none"):
        raise ValueError("correction must be 'holm', 'bonferroni', or 'none'")
    if aggressiveness not in ("increasing_lambda", "decreasing_lambda"):
        raise ValueError("aggressiveness must be 'increasing_lambda' or 'decreasing_lambda'")
    if method not in ("ttest", "hoeffding"):
        raise ValueError("method must be 'ttest' or 'hoeffding'")
    if method == "hoeffding":            # validate bounds even on the NOT_EVALUABLE path (data-contract check)
        _check_bounds(loss_bounds, sl)
    emp = sl.mean(axis=0)
    rule = (f"most_aggressive_passer|{aggressiveness}|{correction}|{method}|alpha={a}|budget={b}")
    if sl.shape[0] < 2:
        nan = np.full(L, np.nan)
        return RiskControlResult(None, None, "NOT_EVALUABLE", nan, nan, emp, nan,
                                 np.zeros(L, dtype=bool), a, b, correction, rule)
    p = one_sided_mean_risk_pvalue(sl, b, method=method, loss_bounds=loss_bounds)
    adj = _adjust(p, correction)
    ucb = _ucb(sl, a, method, loss_bounds)
    passer = adj <= a
    sel = select_most_aggressive_passer(grid, passer, aggressiveness=aggressiveness)
    status = "PASS" if sel is not None else "NO_PASS"
    sel_lam = float(grid[sel]) if sel is not None else None
    return RiskControlResult(sel, sel_lam, status, p, adj, emp, ucb, passer, a, b, correction, rule)
