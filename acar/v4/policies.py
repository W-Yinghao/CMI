"""ACAR v4 (CURB) — label-free selective policy families π_λ.

NON-BINDING / POST-V3 / synthetic-capable: pure deterministic functions over score arrays. They read no real cohort,
fit nothing, freeze nothing, select nothing. The control-first thesis (notes/ACAR_V4_DESIGN_DRAFT.md §4) is that we
should learn and calibrate the policy that is *actually executed*, not upper-bound every action; these families are
that executed policy.

Conventions (frozen from v2/v3, see acar/config.py):
  - Per batch B the NON_IDENTITY actions are columns 0..A-1; identity is the f_0 reference, encoded as choice = -1
    (ΔR_identity ≡ 0).
  - Scores are in ΔR units, "lower is safer/better":
      harm[b, a]    : conservative (upper) estimate of ΔR_a(B)   (higher ⇒ more harmful)
      benefit[b, a] : center estimate of ΔR_a(B)                 (lower  ⇒ more expected reduction)
  - A policy returns `choice` (int array, length n): -1 = identity, else the NON_IDENTITY column index executed.

Accounting invariant (matches v3 S2/S4): the caller passes ALL EVAL batches as rows, including fallback batches
(< MIN_BATCH) whose choice is forced to -1. coverage and reduction then share the same denominator n, so fallback
never inflates utility.
"""
from __future__ import annotations
import numpy as np

IDENTITY = -1  # choice sentinel for the f_0 reference action (ΔR ≡ 0)


def _as2d(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n_batches, n_actions], got shape {a.shape}")
    return a


# ----------------------------------------------------------------------------- policy families (π_λ)

def safe_set_policy(harm, benefit, lam, *, require_benefit=False):
    """A1 primary family (nested safe-action set).

    Γ_λ(B) = {a : harm[B,a] ≤ λ};  π_λ(B) = identity if Γ_λ empty else argmin_{a∈Γ_λ} benefit[B,a].
    Larger λ ⇒ larger (nested) safe set ⇒ coverage non-decreasing — the budget separates safety screening from
    utility selection. With require_benefit=True the policy additionally abstains unless the best admitted action is
    predicted to reduce risk (benefit < 0); default False keeps the primary family exactly as specified.
    """
    harm = _as2d(harm, "harm"); benefit = _as2d(benefit, "benefit")
    if harm.shape != benefit.shape:
        raise ValueError(f"harm {harm.shape} and benefit {benefit.shape} must share shape")
    n, _ = harm.shape
    admit = harm <= lam                                  # [n, A] boolean safe set Γ_λ
    masked = np.where(admit, benefit, np.inf)            # un-admitted actions cannot be chosen
    best = np.argmin(masked, axis=1)
    bestval = masked[np.arange(n), best]
    take = np.isfinite(bestval)                          # at least one admitted action
    if require_benefit:
        take = take & (bestval < 0.0)
    choice = np.full(n, IDENTITY, dtype=int)
    choice[take] = best[take]
    return choice


def benefit_ranked_policy(benefit, tau):
    """Benefit-ranked family (no harm screen): adapt iff the best predicted reduction clears τ
    (argmin_a benefit[B,a] ≤ τ, τ ≤ 0 typically); action = that argmin. Smaller (more negative) τ ⇒ less coverage."""
    benefit = _as2d(benefit, "benefit")
    n, _ = benefit.shape
    best = np.argmin(benefit, axis=1)
    bestval = benefit[np.arange(n), best]
    choice = np.full(n, IDENTITY, dtype=int)
    take = bestval <= tau
    choice[take] = best[take]
    return choice


def direct_selective_policy(gate_score, action_idx, tau):
    """Direct-selective family: a single per-batch confidence `gate_score` (higher ⇒ more willing to adapt) and a
    precomputed per-batch action `action_idx`; adapt iff gate_score ≥ τ. Higher τ ⇒ less coverage."""
    gate_score = np.asarray(gate_score, dtype=float).ravel()
    action_idx = np.asarray(action_idx, dtype=int).ravel()
    if gate_score.shape != action_idx.shape:
        raise ValueError("gate_score and action_idx must share length")
    choice = np.full(gate_score.shape[0], IDENTITY, dtype=int)
    take = gate_score >= tau
    choice[take] = action_idx[take]
    return choice


# ----------------------------------------------------------------------------- deployed-policy accounting

def coverage(choice):
    """Adaptation coverage = fraction of batches with a non-identity action (fallback identity rows in denominator)."""
    choice = np.asarray(choice, dtype=int)
    return float(np.mean(choice != IDENTITY)) if choice.size else 0.0


def realized_dr(choice, dr):
    """Per-batch realized ΔR of `choice` against true ΔR `dr` [n, A]; identity → 0."""
    dr = _as2d(dr, "dr"); choice = np.asarray(choice, dtype=int)
    if choice.shape[0] != dr.shape[0]:
        raise ValueError(f"choice length {choice.shape[0]} != n_batches {dr.shape[0]}")
    out = np.zeros(dr.shape[0], dtype=float)
    adapt = np.where(choice != IDENTITY)[0]
    out[adapt] = dr[adapt, choice[adapt]]
    return out


def reduction(choice, dr):
    """Deployed NLL reduction red(π) = −mean_B ΔR_{π(B)} (identity/fallback contribute 0)."""
    r = realized_dr(choice, dr)
    return -float(np.mean(r)) if r.size else 0.0


def harm_rate(choice, dr):
    """P(ΔR_{π(B)} > 0 | π(B) ≠ identity) — fraction of ADAPTED batches that were actually harmful. NaN if none adapted."""
    dr = _as2d(dr, "dr"); choice = np.asarray(choice, dtype=int)
    adapt = choice != IDENTITY
    if not adapt.any():
        return float("nan")
    return float(np.mean(realized_dr(choice, dr)[adapt] > 0.0))


# ----------------------------------------------------------------------------- score helpers (label-free rankings)

def best_benefit_action(benefit):
    """Per-batch argmin benefit (the action a score-driven policy would pick) and its value."""
    benefit = _as2d(benefit, "benefit")
    idx = np.argmin(benefit, axis=1)
    return idx, benefit[np.arange(benefit.shape[0]), idx]


def adapt_rank_from_harm(harm):
    """Adapt-first ranking for the safe-set family: lower minimum harm ⇒ adapt earlier ⇒ higher rank.
    Returns a per-batch score (higher = adapt first), consistent with the nested Γ_λ ordering."""
    harm = _as2d(harm, "harm")
    return -np.min(harm, axis=1)
