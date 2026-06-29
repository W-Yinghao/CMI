"""ACAR v4 (CURB) — label-free selective policy families π_λ + deployed-policy accounting.

NON-BINDING / POST-V3 / synthetic-capable: pure deterministic functions over score arrays. They read no real cohort,
fit nothing, freeze nothing, select nothing. The control-first thesis (notes/ACAR_V4_DESIGN_DRAFT.md §4) is to learn
and calibrate the policy that is *actually executed*, not upper-bound every action; these families are that policy.

Conventions (frozen from v2/v3, see acar/config.py):
  - Per batch B the NON_IDENTITY actions are columns 0..A-1; identity is the f_0 reference, encoded as choice = -1
    (ΔR_identity ≡ 0).
  - Scores are in ΔR units, "lower is safer/better":
      harm[b, a]    : conservative (upper) estimate of ΔR_a(B)   (higher ⇒ more harmful)
      benefit[b, a] : center estimate of ΔR_a(B)                 (lower  ⇒ more expected reduction)
  - A policy returns `choice` (int array, length n): -1 = identity, else the NON_IDENTITY column index executed.

WEIGHTING (contract). Every accounting function takes `weights=None`. weights=None ⇒ uniform per-batch (batch-level
primitive). For a SUBJECT-MACRO (subject-equal-weighted) DEV number, pass `subject_macro_weights(subject_ids)` so a
subject with many batches does not dominate. Weights are normalised to sum 1 internally, so coverage/red/harm are
genuine weighted means. Fallback rows (forced identity, choice = -1) MUST be kept as rows so they stay in the weight
denominator — they contribute 0 reduction but are counted in coverage. This matches v3 S2/S4 accounting.

FAIL-CLOSED (contract). All numeric inputs must be finite; arrays must be non-empty with ≥1 action; choices and action
indices must be in range; weights must be finite, non-negative, and sum to a positive value. Violations raise
ValueError — nothing is silently coerced.
"""
from __future__ import annotations
import numpy as np

IDENTITY = -1  # choice sentinel for the f_0 reference action (ΔR ≡ 0)


# ----------------------------------------------------------------------------- fail-closed validators

def _as2d(x, name):
    a = np.asarray(x, dtype=float)
    if a.ndim != 2:
        raise ValueError(f"{name} must be 2-D [n_batches, n_actions], got shape {a.shape}")
    if a.shape[0] == 0 or a.shape[1] == 0:
        raise ValueError(f"{name} must have ≥1 batch and ≥1 action, got shape {a.shape}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return a


def _as1d(x, name, *, dtype=float, finite=True):
    a = np.asarray(x, dtype=dtype)
    if a.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError(f"{name} must be non-empty")
    if finite and not np.all(np.isfinite(a.astype(float))):
        raise ValueError(f"{name} contains non-finite values (NaN/Inf)")
    return a


def _as_choice(choice, *, n_actions=None):
    c = np.asarray(choice)
    if c.ndim != 1 or c.shape[0] == 0:
        raise ValueError("choice must be a non-empty 1-D integer array")
    if not np.issubdtype(c.dtype, np.integer):
        raise ValueError("choice must be integer (-1 = identity, else action index)")
    if np.any(c < IDENTITY):
        raise ValueError("choice values < -1 are invalid (only -1 marks identity)")
    if n_actions is not None and np.any(c >= n_actions):
        raise ValueError(f"choice values must be in {{-1, 0..{n_actions - 1}}}")
    return c


def _as_action_idx(action_idx, *, n_actions=None):
    a = np.asarray(action_idx)
    if a.ndim != 1 or a.shape[0] == 0:
        raise ValueError("action_idx must be a non-empty 1-D integer array")
    if not np.issubdtype(a.dtype, np.integer):
        raise ValueError("action_idx must be integer")
    if np.any(a < 0):
        raise ValueError("action_idx must be ≥ 0")
    if n_actions is not None and np.any(a >= n_actions):
        raise ValueError(f"action_idx must be in {{0..{n_actions - 1}}}")
    return a


def _norm_weights(weights, n):
    """Validate and L1-normalise per-batch weights (None ⇒ uniform). Result sums to 1."""
    if weights is None:
        return np.full(n, 1.0 / n)
    w = np.asarray(weights, dtype=float)
    if w.shape != (n,):
        raise ValueError(f"weights must have shape ({n},), got {w.shape}")
    if not np.all(np.isfinite(w)):
        raise ValueError("weights contain non-finite values (NaN/Inf)")
    if np.any(w < 0):
        raise ValueError("weights must be non-negative")
    s = float(w.sum())
    if not (s > 0):
        raise ValueError("weights must sum to a positive value")
    return w / s


def subject_macro_weights(subject_ids):
    """Per-batch weights for SUBJECT-EQUAL (subject-macro) aggregation:
        weight(batch of subject s) = 1 / (n_subjects · n_batches_of_s)
    Each subject contributes total mass 1/n_subjects (sum over all batches = 1). Fallback rows are weighted like any
    other row of their subject, so they remain in the denominator."""
    ids = np.asarray(subject_ids)
    if ids.ndim != 1 or ids.shape[0] == 0:
        raise ValueError("subject_ids must be a non-empty 1-D array")
    uniq, counts = np.unique(ids, return_counts=True)
    n_subj = uniq.shape[0]
    count_of = {u: c for u, c in zip(uniq.tolist(), counts.tolist())}
    return np.array([1.0 / (n_subj * count_of[i]) for i in ids.tolist()], dtype=float)


# ----------------------------------------------------------------------------- policy families (π_λ)

def safe_set_policy(harm, benefit, lam, *, require_benefit=False):
    """A1 primary family (nested safe-action set).

    Γ_λ(B) = {a : harm[B,a] ≤ λ};  π_λ(B) = identity if Γ_λ empty else argmin_{a∈Γ_λ} benefit[B,a].
    Larger λ ⇒ larger (nested) safe set ⇒ coverage non-decreasing — the budget separates safety screening from
    utility selection. With require_benefit=True the policy also abstains unless the best admitted action is predicted
    to reduce risk (benefit < 0); default False keeps the primary family exactly as specified."""
    harm = _as2d(harm, "harm"); benefit = _as2d(benefit, "benefit")
    if harm.shape != benefit.shape:
        raise ValueError(f"harm {harm.shape} and benefit {benefit.shape} must share shape")
    lam = float(lam)
    if not np.isfinite(lam):
        raise ValueError("lam must be finite")
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
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
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
    gate_score = _as1d(gate_score, "gate_score")
    action_idx = _as_action_idx(action_idx)
    if gate_score.shape[0] != action_idx.shape[0]:
        raise ValueError("gate_score and action_idx must share length")
    tau = float(tau)
    if not np.isfinite(tau):
        raise ValueError("tau must be finite")
    choice = np.full(gate_score.shape[0], IDENTITY, dtype=int)
    take = gate_score >= tau
    choice[take] = action_idx[take]
    return choice


# ----------------------------------------------------------------------------- deployed-policy accounting (weighted)

def coverage(choice, *, weights=None):
    """Adaptation coverage = weighted fraction of batches with a non-identity action (fallback identity rows in the
    denominator). weights=None ⇒ batch-level; pass subject_macro_weights(...) for the subject-macro number."""
    c = _as_choice(choice)
    w = _norm_weights(weights, c.shape[0])
    return float(np.sum(w * (c != IDENTITY)))


def realized_dr(choice, dr):
    """Per-batch realized ΔR of `choice` against true ΔR `dr` [n, A]; identity → 0. Validates choice ∈ {-1,0..A-1}."""
    dr = _as2d(dr, "dr")
    c = _as_choice(choice, n_actions=dr.shape[1])
    if c.shape[0] != dr.shape[0]:
        raise ValueError(f"choice length {c.shape[0]} != n_batches {dr.shape[0]}")
    out = np.zeros(dr.shape[0], dtype=float)
    adapt = np.where(c != IDENTITY)[0]
    out[adapt] = dr[adapt, c[adapt]]
    return out


def reduction(choice, dr, *, weights=None):
    """Deployed NLL reduction red(π) = −weighted_mean_B ΔR_{π(B)} (identity/fallback contribute 0)."""
    r = realized_dr(choice, dr)
    w = _norm_weights(weights, r.shape[0])
    return -float(np.sum(w * r))


def harm_rate(choice, dr, *, weights=None):
    """P(ΔR_{π(B)} > 0 | π(B) ≠ identity) — weighted fraction of ADAPTED batches that were actually harmful.
    NaN if nothing is adapted (zero adapted weight)."""
    dr = _as2d(dr, "dr")
    c = _as_choice(choice, n_actions=dr.shape[1])
    w = _norm_weights(weights, c.shape[0])
    adapt = c != IDENTITY
    wa = w[adapt]
    if float(wa.sum()) <= 0.0:
        return float("nan")
    return float(np.sum(wa * (realized_dr(choice, dr)[adapt] > 0.0)) / wa.sum())


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
