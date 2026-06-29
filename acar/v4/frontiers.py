"""ACAR v4 (CURB) — Direction C: information-limit risk–coverage frontiers.

NON-BINDING / POST-V3 / synthetic-capable: pure numpy over arrays of TRUE ΔR (`dr`, [n_batches, n_actions], lower is
better) and label-free scores. Reads no real cohort, fits nothing, selects nothing. Implemented FIRST
(notes/ACAR_V4_DEV_EXPLORATION_PLAN.md §4) because the frontier audit most quickly answers WHY v3 failed: information,
policy-learning, or calibration limit.

Axes (per disease, subject-disjoint OOF cells):
  x = adaptation coverage   (weighted fraction of batches with a non-identity action; fallback identity rows in denom)
  y = deployed NLL reduction red = −weighted_mean_B ΔR_{π(B)}   (identity/fallback → 0; positive = good)
  harm = P(ΔR_{π(B)} > 0 | π(B) ≠ identity)   (weighted, among adapted)

WEIGHTING (contract). Every frontier takes `weights=None`. weights=None ⇒ uniform per-batch (a batch-level primitive,
NOT a DEV conclusion). For a SUBJECT-MACRO DEV number, pass acar.v4.policies.subject_macro_weights(subject_ids) — the
same weights flow through coverage, red, and harm. Fallback rows must be present so they remain in the denominator.

Four frontiers:
  F_true_oracle        : TRUE ΔR selects batches & actions       → ceiling = global max red (any router's upper bound)
  F_single_score_oracle: ONE label-free score ranking + action; TRUE ΔR evaluates; best hindsight coverage
  F_score_oracle (union): UPPER ENVELOPE over a pre-listed set of single-score frontiers = the information ceiling of
                          those observables. A single score is only ONE rule, not the full information ceiling.
  F_policy_family      : the v4 π_λ family's operating points (fitted scores, no oracle threshold)
  F_calibrated         : discrete calibrated operating points (v2 router, v3 C1/C2/C3 as references)

Gap decomposition has TWO parallel outputs (never mixed):
  mode="ceiling" (main diagnostic): ceiling differences, an EXACT telescoping identity
      info_gap + policy_gap + calibration_gap = true_ceiling − calibrated_red
      info_gap = ceil(true) − ceil(score_union) is guaranteed ≥ 0 (true ceiling is the global max red).
      policy_gap / calibration_gap are signed (a negative value flags an informative inversion).
  mode="auc" (descriptive only, NOT pass/fail): area-under-the-frontier (over coverage, on the Pareto envelope) for
      true / score / policy, and the auc info/policy gaps. The calibrated point has no area, so calibration stays the
      ceiling-based reference. AUC never replaces the ceiling diagnostic.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from acar.v4 import policies as P


@dataclass(frozen=True)
class Frontier:
    coverage: np.ndarray   # length m
    red: np.ndarray        # deployed reduction at each coverage
    harm: np.ndarray       # harm rate among adapted at each coverage (nan where nothing adapted)

    def ceiling(self):
        """Max deployed reduction along the frontier (the best hindsight operating point)."""
        return float(np.max(self.red)) if self.red.size else 0.0

    def coverage_at_ceiling(self):
        return float(self.coverage[int(np.argmax(self.red))]) if self.red.size else 0.0


# ----------------------------------------------------------------------------- core builders

def best_action(dr):
    """Per-batch best NON_IDENTITY action (argmin ΔR) and its ΔR value."""
    dr = P._as2d(dr, "dr")
    idx = np.argmin(dr, axis=1)
    return idx, dr[np.arange(dr.shape[0]), idx]


def _topk_frontier(dr, rank, action_idx, weights):
    """Frontier traced by adapting batches in descending `rank` order, each with action_idx[b], evaluated on TRUE ΔR,
    weighted by per-batch `weights` (normalised to sum 1). coverage[k] = Σ_{first k} w; red[k] = −Σ_{first k} w·ΔR;
    harm[k] = weighted fraction of the first k that are harmful. k = 0..n."""
    dr = P._as2d(dr, "dr")
    n = dr.shape[0]
    rank = P._as1d(rank, "rank_score")
    action_idx = P._as_action_idx(action_idx, n_actions=dr.shape[1])
    if rank.shape[0] != n or action_idx.shape[0] != n:
        raise ValueError("rank and action_idx must have length n_batches")
    w = P._norm_weights(weights, n)
    order = np.argsort(-rank, kind="stable")                 # highest rank adapted first (stable, deterministic)
    chosen = dr[order, action_idx[order]]                    # realized ΔR in adopt order
    w_ord = w[order]
    cov = np.concatenate([[0.0], np.cumsum(w_ord)])
    red = -np.concatenate([[0.0], np.cumsum(w_ord * chosen)])
    harm_num = np.concatenate([[0.0], np.cumsum(w_ord * (chosen > 0.0))])
    harm = np.full(n + 1, np.nan)
    denom = cov[1:]
    harm[1:] = np.where(denom > 0.0, harm_num[1:] / np.where(denom > 0.0, denom, 1.0), np.nan)
    return Frontier(coverage=cov, red=red, harm=harm)


# ----------------------------------------------------------------------------- the four frontiers

def frontier_true_oracle(dr, *, weights=None):
    """Choose batches & actions by TRUE ΔR: adapt the most-reducing batches first, each with its best action.
    Its ceiling = the global maximum deployed reduction (adapt every batch whose best action has ΔR<0)."""
    idx, val = best_action(dr)
    return _topk_frontier(dr, rank=-val, action_idx=idx, weights=weights)   # most-reducing (most negative ΔR) first


def frontier_single_score_oracle(dr, rank_score, action_idx, *, weights=None):
    """ONE label-free `rank_score` (higher ⇒ adapt first) + ONE label-free `action_idx`, evaluated on TRUE ΔR. Its
    ceiling is the best hindsight coverage of THIS single rule — not the full information ceiling (see the union)."""
    return _topk_frontier(dr, rank=rank_score, action_idx=action_idx, weights=weights)


def frontier_score_oracle_union(dr, candidates, *, weights=None):
    """Information ceiling of a PRE-LISTED set of label-free observables: the Pareto upper envelope over the
    single-score frontiers of each (rank_score, action_idx) in `candidates`. Its ceiling = max over the listed rules."""
    cands = list(candidates)
    if not cands:
        raise ValueError("candidates must be a non-empty list of (rank_score, action_idx) pairs")
    singles = [frontier_single_score_oracle(dr, r, a, weights=weights) for (r, a) in cands]
    cov = np.concatenate([f.coverage for f in singles])
    red = np.concatenate([f.red for f in singles])
    harm = np.concatenate([f.harm for f in singles])
    return pareto_upper_envelope(Frontier(coverage=cov, red=red, harm=harm))


def operating_point(dr, choice, *, weights=None):
    """(coverage, red, harm) of a single executed policy `choice` against TRUE ΔR."""
    return (P.coverage(choice, weights=weights),
            P.reduction(choice, dr, weights=weights),
            P.harm_rate(choice, dr, weights=weights))


def frontier_policy_family(dr, choices, *, weights=None):
    """Operating points of a family of executed policies (e.g., safe_set_policy over a λ grid), sorted by coverage.
    These are RAW operating points; call pareto_upper_envelope(...) for the plotted frontier."""
    pts = sorted((operating_point(dr, c, weights=weights) for c in choices), key=lambda t: t[0])
    if not pts:
        return Frontier(coverage=np.zeros(0), red=np.zeros(0), harm=np.zeros(0))
    cov = np.array([p[0] for p in pts]); red = np.array([p[1] for p in pts]); harm = np.array([p[2] for p in pts])
    return Frontier(coverage=cov, red=red, harm=harm)


def calibrated_points(dr, named_choices, *, weights=None):
    """{label: (coverage, red, harm)} for reference policies (v2 router, v3 C1/C2/C3, …)."""
    return {k: operating_point(dr, v, weights=weights) for k, v in named_choices.items()}


# ----------------------------------------------------------------------------- envelope / area

def pareto_upper_envelope(frontier):
    """Non-dominated upper-left staircase: sort by coverage ascending (ties: red descending), keep a point only if its
    red strictly exceeds the running max. Drops points dominated by a lower-or-equal-coverage point with ≥ red, so the
    result has strictly increasing coverage and strictly increasing red — the frontier you plot."""
    cov = np.asarray(frontier.coverage, dtype=float)
    red = np.asarray(frontier.red, dtype=float)
    harm = np.asarray(frontier.harm, dtype=float)
    if cov.size == 0:
        return Frontier(coverage=cov, red=red, harm=harm)
    order = np.lexsort((-red, cov))          # primary: coverage asc; secondary: red desc
    keep = []
    best = -np.inf
    for i in order:
        if red[i] > best + 1e-12:
            keep.append(i)
            best = red[i]
    keep = np.array(keep, dtype=int)
    return Frontier(coverage=cov[keep], red=red[keep], harm=harm[keep])


def frontier_auc(frontier, *, envelope=True):
    """Area under the red-vs-coverage frontier (trapezoidal over coverage). envelope=True integrates the Pareto upper
    envelope (the plotted curve); envelope=False integrates the raw points sorted by coverage. DESCRIPTIVE ONLY."""
    f = pareto_upper_envelope(frontier) if envelope else frontier
    cov = np.asarray(f.coverage, dtype=float); red = np.asarray(f.red, dtype=float)
    if cov.size < 2:
        return 0.0
    o = np.argsort(cov, kind="stable")
    return float(np.trapz(red[o], cov[o]))


# ----------------------------------------------------------------------------- gap decomposition

def _as_candidate_list(score_candidates):
    """Accept either a single (rank_score, action_idx) pair or a list of such pairs; return a list."""
    if (isinstance(score_candidates, tuple) and len(score_candidates) == 2
            and np.ndim(score_candidates[0]) == 1 and np.ndim(score_candidates[1]) == 1):
        return [score_candidates]
    return list(score_candidates)


def gap_decomposition(dr, score_candidates, policy_choices, calibrated_choice, *, weights=None, mode="ceiling"):
    """Decompose the measurement→control gap into information / policy-learning / calibration gaps.

    score_candidates : a list of (rank_score, action_idx) label-free rules (or a single pair); the score frontier is
                       their union upper envelope (the information ceiling of those observables).
    policy_choices   : the executed π_λ family (list of choice arrays).
    calibrated_choice: the single frozen calibrated policy.
    weights          : per-batch weights (None=uniform; pass subject_macro_weights for the DEV number).
    mode             : "ceiling" (main, exact telescoping) or "auc" (descriptive area summary, not pass/fail)."""
    cands = _as_candidate_list(score_candidates)
    f_true = frontier_true_oracle(dr, weights=weights)
    f_score = frontier_score_oracle_union(dr, cands, weights=weights)
    f_policy = frontier_policy_family(dr, policy_choices, weights=weights)
    cal_cov, cal_red, cal_harm = operating_point(dr, calibrated_choice, weights=weights)
    ceil_true, ceil_score, ceil_policy = f_true.ceiling(), f_score.ceiling(), f_policy.ceiling()
    out = {
        "mode": mode,
        "true_ceiling": ceil_true,
        "score_ceiling": ceil_score,
        "policy_ceiling": ceil_policy,
        "calibrated_red": cal_red,
        "calibrated_coverage": cal_cov,
        "calibrated_harm": cal_harm,
        "true_coverage_at_ceiling": f_true.coverage_at_ceiling(),
        "score_coverage_at_ceiling": f_score.coverage_at_ceiling(),
        "policy_coverage_at_ceiling": f_policy.coverage_at_ceiling(),
    }
    if mode == "ceiling":
        out.update({
            "info_gap": ceil_true - ceil_score,
            "policy_gap": ceil_score - ceil_policy,
            "calibration_gap": ceil_policy - cal_red,
        })
    elif mode == "auc":
        auc_true = frontier_auc(f_true); auc_score = frontier_auc(f_score); auc_policy = frontier_auc(f_policy)
        out.update({
            "true_auc": auc_true,
            "score_auc": auc_score,
            "policy_auc": auc_policy,
            "info_gap": auc_true - auc_score,           # AUC-based (descriptive)
            "policy_gap": auc_score - auc_policy,        # AUC-based (descriptive)
            "calibration_gap_ceiling": ceil_policy - cal_red,   # calibrated point has no area → ceiling reference
        })
    else:
        raise ValueError(f"mode must be 'ceiling' or 'auc', got {mode!r}")
    return out
