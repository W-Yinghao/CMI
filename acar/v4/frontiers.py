"""ACAR v4 (CURB) — Direction C: information-limit risk–coverage frontiers.

NON-BINDING / POST-V3 / synthetic-capable: pure numpy over arrays of TRUE ΔR (`dr`, [n_batches, n_actions], lower is
better) and label-free scores. Reads no real cohort, fits nothing, selects nothing. Implemented FIRST
(notes/ACAR_V4_DEV_EXPLORATION_PLAN.md §4) because the frontier audit most quickly answers WHY v3 failed: information,
policy-learning, or calibration limit.

Axes (per disease, subject-disjoint OOF cells):
  x = adaptation coverage   (fraction of batches with a non-identity action; fallback identity rows in denominator)
  y = deployed NLL reduction red = −mean_B ΔR_{π(B)}   (identity/fallback → 0; positive = good)
  harm = P(ΔR_{π(B)} > 0 | π(B) ≠ identity)

Four frontiers:
  F_true_oracle  : TRUE ΔR selects batches & actions       → ceiling of ANY router (global max red)
  F_score_oracle : label-free SCORES select; TRUE ΔR evaluates; best hindsight coverage = information ceiling
  F_policy_family: the v4 π_λ family's operating points (fitted scores, no oracle threshold)
  F_calibrated   : discrete calibrated operating points (v2 router, v3 C1/C2/C3 as references)

Gap decomposition (signed diagnostics; usually true ≥ score ≥ policy ≥ calibrated):
  info_gap        = ceil(true)  − ceil(score)      label-free observables lack the info  (always ≥ 0)
  policy_gap      = ceil(score) − ceil(policy)      model/policy fails to exploit info    (signed)
  calibration_gap = ceil(policy) − red(calibrated)  calibration too conservative          (signed)
info_gap ≥ 0 is guaranteed: ceil(true) is the global maximum of red over all policies, so no score-driven frontier can
exceed it. policy_gap / calibration_gap are signed: a negative value flags an informative inversion (a policy beating
the naive score-threshold curve, or a calibrated point above the policy family's own ceiling) and is reported, not
clipped.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np

from acar.v4 import policies as P


@dataclass(frozen=True)
class Frontier:
    coverage: np.ndarray   # non-decreasing, length m
    red: np.ndarray        # deployed reduction at each coverage
    harm: np.ndarray       # harm rate among adapted at each coverage (nan at coverage 0)

    def ceiling(self):
        """Max deployed reduction along the frontier (the best hindsight operating point)."""
        return float(np.max(self.red)) if self.red.size else 0.0

    def coverage_at_ceiling(self):
        return float(self.coverage[int(np.argmax(self.red))]) if self.red.size else 0.0


def _as2d(dr):
    dr = np.asarray(dr, dtype=float)
    if dr.ndim != 2:
        raise ValueError(f"dr must be 2-D [n_batches, n_actions], got shape {dr.shape}")
    if dr.shape[0] == 0:
        raise ValueError("dr must have at least one batch")
    return dr


def _topk_frontier(dr, rank, action_idx):
    """Frontier traced by adapting batches in descending `rank` order, each with action_idx[b], evaluated on TRUE ΔR.
    k = 0..n. coverage[k]=k/n; red[k] = −(1/n) Σ_{first k} ΔR_chosen; harm[k] = fraction of those k with ΔR_chosen>0."""
    dr = _as2d(dr)
    n = dr.shape[0]
    rank = np.asarray(rank, dtype=float).ravel()
    action_idx = np.asarray(action_idx, dtype=int).ravel()
    if rank.shape[0] != n or action_idx.shape[0] != n:
        raise ValueError("rank and action_idx must have length n_batches")
    order = np.argsort(-rank, kind="stable")                 # highest rank adapted first (stable, deterministic)
    chosen = dr[order, action_idx[order]]                    # realized ΔR in adopt order
    k = np.arange(0, n + 1)
    cov = k / n
    red = -np.concatenate([[0.0], np.cumsum(chosen)]) / n
    harm_cnt = np.concatenate([[0.0], np.cumsum((chosen > 0.0).astype(float))])
    harm = np.full(n + 1, np.nan)
    harm[1:] = harm_cnt[1:] / k[1:]
    return Frontier(coverage=cov, red=red, harm=harm)


def best_action(dr):
    """Per-batch best NON_IDENTITY action (argmin ΔR) and its ΔR value."""
    dr = _as2d(dr)
    idx = np.argmin(dr, axis=1)
    return idx, dr[np.arange(dr.shape[0]), idx]


# ----------------------------------------------------------------------------- the four frontiers

def frontier_true_oracle(dr):
    """Choose batches & actions by TRUE ΔR: adapt the most-reducing batches first, each with its best action.
    Its ceiling = the global maximum deployed reduction (adapt every batch whose best action has ΔR<0)."""
    idx, val = best_action(dr)
    return _topk_frontier(dr, rank=-val, action_idx=idx)     # most-reducing (most negative ΔR) first


def frontier_score_oracle(dr, rank_score, action_idx):
    """Choose batches by a label-free `rank_score` (higher ⇒ adapt first) with label-free `action_idx`; evaluate on
    TRUE ΔR. The ceiling = best hindsight coverage = the information ceiling of these observables."""
    return _topk_frontier(dr, rank=rank_score, action_idx=action_idx)


def operating_point(dr, choice):
    """(coverage, red, harm) of a single executed policy `choice` against TRUE ΔR."""
    return P.coverage(choice), P.reduction(choice, dr), P.harm_rate(choice, dr)


def frontier_policy_family(dr, choices):
    """Operating points of a family of executed policies (e.g., safe_set_policy over a λ grid), sorted by coverage."""
    pts = sorted((operating_point(dr, c) for c in choices), key=lambda t: t[0])
    if not pts:
        return Frontier(coverage=np.zeros(0), red=np.zeros(0), harm=np.zeros(0))
    cov = np.array([p[0] for p in pts]); red = np.array([p[1] for p in pts]); harm = np.array([p[2] for p in pts])
    return Frontier(coverage=cov, red=red, harm=harm)


def calibrated_points(dr, named_choices):
    """{label: (coverage, red, harm)} for reference policies (v2 router, v3 C1/C2/C3, …)."""
    return {k: operating_point(dr, v) for k, v in named_choices.items()}


# ----------------------------------------------------------------------------- gap decomposition

def gap_decomposition(dr, rank_score, action_idx, policy_choices, calibrated_choice):
    """Decompose the measurement→control gap into information / policy-learning / calibration gaps.

    rank_score / action_idx are the label-free ranking + per-batch action that the score-oracle uses (typically the
    same scores the policy family is built from). policy_choices is the executed π_λ family; calibrated_choice is the
    single frozen calibrated policy. Returns a flat dict of ceilings, the calibrated operating point, and the three
    signed gaps (info_gap is guaranteed ≥ 0 up to fp error)."""
    f_true = frontier_true_oracle(dr)
    f_score = frontier_score_oracle(dr, rank_score, action_idx)
    f_policy = frontier_policy_family(dr, policy_choices)
    cal_cov, cal_red, cal_harm = operating_point(dr, calibrated_choice)
    ceil_true, ceil_score, ceil_policy = f_true.ceiling(), f_score.ceiling(), f_policy.ceiling()
    return {
        "true_ceiling": ceil_true,
        "score_ceiling": ceil_score,
        "policy_ceiling": ceil_policy,
        "calibrated_red": cal_red,
        "calibrated_coverage": cal_cov,
        "calibrated_harm": cal_harm,
        "true_coverage_at_ceiling": f_true.coverage_at_ceiling(),
        "score_coverage_at_ceiling": f_score.coverage_at_ceiling(),
        "policy_coverage_at_ceiling": f_policy.coverage_at_ceiling(),
        "info_gap": ceil_true - ceil_score,
        "policy_gap": ceil_score - ceil_policy,
        "calibration_gap": ceil_policy - cal_red,
    }
