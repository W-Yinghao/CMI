"""ACAR V5 statistical certification (ENDPOINTS §2, PINNED — Step 2c): subject-clustered one-sided empirical-Bernstein bounds on
per-subject variables in [0,1], and the G1/G3/G4 gate evaluation. Pure (math/stdlib). Subject is the cluster.

    radius(α) = sqrt( 2·var·log(3/α)/n ) + 3·log(3/α)/n     (var = unbiased sample variance clipped to [0, 0.25])
    UCB_α = min(1, xbar + radius)     LCB_α = max(0, xbar − radius)

G4 (harm_among_adapted) is computed ONLY over adapting subjects; if NO subject adapts it is NON-EVALUABLE and the candidate FAILS
(this is the formal close of the v4 low-coverage degeneracy: G3 covers all-batch harm, G4 is not diluted by non-adapting subjects).
"""
from __future__ import annotations
import math
from acar.v5 import protocol as P
from acar.v5 import metrics as M


def _mean(xs):
    return sum(xs) / len(xs)


def _clip_var(xs):
    n = len(xs)
    if n < 2:
        return 0.25                                            # conservative (max Bernoulli variance) when n<2
    mu = _mean(xs)
    v = sum((x - mu) ** 2 for x in xs) / (n - 1)              # unbiased
    return min(0.25, max(0.0, v))


def eb_radius(xs, alpha=P.ALPHA):
    n = len(xs)
    if n == 0:
        raise ValueError("empirical-Bernstein needs n≥1")
    var = _clip_var(xs)
    t = math.log(3.0 / alpha)
    return math.sqrt(2.0 * var * t / n) + 3.0 * t / n


def eb_ucb(xs, alpha=P.ALPHA):
    return min(1.0, _mean(xs) + eb_radius(xs, alpha))


def eb_lcb(xs, alpha=P.ALPHA):
    return max(0.0, _mean(xs) - eb_radius(xs, alpha))


def gate_disease(subject_records, alpha=P.ALPHA):
    """Evaluate the certification gates G1 (coverage LCB), G3 (L_harm_all UCB), G4 (harm_among_adapted UCB) for ONE disease.
    Returns a dict of the bounds + per-gate booleans + H2 evaluability. Does NOT include G2/G5 (those are computed elsewhere)."""
    c = M.collect(subject_records)
    coverage_lcb = eb_lcb(c["coverage"], alpha)
    l_harm_all_ucb = eb_ucb(c["l_harm_all"], alpha)
    h2_evaluable = c["n_adapting"] > 0
    harm_adapted_ucb = eb_ucb(c["harm_among_adapted"], alpha) if h2_evaluable else None
    g1 = coverage_lcb >= P.COVERAGE_MIN
    g3 = l_harm_all_ucb <= P.L_HARM_ALL_MAX
    g4 = bool(h2_evaluable) and (harm_adapted_ucb <= P.HARM_AMONG_ADAPTED_MAX)   # non-evaluable ⇒ FAIL
    return {
        "n_subjects": c["n_subjects"], "n_adapting": c["n_adapting"],
        "coverage_lcb": coverage_lcb, "l_harm_all_ucb": l_harm_all_ucb,
        "harm_among_adapted_ucb": harm_adapted_ucb, "h2_evaluable": h2_evaluable,
        "G1_coverage": g1, "G3_l_harm_all": g3, "G4_harm_among_adapted": g4,
        "certification_pass": bool(g1 and g3 and g4),
    }
