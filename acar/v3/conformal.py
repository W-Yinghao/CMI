"""ACAR v3 subject-clustered conformal + router + harmful-rate test. DESIGN/DEV stage — SYNTHETIC only.

Semantics (notes/ACAR_V3_FREEZE_SKELETON.md S1/S2/S6, Amendment 2/3):
- ONE joint nonconformity score per CAL subject = max over the subject's eligible batches AND all non-identity
  actions of the candidate score. A missing action raises (never silently shrink the simultaneous event).
- conformal rank k = ⌈(m+1)(1−α)⌉ over m = #CAL subjects; k>m ⇒ q = +∞ (strict; no clip/interpolation).
- router: among actions with U_a < −δ pick argmin U_a, ties by canonical action order; else identity.
- harmful-rate: tie-aware paired test on router-adapted batches (exact Wilcoxon only when clean+small, else a
  deterministic sign-flip permutation test); all-zero / too-few pairs ⇒ NOT_EVALUABLE.
- EVAL labels never enter q, predictors, U, or routing; CAL labels affect EVAL only through q.
"""
from __future__ import annotations
import math
import numpy as np

from .set_features import NON_IDENTITY, canonical_tie_break
from .predictors import score, upper_bound


def subject_joint_score(subject_batches, actions=NON_IDENTITY) -> float:
    """subject_batches: list (one per eligible batch) of {action: (CandidatePrediction, delta_r)}.
    Returns max over batches × actions of the candidate nonconformity score."""
    s_max = -math.inf
    for batch in subject_batches:
        for a in actions:
            if a not in batch:
                raise ValueError(f"missing action {a!r} in a CAL batch — would shrink the simultaneous event")
            pred, dr = batch[a]
            s = score(pred, dr)
            if s > s_max:
                s_max = s
    return s_max


def conformal_rank(m, alpha):
    return math.ceil((m + 1) * (1 - alpha))


def conformal_q(subject_scores, alpha):
    """(q, k). m = #CAL subjects; STRICT +inf when k>m (no clip/interpolate)."""
    s = np.sort(np.asarray(subject_scores, float)); m = len(s)
    if m == 0:
        return math.inf, 0
    k = conformal_rank(m, alpha)
    if k > m:
        return math.inf, k
    return float(s[k - 1]), k


def route(preds_by_action, q, delta=0.0):
    """preds_by_action: {action: CandidatePrediction}. Returns (chosen_action, U_by_action). q=+inf -> all U=+inf ->
    identity. Ties among eligible actions broken by canonical action order (not dict order)."""
    U = {a: upper_bound(p, q) for a, p in preds_by_action.items()}
    eligible = [a for a, u in U.items() if u < -delta]
    if not eligible:
        return "identity", U
    mn = min(U[a] for a in eligible)
    chosen = canonical_tie_break([a for a in eligible if U[a] == mn])
    return chosen, U


def harmful_rate_test(router_rates, bestfixed_rates, *, min_pairs=10, exact_max_n=25, n_perm=20000, seed=0):
    """One-sided paired test (H1: router harmful-rate < best-fixed) over subjects WITH ≥1 router-adapted batch.
    Tie-aware: drop zero differences (recorded); exact Wilcoxon only if remaining |diffs| are distinct and n<exact_max_n,
    else a deterministic sign-flip permutation test. all-zero / <min_pairs nonzero ⇒ NOT_EVALUABLE. Holm across sites
    is applied by the caller."""
    router = np.asarray(router_rates, float); bestfixed = np.asarray(bestfixed_rates, float)
    if router.shape != bestfixed.shape or router.ndim != 1:
        raise ValueError("router_rates/bestfixed_rates must be equal-length 1-D arrays (paired by subject)")
    d = router - bestfixed
    n_zero = int(np.sum(d == 0)); dnz = d[d != 0]
    if len(dnz) < min_pairs:
        return dict(evaluable=False, reason="insufficient_nonzero_pairs",
                    n_pairs=int(len(d)), n_zero=n_zero, n_nonzero=int(len(dnz)))
    distinct = len(np.unique(np.abs(dnz))) == len(dnz)
    if distinct and len(dnz) < exact_max_n:
        from scipy.stats import wilcoxon
        r = wilcoxon(dnz, alternative="less", zero_method="wilcox", mode="exact")
        return dict(evaluable=True, method="exact_wilcoxon", p=float(r.pvalue), stat=float(r.statistic),
                    n_nonzero=int(len(dnz)), n_zero=n_zero)
    rng = np.random.default_rng(seed); obs = float(dnz.sum())
    signs = rng.choice((-1.0, 1.0), size=(n_perm, len(dnz)))
    perm = (signs * np.abs(dnz)).sum(1)
    p = float((np.sum(perm <= obs) + 1) / (n_perm + 1))       # one-sided (router lower)
    return dict(evaluable=True, method="signflip_permutation", p=p, stat=obs,
                n_nonzero=int(len(dnz)), n_zero=n_zero, n_perm=int(n_perm), seed=int(seed))
