"""ACAR v3 subject-clustered conformal + router + harmful-rate test. DESIGN/DEV stage — SYNTHETIC only.
FAIL-CLOSED (Amendment 4): malformed inputs raise rather than silently producing low nonconformity / shrinking the
simultaneous action event. ONE harmful-rate estimand (Wilcoxon signed-rank): exact when clean, scipy PermutationMethod
for ties/zeros (no statistic switch). Semantics frozen in notes/ACAR_V3_FREEZE_SKELETON.md S1/S2/S6.
"""
from __future__ import annotations
import math
import numpy as np

from .set_features import NON_IDENTITY, canonical_tie_break
from .predictors import score, upper_bound, CandidatePrediction

_NSET = frozenset(NON_IDENTITY)


def _check_action_dict(d, where):
    if set(d) != _NSET:
        raise ValueError(f"{where}: actions {sorted(d)} != required non-identity set {sorted(NON_IDENTITY)}")


def _consistency(preds, seen):
    for a, p in preds.items():
        if not isinstance(p, CandidatePrediction):
            raise TypeError(f"expected CandidatePrediction for {a!r}")
        if p.action != a:
            raise ValueError(f"prediction.action {p.action!r} != dict key {a!r}")
        seen["cand"].add(p.candidate); seen["dis"].add(p.disease)
    if len(seen["cand"]) > 1 or len(seen["dis"]) > 1:
        raise ValueError(f"mixed candidate/disease within subject: {seen}")


def subject_joint_score(subject_batches) -> float:
    """max over a subject's eligible batches × all non-identity actions of the candidate score. FAIL-CLOSED:
    non-empty; each batch carries EXACTLY the non-identity actions; ΔR/scores finite; action keys + candidate/disease
    consistent."""
    if not subject_batches:
        raise ValueError("subject has no eligible batches")
    seen = {"cand": set(), "dis": set()}
    s_max = -math.inf
    for bi, batch in enumerate(subject_batches):
        _check_action_dict(batch, f"batch[{bi}]")
        preds = {a: batch[a][0] for a in batch}
        _consistency(preds, seen)
        for a in NON_IDENTITY:
            pred, dr = batch[a]
            if not math.isfinite(float(dr)):
                raise ValueError("non-finite ΔR in CAL batch")
            s = score(pred, dr)
            if not math.isfinite(s):
                raise ValueError("non-finite nonconformity score")
            if s > s_max:
                s_max = s
    return s_max


def conformal_rank(m, alpha):
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must be in (0,1)")
    if m < 0:
        raise ValueError("m must be >= 0")
    return math.ceil((m + 1) * (1 - alpha))


def conformal_q(subject_scores, alpha):
    """(q, k). m = #CAL subjects. Empty CAL -> (+inf, k=1) (k>m via the frozen formula). STRICT +inf when k>m."""
    s = np.asarray(list(subject_scores), float)
    if s.size and not np.all(np.isfinite(s)):
        raise ValueError("non-finite CAL subject score")
    m = int(s.size)
    k = conformal_rank(m, alpha)
    if k > m:
        return math.inf, k
    return float(np.sort(s)[k - 1]), k


def route(preds_by_action, q, delta=0.0):
    """preds_by_action MUST be exactly the non-identity actions. Returns (chosen, U) with U in canonical action order.
    q may be +inf (uninformative -> identity); NaN q rejected."""
    _check_action_dict(preds_by_action, "route")
    _consistency(preds_by_action, {"cand": set(), "dis": set()})
    if math.isnan(float(q)):
        raise ValueError("q must not be NaN")
    U = {a: upper_bound(preds_by_action[a], q) for a in NON_IDENTITY}     # canonical order
    eligible = [a for a in NON_IDENTITY if U[a] < -float(delta)]
    if not eligible:
        return "identity", U
    mn = min(U[a] for a in eligible)
    return canonical_tie_break([a for a in eligible if U[a] == mn]), U


def harmful_rate_test(router_rates, bestfixed_rates, *, min_pairs=10, exact_max_n=25, n_perm=20000, seed=0):
    """ONE estimand: one-sided Wilcoxon signed-rank (H1: router < best-fixed) on per-subject rate differences.
    Drop zero diffs (recorded); <min_pairs nonzero ⇒ NOT_EVALUABLE; exact only when |d| distinct & n<exact_max_n,
    else scipy PermutationMethod (deterministic seed). Holm across sites applied by the caller."""
    from scipy.stats import wilcoxon, PermutationMethod
    router = np.asarray(router_rates, float); bestfixed = np.asarray(bestfixed_rates, float)
    if router.ndim != 1 or router.shape != bestfixed.shape:
        raise ValueError("router/bestfixed must be equal-length 1-D arrays (paired by subject)")
    if not (np.all(np.isfinite(router)) and np.all(np.isfinite(bestfixed))):
        raise ValueError("non-finite rate")
    d = router - bestfixed
    n_zero = int(np.sum(d == 0)); dnz = d[d != 0]
    if len(dnz) < min_pairs:
        return dict(evaluable=False, reason="insufficient_nonzero_pairs",
                    n_pairs=int(len(d)), n_zero=n_zero, n_nonzero=int(len(dnz)))
    distinct = len(np.unique(np.abs(dnz))) == len(dnz)
    if distinct and len(dnz) < exact_max_n:
        r = wilcoxon(dnz, alternative="less", zero_method="wilcox", method="exact")
        method = "exact_wilcoxon"
    else:
        r = wilcoxon(dnz, alternative="less", zero_method="wilcox",
                     method=PermutationMethod(n_resamples=n_perm, random_state=seed))
        method = "permutation_wilcoxon"
    return dict(evaluable=True, method=method, p=float(r.pvalue), statistic=float(r.statistic),
                n_nonzero=int(len(dnz)), n_zero=n_zero)
