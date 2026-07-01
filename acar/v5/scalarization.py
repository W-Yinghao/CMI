"""ACAR V5 action-record scalarization & FIT quantile universe (CANDIDATE_SPACE §1.7) — makes the 22 manifest rows bit-executable.
Pure (numpy). Label-free: routing reads ONLY the action-indexed features, never labels.

A synthetic "batch" is: {"batch_id": str, "features": {action -> {feature -> float}}} for every non-identity action in
P.ACTIONS. The proposed action a*(B), the FIT-only quantile thresholds, and the adapt/abstain decision follow the pinned rules;
a candidate with ZERO FIT proposed-action records is NON-EVALUABLE (fails).
"""
from __future__ import annotations
import math
from acar.v5 import protocol as P


class NonEvaluableCandidate(RuntimeError):
    """Raised when a candidate has zero FIT proposed-action records (CANDIDATE_SPACE §1.7)."""


def _feat(batch, action, name):
    return float(batch["features"][action][name])


def _argmax_action(batch, feature):
    """argmax over non-identity actions, ties broken by the fixed ACTION_ORDER (matched_coral ≺ spdim ≺ t3a)."""
    best, best_v = None, None
    for a in sorted(P.ACTIONS, key=lambda x: P.ACTION_ORDER[x]):
        v = _feat(batch, a, feature)
        if best is None or v > best_v + 1e-12:                # strict improvement; equal keeps earlier (tie-break)
            best, best_v = a, v
    return best


def _argmin_action(batch, feature):
    best, best_v = None, None
    for a in sorted(P.ACTIONS, key=lambda x: P.ACTION_ORDER[x]):
        v = _feat(batch, a, feature)
        if best is None or v < best_v - 1e-12:
            best, best_v = a, v
    return best


def proposed_action(candidate, batch):
    """a*(B) per family (CANDIDATE_SPACE §1.7). Returns an action string, or None if no proposal (P4 no agreement)."""
    fam = candidate["family"]
    if fam in ("P1", "P2", "P5"):
        return _argmax_action(batch, "d_margin")
    if fam == "P3":
        return candidate["params"]["action"]
    if fam == "P4":
        cand = {"margin": _argmax_action(batch, "d_margin"),
                "post_sep": _argmax_action(batch, "post_sep"),
                "js": _argmin_action(batch, "JS")}
        votes = {}
        for a in cand.values():
            votes[a] = votes.get(a, 0) + 1
        k = candidate["params"]["k"]
        winners = [a for a, n in votes.items() if n >= k]
        if not winners:
            return None
        return sorted(winners, key=lambda x: P.ACTION_ORDER[x])[0]
    raise ValueError(f"unknown family {fam!r}")


def _Q(xs, level):
    """Explicit linear (Type-7) quantile — PINNED (Step 3b), no dependency on any library's default interpolation. This is
    numpy's default 'linear' method, made bit-stable and permutation-independent (input is sorted)."""
    q = P.QUANTILE_VALUE[level]
    arr = sorted(float(x) for x in xs)
    n = len(arr)
    if n == 0:
        raise NonEvaluableCandidate("empty quantile input")
    if n == 1:
        return arr[0]
    h = (n - 1) * q
    lo = int(math.floor(h))
    hi = int(math.ceil(h))
    if lo == hi:
        return arr[lo]
    w = h - lo
    return (1.0 - w) * arr[lo] + w * arr[hi]


def fit_quantiles(candidate, fit_batches):
    """Compute the FIT-only thresholds over THIS candidate's proposed-action a* records only. Raises NonEvaluableCandidate if
    there are zero such records. Returns a thresholds dict consumed by `decide`."""
    fam, prm = candidate["family"], candidate["params"]
    dmarg, flip, js = [], [], []
    for b in fit_batches:
        a = proposed_action(candidate, b)
        if a is None:                                          # P4 no-agreement batch → no proposed-action record
            continue
        dmarg.append(_feat(b, a, "d_margin"))
        flip.append(_feat(b, a, "flip_rate"))
        js.append(_feat(b, a, "JS"))
    if not dmarg:
        raise NonEvaluableCandidate(f"{candidate['id']}: zero FIT proposed-action records")
    th = {"veto_flip": _Q(flip, prm["veto_q"]), "veto_js": _Q(js, prm["veto_q"])}
    if fam == "P1":
        th["benefit"] = _Q(dmarg, prm["benefit_q"])
    elif fam == "P2":
        th["conf"] = _Q(dmarg, prm["conf_q"])
    elif fam == "P5":
        th["lambda"] = _Q(dmarg, prm["lambda_q"])
    # P3/P4 use only the harm veto
    return th


def _harm_veto_ok(batch, a, th):
    return _feat(batch, a, "flip_rate") <= th["veto_flip"] + 1e-12 and _feat(batch, a, "JS") <= th["veto_js"] + 1e-12


def decide(candidate, batch, thresholds):
    """Return the chosen action for a batch: a non-identity action if the candidate adapts, else IDENTITY (abstain).
    Uses ONLY the frozen FIT thresholds + label-free features."""
    fam = candidate["family"]
    a = proposed_action(candidate, batch)
    if a is None:
        return P.IDENTITY
    if not _harm_veto_ok(batch, a, thresholds):
        return P.IDENTITY
    if fam == "P1":
        return a if _feat(batch, a, "d_margin") >= thresholds["benefit"] - 1e-12 else P.IDENTITY
    if fam == "P2":
        return a if (_feat(batch, a, "d_margin") >= thresholds["conf"] - 1e-12 and _feat(batch, a, "d_entropy") <= 0.0) else P.IDENTITY
    if fam == "P3":
        return a                                              # fixed action, veto already passed
    if fam == "P4":
        return a                                              # agreed action, veto already passed
    if fam == "P5":
        return a if _feat(batch, a, "d_margin") >= thresholds["lambda"] - 1e-12 else P.IDENTITY
    raise ValueError(f"unknown family {fam!r}")
