"""ACAR v3 DEV split-as-ONE-algorithm (S5). DESIGN/DEV stage — SYNTHETIC only until DEV_DESIGN_LOCK.

Deterministic, PERMUTATION-INDEPENDENT subject splits keyed on the canonical dataset-aware SubjectKey:
    outer K folds (each EVAL once; seed_outer)  →  non-EVAL hash-split FIT/CAL (seed_fitcal, fit_frac)
                                                →  FIT hash-split TRAIN/VAL (seed_es, train_frac)
Every split is a pure function of {canon_subject(sk)} and a frozen (salt, seed, ratio) — never of input order. Seeds /
folds / ratios are frozen in `predictors.HP`. FAIL-CLOSED on duplicate / degenerate inputs.
"""
from __future__ import annotations
import hashlib

from .data import SubjectKey, canon_subject
from .predictors import HP


def _u01(sk: SubjectKey, salt: str) -> float:
    """Deterministic uniform-ish [0,1) hash of a canonical SubjectKey under a salt."""
    if not isinstance(sk, SubjectKey):
        raise TypeError("split keys must be SubjectKey")
    h = hashlib.sha256((salt + "|" + canon_subject(sk)).encode()).digest()
    return int.from_bytes(h[:8], "big") / 2.0 ** 64


def _unique(subjects):
    out, seen = [], set()
    for sk in subjects:
        c = canon_subject(sk)
        if c in seen:
            raise ValueError(f"duplicate subject in split input: {c}")
        seen.add(c); out.append(sk)
    if not out:
        raise ValueError("empty subject set")
    return out


def _ordered(subjects, salt):
    """Hash-ordered (then canon-tie-broken) — independent of input order."""
    return sorted(subjects, key=lambda sk: (_u01(sk, salt), canon_subject(sk)))


def outer_folds(subjects, k=None, seed_outer=None):
    """K subject-disjoint folds (each is EVAL in turn). Round-robin over the hash-order -> balanced (sizes differ by
    ≤1), deterministic, permutation-independent."""
    k = HP["k_folds"] if k is None else k
    seed_outer = HP["seed_outer"] if seed_outer is None else seed_outer
    subs = _unique(subjects)
    if isinstance(k, bool) or not isinstance(k, int) or k < 2:
        raise ValueError("k must be an int >= 2")
    if len(subs) < k:
        raise ValueError(f"need >= k={k} subjects to form folds; got {len(subs)}")
    ordered = _ordered(subs, f"outer/{seed_outer}")
    folds = [[] for _ in range(k)]
    for i, sk in enumerate(ordered):
        folds[i % k].append(sk)
    return [sorted(f, key=canon_subject) for f in folds]


def _hash_two_way(subjects, salt, frac):
    if not (0.0 < frac < 1.0):
        raise ValueError("frac must be in (0,1)")
    subs = _unique(subjects)
    if len(subs) < 2:
        raise ValueError("need >= 2 subjects for a two-way split")
    ordered = _ordered(subs, salt)
    n_a = int(round(frac * len(ordered)))
    n_a = max(1, min(len(ordered) - 1, n_a))            # both sides non-empty
    a = sorted(ordered[:n_a], key=canon_subject); b = sorted(ordered[n_a:], key=canon_subject)
    return a, b


def fit_cal_split(non_eval_subjects, fit_frac=None, seed_fitcal=None):
    fit_frac = HP["fit_frac"] if fit_frac is None else fit_frac
    seed_fitcal = HP["seed_fitcal"] if seed_fitcal is None else seed_fitcal
    return _hash_two_way(non_eval_subjects, f"fitcal/{seed_fitcal}", fit_frac)      # (FIT, CAL)


def train_val_split(fit_subjects, train_frac=None, seed_es=None):
    train_frac = HP["train_frac"] if train_frac is None else train_frac
    seed_es = HP["seed_es"] if seed_es is None else seed_es
    return _hash_two_way(fit_subjects, f"es/{seed_es}", train_frac)                 # (TRAIN, VAL)


def cv_assignment(subjects, k=None, seed_outer=None, fit_frac=None, seed_fitcal=None,
                  train_frac=None, seed_es=None, eligible=None):
    """The full S5 split as ONE object: per fold, (EVAL, FIT, CAL, TRAIN, VAL).

    Outer folds cover ALL `subjects` (every subject is EVAL exactly once — incl. fallback-only subjects, which stay in
    EVAL accounting). When `eligible` (a set of canon_subject strings) is given, FIT/CAL are drawn ONLY from the
    non-EVAL ELIGIBLE subjects — so fallback-only subjects never enter FIT/CAL/predictor. EVAL ⟂ (FIT∪CAL);
    FIT ⟂ CAL; TRAIN ⟂ VAL ⊆ FIT."""
    folds = outer_folds(subjects, k, seed_outer)
    all_canon = {canon_subject(sk) for f in folds for sk in f}
    out = []
    for fi, eval_subs in enumerate(folds):
        eval_canon = {canon_subject(sk) for sk in eval_subs}
        non_eval = [sk for f in folds for sk in f if canon_subject(sk) not in eval_canon]
        if eligible is not None:
            non_eval = [sk for sk in non_eval if canon_subject(sk) in eligible]
        fit, cal = fit_cal_split(non_eval, fit_frac, seed_fitcal)
        train, val = train_val_split(fit, train_frac, seed_es)
        out.append(dict(fold=fi, eval=eval_subs, fit=fit, cal=cal, train=train, val=val))
    return out, sorted(all_canon)
