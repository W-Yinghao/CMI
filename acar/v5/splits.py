"""ACAR V5 deterministic subject-disjoint split (SPLITS §5): outer K=5 (each subject EVAL once); non-EVAL → FIT 70% / CAL 30%;
FIT → TRAIN 80% / VAL 20%. Deterministic by canonical-SubjectKey hash (NOT RNG permutation), salt ACAR_V5_SPLIT_V1. Pure/stdlib.

The guard `test_subject_disjoint` asserts FIT/CAL/EVAL are pairwise disjoint, every subject is EVAL in exactly one outer fold,
and the assignment is permutation-independent (hash-based).
"""
from __future__ import annotations
import hashlib
from acar.v5 import protocol as P


def _u01(key):
    """Deterministic uniform-ish [0,1) from a stable salted hash of a subject key (no RNG)."""
    h = hashlib.sha256(f"{P.SPLIT_SALT}|{key}".encode()).hexdigest()
    return int(h[:16], 16) / float(1 << 64)


def _fold_of(key, k=P.OUTER_K):
    h = hashlib.sha256(f"{P.SPLIT_SALT}|fold|{key}".encode()).hexdigest()
    return int(h[:8], 16) % k


def assign_outer_folds(subjects, k=P.OUTER_K):
    """Return {subject: fold} with each subject in exactly one outer fold (that fold's EVAL). Order-independent."""
    subs = sorted(set(subjects))
    if len(subs) != len(list(subjects)):
        raise ValueError("duplicate subjects passed to assign_outer_folds")
    return {s: _fold_of(s, k) for s in subs}


def _threshold_split(subjects, frac, tag):
    """Split subjects into (A, B) with |A| ~ frac by a salted per-subject uniform; deterministic, permutation-independent."""
    a, b = [], []
    for s in sorted(set(subjects)):
        (a if _u01(f"{tag}|{s}") < frac else b).append(s)
    return a, b


def make_fold(subjects, fold):
    """Full fold-contained split for one outer fold (SPLITS §5). Returns dict with subject-DISJOINT eval/cal/train/val lists.
    All are deterministic; no label is read (this operates on subject identifiers only)."""
    folds = assign_outer_folds(subjects)
    eval_s = sorted(s for s, f in folds.items() if f == fold)
    non_eval = sorted(s for s, f in folds.items() if f != fold)
    fit, cal = _threshold_split(non_eval, P.FIT_FRAC, "fitcal")
    train, val = _threshold_split(fit, P.TRAIN_FRAC, "trainval")
    out = {"fold": fold, "eval": eval_s, "cal": cal, "fit": sorted(fit), "train": sorted(train), "val": sorted(val)}
    _assert_disjoint(out)
    return out


def _assert_disjoint(split):
    e, c, f = set(split["eval"]), set(split["cal"]), set(split["fit"])
    if e & c or e & f or c & f:
        raise AssertionError("FIT/CAL/EVAL not subject-disjoint")
    if set(split["train"]) & set(split["val"]):
        raise AssertionError("TRAIN/VAL not subject-disjoint")
    if set(split["train"]) | set(split["val"]) != f:
        raise AssertionError("TRAIN ∪ VAL must equal FIT")
    return True
