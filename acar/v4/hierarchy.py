"""ACAR v4 (CURB) — Direction B: hierarchical / policy-level deployed-risk calibration objects.

NON-BINDING / POST-V3 / synthetic-capable: pure numpy; reads no real cohort, calls no v3 loader, generates no V4 DEV
result, asserts no coverage theorem. These are CALIBRATION OBJECTS — the per-subject quantity a calibrator controls —
not calibrators themselves.

The scientific point (notes/ACAR_V4_DESIGN_DRAFT.md §5): v3's calibration object is an ALL-ACTION joint-max,
≈ max_{B∈B(s)} max_a score_{B,a}, which is dominated by the risk of actions the policy never executes. Direction B
calibrates the risk of the EXECUTED policy only:
    Z_s = (1/|B(s)|) Σ_{B∈B(s)} ℓ(ΔR_{π(B)}, π(B)),
so an unexecuted action's risk must NOT move the object. The decisive guards (tests/test_hierarchy.py) show B0 RESPONDS
to an unexecuted-action risk change while B1/B2 IGNORE it — that is the V4-B vs v3 distinction.

Sign convention (frozen): ΔR_a(B) < 0 = reduced risk (good); identity/fallback (choice = -1) realize ΔR 0 and stay in
the subject denominator. Subjects are the exchangeable unit; aggregation is subject-equal. Fail-closed throughout.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Literal, Optional, Sequence, Tuple

import numpy as np

from acar.v4 import policies as P


@dataclass(frozen=True)
class SubjectRisk:
    subject_ids: Tuple[str, ...]          # canonical (sorted-unique) subject order
    values: np.ndarray                    # [n_subjects], immutable (read-only copy)
    object_name: str                      # "B0_ALL_ACTION_JOINT" | "B1_POLICY_AGG" | "B2_HIER_POLICY"
    loss: Optional[str]                   # None for B0; the loss / "loss|summary" tag otherwise
    n_batches_by_subject: Tuple[int, ...]

    def __post_init__(self):
        v = np.array(self.values, dtype=float)
        v.flags.writeable = False
        object.__setattr__(self, "values", v)
        if v.ndim != 1 or v.shape[0] != len(self.subject_ids):
            raise ValueError("values must be 1-D aligned with subject_ids")
        if len(self.n_batches_by_subject) != len(self.subject_ids):
            raise ValueError("n_batches_by_subject must align with subject_ids")


# ----------------------------------------------------------------------------- validators / helpers

def _check_subject_ids(subject_ids, n):
    ids = np.asarray(subject_ids)
    if ids.ndim != 1 or ids.shape[0] != n:
        raise ValueError(f"subject_ids must be 1-D with length {n}")
    lst = ids.tolist()
    for s in lst:
        if not isinstance(s, str) or s == "":
            raise ValueError("subject_ids must be non-empty strings")
    return lst


def _canonical(ids_list):
    """Canonical sorted-unique subject order + per-batch row→subject index (permutation-independent)."""
    uniq = sorted(set(ids_list))
    index_of = {u: i for i, u in enumerate(uniq)}
    rows = np.array([index_of[i] for i in ids_list], dtype=int)
    return uniq, rows


def _check_loss(loss):
    if loss not in ("mean", "positive", "harm_indicator"):
        raise ValueError("loss must be 'mean', 'positive', or 'harm_indicator'")


def _check_summary(batch_summary):
    if batch_summary not in ("mean", "positive_mean", "harm_rate"):
        raise ValueError("batch_summary must be 'mean', 'positive_mean', or 'harm_rate'")


def _per_batch_loss(realized, choice, loss):
    if loss == "mean":
        return realized
    if loss == "positive":
        return np.maximum(realized, 0.0)
    return ((choice != P.IDENTITY) & (realized > 0.0)).astype(float)     # harm_indicator


def _subject_mean(values, rows, n_subj):
    counts = np.bincount(rows, minlength=n_subj).astype(float)           # batches per subject (incl. fallback)
    sums = np.bincount(rows, weights=values, minlength=n_subj)
    return sums / counts, counts


# ----------------------------------------------------------------------------- B0: v3-style all-action joint max

def all_action_joint_max(scores, subject_ids):
    """B0 comparator (v3-style): Z_s = max_{B∈B(s)} max_a score_{B,a}. Uses EVERY action's score, so it RESPONDS to
    unexecuted-action risk — included only to demonstrate the contrast with B1/B2."""
    s = P._as2d(scores, "scores")                                       # finite, ≥1 batch, ≥1 action
    n = s.shape[0]
    ids = _check_subject_ids(subject_ids, n)
    uniq, rows = _canonical(ids)
    per_batch_max = s.max(axis=1)                                       # max over actions per batch
    values = np.array([per_batch_max[rows == i].max() for i in range(len(uniq))], dtype=float)
    counts = np.bincount(rows, minlength=len(uniq)).astype(int)
    return SubjectRisk(tuple(uniq), values, "B0_ALL_ACTION_JOINT", None, tuple(counts.tolist()))


# ----------------------------------------------------------------------------- B1: policy-only subject aggregate

def policy_subject_risk(choices, dr, subject_ids, *, loss):
    """B1: Z_s = (1/|B(s)|) Σ_B ℓ(ΔR_{π(B)}, π(B)) over the EXECUTED action only (identity/fallback realize 0, kept in
    the denominator). Independent of unexecuted-action risk — the core V4-B object."""
    dr2 = P._as2d(dr, "dr")
    c = P._as_choice(choices, n_actions=dr2.shape[1])
    if c.shape[0] != dr2.shape[0]:
        raise ValueError(f"choices length {c.shape[0]} != n_batches {dr2.shape[0]}")
    ids = _check_subject_ids(subject_ids, dr2.shape[0])
    _check_loss(loss)
    realized = P.realized_dr(c, dr2)
    per_batch = _per_batch_loss(realized, c, loss)
    uniq, rows = _canonical(ids)
    values, counts = _subject_mean(per_batch, rows, len(uniq))
    return SubjectRisk(tuple(uniq), values, "B1_POLICY_AGG", loss, tuple(counts.astype(int).tolist()))


# ----------------------------------------------------------------------------- B2: hierarchical batch→subject

def hierarchical_policy_risk(choices, dr, subject_ids, *, loss, batch_summary):
    """B2: form the batch-level realized policy loss ℓ(ΔR_{π(B)}, π(B)) FIRST, then aggregate to the subject by
    `batch_summary` (mean / positive_mean / harm_rate). Like B1 it depends only on the EXECUTED action, so it ignores
    unexecuted-action risk; the explicit two-stage structure is the extension point for future site/online hierarchical
    calibration. With loss='mean', batch_summary='mean' it equals B1 (mean) numerically."""
    dr2 = P._as2d(dr, "dr")
    c = P._as_choice(choices, n_actions=dr2.shape[1])
    if c.shape[0] != dr2.shape[0]:
        raise ValueError(f"choices length {c.shape[0]} != n_batches {dr2.shape[0]}")
    ids = _check_subject_ids(subject_ids, dr2.shape[0])
    _check_loss(loss)
    _check_summary(batch_summary)
    realized = P.realized_dr(c, dr2)
    per_batch = _per_batch_loss(realized, c, loss)                      # batch-level realized policy loss
    if batch_summary == "mean":
        summarized = per_batch
    elif batch_summary == "positive_mean":
        summarized = np.maximum(per_batch, 0.0)
    else:                                                              # harm_rate
        summarized = (per_batch > 0.0).astype(float)
    uniq, rows = _canonical(ids)
    values, counts = _subject_mean(summarized, rows, len(uniq))
    return SubjectRisk(tuple(uniq), values, "B2_HIER_POLICY", f"{loss}|{batch_summary}",
                       tuple(counts.astype(int).tolist()))
