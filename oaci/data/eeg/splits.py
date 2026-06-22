"""Strict three-role split contract per outer fold:

* ``source_train``  — Stage-1/2, the empirical-risk constraint, and source-only checkpoint
  selection (the ONLY rows any fit statistic may touch);
* ``source_audit``  — never optimized/selected on; source-risk NI + final leakage audit;
* ``target_audit``  — sealed throughout.

The split depends on ``split_seed`` ONLY (never the model seed); the controlled missing-cell
mask uses a separate ``deletion_seed`` and removes rows from ``source_train`` ONLY. A fold with
< 2 active source domains is flagged ``method_inactive`` (OACI has no cross-domain comparison —
report as no-op, not efficacy).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SplitPlan:
    source_train: np.ndarray
    source_audit: np.ndarray
    target_audit: np.ndarray
    target_domain: int
    split_seed: int
    n_active_source_domains: int
    method_inactive: bool
    audit_subjects: list = field(default_factory=list)
    reason: str = ""

    def roles_disjoint(self) -> bool:
        s = [set(self.source_train.tolist()), set(self.source_audit.tolist()), set(self.target_audit.tolist())]
        return s[0].isdisjoint(s[1]) and s[0].isdisjoint(s[2]) and s[1].isdisjoint(s[2])


def make_loso_split(domain, subject, target_domain, split_seed, audit_frac=0.2,
                    ensure_train_per_domain=True) -> SplitPlan:
    """Leave-one-domain-out. ``ensure_train_per_domain=True`` (site-domain clinical): hold out
    audit SUBJECTS within each source site (keep ≥1 train subject per site). ``False`` (MI/SEED,
    domain=subject): audit whole source subjects."""
    domain = np.asarray(domain, int)
    subject = np.asarray(subject, dtype=object)
    target_audit = np.where(domain == target_domain)[0]
    source_mask = domain != target_domain
    rng = np.random.default_rng(split_seed)

    audit_subjects: set = set()
    for dd in sorted(set(domain[source_mask].tolist())):
        subs = np.array(sorted(set(subject[source_mask & (domain == dd)].tolist()), key=str), dtype=object)
        rng.shuffle(subs)
        k = int(round(audit_frac * len(subs)))
        if ensure_train_per_domain:
            k = min(k, len(subs) - 1)            # keep at least one TRAIN subject in this source site
        audit_subjects.update(subs[:k].tolist())

    sa_mask = source_mask & np.isin(subject, list(audit_subjects))
    st_mask = source_mask & ~sa_mask
    active = sorted(set(domain[st_mask].tolist()))
    return SplitPlan(
        source_train=np.where(st_mask)[0], source_audit=np.where(sa_mask)[0],
        target_audit=target_audit, target_domain=int(target_domain), split_seed=int(split_seed),
        n_active_source_domains=len(active), method_inactive=len(active) < 2,
        audit_subjects=sorted(audit_subjects, key=str),
        reason="" if len(active) >= 2 else f"only {len(active)} active source domain(s) -> method-inactive",
    )


def apply_missing_cell_mask(split: SplitPlan, domain, y, deleted_cells) -> SplitPlan:
    """Return a SplitPlan with ``deleted_cells`` (a set of ``(d,y)``) removed from
    ``source_train`` ONLY; ``source_audit`` and ``target_audit`` are byte-identical."""
    domain, y = np.asarray(domain, int), np.asarray(y, int)
    deleted = set((int(a), int(b)) for a, b in deleted_cells)
    keep = np.array([(int(domain[i]), int(y[i])) not in deleted for i in split.source_train], dtype=bool)
    new_st = split.source_train[keep]
    active = sorted(set(domain[new_st].tolist()))
    return SplitPlan(
        source_train=new_st, source_audit=split.source_audit, target_audit=split.target_audit,
        target_domain=split.target_domain, split_seed=split.split_seed,
        n_active_source_domains=len(active), method_inactive=len(active) < 2,
        audit_subjects=list(split.audit_subjects), reason=split.reason,
    )
