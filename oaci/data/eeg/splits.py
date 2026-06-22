"""Strict three-role split contract per outer fold:

* ``source_train``  — Stage-1/2, the empirical-risk constraint, and source-only checkpoint
  selection (the ONLY rows any fit statistic may touch);
* ``source_audit``  — never optimized/selected on; source-risk NI + final leakage audit;
* ``target_audit``  — sealed throughout.

The split depends on ``split_seed`` ONLY (never the model seed); the controlled missing-cell
mask uses a separate ``deletion_seed`` and removes rows from ``source_train`` ONLY.

Two explicit modes (the previous single rule degenerated to an empty source_audit for MI/SEED,
where ``domain == subject`` so ``round(0.2*1)=0`` per domain):

* ``across_source_domains``     — MI/SEED (``domain==subject``): hold out WHOLE source subjects as
  audit; keep ≥2 TRAIN domains; if feasible keep ≥2 AUDIT domains else source-audit leakage/NI is
  flagged non-estimable.
* ``within_each_source_domain`` — clinical (``domain==site``): hold out audit SUBJECTS *within*
  each source site via the composite ``(site_id, subject_id)`` key (a ``sub-01`` in two sites
  never collides), keeping ≥1 train subject per site.
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
    mode: str
    n_active_source_domains: int
    method_inactive: bool
    n_audit_domains: int = 0
    source_audit_estimable: bool = False    # >=2 audit domains -> source-audit leakage/NI estimable
    audit_units: list = field(default_factory=list)
    reason: str = ""

    def roles_disjoint(self) -> bool:
        a, b, c = (set(self.source_train.tolist()), set(self.source_audit.tolist()),
                   set(self.target_audit.tolist()))
        return a.isdisjoint(b) and a.isdisjoint(c) and b.isdisjoint(c)


def make_loso_split(domain, subject, target_domain, split_seed, mode, audit_frac=0.2) -> SplitPlan:
    domain = np.asarray(domain, int)
    subject = np.asarray(subject, dtype=object)
    rng = np.random.default_rng(split_seed)
    target_audit = np.where(domain == target_domain)[0]
    source_mask = domain != target_domain
    src_domains = sorted(set(domain[source_mask].tolist()))

    if mode == "across_source_domains":
        # audit whole source subjects (= whole domains); keep >=2 TRAIN domains
        doms = list(src_domains)
        rng.shuffle(doms)
        k = int(round(audit_frac * len(doms)))
        k = max(0, min(k, len(doms) - 2)) if len(doms) >= 2 else 0
        audit_doms = set(doms[:k])
        sa_mask = source_mask & np.isin(domain, list(audit_doms))
        audit_units = [int(d) for d in sorted(audit_doms)]
        n_audit_domains = len(audit_doms)
    elif mode == "within_each_source_domain":
        # composite (site, subject) so sub-01 in two sites never collide
        audit_keys: set = set()
        audit_doms: set = set()
        for dd in src_domains:
            subs = sorted(set(subject[source_mask & (domain == dd)].tolist()), key=str)
            arr = np.array(subs, dtype=object)
            rng.shuffle(arr)
            kk = min(int(round(audit_frac * len(arr))), len(arr) - 1)   # keep >=1 TRAIN subject/site
            for s in arr[:kk]:
                audit_keys.add((int(dd), s)); audit_doms.add(int(dd))
        sa_mask = np.array([(int(domain[i]), subject[i]) in audit_keys for i in range(len(domain))], dtype=bool)
        audit_units = sorted(f"{d}|{s}" for d, s in audit_keys)
        n_audit_domains = len(audit_doms)
    else:
        raise ValueError(f"mode must be 'across_source_domains' or 'within_each_source_domain', got {mode!r}")

    st_mask = source_mask & ~sa_mask
    active = sorted(set(domain[st_mask].tolist()))
    return SplitPlan(
        source_train=np.where(st_mask)[0], source_audit=np.where(sa_mask)[0], target_audit=target_audit,
        target_domain=int(target_domain), split_seed=int(split_seed), mode=mode,
        n_active_source_domains=len(active), method_inactive=len(active) < 2,
        n_audit_domains=n_audit_domains, source_audit_estimable=n_audit_domains >= 2,
        audit_units=audit_units,
        reason="" if len(active) >= 2 else f"only {len(active)} active source domain(s) -> method-inactive",
    )


def apply_missing_cell_mask(split: SplitPlan, domain, y, deleted_cells) -> SplitPlan:
    """Remove ``deleted_cells`` (a set of ``(d,y)``) from ``source_train`` ONLY; ``source_audit``
    and ``target_audit`` are byte-identical."""
    domain, y = np.asarray(domain, int), np.asarray(y, int)
    deleted = set((int(a), int(b)) for a, b in deleted_cells)
    keep = np.array([(int(domain[i]), int(y[i])) not in deleted for i in split.source_train], dtype=bool)
    new_st = split.source_train[keep]
    active = sorted(set(domain[new_st].tolist()))
    return SplitPlan(
        source_train=new_st, source_audit=split.source_audit, target_audit=split.target_audit,
        target_domain=split.target_domain, split_seed=split.split_seed, mode=split.mode,
        n_active_source_domains=len(active), method_inactive=len(active) < 2,
        n_audit_domains=split.n_audit_domains, source_audit_estimable=split.source_audit_estimable,
        audit_units=list(split.audit_units), reason=split.reason,
    )
