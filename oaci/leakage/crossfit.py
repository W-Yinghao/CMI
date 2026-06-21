"""Strict recording/subject-grouped cross-fit for the conditional-domain probe.

Guarantees (the no-leakage backbone, with ``critic.py``):

* the dependence unit is the **recording group** (a whole recording lands in one fold), so a
  recording never appears in both a probe's train and test — grouped, not sample-level;
* fold assignment is **stratified within domain** (round-robin over a domain's groups) so
  every eligible domain appears across folds, and is computed **once on the original groups**
  (``fold_of_group``). Bootstrap replicates reuse this map, so all duplicate copies of a
  resampled group inherit the SAME fold;
* preprocessing (standardisation; the split itself) happens strictly inside the train fold;
* ``(d,y)`` is the support gate and the stratification key — it is NOT collapsed into one giant
  dependence cluster;
* **feasibility**: if an eligible cell cannot be split into ≥2 grouped folds, the fold count
  is reduced to what is feasible and recorded; if even 2 are impossible, it fails explicitly
  (never silently degrades to a sample-level split).

``FrozenFeatures`` carries the frozen ``Z`` plus the class/domain/group labels.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from ..support_graph import SupportGraph
from .critic import CriticConfig, DomainProbe


@dataclass
class FrozenFeatures:
    """A frozen representation + the labels the probe conditions on."""
    Z: np.ndarray         # [N, dim]
    y: np.ndarray         # [N] class label (int)
    d: np.ndarray         # [N] domain label (int)
    group: np.ndarray     # [N] recording-group id (the dependence unit)

    def __post_init__(self):
        self.Z = np.asarray(self.Z, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int).ravel()
        self.d = np.asarray(self.d, dtype=int).ravel()
        self.group = np.asarray(self.group, dtype=int).ravel()
        n = self.Z.shape[0]
        if not (self.y.shape[0] == self.d.shape[0] == self.group.shape[0] == n):
            raise ValueError("Z/y/d/group length mismatch")

    @property
    def n(self) -> int:
        return self.Z.shape[0]


@dataclass
class FoldPlan:
    """Original-group -> fold map (fixed across bootstrap), plus the feasibility record."""
    fold_of_group: dict[int, int]
    n_folds: int                  # effective fold count actually used
    n_folds_requested: int
    domain_of_group: dict[int, int]
    notes: list[str] = field(default_factory=list)

    @property
    def reduced(self) -> bool:
        return self.n_folds < self.n_folds_requested


def _domain_of_each_group(feat: FrozenFeatures) -> dict[int, int]:
    dom: dict[int, int] = {}
    for g in np.unique(feat.group):
        ds = np.unique(feat.d[feat.group == g])
        if ds.size != 1:
            raise ValueError(f"group {g} spans multiple domains {ds.tolist()} (a recording must be one domain)")
        dom[int(g)] = int(ds[0])
    return dom


def make_fold_plan(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    n_folds: int,
    seed: int = 0,
) -> FoldPlan:
    """Grouped, domain-stratified K-fold plan over the ORIGINAL recording groups."""
    if n_folds < 2:
        raise ValueError(f"n_folds must be >= 2, got {n_folds}")
    dom_of_group = _domain_of_each_group(feat)

    # cells actually used by the estimator: eligible cells of comparable classes.
    used_cells = [(d, y) for y in support_graph.comparable_classes for d in support_graph.support_of_class[y]]
    if not used_cells:
        raise ValueError("no comparable classes with eligible support -> nothing to cross-fit")

    # groups-per-cell determines how finely each cell can be grouped-split.
    groups_per_cell = {
        (d, y): int(np.unique(feat.group[(feat.d == d) & (feat.y == y)]).size) for (d, y) in used_cells
    }
    min_groups = min(groups_per_cell.values())
    n_eff = min(n_folds, min_groups)
    notes: list[str] = []
    if n_eff < 2:
        scarce = min(groups_per_cell, key=groups_per_cell.get)
        raise ValueError(
            f"eligible cell {scarce} has only {groups_per_cell[scarce]} recording group(s); "
            f"cannot form >=2 grouped folds. Refusing to fall back to a sample-level split."
        )
    if n_eff < n_folds:
        scarce = min(groups_per_cell, key=groups_per_cell.get)
        notes.append(f"reduced n_folds {n_folds}->{n_eff}: cell {scarce} has only {groups_per_cell[scarce]} groups")

    # assign: within each domain, seeded-shuffle its groups then round-robin across folds.
    rng = np.random.default_rng(seed)
    by_dom: dict[int, list[int]] = defaultdict(list)
    for g, dom in dom_of_group.items():
        by_dom[dom].append(g)
    fold_of_group: dict[int, int] = {}
    for dom in sorted(by_dom):
        gs = sorted(by_dom[dom])
        rng.shuffle(gs)
        for i, g in enumerate(gs):
            fold_of_group[g] = i % n_eff

    return FoldPlan(
        fold_of_group=fold_of_group,
        n_folds=n_eff,
        n_folds_requested=n_folds,
        domain_of_group=dom_of_group,
        notes=notes,
    )


def oof_nll_by_class(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    fold_plan: FoldPlan,
    capacity: int,
    cfg: CriticConfig,
) -> dict[int, dict]:
    """Out-of-fold mean NLL (nats) of ``q(D|Z,Y=y)`` for each comparable class ``y``.

    For class ``y``: keep only eligible class-``y`` rows (``d ∈ S_y``), relabel domains to the
    ``S_y`` index space, train per fold on the other folds, score the held-out fold. Returns
    ``{y: {nll, n}}`` with ``n`` the number of OOF-scored rows.
    """
    fold = np.array([fold_plan.fold_of_group[int(g)] for g in feat.group])
    out: dict[int, dict] = {}
    for y in support_graph.comparable_classes:
        S = support_graph.support_of_class[y]
        dmap = {int(d): i for i, d in enumerate(S)}
        sel = (feat.y == y) & np.isin(feat.d, S)
        idx = np.where(sel)[0]
        if idx.size == 0:
            out[y] = {"nll": np.nan, "n": 0}
            continue
        Z = feat.Z[idx]
        labels = np.array([dmap[int(d)] for d in feat.d[idx]])
        f = fold[idx]
        nll_sum, n = 0.0, 0
        for k in range(fold_plan.n_folds):
            te = f == k
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            probe = DomainProbe(capacity, len(S), cfg).fit(Z[tr], labels[tr])
            nll = probe.nll(Z[te], labels[te])
            nll_sum += float(nll.sum())
            n += int(te.sum())
        out[y] = {"nll": (nll_sum / n if n > 0 else np.nan), "n": n}
    return out
