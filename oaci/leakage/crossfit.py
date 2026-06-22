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
    """A frozen representation + the labels the probe conditions on, plus per-row base mass
    ``b_i`` (default = ones for synthetic back-compat). All leakage estimands are MASS-weighted by
    ``sample_mass``; never re-infer mass from row counts inside the estimator."""
    Z: np.ndarray              # [N, dim]
    y: np.ndarray              # [N] class label (int)
    d: np.ndarray              # [N] domain label (int)
    group: np.ndarray          # [N] recording-group id (the dependence unit)
    sample_mass: np.ndarray = None   # [N] base mass b_i > 0

    def __post_init__(self):
        self.Z = np.asarray(self.Z, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=int).ravel()
        self.d = np.asarray(self.d, dtype=int).ravel()
        self.group = np.asarray(self.group, dtype=int).ravel()
        n = self.Z.shape[0]
        if not (self.y.shape[0] == self.d.shape[0] == self.group.shape[0] == n):
            raise ValueError("Z/y/d/group length mismatch")
        if self.sample_mass is None:
            self.sample_mass = np.ones(n, dtype=np.float64)
        else:
            self.sample_mass = np.asarray(self.sample_mass, dtype=np.float64).ravel()
            if self.sample_mass.shape[0] != n:
                raise ValueError("sample_mass length mismatch")
            if not np.all(np.isfinite(self.sample_mass)) or np.any(self.sample_mass <= 0):
                raise ValueError("sample_mass must be finite and strictly positive")

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


def _cell_aware_fold_assignment(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    dom_of_group: dict[int, int],
    n_folds: int,
    seed: int,
) -> dict[int, int]:
    """Greedily assign each recording group to a fold, balancing every eligible ``(d,y)`` cell
    across folds (stratified-group-K-fold style). A group is atomic (one fold). With the
    feasibility guarantee (each used cell has >= n_folds groups), a single-cell (clinical)
    group's cells are spread to distinct least-loaded folds, so every cell appears in EVERY
    fold's train and test; multi-class groups are balanced jointly.
    """
    # per-group: the eligible comparable cells it touches, balanced by SAMPLE-MASS sums (not row
    # counts) so duplicating windows with split mass leaves the fold plan unchanged.
    b = feat.sample_mass
    group_cells: dict[int, dict[tuple, float]] = {}
    group_total: dict[int, float] = {}
    for g in np.unique(feat.group):
        g = int(g)
        gm = feat.group == g
        dom = dom_of_group[g]
        cells: dict[tuple, float] = {}
        for y in support_graph.comparable_classes:
            if dom in support_graph.support_of_class[y]:
                mss = float(b[gm & (feat.y == y)].sum())
                if mss > 0:
                    cells[(dom, y)] = mss
        group_cells[g] = cells
        group_total[g] = float(b[gm].sum())

    rng = np.random.default_rng(seed)
    order = list(group_cells)
    rng.shuffle(order)                                   # seed breaks ties
    order.sort(key=lambda g: -sum(group_cells[g].values()))  # stable: heaviest cells first

    fold_cell = [defaultdict(int) for _ in range(n_folds)]
    fold_size = [0] * n_folds
    fold_of_group: dict[int, int] = {}
    for g in order:
        cells = group_cells[g]
        # pick the fold that is currently least loaded for THIS group's cells (ties -> smaller
        # fold, then lower index) so each cell's groups fan out across folds.
        best_key, best_k = None, 0
        for k in range(n_folds):
            cost = sum(fold_cell[k][c] for c in cells)
            key = (cost, fold_size[k], k)
            if best_key is None or key < best_key:
                best_key, best_k = key, k
        fold_of_group[g] = best_k
        for c, cnt in cells.items():
            fold_cell[best_k][c] += cnt
        fold_size[best_k] += group_total[g]
    return fold_of_group


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

    # cell-aware grouped assignment (regression fix): feasibility is measured per (d,y) cell,
    # so the ASSIGNMENT must stratify per cell too — a recording carrying a single clinical
    # label would otherwise let a whole cell land in one fold under domain-only round-robin.
    fold_of_group = _cell_aware_fold_assignment(feat, support_graph, dom_of_group, n_eff, seed)

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
    ``S_y`` index space, train per fold (weighted by ``sample_mass``) on the other folds, score the
    held-out fold. Returns ``{y: {nll, n_rows, mass}}`` where ``nll = Σ b_i·[-log q] / M_y^ov``
    over OOF rows — the weighted numerator over the FIXED mass denominator (NOT a self-normalised
    per-fold mean), so window duplication with split mass is a no-op.
    """
    fold = np.array([fold_plan.fold_of_group[int(g)] for g in feat.group])
    b_all = feat.sample_mass
    out: dict[int, dict] = {}
    for y in support_graph.comparable_classes:
        S = support_graph.support_of_class[y]
        dmap = {int(d): i for i, d in enumerate(S)}
        sel = (feat.y == y) & np.isin(feat.d, S)
        idx = np.where(sel)[0]
        M_y_ov = float(support_graph.cell_mass[S, y].sum())     # FIXED denominator
        if idx.size == 0:
            out[y] = {"nll": np.nan, "n_rows": 0, "mass": 0.0}
            continue
        Z = feat.Z[idx]
        labels = np.array([dmap[int(d)] for d in feat.d[idx]])
        b = b_all[idx]
        f = fold[idx]
        num, mass, n_rows = 0.0, 0.0, 0
        for k in range(fold_plan.n_folds):
            te = f == k
            tr = ~te
            if te.sum() == 0 or tr.sum() == 0:
                continue
            probe = DomainProbe(capacity, len(S), cfg).fit(Z[tr], labels[tr], sample_weight=b[tr])
            nll = probe.nll(Z[te], labels[te])
            num += float((b[te] * nll).sum())
            mass += float(b[te].sum())
            n_rows += int(te.sum())
        out[y] = {"nll": (num / M_y_ov if M_y_ov > 0 else np.nan), "n_rows": n_rows, "mass": mass}
    return out
