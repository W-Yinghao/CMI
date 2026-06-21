"""Domain–class support graph: which conditional-invariance constraints are *identifiable*.

This is the formal core of OACI. The overlap-aware objective compares ``p(z|y,d)`` only on
cells the data can actually estimate. This module turns a table of per-``(domain,class)``
effective counts into exactly that bookkeeping — and refuses to fabricate support where
there is none:

* ``observed[d,y]``      — ``n_{d,y} >= m``. Cells below threshold are **non-identifiable**
                           and are surfaced, never smoothed.
* ``S_y``                — domains with estimable support for class ``y`` (``{d : n_{d,y}>=m}``).
* comparable classes     — ``|S_y| >= 2``: a within-``y`` cross-domain comparison exists.
* singleton classes      — ``|S_y| == 1``: present, but invariance is identifiable *nowhere*.
* the domain–domain support graph — ``d ~ d'`` iff they co-observe some class with support.
* connected components   — conditional invariance propagates **transitively** within a
                           component (a chain of shared-class constraints jointly pins the
                           representation across those domains); **across** components there
                           is no data path, so cross-component invariance is an untestable
                           extrapolation (the *Support theorem*, ``THEORY.md`` §1).

Estimate the leakage / enforce alignment ONLY on the identifiable terms returned by
``overlap_terms()``; report ``non_identifiable_pairs()`` and
``1 - identifiable_mass_fraction()`` so the gap is explicit.

numpy-only, no EEG required. Run ``python -m oaci.support_graph`` for a worked
disconnected-support example.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# --------------------------------------------------------------------------------------
# count table helpers
# --------------------------------------------------------------------------------------
def counts_from_labels(
    domain_labels,
    class_labels,
    n_domains: int | None = None,
    n_classes: int | None = None,
) -> np.ndarray:
    """Build the ``[n_domains, n_classes]`` count table from per-sample integer labels."""
    d = np.asarray(domain_labels).astype(int).ravel()
    y = np.asarray(class_labels).astype(int).ravel()
    if d.shape != y.shape:
        raise ValueError(f"domain/class label length mismatch: {d.shape} vs {y.shape}")
    n_d = (int(d.max()) + 1) if n_domains is None else int(n_domains)
    n_y = (int(y.max()) + 1) if n_classes is None else int(n_classes)
    counts = np.zeros((n_d, n_y), dtype=np.int64)
    np.add.at(counts, (d, y), 1)
    return counts


def _connected_components(n_nodes: int, edges: list[tuple[int, int]]) -> list[list[int]]:
    """Union–find connected components over ``n_nodes`` (isolated nodes are singletons)."""
    parent = list(range(n_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]  # path halving
            x = parent[x]
        return x

    for a, b in edges:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    groups: dict[int, list[int]] = {}
    for i in range(n_nodes):
        groups.setdefault(find(i), []).append(i)
    # deterministic ordering: by smallest member
    return sorted((sorted(v) for v in groups.values()), key=lambda c: c[0])


# --------------------------------------------------------------------------------------
# support graph
# --------------------------------------------------------------------------------------
@dataclass
class SupportGraph:
    """Identifiability bookkeeping for overlap-aware conditional invariance.

    All fields below are *derived* by :func:`build_support_graph`; construct via that
    function rather than by hand.
    """

    counts: np.ndarray                       # [n_domains, n_classes] effective counts
    m: int                                   # min count for a cell to be identifiable
    domain_names: list[str]
    class_names: list[str]

    observed: np.ndarray                     # bool [n_domains, n_classes]
    support_of_class: dict[int, list[int]]   # y -> S_y (sorted domain indices)
    comparable_classes: list[int]            # |S_y| >= 2
    singleton_classes: list[int]             # |S_y| == 1  (non-identifiable everywhere)
    empty_classes: list[int]                 # |S_y| == 0
    components: list[list[int]]              # connected components (domain indices)
    component_of: np.ndarray                 # [n_domains] -> component id

    # ---- shapes ----
    @property
    def n_domains(self) -> int:
        return self.counts.shape[0]

    @property
    def n_classes(self) -> int:
        return self.counts.shape[1]

    # ---- cell / class queries ----
    def is_observed(self, d: int, y: int) -> bool:
        return bool(self.observed[d, y])

    def is_comparable(self, y: int) -> bool:
        return y in self.comparable_classes

    def unobserved_cells(self) -> list[tuple[int, int]]:
        """All ``(d, y)`` below the support threshold — explicitly non-identifiable cells."""
        ds, ys = np.where(~self.observed)
        return list(zip(ds.tolist(), ys.tolist()))

    # ---- domain-pair identifiability ----
    def identifiable_pair(self, d: int, dp: int) -> bool:
        """True iff cross-domain conditional invariance between ``d`` and ``dp`` is reachable
        from data (same component → a chain of shared-class constraints links them)."""
        if d == dp:
            return True
        return bool(self.component_of[d] == self.component_of[dp])

    def identifiable_pairs(self) -> list[tuple[int, int]]:
        return [
            (d, dp)
            for d in range(self.n_domains)
            for dp in range(d + 1, self.n_domains)
            if self.identifiable_pair(d, dp)
        ]

    def non_identifiable_pairs(self) -> list[tuple[int, int]]:
        """Domain pairs in different components — cross-component invariance is untestable."""
        return [
            (d, dp)
            for d in range(self.n_domains)
            for dp in range(d + 1, self.n_domains)
            if not self.identifiable_pair(d, dp)
        ]

    # ---- the overlap-aware objective's identifiable terms ----
    def overlap_terms(self) -> list[dict]:
        """The identifiable terms of ``I_ov(Z;D|Y)`` — one per comparable class.

        Each term is ``{y, support: S_y, n_obs, weight}`` where ``weight`` is the class
        prior ``p(y)`` renormalised over the comparable classes (the mass the objective
        actually constrains). Unsupported / singleton classes contribute **no** term.
        """
        n_obs = {y: int(self.counts[self.support_of_class[y], y].sum()) for y in self.comparable_classes}
        total = sum(n_obs.values())
        return [
            {
                "y": y,
                "class_name": self.class_names[y],
                "support": list(self.support_of_class[y]),
                "n_obs": n_obs[y],
                "weight": (n_obs[y] / total) if total > 0 else 0.0,
            }
            for y in self.comparable_classes
        ]

    def identifiable_mass_fraction(self) -> float:
        """Fraction of all samples that live in a constrained ``(d,y)`` cell (observed AND
        in a comparable class). ``1 - this`` is the honest non-identifiable remainder."""
        total = float(self.counts.sum())
        if total <= 0:
            return 0.0
        constrained = 0
        for y in self.comparable_classes:
            constrained += int(self.counts[self.support_of_class[y], y].sum())
        return constrained / total

    # ---- reporting ----
    def summary(self) -> dict:
        return {
            "n_domains": self.n_domains,
            "n_classes": self.n_classes,
            "m": self.m,
            "n_observed_cells": int(self.observed.sum()),
            "n_cells": self.n_domains * self.n_classes,
            "comparable_classes": list(self.comparable_classes),
            "singleton_classes": list(self.singleton_classes),
            "empty_classes": list(self.empty_classes),
            "n_components": len(self.components),
            "components": [list(c) for c in self.components],
            "identifiable_mass_fraction": self.identifiable_mass_fraction(),
            "n_non_identifiable_domain_pairs": len(self.non_identifiable_pairs()),
        }

    def report(self) -> str:
        s = self.summary()
        lines = [
            f"Support graph (m={self.m}): {s['n_domains']} domains x {s['n_classes']} classes",
            f"  observed cells: {s['n_observed_cells']}/{s['n_cells']}"
            f"   identifiable mass: {s['identifiable_mass_fraction']:.3f} of N={int(self.counts.sum())}",
            f"  comparable classes (|S_y|>=2): {s['comparable_classes']}",
            f"  singleton-support classes (|S_y|==1, NON-IDENTIFIABLE): {s['singleton_classes']}",
            f"  empty classes (|S_y|==0): {s['empty_classes']}",
            f"  components: {s['n_components']} -> {s['components']}",
            f"  cross-component (NON-IDENTIFIABLE) domain pairs: {self.non_identifiable_pairs()}",
        ]
        return "\n".join(lines)


def build_support_graph(
    counts,
    m: int,
    domain_names: list[str] | None = None,
    class_names: list[str] | None = None,
) -> SupportGraph:
    """Build the support graph from a ``[n_domains, n_classes]`` effective-count table.

    ``m`` is the minimum effective sample count for a ``(d,y)`` cell to be treated as
    estimable. Pass *effective* sample sizes if counts are correlated within a domain
    (e.g. trials from one recording); ``m`` then guards estimator variance, not raw n.
    """
    counts = np.asarray(counts)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D [n_domains, n_classes], got shape {counts.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1 (a cell needs at least one sample to be observed), got {m}")
    n_d, n_y = counts.shape

    domain_names = list(domain_names) if domain_names is not None else [f"d{d}" for d in range(n_d)]
    class_names = list(class_names) if class_names is not None else [f"y{y}" for y in range(n_y)]
    if len(domain_names) != n_d or len(class_names) != n_y:
        raise ValueError("domain_names / class_names length must match counts shape")

    observed = counts >= m
    support_of_class = {y: sorted(np.where(observed[:, y])[0].tolist()) for y in range(n_y)}
    comparable = [y for y in range(n_y) if len(support_of_class[y]) >= 2]
    singleton = [y for y in range(n_y) if len(support_of_class[y]) == 1]
    empty = [y for y in range(n_y) if len(support_of_class[y]) == 0]

    # domain–domain edges: a star within each comparable class's support is enough for
    # union–find connectivity (cheaper than all pairs, same components).
    edges: list[tuple[int, int]] = []
    for y in comparable:
        s = support_of_class[y]
        for k in range(1, len(s)):
            edges.append((s[0], s[k]))
    components = _connected_components(n_d, edges)

    component_of = np.full(n_d, -1, dtype=np.int64)
    for cid, comp in enumerate(components):
        for d in comp:
            component_of[d] = cid

    return SupportGraph(
        counts=counts,
        m=int(m),
        domain_names=domain_names,
        class_names=class_names,
        observed=observed,
        support_of_class=support_of_class,
        comparable_classes=comparable,
        singleton_classes=singleton,
        empty_classes=empty,
        components=components,
        component_of=component_of,
    )


# --------------------------------------------------------------------------------------
# self-demo: a deliberately disconnected support pattern
# --------------------------------------------------------------------------------------
def _demo() -> None:
    # 4 domains (sites), 3 classes. Sites 0,1 co-observe class 0; sites 2,3 co-observe
    # class 1; class 2 lives only in site 0 (singleton); site 3 is short on class 1.
    counts = np.array(
        [
            # y0   y1   y2
            [120, 0, 40],   # site 0
            [110, 0, 0],    # site 1
            [0, 90, 0],     # site 2
            [0, 5, 0],      # site 3  (class-1 count below threshold)
        ]
    )
    sites = ["BNCI-2a-s1", "BNCI-2a-s2", "PD-cohort", "SCZ-cohort"]
    classes = ["rest", "task", "rare-event"]
    sg = build_support_graph(counts, m=20, domain_names=sites, class_names=classes)
    print(sg.report())
    print("\nidentifiable overlap terms (only these enter I_ov):")
    for t in sg.overlap_terms():
        names = [sites[d] for d in t["support"]]
        print(f"  y={t['class_name']:<10} S_y={names}  n_obs={t['n_obs']:<5} weight={t['weight']:.3f}")
    print(
        "\nidentifiable_pair(site0,site1) =", sg.identifiable_pair(0, 1),
        "| identifiable_pair(site0,site2) =", sg.identifiable_pair(0, 2),
        "(cross-component → non-identifiable)",
    )


if __name__ == "__main__":
    _demo()
