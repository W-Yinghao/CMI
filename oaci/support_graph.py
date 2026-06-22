"""Domain–class support graph: which conditional-invariance constraints are *identifiable*,
and how the constraint system *decomposes*.

This is the formal core of OACI. Two distinct notions, kept separate on purpose (the v1
scaffold conflated them — see THEORY §1):

* **Per-class identifiability.** The equality ``p(z|y,d) = p(z|y,d')`` for a FIXED class
  ``y`` is identifiable iff both cells are estimable, i.e. ``d, d' ∈ S_y``. A path of
  *other* classes does NOT transfer a fixed-``y`` equality — there is no transitive reach
  across classes. Use :meth:`SupportGraph.is_estimable_pair` (takes ``y``) — note this is
  finite-sample *estimability* (the ``m``-gate), NOT population identifiability (THEORY §0/§1).
* **Coupling / decomposability.** The domain–domain graph ``d ~ d'`` iff they co-observe
  *some* class ties domains into **coupling components**. Within a component the per-class
  constraints share parameters (a single encoder is jointly constrained), so the problem
  does NOT decompose into independent subproblems; across components it does. This is about
  optimization structure, NOT about identifiability of any particular equality. Use
  :meth:`SupportGraph.coupled` / :attr:`coupling_components`.

Three support notions are also kept distinct (THEORY §1, point 2):

* **present** (``n_{d,y} > 0``)   — observed support;
* **eligible** (``n_{d,y} ≥ m``)  — *estimator-eligibility* at finite sample (what gates
  whether we form a leakage/alignment term — a variance guard, not a population fact);
* structural support (can the DGP produce ``y`` in ``d`` at all) is unobservable and is
  never asserted here.

The overlap-aware estimand is reported under a **fixed reference prior** ``p_ref(y)`` so it
does not drift as cells are deleted in the missing-cell sweep (THEORY §Estimand):

* ``L_abs  = Σ_{y∈C_cmp} p_ref(y)            · L_y``   (primary; report with the
  identifiable mass fraction ``Σ_{y∈C_cmp} p_ref(y)``),
* ``L_cond = Σ_{y∈C_cmp} p_ref(y|y∈C_cmp)    · L_y``   (diagnostic only).

numpy-only, no EEG required. Run ``python -m oaci.support_graph`` for a worked example.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def empirical_class_prior(counts) -> np.ndarray:
    """``p_ref(y) = N_y / N`` from a count table (uniform if empty)."""
    counts = np.asarray(counts, dtype=np.float64)
    per_class = counts.sum(axis=0)
    total = per_class.sum()
    if total <= 0:
        n_y = counts.shape[1]
        return np.full(n_y, 1.0 / n_y) if n_y else per_class
    return per_class / total


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
    return sorted((sorted(v) for v in groups.values()), key=lambda c: c[0])


# --------------------------------------------------------------------------------------
# support graph
# --------------------------------------------------------------------------------------
@dataclass
class SupportGraph:
    """Identifiability + decomposability bookkeeping. Build via :func:`build_support_graph`."""

    counts: np.ndarray                       # [D,C] ELIGIBILITY counts (unique support units; m-gate ONLY)
    m: int                                   # eligibility threshold (n^elig_{d,y} >= m)
    reference_prior: np.ndarray              # FIXED p_ref(y), len n_classes, sums to 1
    domain_names: list[str]
    class_names: list[str]
    cell_mass: np.ndarray                    # [D,C] ESTIMAND mass M_{d,y}: drives p(d|y), H_ref, weights

    present: np.ndarray                      # bool [D,C]  n > 0   (observed support)
    eligible: np.ndarray                     # bool [D,C]  n >= m  (estimator-eligibility)
    support_of_class: dict[int, list[int]]   # y -> S_y (eligible domains, sorted)
    comparable_classes: list[int]            # |S_y| >= 2  (a within-y comparison exists)
    singleton_classes: list[int]             # |S_y| == 1
    empty_classes: list[int]                 # |S_y| == 0
    coupling_components: list[list[int]]     # decomposition structure, NOT identifiability
    component_of: np.ndarray                 # [D] -> coupling-component id

    # ---- shapes ----
    @property
    def n_domains(self) -> int:
        return self.counts.shape[0]

    @property
    def n_classes(self) -> int:
        return self.counts.shape[1]

    # ---- cell / class queries ----
    def is_present(self, d: int, y: int) -> bool:
        return bool(self.present[d, y])

    def is_eligible(self, d: int, y: int) -> bool:
        """Estimator-eligibility (``n_{d,y} >= m``) — gates term formation, not a population
        identifiability statement."""
        return bool(self.eligible[d, y])

    def is_comparable(self, y: int) -> bool:
        return y in self.comparable_classes

    def ineligible_cells(self) -> list[tuple[int, int]]:
        """Cells below the eligibility threshold — no term is formed on these."""
        ds, ys = np.where(~self.eligible)
        return list(zip(ds.tolist(), ys.tolist()))

    # ---- per-class ESTIMABILITY (finite-sample, m-gated) — NOT population identifiability ----
    def is_estimable_pair(self, d: int, dp: int, y: int) -> bool:
        """Is the fixed-``y`` two-sample comparison of ``p(z|y,d)`` vs ``p(z|y,d')`` ESTIMABLE
        at this sample size, i.e. both cells eligible for THIS class (``d, d' ∈ S_y``, the
        finite-sample ``m``-gate)? No cross-class transitivity.

        This is an *operational estimability* statement, NOT population identifiability: the
        population theorem (THEORY §1) conditions on positive-probability **structural**
        support; this method reports whether we have enough samples (``m``) to estimate it.
        """
        if d == dp:
            return True
        return bool(self.eligible[d, y] and self.eligible[dp, y])

    def estimable_pairs(self, y: int) -> list[tuple[int, int]]:
        """All within-class-``y`` domain pairs estimable at this sample size (both in ``S_y``)."""
        s = self.support_of_class[y]
        return [(s[i], s[j]) for i in range(len(s)) for j in range(i + 1, len(s))]

    def all_estimable_constraints(self) -> list[tuple[int, int, int]]:
        """Every estimable constraint as ``(y, d, d')`` over comparable classes."""
        return [(y, d, dp) for y in self.comparable_classes for (d, dp) in self.estimable_pairs(y)]

    # deprecated aliases (do NOT use in paper output — they imply identifiability, not estimability)
    def is_identifiable_pair(self, d: int, dp: int, y: int) -> bool:
        return self.is_estimable_pair(d, dp, y)

    def identifiable_pairs(self, y: int) -> list[tuple[int, int]]:
        return self.estimable_pairs(y)

    def all_identifiable_constraints(self) -> list[tuple[int, int, int]]:
        return self.all_estimable_constraints()

    # ---- coupling / decomposability (NOT identifiability) ----
    def coupled(self, d: int, dp: int) -> bool:
        """True iff ``d`` and ``d'`` lie in the same coupling component, i.e. the constraint
        system does not decompose across them. This is an OPTIMIZATION-structure statement;
        it does NOT imply any particular ``p(z|y,·)`` equality between them is identifiable."""
        return bool(self.component_of[d] == self.component_of[dp])

    def decoupled_pairs(self) -> list[tuple[int, int]]:
        """Domain pairs that share no class — fully independent subproblems."""
        return [
            (d, dp)
            for d in range(self.n_domains)
            for dp in range(d + 1, self.n_domains)
            if not self.coupled(d, dp)
        ]

    # ---- the overlap-aware estimand's identifiable terms (fixed reference prior) ----
    def overlap_terms(self) -> list[dict]:
        """One term per comparable class ``y``, weighted by the FIXED reference prior.

        ``w_abs = p_ref(y)`` (primary, L_abs; mass that is NOT comparable is simply absent,
        so Σ w_abs = identifiable_mass_fraction <= 1). ``w_cond = p_ref(y|y∈C_cmp)``
        (renormalised over comparable classes; diagnostic only, L_cond — it moves the
        estimand as support fragments, so never the headline).
        """
        z = float(sum(self.reference_prior[y] for y in self.comparable_classes))
        out = []
        for y in self.comparable_classes:
            n_obs = float(self.cell_mass[self.support_of_class[y], y].sum())   # estimand MASS, not unit count
            out.append(
                {
                    "y": y,
                    "class_name": self.class_names[y],
                    "support": list(self.support_of_class[y]),
                    "n_obs": n_obs,
                    "w_abs": float(self.reference_prior[y]),
                    "w_cond": float(self.reference_prior[y] / z) if z > 0 else 0.0,
                }
            )
        return out

    def eligible_comparable_mass_fraction(self) -> float:
        """``Σ_{y∈C_cmp} p_ref(y)`` — the reference-prior mass on classes with an estimable
        cross-domain comparison (``|S_y|≥2`` under the ``m``-gate). Stable under the missing-cell
        sweep (``p_ref`` fixed). ``1 - this`` is the mass excluded for **low sample size** — it
        is NOT (necessarily) population-non-identifiable mass; report it alongside ``L_abs``."""
        return float(sum(self.reference_prior[y] for y in self.comparable_classes))

    def identifiable_mass_fraction(self) -> float:
        """Deprecated alias of :meth:`eligible_comparable_mass_fraction` — do NOT use in paper
        output (the excluded mass is low-sample, not proven non-identifiable)."""
        return self.eligible_comparable_mass_fraction()

    def observed_mass_fraction(self) -> float:
        """Sample-based companion (descriptive): fraction of *current* samples in a
        comparable, eligible cell. Unlike :meth:`eligible_comparable_mass_fraction` this moves
        with the counts, so it is a description of the data, not the estimand weight."""
        total = float(self.cell_mass.sum())
        if total <= 0:
            return 0.0
        constrained = sum(float(self.cell_mass[self.support_of_class[y], y].sum()) for y in self.comparable_classes)
        return constrained / total

    # ---- reporting ----
    def summary(self) -> dict:
        return {
            "n_domains": self.n_domains,
            "n_classes": self.n_classes,
            "m": self.m,
            "n_eligible_cells": int(self.eligible.sum()),
            "n_present_cells": int(self.present.sum()),
            "n_cells": self.n_domains * self.n_classes,
            "comparable_classes": list(self.comparable_classes),
            "singleton_classes": list(self.singleton_classes),
            "empty_classes": list(self.empty_classes),
            "n_coupling_components": len(self.coupling_components),
            "coupling_components": [list(c) for c in self.coupling_components],
            "eligible_comparable_mass_fraction": self.eligible_comparable_mass_fraction(),
            "observed_mass_fraction": self.observed_mass_fraction(),
            "n_decoupled_domain_pairs": len(self.decoupled_pairs()),
        }

    def report(self) -> str:
        s = self.summary()
        return "\n".join(
            [
                f"Support graph (m={self.m}): {s['n_domains']} domains x {s['n_classes']} classes",
                f"  eligible cells: {s['n_eligible_cells']}/{s['n_cells']}"
                f"  (present: {s['n_present_cells']})",
                f"  eligible-comparable mass (Σ p_ref over comparable): {s['eligible_comparable_mass_fraction']:.3f}"
                f"   observed mass: {s['observed_mass_fraction']:.3f}",
                f"  comparable classes (|S_y|>=2): {s['comparable_classes']}",
                f"  singleton-support classes (|S_y|==1): {s['singleton_classes']}",
                f"  empty classes (|S_y|==0): {s['empty_classes']}",
                f"  coupling components (decomposition, NOT identifiability):"
                f" {s['n_coupling_components']} -> {s['coupling_components']}",
                f"  fully-decoupled domain pairs (share no class): {self.decoupled_pairs()}",
            ]
        )


def build_support_graph(
    eligibility_counts,
    m: int,
    cell_mass=None,
    reference_prior=None,
    domain_names: list[str] | None = None,
    class_names: list[str] | None = None,
) -> SupportGraph:
    """Build the support graph, SEPARATING the eligibility count from the estimand mass.

    ``eligibility_counts[d,y]`` are **independent support units** (e.g. subjects/trials) and gate
    *only* the ``m``-eligibility — never let dozens of correlated windows of one subject count as
    independent support. ``cell_mass[d,y]`` (default = ``eligibility_counts``) is the estimand
    MASS ``M_{d,y}`` that drives ``p(d|y)``, the reference entropy ``H_ref``, and the sampler
    target. ``reference_prior`` defaults to the empirical prior of ``cell_mass``.

    ``m`` is the estimator-eligibility threshold (a finite-sample variance guard, not a
    population fact). ``reference_prior`` is the FIXED ``p_ref(y)`` used for the overlap
    estimand — pass the SAME prior (e.g. the pre-deletion empirical one) across a
    missing-cell sweep so the estimand does not drift; ``None`` defaults to the empirical
    prior of THIS ``counts`` table (correct for a one-off graph, wrong for a sweep).
    """
    counts = np.asarray(eligibility_counts)
    if counts.ndim != 2:
        raise ValueError(f"eligibility_counts must be 2D [n_domains, n_classes], got shape {counts.shape}")
    if m < 1:
        raise ValueError(f"m must be >= 1, got {m}")
    n_d, n_y = counts.shape
    mass = counts.astype(np.float64) if cell_mass is None else np.asarray(cell_mass, dtype=np.float64)
    if mass.shape != counts.shape:
        raise ValueError(f"cell_mass shape {mass.shape} must match eligibility_counts {counts.shape}")

    domain_names = list(domain_names) if domain_names is not None else [f"d{d}" for d in range(n_d)]
    class_names = list(class_names) if class_names is not None else [f"y{y}" for y in range(n_y)]
    if len(domain_names) != n_d or len(class_names) != n_y:
        raise ValueError("domain_names / class_names length must match counts shape")

    if reference_prior is None:
        p_ref = empirical_class_prior(mass)          # estimand prior comes from MASS, not unit counts
    else:
        p_ref = np.asarray(reference_prior, dtype=np.float64).ravel()
        if p_ref.shape != (n_y,):
            raise ValueError(f"reference_prior must have length n_classes={n_y}, got {p_ref.shape}")
        if np.any(p_ref < 0):
            raise ValueError("reference_prior must be non-negative")
        tot = p_ref.sum()
        if tot <= 0:
            raise ValueError("reference_prior must have positive mass")
        p_ref = p_ref / tot  # defensive normalisation to a proper prior

    present = counts > 0
    eligible = counts >= m
    support_of_class = {y: sorted(np.where(eligible[:, y])[0].tolist()) for y in range(n_y)}
    comparable = [y for y in range(n_y) if len(support_of_class[y]) >= 2]
    singleton = [y for y in range(n_y) if len(support_of_class[y]) == 1]
    empty = [y for y in range(n_y) if len(support_of_class[y]) == 0]

    # coupling edges: a star within each comparable class's support suffices for union–find.
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
        reference_prior=p_ref,
        domain_names=domain_names,
        class_names=class_names,
        cell_mass=mass,
        present=present,
        eligible=eligible,
        support_of_class=support_of_class,
        comparable_classes=comparable,
        singleton_classes=singleton,
        empty_classes=empty,
        coupling_components=components,
        component_of=component_of,
    )


# --------------------------------------------------------------------------------------
# self-demo
# --------------------------------------------------------------------------------------
def _demo() -> None:
    # 4 domains, 3 classes. Sites 0,1 co-observe class 0; sites 2,3 co-observe class 1;
    # class 2 only in site 0 (singleton); site 3 short on class 1 (below threshold).
    counts = np.array(
        [
            # y0   y1   y2
            [120, 0, 40],
            [110, 0, 0],
            [0, 90, 0],
            [0, 5, 0],
        ]
    )
    sites = ["BNCI-2a-s1", "BNCI-2a-s2", "PD-cohort", "SCZ-cohort"]
    classes = ["rest", "task", "rare-event"]
    sg = build_support_graph(counts, m=20, domain_names=sites, class_names=classes)
    print(sg.report())
    print("\nidentifiable overlap terms (fixed p_ref):")
    for t in sg.overlap_terms():
        names = [sites[d] for d in t["support"]]
        print(f"  y={t['class_name']:<10} S_y={names}  n_obs={t['n_obs']:<5}"
              f" w_abs={t['w_abs']:.3f}  w_cond={t['w_cond']:.3f}")
    print(
        "\nper-class estimability (m-gate, not identifiability) — is_estimable_pair(s0,s1,y=rest) =",
        sg.is_estimable_pair(0, 1, 0),
        "| (s0,s2,y=rest) =", sg.is_estimable_pair(0, 2, 0),
    )
    print("coupling (decomposition) — coupled(s0,s2) =", sg.coupled(0, 2),
          "(no shared class -> independent subproblems)")


if __name__ == "__main__":
    _demo()
