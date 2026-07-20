"""Domain-factor DAG primitives.

A ``DomainFactor`` is one acquisition/biological nuisance variable (site, device,
montage, subject, session, block, medication-state, ...).  A ``DomainDAG`` wires them
into a directed acyclic graph of parent->child relations and validates it.  A
``DomainLabels`` object carries the per-sample integer level of every factor as an
``[N, n_factors]`` matrix aligned to the DAG's factor order.

Handling policy (review section 5.5) per factor:

  invariant       amplifier / montage / reference / channel gain / site protocol:
                  drive leakage to (near) zero -> small budget.
  random_effect   session impedance, subject anatomy: allow a per-level offset /
                  partial leakage -> larger budget, not fully removed.
  conditional     age / sex / medication / fatigue: may be genuinely predictive of Y
                  or of brain physiology; CONDITION on it, do not erase -> large budget.
  label_mechanism rater / diagnostic-site: affects the LABEL channel p(Ytilde|Ystar,D),
                  modelled by ``h2cmi.label.site_mechanism`` rather than encoder
                  invariance -> excluded from the encoder CMI penalty.

``determines_label`` flags the SCPS degeneracy (review P0-4): when a factor fixes Y
(e.g. D=subject and disease label is constant per subject) the decoder CMI I(Y;D|Z)
collapses to label predictability H(Y|Z) and is NOT a concept-shift measurement.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Sequence

import numpy as np

HANDLING = ("invariant", "random_effect", "conditional", "label_mechanism")


@dataclass(frozen=True)
class DomainFactor:
    """One nuisance factor in the domain DAG."""

    name: str
    n_levels: int                      # cardinality of the categorical factor
    parents: tuple[str, ...] = ()      # parent factor names (must be defined earlier)
    handling: str = "invariant"        # one of HANDLING
    budget: float = 0.02               # leakage budget epsilon_j in nats
    determines_label: bool = False     # D=subject style Y=g(D) degeneracy flag
    description: str = ""

    def __post_init__(self) -> None:
        if self.handling not in HANDLING:
            raise ValueError(f"factor {self.name}: handling must be one of {HANDLING}")
        if self.n_levels < 1:
            raise ValueError(f"factor {self.name}: n_levels must be >= 1")
        if self.budget < 0:
            raise ValueError(f"factor {self.name}: budget must be >= 0")

    @property
    def penalised(self) -> bool:
        """Whether the encoder CMI penalty acts on this factor.

        Label-mechanism factors are handled by the label model, not encoder invariance,
        so they are excluded from the leakage budget machinery.
        """
        return self.handling in ("invariant", "random_effect")


class DomainDAG:
    """An ordered, validated DAG of ``DomainFactor`` objects.

    Factors are stored in the order given; every parent must appear before its child
    (so the stored order is already a topological order).  The chain-rule CMI
    decomposition (``h2cmi.cmi.hierarchical``) walks factors in this order, which is why
    the topological invariant is enforced at construction time.
    """

    def __init__(self, factors: Sequence[DomainFactor]):
        self.factors: list[DomainFactor] = list(factors)
        self._index = {f.name: i for i, f in enumerate(self.factors)}
        if len(self._index) != len(self.factors):
            raise ValueError("duplicate factor names in DomainDAG")
        seen: set[str] = set()
        for f in self.factors:
            for p in f.parents:
                if p not in self._index:
                    raise ValueError(f"factor {f.name}: unknown parent '{p}'")
                if p not in seen:
                    raise ValueError(
                        f"factor {f.name}: parent '{p}' must be declared before it "
                        "(DAG must be given in topological order)"
                    )
                if p == f.name:
                    raise ValueError(f"factor {f.name}: self-parent not allowed")
            seen.add(f.name)

    # -- lookups -----------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.factors)

    def __iter__(self):
        return iter(self.factors)

    @property
    def names(self) -> list[str]:
        return [f.name for f in self.factors]

    def index(self, name: str) -> int:
        return self._index[name]

    def get(self, name: str) -> DomainFactor:
        return self.factors[self._index[name]]

    def parent_indices(self, name: str) -> list[int]:
        return [self._index[p] for p in self.get(name).parents]

    def penalised_factors(self) -> list[DomainFactor]:
        """Factors the encoder CMI budget machinery should penalise."""
        return [f for f in self.factors if f.penalised]

    # -- convenience constructors ------------------------------------------------
    @staticmethod
    def hierarchical_site_subject_session(
        n_sites: int,
        subjects_per_site: int,
        sessions_per_subject: int,
        *,
        subject_determines_label: bool = False,
        budgets: dict[str, float] | None = None,
    ) -> "DomainDAG":
        """Canonical nested EEG DAG: site -> subject -> session.

        Mirrors review section 5.4.  ``subject_determines_label`` sets the SCPS
        degeneracy flag (clinical: disease label fixed per subject).
        """
        b = {"site": 0.02, "subject": 0.05, "session": 0.10}
        if budgets:
            b.update(budgets)
        n_subjects = n_sites * subjects_per_site
        n_sessions = n_subjects * sessions_per_subject
        return DomainDAG([
            DomainFactor("site", n_sites, (), "invariant", b["site"],
                         description="acquisition site / hardware+protocol cluster"),
            DomainFactor("subject", n_subjects, ("site",), "random_effect", b["subject"],
                         determines_label=subject_determines_label,
                         description="subject anatomy (nested in site)"),
            DomainFactor("session", n_sessions, ("subject",), "random_effect", b["session"],
                         description="session drift (nested in subject)"),
        ])

    def __repr__(self) -> str:
        parts = []
        for f in self.factors:
            par = ",".join(f.parents) if f.parents else "-"
            parts.append(f"{f.name}[{f.n_levels}]<-({par}):{f.handling}/eps={f.budget:g}")
        return "DomainDAG(" + "; ".join(parts) + ")"


@dataclass
class DomainLabels:
    """Per-sample integer levels for every factor, aligned to ``dag.names``.

    ``levels`` is an ``[N, n_factors]`` int array.  ``levels[:, j]`` are the level
    indices of factor ``dag.factors[j]`` (in ``0 .. n_levels-1``).
    """

    dag: DomainDAG
    levels: np.ndarray                  # [N, n_factors] int

    def __post_init__(self) -> None:
        self.levels = np.asarray(self.levels, dtype=np.int64)
        if self.levels.ndim != 2 or self.levels.shape[1] != len(self.dag):
            raise ValueError(
                f"levels must be [N, {len(self.dag)}]; got {self.levels.shape}")
        for j, f in enumerate(self.dag.factors):
            col = self.levels[:, j]
            if col.min() < 0 or col.max() >= f.n_levels:
                raise ValueError(
                    f"factor {f.name}: levels out of range [0,{f.n_levels})")

    @property
    def n(self) -> int:
        return self.levels.shape[0]

    def factor(self, name: str) -> np.ndarray:
        return self.levels[:, self.dag.index(name)]

    def subset(self, idx: np.ndarray) -> "DomainLabels":
        return DomainLabels(self.dag, self.levels[idx])

    def parent_key(self, name: str) -> np.ndarray:
        """Encode the joint level of a factor's parents as a single int per sample.

        Used as the conditioning context Pa(D_j) for the hierarchical CMI critic.
        Returns an all-zero column when the factor has no parents.
        """
        pidx = self.dag.parent_indices(name)
        if not pidx:
            return np.zeros(self.n, dtype=np.int64)
        key = np.zeros(self.n, dtype=np.int64)
        for j in pidx:
            key = key * self.dag.factors[j].n_levels + self.levels[:, j]
        return key


def compact_domain_labels(domains: "DomainLabels"):
    """Build a SOURCE-ONLY DAG with contiguous levels for every factor (review P0-1).

    After an outer (or inner pseudo-target) split, the subset still references the full
    DAG, whose factor cardinalities include levels that never appear in the subset (e.g.
    the held-out target site/subjects/sessions). That (a) leaks the target cardinality
    into a "strict DG" run and (b) makes the Laplace-smoothed reference entropy
    H(D_j|Y,Pa) and the critic CE live on different effective supports.

    This relabels each factor's observed levels to ``0..K-1`` and shrinks the factor
    cardinalities accordingly, returning ``(compact_dag, compact_labels, level_maps)``
    where ``level_maps[name]`` are the original level ids in compact order (so the mapping
    can be inverted). MUST be called on every fold/pseudo-target subset, not once.
    """
    levels = np.asarray(domains.levels, dtype=np.int64).copy()
    factors, level_maps = [], {}
    for j, factor in enumerate(domains.dag.factors):
        unique, inverse = np.unique(levels[:, j], return_inverse=True)
        levels[:, j] = inverse
        level_maps[factor.name] = unique
        factors.append(replace(factor, n_levels=int(len(unique))))
    dag = DomainDAG(factors)
    return dag, DomainLabels(dag, levels), level_maps
