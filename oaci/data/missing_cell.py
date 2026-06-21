"""Controlled missing-cell stress test — the direct, falsifiable carrier of OACI's claim.

Start from a fully-supported configuration and **systematically delete site×class cells**,
sweeping the support graph from connected toward fragmented. At every deletion level the
harness emits the artifacts that ALL downstream components (critic, bootstrap, sampler,
trainer, eval) must share so the comparison ERM / global-LPC / uniform / OACI is on
identical ground:

* a **cell mask** ``keep[d,y]`` (monotone: deletions accumulate);
* a **support graph** built under a **FIXED reference prior** ``p_ref`` (computed once on the
  base config) — so the overlap estimand ``L_abs`` does not drift as cells vanish
  (THEORY §Estimand);
* **group IDs** for the clustered bootstrap / grouped probe (the dependence unit);
* a deterministic **deletion schedule** (``random`` / ``rare_first`` / ``bridge_first``).

The key readout is the level at which the support graph first **fragments** (a new coupling
component appears) — align it with where the global/uniform routes start to hurt worst-domain
accuracy / calibration.

numpy-only, EEG-free, fully testable. Operates on count tables + per-sample label arrays.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..support_graph import SupportGraph, build_support_graph, empirical_class_prior


# --------------------------------------------------------------------------------------
# cell mask + schedule data structures
# --------------------------------------------------------------------------------------
@dataclass
class CellMask:
    """``keep[d,y]`` — True iff cell ``(d,y)`` is retained at this deletion level."""

    keep: np.ndarray  # bool [n_domains, n_classes]

    @property
    def n_deleted(self) -> int:
        return int((~self.keep).sum())

    def deleted_cells(self) -> list[tuple[int, int]]:
        ds, ys = np.where(~self.keep)
        return list(zip(ds.tolist(), ys.tolist()))

    def sample_keep(self, domain_labels, class_labels) -> np.ndarray:
        """Per-sample boolean: keep samples whose ``(d,y)`` cell is retained."""
        d = np.asarray(domain_labels).astype(int).ravel()
        y = np.asarray(class_labels).astype(int).ravel()
        return self.keep[d, y]


@dataclass
class DeletionStep:
    """One level of the sweep: the mask + the support graph it induces (fixed ``p_ref``)."""

    level: int                       # cumulative cells deleted vs base (0 = base)
    deleted_cell: tuple | None       # the (d,y) removed AT this step (None for base)
    mask: CellMask
    support_graph: SupportGraph

    @property
    def n_components(self) -> int:
        return len(self.support_graph.coupling_components)

    @property
    def identifiable_mass_fraction(self) -> float:
        return self.support_graph.identifiable_mass_fraction()

    @property
    def n_comparable_classes(self) -> int:
        return len(self.support_graph.comparable_classes)


@dataclass
class MissingCellSchedule:
    """A monotone deletion sweep with a fixed reference prior shared across all steps."""

    base_counts: np.ndarray
    reference_prior: np.ndarray      # FIXED across the whole sweep
    m: int
    strategy: str
    steps: list[DeletionStep]
    domain_names: list[str]
    class_names: list[str]

    def __len__(self) -> int:
        return len(self.steps)

    def __getitem__(self, i: int) -> DeletionStep:
        return self.steps[i]

    @property
    def base_n_components(self) -> int:
        return self.steps[0].n_components

    def first_fragmentation_level(self) -> int | None:
        """Level at which a NEW coupling component first appears (support fragments).
        ``None`` if it never fragments within the schedule."""
        base = self.base_n_components
        for s in self.steps:
            if s.n_components > base:
                return s.level
        return None

    def as_table(self) -> list[dict]:
        """Flat per-level log (level, deleted_cell, #components, fragmented?, id-mass, #comparable)."""
        base = self.base_n_components
        return [
            {
                "level": s.level,
                "deleted_cell": s.deleted_cell,
                "n_components": s.n_components,
                "fragmented": s.n_components > base,
                "identifiable_mass_fraction": round(s.identifiable_mass_fraction, 4),
                "n_comparable_classes": s.n_comparable_classes,
            }
            for s in self.steps
        ]


# --------------------------------------------------------------------------------------
# group IDs for clustered inference
# --------------------------------------------------------------------------------------
def make_group_ids(*label_arrays) -> np.ndarray:
    """Contiguous integer group id per sample for the dependence unit of the clustered
    bootstrap / grouped probe. Pass the label array(s) that define a cluster, e.g.
    ``make_group_ids(subject)`` or ``make_group_ids(subject, session)``."""
    if not label_arrays:
        raise ValueError("make_group_ids needs at least one label array")
    arrs = [np.asarray(a).ravel() for a in label_arrays]
    n = arrs[0].shape[0]
    if any(a.shape[0] != n for a in arrs):
        raise ValueError("all label arrays must have the same length")
    keys = list(zip(*[a.tolist() for a in arrs]))
    order = {k: i for i, k in enumerate(sorted(set(keys)))}
    return np.array([order[k] for k in keys], dtype=np.int64)


def apply_to_samples(mask: CellMask, domain_labels, class_labels) -> np.ndarray:
    """Boolean keep-mask over samples for a given :class:`CellMask` (convenience alias)."""
    return mask.sample_keep(domain_labels, class_labels)


# --------------------------------------------------------------------------------------
# deletion-cell selection strategies
# --------------------------------------------------------------------------------------
def _eligible_kept_cells(counts: np.ndarray, keep: np.ndarray, m: int) -> list[tuple[int, int]]:
    elig = (counts >= m) & keep
    ds, ys = np.where(elig)
    return list(zip(ds.tolist(), ys.tolist()))


def _select_cell(
    candidates: list[tuple[int, int]],
    strategy: str,
    counts: np.ndarray,
    keep: np.ndarray,
    build,
    rng,
) -> tuple[int, int]:
    if strategy == "random":
        return candidates[int(rng.integers(len(candidates)))]
    if strategy == "rare_first":
        # smallest eligible cell first; deterministic lexicographic tie-break
        return min(candidates, key=lambda c: (int(counts[c]), c[0], c[1]))
    if strategy == "bridge_first":
        # maximise resulting #coupling-components; tie -> smaller count -> lexicographic
        best_key = None
        best_cell = None
        for c in candidates:
            keep[c] = False
            ncomp = len(build(keep).coupling_components)
            keep[c] = True
            key = (ncomp, -int(counts[c]), -c[0], -c[1])  # max: more comps, smaller count, lex
            if best_key is None or key > best_key:
                best_key, best_cell = key, c
        return best_cell
    raise ValueError(f"unknown strategy: {strategy!r} (use random | rare_first | bridge_first)")


# --------------------------------------------------------------------------------------
# schedule builder
# --------------------------------------------------------------------------------------
def make_schedule(
    counts,
    m: int,
    strategy: str = "bridge_first",
    n_steps: int | None = None,
    reference_prior=None,
    seed: int = 0,
    domain_names: list[str] | None = None,
    class_names: list[str] | None = None,
    stop_when_no_comparable: bool = True,
) -> MissingCellSchedule:
    """Build a monotone missing-cell sweep.

    ``reference_prior`` is fixed once (default: empirical prior of the base ``counts``) and
    threaded through every step's support graph, so the ``L_abs`` estimand is stable across
    the sweep. ``n_steps`` caps the number of deletions (default: until no comparable class
    remains, or all eligible cells are gone). Deletion is one eligible cell per step.
    """
    counts = np.asarray(counts).astype(np.int64)
    if counts.ndim != 2:
        raise ValueError(f"counts must be 2D, got shape {counts.shape}")
    n_d, n_y = counts.shape
    domain_names = list(domain_names) if domain_names is not None else [f"d{d}" for d in range(n_d)]
    class_names = list(class_names) if class_names is not None else [f"y{y}" for y in range(n_y)]

    p_ref = empirical_class_prior(counts) if reference_prior is None else np.asarray(reference_prior, float).ravel()
    rng = np.random.default_rng(seed)

    keep = np.ones((n_d, n_y), dtype=bool)

    def build(mask_keep: np.ndarray) -> SupportGraph:
        masked = np.where(mask_keep, counts, 0)
        return build_support_graph(
            masked, m, reference_prior=p_ref, domain_names=domain_names, class_names=class_names
        )

    steps = [DeletionStep(level=0, deleted_cell=None, mask=CellMask(keep.copy()), support_graph=build(keep))]
    max_steps = n_steps if n_steps is not None else int((counts >= m).sum())

    for _ in range(max_steps):
        if stop_when_no_comparable and not steps[-1].support_graph.comparable_classes:
            break
        candidates = _eligible_kept_cells(counts, keep, m)
        if not candidates:
            break
        cell = _select_cell(candidates, strategy, counts, keep, build, rng)
        keep[cell] = False
        steps.append(
            DeletionStep(
                level=len(steps),
                deleted_cell=cell,
                mask=CellMask(keep.copy()),
                support_graph=build(keep),
            )
        )

    return MissingCellSchedule(
        base_counts=counts,
        reference_prior=p_ref,
        m=int(m),
        strategy=strategy,
        steps=steps,
        domain_names=domain_names,
        class_names=class_names,
    )


# --------------------------------------------------------------------------------------
# self-demo
# --------------------------------------------------------------------------------------
def _demo() -> None:
    # d2 hangs off the rest only through class 2; deleting that bridge cell isolates it.
    counts = np.array(
        [
            # y0   y1   y2
            [100, 100, 0],
            [100, 100, 200],
            [0, 0, 200],
        ]
    )
    sched = make_schedule(counts, m=10, strategy="bridge_first")
    print(f"strategy=bridge_first  fixed p_ref={np.round(sched.reference_prior, 3)}")
    print(f"first fragmentation at level: {sched.first_fragmentation_level()}")
    for row in sched.as_table():
        print(f"  L{row['level']}: del={row['deleted_cell']} comps={row['n_components']}"
              f" frag={row['fragmented']} id-mass={row['identifiable_mass_fraction']}"
              f" #cmp-classes={row['n_comparable_classes']}")


if __name__ == "__main__":
    _demo()
