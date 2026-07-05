"""C18 — deterministic support-stress plans. Each regime S0-S7 maps a fold-level's base support (counts,
cell_mass, eligible mask, m) to a CONCRETE, DETERMINISTIC modification: a set of perturbed (domain,class)
cells + a per-cell action (delete / drop-below-m / rare / skew-reweight). No Math.random / no wall-clock:
S7's "random" mask is seeded from the fold key so it is reproducible and order-invariant.

The key contrast is S6 (boundary-aligned: perturb cells in the C16 class-boundary rotation classes) vs S7
(random mask matched to S6 in the NUMBER of perturbed cells). Boundary classes are loaded from committed C16
evidence, never hardcoded blindly.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field

import numpy as np

from . import schema


@dataclass(frozen=True)
class CellAction:
    domain: int                      # domain index into counts rows
    cls: int                         # class index
    action: str                      # 'delete' | 'nonestimable' | 'rare' | 'skew'
    target_count: int = 0            # resulting eligibility count for rare/nonestimable
    weight: float = 1.0              # cell_mass multiplier for skew


@dataclass(frozen=True)
class RegimePlan:
    regime: str
    actions: tuple                   # tuple[CellAction]
    boundary_classes: tuple          # classes driving S6
    severity_n_cells: int
    note: str = ""

    def deleted_cells(self) -> list:
        return [(a.domain, a.cls) for a in self.actions if a.action in ("delete", "nonestimable")]

    def as_row(self) -> dict:
        return {"regime": self.regime, "n_perturbed_cells": len(self.actions),
                "severity_n_cells": self.severity_n_cells, "boundary_classes": list(self.boundary_classes),
                "actions": [{"domain": a.domain, "cls": a.cls, "action": a.action,
                             "target_count": a.target_count, "weight": a.weight} for a in self.actions]}


def _fold_seed(seed, target, level, salt="") -> int:
    h = hashlib.sha256(f"{seed}|{target}|{level}|{salt}".encode()).digest()
    return int.from_bytes(h[:4], "big")


def boundary_classes_from_c16(per_class_recall_delta: dict, k=2) -> tuple:
    """The k classes with the most negative OACI-ERM recall delta (the classes OACI sacrifices) — the
    'opposite-direction' losers of the boundary rotation. Deterministic tie-break by class index."""
    items = sorted(((int(c), float(v["mean_recall_delta"])) for c, v in per_class_recall_delta.items()),
                   key=lambda t: (t[1], t[0]))
    return tuple(c for c, _ in items[:k])


def _eligible_source_cells(counts, eligible, m) -> list:
    """(domain,class) cells that are eligible (n>=m) and have slack to perturb, sorted deterministically."""
    D, C = counts.shape
    return [(d, c) for d in range(D) for c in range(C) if eligible[d, c]]


def build_regime_plan(regime, counts, cell_mass, eligible, m, *, boundary_classes, seed, target, level,
                      n_perturb=2) -> RegimePlan:
    counts = np.asarray(counts); cell_mass = np.asarray(cell_mass); eligible = np.asarray(eligible, dtype=bool)
    elig = _eligible_source_cells(counts, eligible, m)
    bnd = tuple(int(c) for c in boundary_classes)

    if regime == "S0_full_support":
        return RegimePlan(regime, (), bnd, 0, "no perturbation; exact reproduction baseline")

    if regime == "S1_label_marginal_skew":
        # skew class marginals within each source domain (down-weight boundary classes' mass), no deletion
        acts = tuple(CellAction(d, c, "skew", weight=0.5) for (d, c) in elig if c in bnd)
        return RegimePlan(regime, acts, bnd, 0, "cell_mass reweight (skew); all cells remain estimable")

    if regime in ("S2_rare_cells", "S3_nonestimable_cells", "S4_missing_cells"):
        # perturb the n_perturb eligible cells with the LARGEST slack (count - m), deterministic
        ranked = sorted(elig, key=lambda dc: (-(int(counts[dc]) - m), dc[0], dc[1]))[:n_perturb]
        if regime == "S2_rare_cells":
            acts = tuple(CellAction(d, c, "rare", target_count=m) for (d, c) in ranked)
        elif regime == "S3_nonestimable_cells":
            acts = tuple(CellAction(d, c, "nonestimable", target_count=max(m - 1, 0)) for (d, c) in ranked)
        else:
            acts = tuple(CellAction(d, c, "delete") for (d, c) in ranked)
        return RegimePlan(regime, acts, bnd, len(acts), f"{regime}: {len(acts)} cells")

    if regime == "S5_block_class_by_domain":
        # make ONE class (the top boundary-loser) non-estimable across a subset of source domains
        cls = bnd[0] if bnd else 0
        doms = sorted({d for (d, c) in elig if c == cls})
        take = doms[: max(1, len(doms) // 2)]
        acts = tuple(CellAction(d, cls, "nonestimable", target_count=max(m - 1, 0)) for d in take)
        return RegimePlan(regime, acts, bnd, len(acts), f"class {cls} non-estimable across {len(take)} domains")

    if regime == "S6_boundary_aligned_mask":
        cells = [dc for dc in elig if dc[1] in bnd]
        rng = np.random.RandomState(_fold_seed(seed, target, level, "S6"))
        take = _det_take(cells, min(n_perturb + 1, len(cells)), rng)
        acts = tuple(CellAction(d, c, "delete") for (d, c) in take)
        return RegimePlan(regime, acts, bnd, len(acts), f"boundary-aligned delete of {len(acts)} cells in classes {bnd}")

    if regime == "S7_random_matched_mask":
        # severity-matched to S6 (same #cells) but random cells NOT restricted to boundary classes
        s6 = build_regime_plan("S6_boundary_aligned_mask", counts, cell_mass, eligible, m,
                               boundary_classes=bnd, seed=seed, target=target, level=level, n_perturb=n_perturb)
        k = s6.severity_n_cells
        rng = np.random.RandomState(_fold_seed(seed, target, level, "S7"))
        take = _det_take(list(elig), min(k, len(elig)), rng)
        acts = tuple(CellAction(d, c, "delete") for (d, c) in take)
        return RegimePlan(regime, acts, bnd, len(acts), f"random severity-matched delete of {len(acts)} cells")

    raise ValueError(f"unknown regime {regime}")


def _det_take(cells, k, rng) -> list:
    cells = sorted(cells)
    if k >= len(cells):
        return cells
    idx = rng.choice(len(cells), size=k, replace=False)
    return sorted(cells[i] for i in idx.tolist())


def all_regime_plans(counts, cell_mass, eligible, m, *, boundary_classes, seed, target, level, n_perturb=2) -> dict:
    return {r: build_regime_plan(r, counts, cell_mass, eligible, m, boundary_classes=boundary_classes,
                                 seed=seed, target=target, level=level, n_perturb=n_perturb)
            for r in schema.REGIME_ORDER}
