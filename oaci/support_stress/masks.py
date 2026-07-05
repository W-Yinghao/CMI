"""C18 — apply a RegimePlan to (a) a support graph (masked counts/cell_mass -> rebuilt SupportGraph, for
estimability + leakage recompute) and (b) persisted per-unit source predictions (keep/reweight mask over
units, for masked worst-domain source-metric recompute). Cell actions are resolved to domain NAMES so the
same plan applies role-agnostically to source_guard / source_audit units and to source/audit support graphs.

The FIXED reference prior p_ref is NEVER changed by masking (support degradation must not move the estimand).
Non-estimable / rare cells are subsampled deterministically (fold-seeded); deleted cells are removed; skewed
cells are reweighted. Unsupported cells are NEVER smoothed or imputed.
"""
from __future__ import annotations

import numpy as np

from ..support_graph import build_support_graph
from .stress_plan import _fold_seed


def actions_by_name(plan, source_domain_names) -> dict:
    """(domain_name, class) -> CellAction, from a plan whose actions carry source-domain INDICES."""
    out = {}
    for a in plan.actions:
        out[(str(source_domain_names[a.domain]), int(a.cls))] = a
    return out


def apply_to_support_graph(name_actions, base_sg):
    """Rebuild a masked SupportGraph, resolving each perturbed cell by domain NAME against base_sg (so a plan
    built on the source graph applies correctly to the differently-indexed audit graph; cells whose domain is
    absent from base_sg are skipped). delete->count/mass 0; nonestimable->count target(<m); rare->count m;
    skew->cell_mass*weight. reference_prior FIXED. Domains/classes/m unchanged."""
    counts = np.array(base_sg.counts, dtype=np.int64, copy=True)
    mass = np.array(base_sg.cell_mass, dtype=np.float64, copy=True)
    name_to_idx = {str(n): i for i, n in enumerate(base_sg.domain_names)}
    for (dname, c), a in name_actions.items():
        d = name_to_idx.get(str(dname))
        if d is None:
            continue                                       # cell's domain not in this graph (e.g. audit subset)
        if a.action == "delete":
            counts[d, c] = 0; mass[d, c] = 0.0
        elif a.action in ("nonestimable", "rare"):
            counts[d, c] = int(a.target_count)
        elif a.action == "skew":
            mass[d, c] = mass[d, c] * float(a.weight)
    return build_support_graph(counts, int(base_sg.m), cell_mass=mass,
                               reference_prior=np.array(base_sg.reference_prior, dtype=np.float64),
                               domain_names=list(base_sg.domain_names), class_names=list(base_sg.class_names))


def unit_keep_weight(name_actions, domain_raw, y, *, seed, target, level):
    """Boolean keep-mask + per-unit weight over persisted units for a regime.
      delete       -> drop all units in the cell
      nonestimable -> subsample the cell to target_count units (deterministic)
      rare         -> subsample to target_count (== m)
      skew         -> keep all, multiply weight
    Units not in any perturbed cell are kept at weight 1."""
    domain_raw = np.asarray([str(x) for x in domain_raw]); y = np.asarray(y).astype(int)
    keep = np.ones(len(y), dtype=bool); weight = np.ones(len(y), dtype=np.float64)
    rng = np.random.RandomState(_fold_seed(seed, target, level, "unit_subsample"))
    for (dname, cls), a in sorted(name_actions.items()):
        idx = np.where((domain_raw == dname) & (y == cls))[0]
        if len(idx) == 0:
            continue
        if a.action == "delete":
            keep[idx] = False
        elif a.action in ("nonestimable", "rare"):
            tc = int(a.target_count)
            if tc < len(idx):
                drop = rng.choice(idx, size=len(idx) - tc, replace=False)
                keep[drop] = False
        elif a.action == "skew":
            weight[idx] = weight[idx] * float(a.weight)
    return keep, weight


def subset_rows_by_cells(name_actions, domain_raw, y, *, seed, target, level):
    """Row-index keep set for Z-feature leakage recompute (delete/nonestimable/rare reduce membership;
    skew keeps all)."""
    keep, _ = unit_keep_weight(name_actions, domain_raw, y, seed=seed, target=target, level=level)
    return np.where(keep)[0]
