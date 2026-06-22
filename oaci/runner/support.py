"""Deletion schedule + per-level support state (2-level: source-train cells deleted, audit/target
fixed). Level-0 freezes p_ref / D0 / class map; later levels recompute eligibility counts and cell
mass on the shrunken source-train but always pass the FIXED level-0 prior to build_support_graph.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np

from ..methods.activity import all_method_status
from ..support_graph import build_support_graph
from .keys import feed_int64, feed_string


@dataclass(frozen=True)
class DeletionCell:
    domain_id: str
    class_name: str


@dataclass(frozen=True)
class DeletionSchedule:
    cells: tuple
    schedule_hash: str
    require_deleted_domain_retained: bool = True


def _source_cells(fold_data) -> set:
    out = set()
    for i in fold_data.source_train_idx.tolist():
        out.add((fold_data.domain_id[i], fold_data.class_names[int(fold_data.y[i])]))
    return out


def make_deletion_schedule(cells, fold_data, maps, require_deleted_domain_retained=True) -> DeletionSchedule:
    cells = tuple(DeletionCell(str(c.domain_id), str(c.class_name)) for c in cells)
    if len({(c.domain_id, c.class_name) for c in cells}) != len(cells):
        raise ValueError("duplicate deletion cell")
    observed = _source_cells(fold_data)
    for c in cells:
        if c.domain_id not in maps.source_domain_to_index:
            raise ValueError(f"deletion domain {c.domain_id!r} not a source domain")
        if c.class_name not in maps.class_to_index:
            raise ValueError(f"deletion class {c.class_name!r} not in class map")
        if (c.domain_id, c.class_name) not in observed:
            raise ValueError(f"deletion cell {(c.domain_id, c.class_name)} not observed in level-0 source train")
    h = hashlib.sha256(); h.update(maps.maps_hash.encode())
    h.update(fold_data.source_train_population_hash.encode())
    for c in cells:
        feed_string(h, c.domain_id); feed_string(h, c.class_name)
    return DeletionSchedule(cells, h.hexdigest()[:16], bool(require_deleted_domain_retained))


def level0_reference_prior(fold_data, maps) -> np.ndarray:
    nc = len(maps.class_names)
    m = np.zeros(nc, dtype=np.float64)
    for i in fold_data.source_train_idx.tolist():
        m[int(fold_data.y[i])] += float(fold_data.sample_mass[i])
    tot = m.sum()
    if tot <= 0:
        raise ValueError("level-0 source train has zero total mass")
    return m / tot


@dataclass(frozen=True)
class LevelSupportState:
    level: int
    source_train_idx: np.ndarray
    source_train_sample_ids: tuple
    source_train_population_hash: str
    deleted_cells: tuple
    eligibility_counts: np.ndarray
    cell_mass: np.ndarray
    support_graph: object
    observed_domain_ids: tuple
    method_status_items: tuple
    support_hash: str
    level_support_hash: str


def build_level_support(fold_data, maps, level, schedule: DeletionSchedule, level0_ref_prior,
                        support_m) -> LevelSupportState:
    if not (0 <= level <= len(schedule.cells)):
        raise ValueError(f"level {level} out of 0..{len(schedule.cells)}")
    deleted = schedule.cells[:level]
    deleted_keys = {(c.domain_id, c.class_name) for c in deleted}
    keep = [i for i in fold_data.source_train_idx.tolist()
            if (fold_data.domain_id[i], fold_data.class_names[int(fold_data.y[i])]) not in deleted_keys]
    keep = np.array(sorted(keep), dtype=np.int64); keep.setflags(write=False)

    nd, nc = len(maps.source_domain_ids), len(maps.class_names)
    counts = np.zeros((nd, nc), dtype=np.int64)
    mass = np.zeros((nd, nc), dtype=np.float64)
    cell_units: dict = {}
    observed_dom = set()
    for i in keep.tolist():
        d = maps.source_domain_to_index[fold_data.domain_id[i]]; yy = int(fold_data.y[i])
        cell_units.setdefault((d, yy), set()).add(fold_data.support_unit_id[i])
        mass[d, yy] += float(fold_data.sample_mass[i]); observed_dom.add(fold_data.domain_id[i])
    for (d, yy), us in cell_units.items():
        counts[d, yy] = len(us)
    counts.setflags(write=False); mass.setflags(write=False)
    if not np.array_equal(counts > 0, mass > 0):
        raise ValueError("counts>0 and cell_mass>0 disagree")
    for c in range(nc):
        if mass[:, c].sum() <= 0:
            raise ValueError(f"class {maps.class_names[c]} has zero mass at level {level}")

    sg = build_support_graph(eligibility_counts=counts, cell_mass=mass, m=int(support_m),
                             reference_prior=np.asarray(level0_ref_prior, dtype=np.float64),
                             domain_names=list(maps.source_domain_ids), class_names=list(maps.class_names))

    for c in deleted:                                          # declared deleted cell must be empty
        d = maps.source_domain_to_index[c.domain_id]; yy = maps.class_to_index[c.class_name]
        if counts[d, yy] != 0 or mass[d, yy] != 0:
            raise ValueError(f"deleted cell {(c.domain_id, c.class_name)} still has count/mass")
        if any(fold_data.domain_id[i] == c.domain_id and int(fold_data.y[i]) == yy for i in keep.tolist()):
            raise ValueError("deleted cell still has source-train rows")
        if schedule.require_deleted_domain_retained and c.domain_id not in observed_dom:
            raise ValueError(f"deleted domain {c.domain_id} no longer present via another class")

    sids = tuple(fold_data.sample_id[i] for i in keep.tolist())
    pop = hashlib.sha256(); feed_string(pop, "source_train_level")
    for s in sorted(sids):
        feed_string(pop, s)
    st_pop = pop.hexdigest()[:16]
    ms = tuple((m, all_method_status(sg, nd, len(observed_dom))[m]) for m in ("ERM", "OACI", "global_lpc", "uniform"))
    lh = hashlib.sha256(); lh.update(st_pop.encode()); lh.update(sg.support_hash().encode())
    feed_int64(lh, level)
    for c in deleted:
        feed_string(lh, c.domain_id); feed_string(lh, c.class_name)
    return LevelSupportState(level, keep, sids, st_pop, deleted, counts, mass, sg,
                             tuple(sorted(observed_dom)), ms, sg.support_hash(), lh.hexdigest()[:16])
