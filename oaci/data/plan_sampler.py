"""String-preserving mass-unit samplers that emit ID-based plan steps.

These are the kernels the runner's plan materialisers use (the legacy ``RareCellSampler`` int-casts
groups and cannot make a task stream without a comparable class). All ids — ``sample_id``,
``group_id``, ``mass_unit_id`` — stay STABLE STRINGS end to end. A mass unit lives in exactly one
``(domain, class, group)`` and its base mass sums to 1; rows inside a unit are canonical-sorted by
``sample_id`` so input row order never changes a plan.
"""
from __future__ import annotations

import numpy as np

from ..train.rng import derive_seed


class UnitIndex:
    def __init__(self, sample_id, y, d, group_id, mass_unit_id, sample_mass):
        self.sample_id = tuple(str(s) for s in sample_id)
        n = len(self.sample_id)
        self.y = np.asarray(y, dtype=int); self.d = np.asarray(d, dtype=int)
        self.group = tuple(str(g) for g in group_id)
        self.unit = tuple(str(u) for u in mass_unit_id)
        self.b = np.asarray(sample_mass, dtype=np.float64)
        if not (len(self.group) == len(self.unit) == self.y.shape[0] == self.d.shape[0] == self.b.shape[0] == n):
            raise ValueError("plan-sampler array lengths disagree")
        if len(set(self.sample_id)) != n:
            raise ValueError("sample_id must be unique")
        if not np.all(np.isfinite(self.b)) or np.any(self.b <= 0):
            raise ValueError("sample_mass must be finite and strictly positive")
        # unit -> rows (canonical-sorted by sample_id), unit -> (d, y, group), unit mass == 1
        rows: dict = {}
        for i in range(n):
            rows.setdefault(self.unit[i], []).append(i)
        self.unit_rows: dict = {}
        self.unit_cell: dict = {}
        self.unit_class: dict = {}
        for u, ix in rows.items():
            ix = sorted(ix, key=lambda i: self.sample_id[i])
            cells = {(int(self.d[i]), int(self.y[i]), self.group[i]) for i in ix}
            if len(cells) != 1:
                raise ValueError(f"mass unit {u!r} spans multiple (domain,class,group) cells {cells}")
            dom, cls, grp = next(iter(cells))
            if abs(float(self.b[ix].sum()) - 1.0) > 1e-9:
                raise ValueError(f"mass unit {u!r} base mass {float(self.b[ix].sum())} != 1")
            self.unit_rows[u] = ix
            self.unit_cell[u] = (dom, cls)
            self.unit_class[u] = cls
        self.units = sorted(self.unit_rows)
        self.class_units = {}
        self.cell_units = {}
        for u in self.units:
            self.class_units.setdefault(self.unit_class[u], []).append(u)
            self.cell_units.setdefault(self.unit_cell[u], []).append(u)

    def present_classes(self) -> list:
        return sorted(self.class_units)

    def observed_cells(self) -> list:
        return sorted(self.cell_units)

    def draw_row(self, unit, seed: int) -> int:
        """Pick one row of a unit by its within-unit base mass (canonical order; derived seed)."""
        ix = self.unit_rows[unit]
        if len(ix) == 1:
            return ix[0]
        p = self.b[ix] / self.b[ix].sum()
        return int(np.random.default_rng(int(seed)).choice(ix, p=p))


def _draw_units(pool, k, replacement_mode, rng):
    pool = list(pool)
    if replacement_mode == "never" and k > len(pool):
        raise ValueError(f"replacement_mode='never' but {len(pool)} units < k={k}")
    if replacement_mode == "never" or (replacement_mode == "auto" and k <= len(pool)):
        return [pool[i] for i in rng.permutation(len(pool))[:k]]
    return [pool[i] for i in rng.integers(0, len(pool), size=k)]


class MassUnitTaskSampler:
    """Class-stratified task stream — works WITHOUT any comparable class. Each step assigns
    ``m_y >= 1`` per present class summing to ``batch_size`` (remainder rotated deterministically),
    draws ``m_y`` units and weights each drawn row ``U_y / m_y`` so the weighted class mass restores
    the fixed unit count ``U_y``."""

    def __init__(self, idx: UnitIndex, batch_size: int, base_seed: int, replacement_mode="auto"):
        self.idx = idx
        self.classes = idx.present_classes()
        if batch_size < len(self.classes):
            raise ValueError(f"task_batch_size {batch_size} < present classes {len(self.classes)}")
        self.batch_size = int(batch_size)
        self.base_seed = int(base_seed)
        self.replacement_mode = replacement_mode

    def per_class_counts(self, step_index: int) -> dict:
        C = len(self.classes); base = self.batch_size // C; rem = self.batch_size - base * C
        m = {c: base for c in self.classes}
        for j in range(rem):                                  # rotate the remainder by step index
            m[self.classes[(step_index + j) % C]] += 1
        return m

    def step(self, step_index: int):
        m = self.per_class_counts(step_index)
        sids, ws = [], []
        for c in self.classes:
            U = len(self.idx.class_units[c]); mc = m[c]
            rng = np.random.default_rng(derive_seed(self.base_seed, "task_units", step_index, c))
            for j, u in enumerate(_draw_units(self.idx.class_units[c], mc, self.replacement_mode, rng)):
                sids.append(self.idx.sample_id[self.idx.draw_row(u, derive_seed(self.base_seed, "task_row", step_index, c, j))])
                ws.append(U / mc)
        return tuple(sids), tuple(ws), derive_seed(self.base_seed, "task_dropout", step_index)


class _CellSampler:
    """Cell-stratified alignment over a fixed cell list; weights ``U_cell / k`` restore cell mass."""
    namespace = "cell"

    def __init__(self, idx: UnitIndex, cells, per_cell: int, base_seed: int, replacement_mode="auto"):
        self.idx = idx
        self.cells = list(cells)
        self.per_cell = int(per_cell)
        self.base_seed = int(base_seed)
        self.replacement_mode = replacement_mode

    def step(self, step_index: int):
        sids, ws = [], []
        for ci, cell in enumerate(self.cells):
            pool = self.idx.cell_units[cell]; U = len(pool); k = self.per_cell
            rng = np.random.default_rng(derive_seed(self.base_seed, self.namespace, step_index, ci))
            for j, u in enumerate(_draw_units(pool, k, self.replacement_mode, rng)):
                sids.append(self.idx.sample_id[self.idx.draw_row(u, derive_seed(self.base_seed, self.namespace + "_row", step_index, ci, j))])
                ws.append(U / k)
        return tuple(sids), tuple(ws), derive_seed(self.base_seed, self.namespace + "_dropout", step_index)


class RareEligibleCellSampler(_CellSampler):
    """OACI alignment: only eligible comparable cells ``{(d,y): y∈C_cmp, d∈S_y}``."""
    namespace = "oaci_align"

    def __init__(self, idx: UnitIndex, support_graph, per_cell, base_seed, replacement_mode="auto"):
        cells = [(int(dd), int(y)) for y in support_graph.comparable_classes
                 for dd in support_graph.support_of_class[y] if (int(dd), int(y)) in idx.cell_units]
        super().__init__(idx, cells, per_cell, base_seed, replacement_mode)


class ObservedCellSampler(_CellSampler):
    """Full-domain alignment: every observed cell (``cell_mass > 0``), incl. low-sample cells; no
    missing-cell rows. Uses observed cells, NOT the OACI m-eligibility gate."""
    namespace = "full_domain_align"

    def __init__(self, idx: UnitIndex, per_cell, base_seed, replacement_mode="auto"):
        super().__init__(idx, idx.observed_cells(), per_cell, base_seed, replacement_mode)
