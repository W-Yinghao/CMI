"""Paired-stream rare-cell sampler — over MASS UNITS, not windows.

Each row carries base mass ``b_i > 0`` and belongs to a ``mass_unit_id`` (default: each row is its
own unit); within a unit ``Σ b_i = 1``. The sampler's queues are over **units** (a subject / trial
/ film clip), so duplicating a window inside a unit leaves the sampled MEASURE unchanged:

* task stream      — draw class-stratified UNITS (incl. ineligible cells);
* adversary stream — draw UNITS per eligible ``(d,y)`` cell (covers all eligible cells per logical
  step); for a drawn unit, sample ONE row within it ∝ ``b_i``.

In the unit-equal case (``B_u=1``) a stratum with ``U_s`` units drawn ``m_s`` per batch gives the
importance weight ``w_i = U_s / m_s`` (the general form is ``w_i = b_i/(m q_i)``); the per-logical-
batch weighted stratum mass is then exactly ``U_s = M_s``. ``WeightedBatch.weight`` already includes
that proposal correction — do NOT multiply by ``sample_mass`` again at train time.

Support graph / ``S_y`` / ``p_ref`` / ``cell_mass`` are FIXED; a batch never redefines them.
"""
from __future__ import annotations

import numpy as np

from ..config import SamplerConfig
from ..support_graph import SupportGraph
from .batch import AdvLogicalBatch, WeightedBatch


class _Queue:
    """Seeded draw from a fixed pool of UNIT ids (no repeat until exhausted; rare cells replace
    only when forced)."""

    def __init__(self, items, rng, mode: str):
        self.items = np.asarray(items, dtype=object)
        self.rng = rng
        self.mode = mode
        self._order = self.items.copy()
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self):
        self._order = self.items.copy()
        self.rng.shuffle(self._order)
        self._cursor = 0

    def draw(self, k: int):
        n = len(self.items)
        if self.mode == "always":
            sel = self.rng.choice(self.items, size=k, replace=True)
            return list(sel), int(k - np.unique(sel).size)
        if self.mode == "never":
            if k > n:
                raise ValueError(f"replacement_mode='never' but a cell/class has {n} units < k={k}")
            if self._cursor + k > n:
                self._reshuffle()
            sel = self._order[self._cursor:self._cursor + k]
            self._cursor += k
            return list(sel), 0
        out, seen, reps = [], set(), 0
        for _ in range(k):
            if self._cursor >= n:
                self._reshuffle()
            it = self._order[self._cursor]
            self._cursor += 1
            if it in seen:
                reps += 1
            seen.add(it)
            out.append(it)
        return out, reps


class RareCellSampler:
    def __init__(self, y, d, group, support_graph: SupportGraph, cfg: SamplerConfig,
                 sample_mass=None, mass_unit_id=None):
        self.cfg = cfg.validate()
        self.sg = support_graph
        self.y = np.asarray(y, int)
        self.d = np.asarray(d, int)
        self.group = np.asarray(group, int)
        n = self.y.shape[0]
        self.b = np.ones(n) if sample_mass is None else np.asarray(sample_mass, float)
        if self.b.shape[0] != n or not np.all(np.isfinite(self.b)) or np.any(self.b <= 0):
            raise ValueError("sample_mass must be finite, strictly positive, length N")
        self.unit = (np.arange(n).astype(object) if mass_unit_id is None
                     else np.asarray(mass_unit_id, dtype=object))

        self.comparable = list(support_graph.comparable_classes)
        self.cells = [(int(dd), int(yy)) for yy in self.comparable for dd in support_graph.support_of_class[yy]]
        self.K_ov = len(self.cells)
        if self.K_ov == 0:
            raise ValueError("no eligible (d,y) cells -> adversary stream empty (no comparable class)")
        cfg.assert_capacity(self.K_ov)

        # unit -> rows; validate each unit nested in one (d,y,group) and unit mass == 1
        self.unit_rows: dict = {}
        self.unit_prob: dict = {}
        self.unit_cell: dict = {}
        self.unit_class: dict = {}
        for u in np.unique(self.unit):
            rows = np.where(self.unit == u)[0]
            cells = {(int(self.d[i]), int(self.y[i])) for i in rows}
            groups = {int(self.group[i]) for i in rows}
            if len(cells) != 1 or len(groups) != 1:
                raise ValueError(f"mass_unit {u!r} spans multiple (d,y) cells or groups: {cells} {groups}")
            mass = float(self.b[rows].sum())
            if abs(mass - 1.0) > 1e-6:
                raise ValueError(f"mass_unit {u!r} base-mass sum {mass} != 1")
            self.unit_rows[u] = rows
            self.unit_prob[u] = self.b[rows] / mass
            self.unit_cell[u] = next(iter(cells))
            self.unit_class[u] = int(self.y[rows[0]])

        # rows' cell mass must equal the FIXED support-graph cell_mass
        for (dd, yy) in self.cells:
            mss = float(self.b[(self.d == dd) & (self.y == yy)].sum())
            if abs(mss - float(support_graph.cell_mass[dd, yy])) > 1e-6:
                raise ValueError(f"cell ({dd},{yy}) row mass {mss} != support_graph.cell_mass "
                                 f"{float(support_graph.cell_mass[dd, yy])}")

        self.n_classes = int(support_graph.n_classes)
        self.cell_units = {c: [u for u in self.unit_rows if self.unit_cell[u] == c] for c in self.cells}
        self.U_cell = {c: len(self.cell_units[c]) for c in self.cells}
        self.class_units = {cl: [u for u in self.unit_rows if self.unit_class[u] == cl]
                            for cl in range(self.n_classes)}
        self.U_class = {cl: len(self.class_units[cl]) for cl in range(self.n_classes)}

        self.cell_queue = {c: _Queue(self.cell_units[c], np.random.default_rng([cfg.seed, 1, i]), cfg.replacement_mode)
                           for i, c in enumerate(self.cells)}
        self.class_queue = {cl: _Queue(self.class_units[cl], np.random.default_rng([cfg.seed, 2, cl]), cfg.replacement_mode)
                            for cl in range(self.n_classes) if self.U_class[cl] > 0}
        self._row_rng = np.random.default_rng([cfg.seed, 9])
        self._drawn = 0
        self._replaced = 0

    def _draw_row(self, u) -> int:
        rows = self.unit_rows[u]
        if rows.size == 1:
            return int(rows[0])
        return int(self._row_rng.choice(rows, p=self.unit_prob[u]))

    def adv_logical_batch(self) -> AdvLogicalBatch:
        k = self.cfg.min_per_eligible_cell
        idx, wt = [], []
        for c in self.cells:
            units, reps = self.cell_queue[c].draw(k)
            w = self.U_cell[c] / k                       # w^adv = U_cell/m  (unit-equal)
            for u in units:
                idx.append(self._draw_row(u)); wt.append(w)
            self._drawn += k; self._replaced += reps
        idx = np.array(idx, int); wt = np.array(wt, float)
        mb = self.cfg.adv_microbatch_size
        micro = [WeightedBatch(idx[s:s + mb], wt[s:s + mb]) for s in range(0, len(idx), mb)]
        if len(micro) > self.cfg.adv_accumulation_steps:
            raise ValueError(f"logical batch needs {len(micro)} microbatches > adv_accumulation_steps "
                             f"{self.cfg.adv_accumulation_steps}; raise capacity")
        return AdvLogicalBatch(microbatches=micro)

    def task_batch(self) -> WeightedBatch:
        present = [c for c in range(self.n_classes) if self.U_class[c] > 0]
        per = max(1, self.cfg.task_batch_size // len(present))
        idx, wt = [], []
        for c in present:
            units, reps = self.class_queue[c].draw(per)
            w = self.U_class[c] / per                    # w^task = U_class/m
            for u in units:
                idx.append(self._draw_row(u)); wt.append(w)
            self._drawn += per; self._replaced += reps
        return WeightedBatch(np.array(idx, int), np.array(wt, float))

    # ---- reporting ----
    @property
    def logical_adv_batch_size(self) -> int:
        return self.K_ov * self.cfg.min_per_eligible_cell

    @property
    def replacement_rate(self) -> float:
        return float(self._replaced / self._drawn) if self._drawn else 0.0

    def eligible_cell_coverage(self, batch) -> float:
        covered = {(int(self.d[i]), int(self.y[i])) for i in np.asarray(batch.idx)}
        return len(covered & set(self.cells)) / self.K_ov

    def unique_recording_fraction(self, batch) -> float:
        idx = np.asarray(batch.idx)
        return float(np.unique(self.group[idx]).size / idx.size) if idx.size else 0.0


if __name__ == "__main__":
    from .sampler_demo import _demo  # type: ignore
    _demo()
