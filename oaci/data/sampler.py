"""Paired-stream rare-cell sampler.

Two streams over the SAME fixed dataset (see ``batch.py`` for the invariants):

* **task stream** — class-stratified over ALL retained rows (including unsupported / ineligible
  cells); feeds Stage-1 ERM and the Stage-2 task-risk gradient. Weight ``w^task_i = n_y/m_y``
  restores the fixed class prior ``p(y)``.
* **adversary stream** — only eligible ``(d,y)`` (``d∈S_y``); every logical update covers ALL
  eligible cells (``k_min`` rows each). Weight ``w^adv_i = n_{d,y}/m_{d,y}`` restores the fixed
  empirical ``p(d|y, d∈S_y)`` — NOT the sampler's near-uniform per-cell draw.

Eligibility / ``S_y`` / ``p_ref`` / ``n_{d,y}`` come from the fixed full-data support graph and
are never recomputed from a batch. ``group`` is used only for reporting unique-recording
coverage; it does not change the row-level empirical target. Per-cell/per-class draws use a
seeded shuffled queue (no repeat until exhausted; rare cells use replacement only when forced).
"""
from __future__ import annotations

import numpy as np

from ..config import SamplerConfig
from ..support_graph import SupportGraph
from .batch import AdvLogicalBatch, WeightedBatch


class _Queue:
    """Seeded draw from a fixed row pool. 'auto': cycle a shuffled order (repeat only when the
    draw exceeds the pool); 'always': i.i.d. with replacement; 'never': without replacement
    (fails if a draw exceeds the pool)."""

    def __init__(self, rows, rng, mode: str):
        self.rows = np.asarray(rows, int)
        self.rng = rng
        self.mode = mode
        self._order = self.rows.copy()
        self._cursor = 0
        self._reshuffle()

    def _reshuffle(self):
        self._order = self.rows.copy()
        self.rng.shuffle(self._order)
        self._cursor = 0

    def draw(self, k: int):
        n = len(self.rows)
        if self.mode == "always":
            sel = self.rng.choice(self.rows, size=k, replace=True)
            return sel, int(k - np.unique(sel).size)
        if self.mode == "never":
            if k > n:
                raise ValueError(f"replacement_mode='never' but a cell/class has {n} rows < k={k}")
            if self._cursor + k > n:
                self._reshuffle()
            sel = self._order[self._cursor:self._cursor + k]
            self._cursor += k
            return sel, 0
        # auto
        sel = np.empty(k, dtype=int)
        seen: set[int] = set()
        reps = 0
        for j in range(k):
            if self._cursor >= n:
                self._reshuffle()
            r = int(self._order[self._cursor])
            self._cursor += 1
            if r in seen:
                reps += 1
            seen.add(r)
            sel[j] = r
        return sel, reps


class RareCellSampler:
    """Paired-stream sampler over fixed ``(y, d, group)`` and a fixed support graph."""

    def __init__(self, y, d, group, support_graph: SupportGraph, cfg: SamplerConfig):
        self.cfg = cfg.validate()
        self.sg = support_graph
        self.y = np.asarray(y, int)
        self.d = np.asarray(d, int)
        self.group = np.asarray(group, int)
        self.comparable = list(support_graph.comparable_classes)
        self.cells = [(int(dd), int(yy)) for yy in self.comparable for dd in support_graph.support_of_class[yy]]
        self.K_ov = len(self.cells)
        if self.K_ov == 0:
            raise ValueError("no eligible (d,y) cells -> adversary stream is empty (no comparable class)")
        cfg.assert_capacity(self.K_ov)
        # FIXED counts from the support graph (NOT recomputed from the row arrays).
        self.n_cell = {c: int(support_graph.counts[c[0], c[1]]) for c in self.cells}
        self.n_classes = int(support_graph.n_classes)
        self.n_class = {c: int((self.y == c).sum()) for c in range(self.n_classes)}

        self.cell_queue = {
            c: _Queue(np.where((self.d == c[0]) & (self.y == c[1]))[0],
                      np.random.default_rng([cfg.seed, 1, i]), cfg.replacement_mode)
            for i, c in enumerate(self.cells)
        }
        self.class_queue = {
            c: _Queue(np.where(self.y == c)[0], np.random.default_rng([cfg.seed, 2, c]), cfg.replacement_mode)
            for c in range(self.n_classes) if self.n_class[c] > 0
        }
        self._drawn = 0
        self._replaced = 0

    # ---- streams ----
    def adv_logical_batch(self) -> AdvLogicalBatch:
        """All eligible cells, ``k_min`` rows each, weighted to restore ``n_{d,y}/N_y^ov``."""
        k = self.cfg.min_per_eligible_cell
        idx, wt = [], []
        for c in self.cells:
            rows, reps = self.cell_queue[c].draw(k)
            idx.append(rows)
            wt.append(np.full(k, self.n_cell[c] / k, dtype=float))   # w^adv = n_{d,y}/m
            self._drawn += k
            self._replaced += reps
        idx = np.concatenate(idx)
        wt = np.concatenate(wt)
        mb = self.cfg.adv_microbatch_size
        micro = [WeightedBatch(idx[s:s + mb], wt[s:s + mb]) for s in range(0, len(idx), mb)]
        if len(micro) > self.cfg.adv_accumulation_steps:   # guarded by assert_capacity, belt-and-braces
            raise ValueError(f"logical batch needs {len(micro)} microbatches > adv_accumulation_steps "
                             f"{self.cfg.adv_accumulation_steps}; raise capacity")
        return AdvLogicalBatch(microbatches=micro)

    def task_batch(self) -> WeightedBatch:
        """Class-stratified over ALL rows (incl. ineligible cells); ``w^task = n_y/m_y``."""
        per = max(1, self.cfg.task_batch_size // len([c for c in range(self.n_classes) if self.n_class[c] > 0]))
        idx, wt = [], []
        for c in range(self.n_classes):
            if self.n_class[c] == 0:
                continue
            rows, reps = self.class_queue[c].draw(per)
            idx.append(rows)
            wt.append(np.full(per, self.n_class[c] / per, dtype=float))
            self._drawn += per
            self._replaced += reps
        return WeightedBatch(np.concatenate(idx), np.concatenate(wt))

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
