"""Logical-batch containers + exact effective-prior utilities for the rare-cell sampler.

A logical **adversary** batch covers ALL eligible ``(d,y)`` cells; the importance weight
``w^adv = n_{d,y}/m_{d,y}`` restores the FIXED empirical ``p(d|y, d∈S_y)`` (NOT the sampler's
near-uniform per-cell draw). A **task** batch is class-stratified (and includes ineligible
cells); ``w^task = n_y/m_y`` restores the FIXED class prior ``p(y)``. Eligibility, ``S_y``,
``p_ref``, ``n_{d,y}`` are properties of the full-data support graph and are NEVER recomputed
from a batch.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..support_graph import SupportGraph


@dataclass
class WeightedBatch:
    idx: np.ndarray       # row indices into the full dataset
    weight: np.ndarray    # per-row importance/task weight

    def __len__(self) -> int:
        return int(len(self.idx))


@dataclass
class AdvLogicalBatch:
    """One adversary update's worth of rows, split into memory-sized microbatches. Summing
    ``domain_ce_contribution`` over the microbatches == the importance-weighted ``C_D``."""
    microbatches: list[WeightedBatch]

    @property
    def idx(self) -> np.ndarray:
        return np.concatenate([m.idx for m in self.microbatches]) if self.microbatches else np.array([], int)

    @property
    def weight(self) -> np.ndarray:
        return np.concatenate([m.weight for m in self.microbatches]) if self.microbatches else np.array([], float)


def effective_prior_domain_given_y(idx, weight, y_all, d_all, support_graph: SupportGraph) -> dict:
    """Per comparable class ``y``: the weighted domain mass over ``S_y`` normalised to a prior.
    With ``w^adv`` this equals the FIXED ``n_{d,y}/N_y^ov`` exactly (mass-based, classifier-free)."""
    idx = np.asarray(idx, int)
    w = np.asarray(weight, float)
    yb, db = np.asarray(y_all)[idx], np.asarray(d_all)[idx]
    out = {}
    for yy in support_graph.comparable_classes:
        S = support_graph.support_of_class[yy]
        mass = np.array([w[(yb == yy) & (db == dd)].sum() for dd in S], float)
        tot = mass.sum()
        out[yy] = (list(S), mass / tot if tot > 0 else mass)
    return out


def fixed_prior_domain_given_y(support_graph: SupportGraph) -> dict:
    """The FIXED empirical ``p(d|y, d∈S_y) = n_{d,y}/N_y^ov`` from the support-graph counts."""
    out = {}
    for yy in support_graph.comparable_classes:
        S = support_graph.support_of_class[yy]
        n = support_graph.counts[S, yy].astype(float)
        out[yy] = (list(S), n / n.sum())
    return out


def effective_prior_y(idx, weight, y_all, n_classes: int) -> np.ndarray:
    """Weighted class mass normalised to a prior (``w^task`` restores ``p(y)=n_y/N`` exactly)."""
    idx = np.asarray(idx, int)
    w = np.asarray(weight, float)
    yb = np.asarray(y_all)[idx]
    mass = np.array([w[yb == c].sum() for c in range(n_classes)], float)
    tot = mass.sum()
    return mass / tot if tot > 0 else mass


def weighted_ess(weight) -> float:
    """Kish effective sample size ``(Σw)^2 / Σw^2`` (=count when weights are uniform)."""
    w = np.asarray(weight, float)
    denom = float((w * w).sum())
    return float(w.sum() ** 2 / denom) if denom > 0 else 0.0
