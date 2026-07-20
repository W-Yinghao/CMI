"""Eval-unit aggregation + the eligibility/mass split.

Clinical & SEED data give MANY correlated windows per subject/clip; they must NOT count as
independent support. Each row gets base mass ``b_i = 1/|windows in its eval unit|`` so every
subject/trial/clip totals mass 1. Then:

* ``n^elig_{d,y} = #unique support units`` in cell (d,y)  — the ``m``-gate ONLY;
* ``M_{d,y} = Σ_{i∈cell} b_i``                             — the estimand mass (p(d|y), H_ref, weights).

Model outputs are aggregated to the eval unit by MEAN PROBABILITY,
``p̄_u = mean_w softmax(ℓ_w)``, ``ℓ^agg_u = log clip(p̄_u)`` — all formal accuracy/NLL/ECE and
clustered inference run on aggregated units; window-level numbers are diagnostic only.
"""
from __future__ import annotations

import numpy as np


def base_mass(eval_unit_id) -> np.ndarray:
    """``b_i = 1/|windows in eval_unit(i)|`` so each eval unit totals mass 1."""
    u = np.asarray(eval_unit_id, dtype=object)
    counts = {}
    for v in u:
        counts[v] = counts.get(v, 0) + 1
    return np.array([1.0 / counts[v] for v in u], dtype=float)


def eligibility_counts(domain, y, support_unit_id, n_domains, n_classes) -> np.ndarray:
    """``n^elig_{d,y}`` = number of UNIQUE support units in each cell (windows do not inflate it)."""
    domain, y, sid = np.asarray(domain, int), np.asarray(y, int), np.asarray(support_unit_id, dtype=object)
    out = np.zeros((n_domains, n_classes), dtype=np.int64)
    for d in range(n_domains):
        for c in range(n_classes):
            m = (domain == d) & (y == c)
            out[d, c] = np.unique(sid[m]).size if m.any() else 0
    return out


def cell_mass(domain, y, base, n_domains, n_classes) -> np.ndarray:
    """``M_{d,y} = Σ_{i∈(d,y)} b_i`` (≈ number of eval units in the cell)."""
    domain, y, base = np.asarray(domain, int), np.asarray(y, int), np.asarray(base, float)
    out = np.zeros((n_domains, n_classes), dtype=np.float64)
    np.add.at(out, (domain, y), base)
    return out


def aggregate_mean_prob(logits, eval_unit_id, prob_floor=1e-6, sample_mass=None):
    """Aggregate window logits to eval units by MASS-weighted mean softmax. Returns
    (unit_ids, agg_logits[U,C], unit_y_index_into_rows) — the first row index per unit (labels are
    constant within a unit, so any row's y is the unit label). With ``sample_mass=None`` this is the
    arithmetic mean; with the natural ``b_i = 1/|u|`` it is identical, and duplicating a window while
    splitting its mass leaves the aggregate unchanged."""
    logits = np.asarray(logits, dtype=np.float64)
    u = np.asarray(eval_unit_id, dtype=object)
    b = (np.ones(logits.shape[0]) if sample_mass is None
         else np.asarray(sample_mass, dtype=np.float64).ravel())
    if b.shape[0] != logits.shape[0]:
        raise ValueError("sample_mass length != logits")
    z = logits - logits.max(axis=1, keepdims=True)
    p = np.exp(z); p /= p.sum(axis=1, keepdims=True)
    units = sorted(set(u.tolist()), key=str)
    agg = np.zeros((len(units), logits.shape[1]))
    rep = np.zeros(len(units), dtype=int)
    for i, uu in enumerate(units):
        idx = np.where(u == uu)[0]
        w = b[idx]
        pbar = (w[:, None] * p[idx]).sum(axis=0) / w.sum()    # MASS-weighted unit posterior
        pbar = np.clip(pbar, prob_floor, None)
        pbar = pbar / pbar.sum()                 # RENORMALISE after the floor (proper distribution)
        agg[i] = np.log(pbar)
        rep[i] = idx[0]
    return np.array(units, dtype=object), agg, rep
