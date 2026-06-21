"""Upper confidence bound on ``L_Q^ov`` via a recording-clustered bootstrap.

Discipline (the hard constraints):

* resample **whole recording groups** with replacement, **stratified within domain** (each
  domain keeps its group count) — the design ``p(D)`` is held fixed; we estimate the
  variability of the conditional from recording-level resampling;
* the ``SupportGraph``, ``S_y``, ``p_ref`` and the ``fold_of_group`` map are **fixed outside**
  the bootstrap; a replicate whose per-cell count drops below ``m`` does NOT redefine the
  estimand (the label space stays ``S_y``);
* duplicate copies of a resampled group inherit the original group's fold (guaranteed: the
  rebuilt rows keep the original group id and ``oof_nll_by_class`` looks fold up by it);
* **all capacities are retrained and the capacity sup is re-selected inside every replicate**
  (``estimate_extractable_leakage`` does the argmax) — selecting capacity once on the full
  sample would make the bound optimistic;
* negative estimates are kept (no clipping).

Main bound = one-sided **basic** cluster bootstrap upper limit
``bootstrap_ucl = 2·L̂_Q − Q_α(L̂*_Q)``; the percentile endpoint ``Q_{1-α}(L̂*_Q)`` is
reported separately as a sensitivity analysis (the two are never mixed). Output keys are
``extractable_LQ_ov`` / ``L_abs`` / ``L_cond`` / ``bootstrap_ucl`` — never "CMI" or
"true-I_ov upper bound".
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from ..support_graph import SupportGraph
from .critic import CriticConfig
from .crossfit import FoldPlan, FrozenFeatures
from .estimate import estimate_extractable_leakage


def _group_to_rows(feat: FrozenFeatures) -> dict[int, np.ndarray]:
    rows: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(feat.group):
        rows[int(g)].append(i)
    return {g: np.array(ix, dtype=int) for g, ix in rows.items()}


def within_domain_group_bootstrap(fold_plan: FoldPlan, rng) -> list[int]:
    """Resample recording groups with replacement, holding each domain's group count fixed.
    Returns a list of ORIGINAL group ids (with duplicates)."""
    by_dom: dict[int, list[int]] = defaultdict(list)
    for g, dom in fold_plan.domain_of_group.items():
        by_dom[dom].append(g)
    resampled: list[int] = []
    for dom in sorted(by_dom):
        gs = np.array(sorted(by_dom[dom]))
        resampled.extend(gs[rng.integers(0, gs.size, size=gs.size)].tolist())
    return resampled


def _rebuild(feat: FrozenFeatures, group_rows: dict[int, np.ndarray], resampled_groups: list[int]) -> FrozenFeatures:
    take = np.concatenate([group_rows[g] for g in resampled_groups])
    return FrozenFeatures(Z=feat.Z[take], y=feat.y[take], d=feat.d[take], group=feat.group[take])


def bootstrap_ucb(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    fold_plan: FoldPlan,
    cfg: CriticConfig,
    alpha: float = 0.1,
    n_bootstrap: int = 200,
    seed: int = 0,
) -> dict:
    """Point estimate + one-sided basic bootstrap UCL (and percentile sensitivity)."""
    point = estimate_extractable_leakage(feat, support_graph, fold_plan, cfg)
    Lhat = point["extractable_LQ_ov"]

    group_rows = _group_to_rows(feat)
    rng = np.random.default_rng(seed)
    reps = np.empty(n_bootstrap, dtype=np.float64)
    rep_caps: list[int] = []
    for b in range(n_bootstrap):
        resampled = within_domain_group_bootstrap(fold_plan, rng)
        feat_b = _rebuild(feat, group_rows, resampled)
        est_b = estimate_extractable_leakage(feat_b, support_graph, fold_plan, cfg)
        reps[b] = est_b["extractable_LQ_ov"]
        rep_caps.append(est_b["selected_capacity"])

    q_low = float(np.quantile(reps, alpha))         # α-quantile (lower tail)
    bootstrap_ucl = float(2.0 * Lhat - q_low)       # basic one-sided upper limit
    percentile_ucl = float(np.quantile(reps, 1.0 - alpha))

    return {
        "extractable_LQ_ov": float(Lhat),
        "L_abs": float(point["L_abs"]),
        "L_cond": float(point["L_cond"]),
        "selected_capacity": point["selected_capacity"],
        "bootstrap_ucl": bootstrap_ucl,             # MAIN (basic, one-sided)
        "percentile_ucl": percentile_ucl,           # sensitivity only (do not mix)
        "alpha": float(alpha),
        "n_bootstrap": int(n_bootstrap),
        "replicates": reps,
        "replicate_capacities": rep_caps,
        "reference_entropy": point["reference_entropy"],
        "fold_notes": fold_plan.notes,
    }
