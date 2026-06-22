"""Point estimate of the probe-class-extractable overlap leakage ``L_Q^ov``.

For each capacity ``c`` and comparable class ``y``:

    ``L_{y,c} = Ĥ_y(D) - NLL^OOF_{y,c}``                         (nats)
    ``L_abs,c = Σ_{y∈C_cmp} p_ref(y) · L_{y,c}``                 (primary, fixed p_ref)
    ``L_cond,c = Σ_{y∈C_cmp} p_ref(y|y∈C_cmp) · L_{y,c}``        (diagnostic)

Capacity selection happens **after** the class-weighted aggregation:

    ``extractable_LQ_ov = L̂_Q^abs = max_c L_abs,c``

NOT a per-class capacity argmax — that would correspond to a different, larger probe family.
``Ĥ_y(D)`` is the reference conditional entropy of ``D`` on ``S_y`` under the **fixed**
support graph (held constant across the bootstrap; a resample never redefines it). Negative
``L`` values are kept (no ``max(0,·)``).
"""
from __future__ import annotations

import numpy as np

from ..support_graph import SupportGraph
from .critic import CriticConfig
from .crossfit import FoldPlan, FrozenFeatures, oof_nll_by_class


def reference_conditional_entropy(support_graph: SupportGraph) -> dict[int, float]:
    """``Ĥ_y(D)`` in **nats** for each comparable class, over ``S_y`` only, from the FIXED
    support-graph counts (``p(d|y, d∈S_y) = n_{d,y} / Σ_{d∈S_y} n_{d,y}``)."""
    H: dict[int, float] = {}
    for y in support_graph.comparable_classes:
        S = support_graph.support_of_class[y]
        n = support_graph.cell_mass[S, y].astype(np.float64)   # estimand MASS, not unit counts
        p = n / n.sum()
        p = p[p > 0]
        H[y] = float(-np.sum(p * np.log(p)))
    return H


def _conditional_weights(support_graph: SupportGraph) -> dict[int, float]:
    z = float(sum(support_graph.reference_prior[y] for y in support_graph.comparable_classes))
    return {y: (float(support_graph.reference_prior[y] / z) if z > 0 else 0.0)
            for y in support_graph.comparable_classes}


def estimate_extractable_leakage(
    feat: FrozenFeatures,
    support_graph: SupportGraph,
    fold_plan: FoldPlan,
    cfg: CriticConfig,
) -> dict:
    """Compute ``extractable_LQ_ov`` (= ``L_abs`` at the selected capacity) and diagnostics."""
    H = reference_conditional_entropy(support_graph)
    p_ref = support_graph.reference_prior
    w_cond = _conditional_weights(support_graph)
    comparable = support_graph.comparable_classes

    L_abs_by_c: dict[int, float] = {}
    L_cond_by_c: dict[int, float] = {}
    L_y_by_c: dict[int, dict[int, float]] = {}
    nll_by_c: dict[int, dict[int, float]] = {}

    for c in cfg.capacities:
        nll = oof_nll_by_class(feat, support_graph, fold_plan, c, cfg)
        L_y = {y: (H[y] - nll[y]["nll"]) for y in comparable}
        L_y_by_c[c] = L_y
        nll_by_c[c] = {y: nll[y]["nll"] for y in comparable}
        # NaN guard: a class with no OOF rows cannot contribute (should not happen post-feasibility)
        L_abs_by_c[c] = float(np.nansum([p_ref[y] * L_y[y] for y in comparable]))
        L_cond_by_c[c] = float(np.nansum([w_cond[y] * L_y[y] for y in comparable]))

    sel = max(L_abs_by_c, key=L_abs_by_c.get)  # capacity sup AFTER aggregation
    return {
        "extractable_LQ_ov": L_abs_by_c[sel],
        "L_abs": L_abs_by_c[sel],
        "L_cond": L_cond_by_c[sel],
        "selected_capacity": sel,
        "L_abs_by_capacity": L_abs_by_c,
        "L_cond_by_capacity": L_cond_by_c,
        "L_y": L_y_by_c[sel],
        "nll_by_class": nll_by_c[sel],
        "reference_entropy": H,
    }
