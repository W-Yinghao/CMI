"""Synthetic frozen-representation generators for the leakage estimator's tests + demo.

Each returns ``(FrozenFeatures, SupportGraph)`` with a recording-grouped layout (multiple
recordings per domain, every recording one domain, all classes present). Scenarios:

* ``make_null``               — ``Z ⟂ D | Y``: leakage should be ~0.
* ``make_perfect``            — ``D`` linearly encoded within class: leakage ≈ Σ p_ref·Ĥ_y.
* ``make_group_memorization`` — ``Z`` encodes recording identity only (no shared domain
  signal): a grouped probe must read ~0 (the identity does not generalise to held-out
  recordings), whereas a sample-level split would falsely flag leakage.
* ``make_nonlinear``          — ``D`` separable only nonlinearly (XOR): capacity 0 fails,
  capacity>0 succeeds (drives the after-aggregation capacity sup).
"""
from __future__ import annotations

import numpy as np

from ..support_graph import build_support_graph, counts_from_labels, empirical_class_prior
from .crossfit import FrozenFeatures


def _layout(n_domains: int, n_classes: int, recs_per_domain: int, per_cell: int):
    y, d, g = [], [], []
    gid = 0
    for dom in range(n_domains):
        for _ in range(recs_per_domain):
            for c in range(n_classes):
                y += [c] * per_cell
                d += [dom] * per_cell
                g += [gid] * per_cell
            gid += 1
    return np.array(y), np.array(d), np.array(g)


def _support_graph(y, d, m, n_domains, n_classes):
    counts = counts_from_labels(d, y, n_domains=n_domains, n_classes=n_classes)
    return build_support_graph(counts, m=m, reference_prior=empirical_class_prior(counts))


def make_null(seed=0, n_domains=2, n_classes=2, recs_per_domain=4, per_cell=25, dim=8, m=20):
    rng = np.random.default_rng(seed)
    y, d, g = _layout(n_domains, n_classes, recs_per_domain, per_cell)
    class_mean = rng.standard_normal((n_classes, dim)) * 2.0
    Z = class_mean[y] + 0.5 * rng.standard_normal((y.size, dim))  # depends on Y only
    return FrozenFeatures(Z, y, d, g), _support_graph(y, d, m, n_domains, n_classes)


def make_perfect(seed=0, n_domains=2, n_classes=2, recs_per_domain=4, per_cell=25, dim=8, m=20):
    rng = np.random.default_rng(seed)
    y, d, g = _layout(n_domains, n_classes, recs_per_domain, per_cell)
    class_mean = rng.standard_normal((n_classes, dim)) * 2.0
    domain_dir = rng.standard_normal((n_domains, dim))
    domain_dir /= np.linalg.norm(domain_dir, axis=1, keepdims=True)
    Z = class_mean[y] + 4.0 * domain_dir[d] + 0.2 * rng.standard_normal((y.size, dim))  # D linearly encoded
    return FrozenFeatures(Z, y, d, g), _support_graph(y, d, m, n_domains, n_classes)


def make_group_memorization(seed=0, n_domains=2, n_classes=2, recs_per_domain=4, per_cell=25, dim=16, m=20):
    rng = np.random.default_rng(seed)
    y, d, g = _layout(n_domains, n_classes, recs_per_domain, per_cell)
    n_groups = int(g.max()) + 1
    rec_embed = rng.standard_normal((n_groups, dim)) * 3.0   # identity, NOT a shared domain signal
    class_mean = rng.standard_normal((n_classes, dim)) * 1.0
    Z = rec_embed[g] + class_mean[y] + 0.1 * rng.standard_normal((y.size, dim))
    return FrozenFeatures(Z, y, d, g), _support_graph(y, d, m, n_domains, n_classes)


def make_nonlinear(seed=0, n_classes=2, recs_per_domain=4, per_cell=40, m=20):
    """XOR domain encoding: D separable only with a nonlinear probe."""
    rng = np.random.default_rng(seed)
    n_domains = 2
    y, d, g = _layout(n_domains, n_classes, recs_per_domain, per_cell)
    a = 2.0
    centers = {
        0: np.array([[a, a], [-a, -a]]),     # domain 0 sub-clusters
        1: np.array([[a, -a], [-a, a]]),     # domain 1 sub-clusters (XOR)
    }
    xor = np.zeros((y.size, 2))
    for i in range(y.size):
        ctr = centers[int(d[i])][rng.integers(2)]
        xor[i] = ctr + 0.4 * rng.standard_normal(2)
    class_mean = rng.standard_normal((n_classes, 2)) * 2.0
    Z = np.concatenate([xor, class_mean[y] + 0.3 * rng.standard_normal((y.size, 2))], axis=1)
    return FrozenFeatures(Z, y, d, g), _support_graph(y, d, m, n_domains, n_classes)


# --------------------------------------------------------------------------------------
def _demo() -> None:
    from .critic import CriticConfig
    from .crossfit import make_fold_plan
    from .ucb import bootstrap_ucb

    cfg = CriticConfig(capacities=(0, 64))
    for name, maker in [("NULL  (Z ⟂ D | Y)", make_null), ("PERFECT (D encoded in Z)", make_perfect)]:
        feat, sg = maker(seed=0)
        plan = make_fold_plan(feat, sg, n_folds=4, seed=0)
        res = bootstrap_ucb(feat, sg, plan, cfg, alpha=0.1, n_bootstrap=200, seed=0)
        Hbar = sum(sg.reference_prior[y] * res["reference_entropy"][y] for y in sg.comparable_classes)
        print(f"\n{name}")
        print(f"  folds={plan.n_folds}  selected_capacity={res['selected_capacity']}"
              f"  Σ p_ref·Ĥ_y (max possible L_abs)={Hbar:.3f} nats")
        print(f"  extractable_LQ_ov (L_abs) = {res['L_abs']:+.4f}   L_cond = {res['L_cond']:+.4f}")
        print(f"  bootstrap_ucl (basic, 90%) = {res['bootstrap_ucl']:+.4f}"
              f"   percentile_ucl = {res['percentile_ucl']:+.4f}")


if __name__ == "__main__":
    _demo()
