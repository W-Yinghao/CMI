"""Reference-prior marginal alignment + corrected GLS reference distribution.

Review section 4 / P0-4: forcing raw marginals p_d(z) equal is wrong under label shift,
because p_d(z) = sum_y p_d(y) p_d(z|y) differs across domains whenever the class priors
p_d(y) differ, EVEN IF every class-conditional p_d(z|y) is shared.  The fix is to align
the marginal under a FIXED reference prior:

    p_d^*(z) = sum_y pi^*(y) p_d(z|y)            (align this, not raw p_d(z))

A sufficient and cleaner realisation -- consistent with the shared class-conditional
density head -- is to align each domain's class-conditional p_d(z|y) to the pooled
class-conditional p(z|y), weighted by pi^*(y).  This never forces the raw marginals equal
and so does not distort class structure under label shift.

Review P0-5: the GLS-reweighted reference DOMAIN marginal must be computed as

    p_ref(d) = normalise( sum_i [ y_i has weight pi^*(y_i)/p(y_i|d_i) ] )

i.e. a bincount of d weighted by the GLS importance weights -- NOT ``mean_y p(d|y)``,
which is a different probability measure in general.
"""
from __future__ import annotations

import numpy as np
import torch

from h2cmi.config import AlignConfig
from h2cmi.align.distances import sliced_wasserstein, energy_distance, gauss_w2


# --------------------------------------------------------------------------- GLS
def class_given_domain(y: np.ndarray, d: np.ndarray, n_dom: int, n_cls: int,
                       alpha: float = 1.0) -> np.ndarray:
    """p(y|d) Laplace-smoothed -> [n_dom, n_cls]."""
    counts = np.zeros((n_dom, n_cls), dtype=np.float64)
    np.add.at(counts, (d.astype(np.int64), y.astype(np.int64)), 1.0)
    return (counts + alpha) / (counts.sum(1, keepdims=True) + alpha * n_cls)


def gls_weights(y: np.ndarray, d: np.ndarray, n_dom: int, n_cls: int,
                pi_star: np.ndarray | None = None) -> np.ndarray:
    """Per-sample GLS importance weight w_i = pi^*(y_i) / p(y_i | d_i).  Mean-normalised."""
    if pi_star is None:
        pi_star = np.full(n_cls, 1.0 / n_cls)
    p_y_d = class_given_domain(y, d, n_dom, n_cls)
    w = pi_star[y.astype(np.int64)] / p_y_d[d.astype(np.int64), y.astype(np.int64)]
    return (w / w.mean()).astype(np.float32)


def gls_reference_domain_marginal(y: np.ndarray, d: np.ndarray, n_dom: int, n_cls: int,
                                  pi_star: np.ndarray | None = None) -> np.ndarray:
    """CORRECTED (P0-5) GLS reference domain marginal p_ref(d).

    p_ref(d) = normalise( bincount(d, weights = pi^*(y)/p(y|d)) ).
    This is the measure the GLS-reweighted dataset actually induces; under fully
    within-domain GLS it equals the raw p(d).  Contrast the buggy ``mean_y p(d|y)``.
    """
    w = gls_weights(y, d, n_dom, n_cls, pi_star)
    ref = np.bincount(d.astype(np.int64), weights=w, minlength=n_dom).astype(np.float64)
    s = ref.sum()
    return ref / s if s > 0 else np.full(n_dom, 1.0 / n_dom)


# ---------------------------------------------------- reference-prior marginal align
_DIST = {
    "sliced_wasserstein": lambda a, b, cfg: sliced_wasserstein(a, b, cfg.n_projections),
    "energy": lambda a, b, cfg: energy_distance(a, b),
    "gauss_w2": lambda a, b, cfg: gauss_w2(a, b),
}


class ReferenceMarginalAlignment:
    """Align class-conditional latents across domains under a reference prior pi^*.

    Callable: ``align(z_c, y, d, n_cls) -> (loss, info)``.  For every (domain, class)
    cell with enough samples, measure the distance to the POOLED class-conditional (over
    all domains in the batch) and average weighted by pi^*(y).  Minimising this aligns
    p_d^*(z) = sum_y pi^*(y) p_d(z|y) to a common p^*(z) WITHOUT collapsing class
    structure under label shift.
    """

    def __init__(self, cfg: AlignConfig, n_classes: int, pi_star: np.ndarray | None = None,
                 min_per_cell: int = 4):
        self.cfg = cfg
        self.n_classes = n_classes
        if pi_star is None:
            pi_star = np.full(n_classes, 1.0 / n_classes)
        self.pi_star = np.asarray(pi_star, dtype=np.float64)
        self.min_per_cell = min_per_cell
        self.dist = _DIST.get(cfg.distance, _DIST["sliced_wasserstein"])

    def __call__(self, z_c: torch.Tensor, y: torch.Tensor, d: torch.Tensor, n_cls: int | None = None):
        n_cls = n_cls or self.n_classes
        y_np = y.detach().cpu().numpy()
        d_np = d.detach().cpu().numpy()
        doms = np.unique(d_np)
        if len(doms) < 2:
            zero = z_c.sum() * 0.0
            return zero, dict(align=0.0, n_cells=0)
        total = z_c.sum() * 0.0
        n_cells = 0
        wsum = 0.0
        for c in range(n_cls):
            cls_mask = y_np == c
            if cls_mask.sum() < self.min_per_cell:
                continue
            pooled = z_c[torch.as_tensor(cls_mask, device=z_c.device)]   # p(z|y=c) pooled over domains
            wc = float(self.pi_star[c])
            for dom in doms:
                cell = (y_np == c) & (d_np == dom)
                if cell.sum() < self.min_per_cell:
                    continue
                zc_cell = z_c[torch.as_tensor(cell, device=z_c.device)]
                total = total + wc * self.dist(zc_cell, pooled, self.cfg)
                wsum += wc
                n_cells += 1
        if wsum > 0:
            total = total / wsum
        return total, dict(align=float(total.detach()), n_cells=n_cells)


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    n = 400
    y = rng.integers(0, 3, n)
    d = rng.integers(0, 4, n)
    # corrected GLS reference vs the buggy mean_y p(d|y)
    n_dom, n_cls = 4, 3
    ref_correct = gls_reference_domain_marginal(y, d, n_dom, n_cls)
    pi_y = class_given_domain(d, y, n_cls, n_dom)   # note: shape [n_cls?]; just demo p(d|y)
    print("GLS reference p_ref(d):", np.round(ref_correct, 3), "sum", round(ref_correct.sum(), 4))
    # alignment on synthetic latents with a per-domain shift
    z = torch.randn(n, 8)
    for dom in range(n_dom):
        z[torch.as_tensor(d == dom)] += 0.5 * dom
    z.requires_grad_(True)
    align = ReferenceMarginalAlignment(AlignConfig(distance="sliced_wasserstein"), n_cls)
    loss, info = align(z, torch.as_tensor(y), torch.as_tensor(d), n_cls)
    loss.backward()
    print("align loss:", round(float(loss.detach()), 4), info, "grad ok:", bool(torch.isfinite(z.grad).all()))
