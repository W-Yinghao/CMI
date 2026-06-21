"""Label Fisher F_Y and conditional-domain Fisher F_{D|Y}.

These are the two quadratic forms whose generalized spectrum defines the
domain-rich / label-light directions (see `subspace.py`). Both are *between-group
scatter* matrices: how far the group means sit from the (conditional) grand mean,
weighted by the group probability.

    F_Y     = sum_y  p(y)  (mu_y - mu)(mu_y - mu)^T                       [d, d]
    F_{D|Y} = sum_y  p(y)  sum_d p(d|y) (mu_{d,y} - mu_y)(mu_{d,y} - mu_y)^T

`F_{D|Y}` is the class-CONDITIONAL between-domain scatter: it only sees how the
*within-class* domain means spread, so a direction with high `v^T F_{D|Y} v` carries
domain information that survives after conditioning on the label -- exactly the
leakage `I(Z;D|Y)` is meant to remove.

Estimation is empirical and ridge-free here; the ridge lives on F_Y inside the
generalized eig. Cells (a class, or a (domain,class) pair) with fewer than
`min_per_cell` samples are dropped so a class observed in a single domain contributes
zero conditional-domain scatter instead of noise. Everything is torch and runs on the
same device as Z; we accumulate in float64 when `dtype64` for eig stability.
"""
from __future__ import annotations
import torch

from .config import FisherConfig


def _as2d(Z: torch.Tensor) -> torch.Tensor:
    if Z.dim() != 2:
        raise ValueError(f"expected Z of shape [N, d], got {tuple(Z.shape)}")
    return Z


def _group_means(Z, g, n_groups, min_per_cell):
    """Return (means [G,d], counts [G]) of Z grouped by integer label g in [0,n_groups).
    Groups with count < min_per_cell get a NaN mean row and count 0 (caller skips them)."""
    d = Z.shape[1]
    means = Z.new_zeros((n_groups, d))
    counts = torch.zeros(n_groups, dtype=torch.long, device=Z.device)
    for gi in range(n_groups):
        mask = g == gi
        c = int(mask.sum())
        counts[gi] = c
        if c >= min_per_cell:
            means[gi] = Z[mask].mean(0)
        else:
            means[gi] = float("nan")
    return means, counts


def _scatter(means, weights, center):
    """sum_g w_g (mu_g - center)(mu_g - center)^T over rows with finite mean & w>0."""
    d = means.shape[1]
    S = means.new_zeros((d, d))
    for mu, w in zip(means, weights):
        if w <= 0 or not torch.isfinite(mu).all():
            continue
        diff = (mu - center).unsqueeze(1)        # [d,1]
        S = S + w * (diff @ diff.t())
    return S


def label_fisher(Z, y, n_cls, cfg: FisherConfig = FisherConfig()) -> torch.Tensor:
    """Between-class scatter F_Y = sum_y p(y) (mu_y - mu)(mu_y - mu)^T  [d, d]."""
    Z = _as2d(Z)
    if cfg.dtype64:
        Z = Z.double()
    mu = Z.mean(0) if cfg.center_global else torch.zeros_like(Z[0])
    means, counts = _group_means(Z, y, n_cls, cfg.min_per_cell)
    p = counts.to(Z.dtype) / counts.sum().clamp(min=1)
    return _scatter(means, p, mu)


def conditional_domain_fisher(Z, y, d, n_cls, n_dom,
                              cfg: FisherConfig = FisherConfig()) -> torch.Tensor:
    """Class-conditional between-domain scatter
        F_{D|Y} = sum_y p(y) [ sum_d p(d|y) (mu_{d,y} - mu_y)(mu_{d,y} - mu_y)^T ]   [d, d].
    Domains absent / too thin within a class contribute nothing; classes seen in a single
    domain contribute zero (no within-class domain spread to measure)."""
    Z = _as2d(Z)
    if cfg.dtype64:
        Z = Z.double()
    dim = Z.shape[1]
    F = Z.new_zeros((dim, dim))
    n = Z.shape[0]
    for yi in range(n_cls):
        cls_mask = y == yi
        c_y = int(cls_mask.sum())
        if c_y < cfg.min_per_cell:
            continue
        p_y = c_y / n
        Zy, dy = Z[cls_mask], d[cls_mask]
        mu_y = Zy.mean(0)                                  # within-class grand mean
        dmeans, dcounts = _group_means(Zy, dy, n_dom, cfg.min_per_cell)
        # p(d|y) over the domains that actually pass the min-cell filter
        valid = dcounts >= cfg.min_per_cell
        denom = dcounts[valid].sum().clamp(min=1)
        p_d_given_y = torch.zeros(n_dom, dtype=Z.dtype, device=Z.device)
        p_d_given_y[valid] = dcounts[valid].to(Z.dtype) / denom
        if int(valid.sum()) < 2:
            continue                                       # need >=2 domains to have spread
        F = F + p_y * _scatter(dmeans, p_d_given_y, mu_y)
    return F


def fisher_pair(Z, y, d, n_cls, n_dom, cfg: FisherConfig = FisherConfig()):
    """Convenience: (F_{D|Y}, F_Y) in one pass-pair, both symmetrized [d,d] (float64 if cfg)."""
    F_DgY = conditional_domain_fisher(Z, y, d, n_cls, n_dom, cfg)
    F_Y = label_fisher(Z, y, n_cls, cfg)
    sym = lambda A: 0.5 * (A + A.t())
    return sym(F_DgY), sym(F_Y)


def _shuffle_d_within_y(y, d, generator):
    """Permute domain labels independently within each class -- breaks D|Y dependence while
    preserving p(d|y) marginals. Returns a shuffled copy of d (torch long)."""
    d_perm = d.clone()
    for yi in torch.unique(y):
        idx = (y == yi).nonzero(as_tuple=True)[0]
        d_perm[idx] = d[idx][torch.randperm(idx.numel(), generator=generator)]
    return d_perm


def null_domain_energy_floor(Z, y, d, n_cls, n_dom,
                             cfg: FisherConfig = FisherConfig(),
                             n_perm: int = 5, seed: int = 0) -> float:
    """Largest eigenvalue of F_{D|Y} when D is shuffled within each class, over `n_perm`
    permutations. Under the null D ⊥ Z | Y this is pure sampling noise -- the floor a real
    domain-rich direction must clear. Returns 0.0 if n_perm<=0."""
    if n_perm <= 0:
        return 0.0
    g = torch.Generator(device="cpu").manual_seed(int(seed))
    floor = 0.0
    yc, dc = y.cpu(), d.cpu()
    for _ in range(n_perm):
        d_perm = _shuffle_d_within_y(yc, dc, g).to(Z.device)
        Fn = conditional_domain_fisher(Z, y, d_perm, n_cls, n_dom, cfg)
        Fn = 0.5 * (Fn + Fn.t())
        lam = float(torch.linalg.eigvalsh(Fn).max())
        floor = max(floor, lam)
    return floor
