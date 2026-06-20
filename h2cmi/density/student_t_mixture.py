"""Class-conditional latent density head p_phi(z_c | y) as a Student-t mixture, the
Bayes generative classifier it induces, and the discriminative/density/consistency
hybrid head (review section 5.3).

Design choices (from the review):
  * Student-t components (heavy tails -> robust to artifacts/outliers, vs Gaussian).
  * low-rank + diagonal scale  Sigma = L L^T + diag(softplus(s) + floor)  (small per-cell
    sample counts) with an eigenvalue/variance floor.
  * K components per class (default 1-2 to start).
  * a generative classifier p_phi(y|z; pi*) by Bayes under a REFERENCE prior pi*; the
    target prior is estimated separately at test time (TTA).
  * a parallel discriminative head h_omega(y|z) and a JS consistency term, because a pure
    density model is misspecification-sensitive and a pure discriminative model has no
    explicit p(z|y) to drive class-conditional TTA.

The Student-t log-density uses Woodbury / matrix-determinant-lemma so the per-component
inverse and log-det cost O(d r^2 + r^3), never a dense d x d solve.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from h2cmi.config import DensityConfig


def _student_t_logpdf(u, L, log_d_floor, df):
    """Multivariate Student-t log density of residuals u under scale Sigma = L L^T + D.

    u           [..., d]   residual (z - mu)
    L           [d, r]     low-rank factor
    log_d_floor [d]        log of the diagonal (already floored, positive)
    df          scalar     degrees of freedom
    returns     [...]      log pdf
    """
    d = u.shape[-1]
    r = L.shape[-1]
    Dinv = torch.exp(-log_d_floor)                                  # [d]
    # M = I_r + L^T D^{-1} L
    LtDinv = L.transpose(-1, -2) * Dinv                            # [r,d]
    M = torch.eye(r, device=u.device, dtype=u.dtype) + LtDinv @ L  # [r,r]
    Lchol = torch.linalg.cholesky(M)
    # logdet Sigma = sum log d + logdet M
    logdet = log_d_floor.sum() + 2.0 * torch.log(torch.diagonal(Lchol)).sum()
    # Mahalanobis: u^T Dinv u - (L^T Dinv u)^T M^{-1} (L^T Dinv u)
    uDinv = u * Dinv                                               # [...,d]
    quad0 = (u * uDinv).sum(-1)                                    # [...]
    w = torch.einsum("rd,...d->...r", LtDinv, u)                   # [...,r]
    sol = torch.cholesky_solve(w.unsqueeze(-1), Lchol).squeeze(-1) # [...,r]
    quad = quad0 - (w * sol).sum(-1)
    quad = quad.clamp_min(0.0)
    half = 0.5 * (df + d)
    return (math.lgamma(0.5 * (df + d)) - math.lgamma(0.5 * df)
            - 0.5 * d * math.log(df * math.pi) - 0.5 * logdet
            - half * torch.log1p(quad / df))


class ClassConditionalDensity(nn.Module):
    """p_phi(z | y) Student-t mixture, one mixture per class."""

    def __init__(self, dim: int, n_classes: int, cfg: DensityConfig):
        super().__init__()
        self.dim, self.n_classes, self.cfg = dim, n_classes, cfg
        K = cfg.n_components
        self.K = K
        self.df = float(cfg.df)
        self.eig_floor = float(cfg.eig_floor)
        scale = cfg.init_scale
        self.mu = nn.Parameter(scale * torch.randn(n_classes, K, dim))
        self.L = nn.Parameter(0.1 * torch.randn(n_classes, K, dim, cfg.cov_rank))
        self.log_s = nn.Parameter(torch.zeros(n_classes, K, dim))      # raw diag (pre-softplus)
        self.mix_logits = nn.Parameter(torch.zeros(n_classes, K))

    def _comp_logpdf(self, z, c, k):
        """log N_t(z | mu_{c,k}, Sigma_{c,k}) for all z [B,d]."""
        u = z - self.mu[c, k]                                          # [B,d]
        diag = F.softplus(self.log_s[c, k]) + self.eig_floor          # [d] > 0
        return _student_t_logpdf(u, self.L[c, k], torch.log(diag), self.df)

    def log_prob_all(self, z: torch.Tensor) -> torch.Tensor:
        """log p(z | y=c) for every class -> [B, n_classes]."""
        cols = []
        for c in range(self.n_classes):
            logw = F.log_softmax(self.mix_logits[c], dim=0)           # [K]
            comp = torch.stack([self._comp_logpdf(z, c, k) for k in range(self.K)], dim=1)  # [B,K]
            cols.append(torch.logsumexp(comp + logw.unsqueeze(0), dim=1))                  # [B]
        return torch.stack(cols, dim=1)                               # [B,n_classes]

    def log_prob(self, z: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return self.log_prob_all(z).gather(1, y.view(-1, 1)).squeeze(1)

    def class_posterior(self, z: torch.Tensor, log_prior: torch.Tensor) -> torch.Tensor:
        """Bayes generative posterior p(y|z; prior) -> [B, n_classes] (probabilities).

        ``log_prior`` is [n_classes] log pi (reference or estimated target prior).
        """
        joint = self.log_prob_all(z) + log_prior.view(1, -1)
        return F.softmax(joint, dim=1)

    @torch.no_grad()
    def ema_update(self, z: torch.Tensor, y: torch.Tensor, momentum: float):
        """Optional EMA of the class means toward batch class means (review: EMA prototypes)."""
        for c in range(self.n_classes):
            m = y == c
            if m.sum() >= 2:
                self.mu[c, 0].mul_(momentum).add_(z[m].mean(0), alpha=1 - momentum)


class HybridHead(nn.Module):
    """Discriminative head + class-conditional density + Bayes generative classifier.

    Hybrid loss (review 5.3):
        L = CE(h_omega(z), y) + beta * (-log p_phi(z|y)) + gamma * JS(h_omega || p_gen)
    ``predict_proba`` exposes the discriminative, generative and blended posteriors; the
    generative one is what test-time class-conditional adaptation acts on.
    """

    def __init__(self, dim: int, n_classes: int, cfg: DensityConfig,
                 reference_prior: torch.Tensor | None = None, disc_hidden: int = 128):
        super().__init__()
        self.n_classes = n_classes
        self.cfg = cfg
        self.density = ClassConditionalDensity(dim, n_classes, cfg)
        self.disc = nn.Sequential(nn.Linear(dim, disc_hidden), nn.ELU(),
                                  nn.Linear(disc_hidden, n_classes))
        if reference_prior is None:
            reference_prior = torch.full((n_classes,), 1.0 / n_classes)
        self.register_buffer("log_pi_star", torch.log(reference_prior.clamp_min(1e-8)))

    def disc_logits(self, z):
        return self.disc(z)

    @staticmethod
    def _js(p, q):
        p = p.clamp_min(1e-8); q = q.clamp_min(1e-8)
        m = (0.5 * (p + q)).clamp_min(1e-8)
        return 0.5 * (p * (p.log() - m.log())).sum(1) + 0.5 * (q * (q.log() - m.log())).sum(1)

    def loss(self, z: torch.Tensor, y: torch.Tensor, ce_weight=None):
        logits = self.disc(z)
        ce = F.cross_entropy(logits, y, weight=ce_weight)
        # per-dimension NLL so beta_density is dimension-robust (raw multivariate NLL grows
        # ~linearly with z_c_dim and would otherwise swamp the CE gradient).
        nll = -self.density.log_prob(z, y).mean() / self.density.dim
        p_disc = F.softmax(logits, dim=1)
        p_gen = self.density.class_posterior(z, self.log_pi_star)
        js = self._js(p_disc, p_gen).mean()
        total = ce + self.cfg.beta_density * nll + self.cfg.gamma_consistency * js
        return total, dict(ce=float(ce.detach()), density_nll=float(nll.detach()),
                           js_consistency=float(js.detach()))

    def predict_proba(self, z: torch.Tensor, prior: torch.Tensor | None = None, mode: str = "blend"):
        log_prior = self.log_pi_star if prior is None else torch.log(prior.clamp_min(1e-8))
        p_gen = self.density.class_posterior(z, log_prior)
        if mode == "gen":
            return p_gen
        p_disc = F.softmax(self.disc(z), dim=1)
        if mode == "disc":
            return p_disc
        return 0.5 * (p_disc + p_gen)


if __name__ == "__main__":
    torch.manual_seed(0)
    d, K = 16, 3
    head = HybridHead(d, K, DensityConfig(n_components=2, cov_rank=4))
    z = torch.randn(64, d, requires_grad=True)
    y = torch.randint(0, K, (64,))
    lp = head.density.log_prob_all(z)
    assert lp.shape == (64, K) and torch.isfinite(lp).all(), "log_prob_all bad"
    loss, info = head.loss(z, y)
    loss.backward()
    p = head.predict_proba(z)
    print("loss", round(float(loss), 4), {k: round(v, 4) for k, v in info.items()})
    print("proba shape", tuple(p.shape), "rows sum~1:", float(p.sum(1).mean()),
          "grad ok:", bool(torch.isfinite(z.grad).all()))
