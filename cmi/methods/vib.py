"""Variational Information Bottleneck (Alemi et al. 2017) as a drop-in backbone wrapper.

Adds an I(X;Z) compression term on top of ANY base backbone. The base net is used as a feature
extractor (its own classifier head is ignored); we put a stochastic Gaussian bottleneck on its
feature z and a fresh linear classifier on the SAMPLED code z'. The variational bound

    I(X;Z) <= E_x KL( q(z'|x)=N(mu(x),sigma^2(x)) || r(z)=N(0,I) )

is the per-batch KL stashed in `self.last_kl`; the trainer adds `beta * last_kl` to the loss.
Because forward() returns (logits, z') exactly like a normal backbone (deterministic mu at eval),
it composes with the LPC-CMI penalty (which then operates on the compressed code z') and with the
standard predict() path with no other change."""
import torch
import torch.nn as nn


class VIBBackbone(nn.Module):
    def __init__(self, base, n_cls):
        super().__init__()
        self.base = base
        self.z_dim = base.z_dim
        self.to_stats = nn.Linear(base.z_dim, 2 * base.z_dim)   # -> (mu, logvar)
        self.clf = nn.Linear(base.z_dim, n_cls)
        self.last_kl = None

    def forward(self, x, *a, **k):
        out = self.base(x)
        z = out[1] if isinstance(out, (tuple, list)) else out      # base features
        mu, logvar = self.to_stats(z).chunk(2, dim=1)
        logvar = logvar.clamp(-8.0, 8.0)
        if self.training:
            zs = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu)   # reparameterized sample
        else:
            zs = mu                                                    # deterministic at eval
        # KL( N(mu,sigma^2) || N(0,I) ) per sample, averaged over the batch
        self.last_kl = 0.5 * (mu.pow(2) + logvar.exp() - logvar - 1.0).sum(1).mean()
        return self.clf(zs), zs
