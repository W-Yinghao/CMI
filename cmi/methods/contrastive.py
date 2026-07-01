"""Contrastive framework pieces for the conditional-leakage exploration.

Supervised contrastive loss (Khosla et al., arXiv 2004.11362), reimplemented MIT-clean.
The DG-relevant "domain-aware" variant restricts positives to *same class, different
domain* (same-Y cross-D pairs) — the geometric counterpart of minimizing I(Z;D|Y):
pulling a class together across domains suppresses domain style while preserving Y.

Used as: (a) a standalone contrastive framework baseline (`supcon`), and (b) an
alternative host for our estimator (`lpc_supcon` = SupCon + LPC-CMI).
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def sup_con_loss(z, y, d=None, temperature=0.1, cross_domain=True):
    """Supervised contrastive loss on (optionally domain-aware) positives.

    z [B, dim] embeddings; y [B] labels; d [B] domain ids. If cross_domain and d given,
    positives are same-label AND different-domain pairs.
    """
    z = F.normalize(z, dim=1)
    sim = z @ z.t() / temperature                       # [B, B]
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    B = z.size(0)
    eye = torch.eye(B, dtype=torch.bool, device=z.device)
    not_self = ~eye
    pos = (y[:, None] == y[None, :]) & not_self
    if cross_domain and d is not None:
        pos = pos & (d[:, None] != d[None, :])
    exp = torch.exp(sim) * not_self
    log_prob = sim - torch.log(exp.sum(dim=1, keepdim=True) + 1e-12)
    pos_cnt = pos.sum(dim=1)
    valid = pos_cnt > 0
    if not valid.any():
        return z.new_zeros(())
    loss = -(log_prob * pos).sum(dim=1)[valid] / pos_cnt[valid]
    return loss.mean()


# which frameworks use the posterior-KL CMI term vs the contrastive term
CMI_METHODS = {"marginal", "chain", "lpc_uniform", "lpc_prior", "lpc_supcon"}
SUPCON_METHODS = {"supcon", "lpc_supcon"}
