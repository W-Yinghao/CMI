"""SPDIM-style comparator (review W1_W2_FROZEN): source-free Information-Maximization recentering.

Faithful to SPDIM's essence -- source-free recentering driven by information maximization -- realized
as a constrained diagonal affine transform on the FROZEN H2 latent, optimized (no labels, no encoder
gradient) to maximize IM = H(mean prediction) - mean H(prediction) of the frozen class-conditional
classifier. Fit on the unlabeled adaptation split, applied to the held-out eval split. Like the other
operators it is near-identity (trust region). NOTE: the IM marginal-entropy term pulls the predicted
marginal toward uniform (the standard IM/SHOT assumption), which is a meaningful contrast to the
metadata class-conditional route under NATURAL class imbalance.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.tta.class_conditional import Transform


def spdim_fit(density, U_adapt, prior, device, steps=40, lr=5e-2, trust=1.0):
    """Return a diagonal-affine Transform optimized for information maximization on U_adapt."""
    U = U_adapt.detach().to(device)
    d = U.shape[1]
    T = Transform(d, "diag_affine", device=device)
    log_prior = torch.log(torch.as_tensor(np.asarray(prior), dtype=torch.float32, device=device).clamp_min(1e-8))
    I = torch.eye(d, device=device)
    opt = torch.optim.Adam(T.params, lr=lr)
    frozen = [(p, p.requires_grad) for p in density.parameters()]
    for p, _ in frozen:
        p.requires_grad_(False)
    try:
        for _ in range(steps):
            z = T.apply(U)
            logits = density.log_prob_all(z) + log_prior.view(1, -1)
            p = F.softmax(logits, dim=1)
            mean_p = p.mean(0)
            h_marg = -(mean_p * torch.log(mean_p.clamp_min(1e-8))).sum()        # maximize (balance)
            h_cond = -(p * torch.log(p.clamp_min(1e-8))).sum(1).mean()          # minimize (confidence)
            im = h_marg - h_cond
            reg = trust * ((T.matrix() - I) ** 2).sum() / d + trust * (T.b ** 2).sum() / d
            loss = -im + reg
            opt.zero_grad(); loss.backward(); opt.step()
    finally:
        for p, req in frozen:
            p.requires_grad_(req)
    return T
