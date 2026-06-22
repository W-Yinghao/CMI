"""Full-domain posterior critic shared by global-LPC and uniform.

Per present class ``y`` a head outputs ``q_ψ,y(D | z)`` over the FIXED level-0 domain universe (it
is NOT re-encoded when a cell is deleted). The critic minimises the mass-weighted domain CE over
ALL observed rows (incl. low-sample cells):

    C_D^full = Σ_y p_ref(y) · (Σ_{i:y_i=y} b_i [-log q_ψ,y(d_i|z_i)]) / M_y .

The encoder minimises the mass-weighted KL of that posterior to a target prior π_y:

    A_m = Σ_y p_ref(y) · (Σ_{i:y_i=y} b_i KL(q_ψ,y(·|z_i) ‖ π_y)) / M_y .

global-LPC sets π_y = π_y^α (cell-mass smoothed); uniform sets π_y = 1/|D0|. Missing cells appear
ONLY in π_y, never as fabricated rows.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..train.bn import all_eval
from ..train.objective import ActiveStatus, BatchView


class PosteriorDomainCritic(nn.Module):
    def __init__(self, feat_dim: int, present_classes, n_level0: int, hidden: int = 0):
        super().__init__()
        self.heads = nn.ModuleDict()
        for y in present_classes:
            self.heads[str(int(y))] = (
                nn.Linear(feat_dim, n_level0) if hidden <= 0
                else nn.Sequential(nn.Linear(feat_dim, hidden), nn.ReLU(), nn.Linear(hidden, n_level0)))

    def logits_for(self, z, yy: int):
        return self.heads[str(int(yy))](z)


class PosteriorObjective:
    """Base for the two full-domain baselines; subclasses supply the per-class log-prior."""

    def __init__(self, name, level0_domains, class_mass, reference_prior, present_classes, active=True,
                 inactive_reason=None, hidden: int = 0):
        self.name = name
        self.level0 = [int(d) for d in level0_domains]
        self.dmap = {d: i for i, d in enumerate(self.level0)}
        self.present = [int(y) for y in present_classes]
        self.M_y = {int(y): float(class_mass[int(y)]) for y in self.present}
        self.p_ref = {int(y): float(reference_prior[int(y)]) for y in self.present}
        self.hidden = hidden
        self._active = active
        self._reason = inactive_reason
        self._critic = None
        self._log_priors = {y: torch.log(torch.as_tensor(self._prior_vector(y), dtype=torch.float32))
                            for y in self.present}

    # ---- subclass hook ----
    def _prior_vector(self, yy: int) -> np.ndarray:               # pragma: no cover - overridden
        raise NotImplementedError

    def prior_vector(self, yy: int) -> np.ndarray:
        return self._prior_vector(yy)

    # ---- objective protocol ----
    def active_status(self):
        return ActiveStatus(self._active, self._reason)

    def build_critic(self, feat_dim, device):
        self._critic = PosteriorDomainCritic(int(feat_dim), self.present, len(self.level0), self.hidden).to(device)
        return self._critic

    def _labels(self, batch, mask, device):
        return torch.tensor([self.dmap[int(dd)] for dd in batch.d[mask].tolist()], device=device)

    def _accumulate(self, critic, z, batch, fn):
        total = z.new_zeros(())
        for yy in self.present:
            m = batch.y == yy
            if not bool(m.any()) or self.M_y[yy] <= 0:
                continue
            logits = critic.logits_for(z[m], yy)
            labels = self._labels(batch, m, z.device)
            total = total + (self.p_ref[yy] / self.M_y[yy]) * (batch.w[m].to(z.dtype) * fn(logits, labels, yy)).sum()
        return total

    def critic_loss(self, critic, z_detached, batch):
        return self._accumulate(critic, z_detached, batch,
                                lambda lg, lab, yy: F.cross_entropy(lg, lab, reduction="none"))

    def _kl(self, logits, yy):
        logq = F.log_softmax(logits, dim=1)
        logpi = self._log_priors[yy].to(logits.device)
        return (logq.exp() * (logq - logpi.unsqueeze(0))).sum(dim=1)

    def encoder_penalty(self, critic, z, batch):
        return self._accumulate(critic, z, batch, lambda lg, lab, yy: self._kl(lg, yy))

    def full_surrogate(self, model, data, device, chunk_size):
        with all_eval(model), torch.no_grad():
            z = model(data.X.to(device)).z
            bv = BatchView(data.y.to(device), data.d.to(device), data.sample_mass.to(device))
            return float(self._accumulate(self._critic, z, bv, lambda lg, lab, yy: self._kl(lg, yy)))

    def weighted_domain_ce(self, model, data, device):
        """Diagnostic ``C_D^full`` (the critic objective) over the full set."""
        with all_eval(model), torch.no_grad():
            z = model(data.X.to(device)).z
            bv = BatchView(data.y.to(device), data.d.to(device), data.sample_mass.to(device))
            return float(self._accumulate(self._critic, z, bv,
                                          lambda lg, lab, yy: F.cross_entropy(lg, lab, reduction="none")))

    def diagnostics(self):
        return {"name": self.name, "level0_domains": list(self.level0), "present_classes": list(self.present)}
