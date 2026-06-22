"""Formal OACI objective: ``A_OACI = -C_D^ov`` (support-aware conditional-domain adversary).

Active iff at least one comparable class. EVERY critic / encoder alignment row must satisfy
``d ∈ S_y`` — an ineligible row is a hard FAILURE (the alignment plan already filters to eligible
rows, so a wrong batch must crash, not be silently dropped); ``rejected_ineligible_rows`` is 0 in a
correct runner. The denominator is the fixed ``M_y^ov``; the task risk still uses all source rows.
"""
from __future__ import annotations

import hashlib

import torch

from ..train.adversary import ConditionalDomainAdversary, reference_entropy_bar
from ..train.bn import all_eval
from ..train.objective import ActiveStatus


def support_hash(support_graph) -> str:
    h = hashlib.sha256()
    for y in support_graph.comparable_classes:
        h.update(f"{int(y)}:{sorted(int(d) for d in support_graph.support_of_class[y])}".encode())
    return h.hexdigest()


class OACIObjective:
    name = "OACI"

    def __init__(self, support_graph, adv_hidden: int = 16):
        self.sg = support_graph
        self.adv_hidden = adv_hidden
        self.H_ref = reference_entropy_bar(support_graph)
        self.support_hash = support_hash(support_graph)
        self._critic = None
        self._eligible_rows = 0
        self._rejected = 0

    def active_status(self):
        if self.sg.comparable_classes:
            return ActiveStatus(True, None)
        return ActiveStatus(False, "no_comparable_class")

    def build_critic(self, feat_dim, device):
        self._critic = ConditionalDomainAdversary(int(feat_dim), self.sg, hidden=self.adv_hidden).to(device)
        return self._critic

    def _assert_eligible(self, batch):
        ys = batch.y.tolist(); ds = batch.d.tolist()
        for yy, dd in zip(ys, ds):
            if int(yy) in self.sg.comparable_classes and int(dd) in self.sg.support_of_class[int(yy)]:
                self._eligible_rows += 1
            else:
                self._rejected += 1
                raise ValueError(f"OACI alignment batch contains an ineligible row (y={yy}, d={dd})")

    def critic_loss(self, critic, z_detached, batch):
        self._assert_eligible(batch)
        return critic.domain_ce_contribution(z_detached, batch.y, batch.d, batch.w)

    def encoder_penalty(self, critic, z, batch):
        self._assert_eligible(batch)
        return -critic.domain_ce_contribution(z, batch.y, batch.d, batch.w)

    def _extract_z(self, model, data, device, chunk_size):
        n = int(data.X.shape[0]); cs = n if chunk_size is None else int(chunk_size)
        zs = []
        with all_eval(model), torch.no_grad():
            for a in range(0, n, cs):
                zs.append(model(data.X[a:a + cs].to(device)).z)
        return torch.cat(zs, 0)

    def full_surrogate(self, model, data, device, chunk_size):
        z = self._extract_z(model, data, device, chunk_size)
        with torch.no_grad():
            cd = float(self._critic.domain_ce(z, data.y.to(device), data.d.to(device),
                                              importance_weight=data.sample_mass.to(device)).item())
        return self.H_ref - cd

    def diagnostics(self):
        return {"eligible_alignment_rows": self._eligible_rows, "rejected_ineligible_rows": self._rejected,
                "comparable_classes": list(self.sg.comparable_classes), "support_hash": self.support_hash,
                "H_ref_bar": self.H_ref}
