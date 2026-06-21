"""PyTorch conditional-domain **adversary** for the trainer's inner min–max game.

NOT the sklearn ``extractable_LQ_ov`` estimator (that is the non-differentiable OUTER score on
frozen representations — see ``oaci/leakage/``). This is a differentiable per-class domain
classifier whose cross-entropy is ``C_D``:

* the **critic** minimises ``C_D`` (predict ``D`` from ``Z``);
* the **encoder** maximises it (the ``-C_D`` term reduces leakage).

Per comparable class ``y`` a head predicts ``D`` over the label space ``S_y``. Unsupported cells
(``d ∉ S_y`` or non-comparable ``y``) contribute **no** label, loss, or gradient. The class
aggregation uses the **fixed** ``p_ref(y)``; optional per-example importance weights (reserved
for the rare-cell sampler) reweight WITHIN a class only — they never change the ``p_ref`` class
weights, the support sets, or eligibility.

``train_leakage_surrogate = H_ref_bar − C_D`` (nats), where ``H_ref_bar = Σ_y p_ref(y) Ĥ_y`` is
a constant from the fixed support graph. It is a training-side surrogate, distinct from the
audited ``L_Q^ov``.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..leakage.estimate import reference_conditional_entropy
from ..support_graph import SupportGraph


def reference_entropy_bar(support_graph: SupportGraph) -> float:
    """``H_ref_bar = Σ_{y∈C_cmp} p_ref(y) Ĥ_y`` (nats), from the fixed support graph."""
    H = reference_conditional_entropy(support_graph)
    return float(sum(support_graph.reference_prior[y] * H[y] for y in support_graph.comparable_classes))


class ConditionalDomainAdversary(nn.Module):
    """One domain head per comparable class, label space ``S_y``; aggregates by fixed ``p_ref``."""

    def __init__(self, z_dim: int, support_graph: SupportGraph, hidden: int = 0):
        super().__init__()
        self.comparable = list(support_graph.comparable_classes)
        self.support_of_class = {y: list(support_graph.support_of_class[y]) for y in self.comparable}
        self.p_ref = {y: float(support_graph.reference_prior[y]) for y in self.comparable}
        self.dmap = {y: {int(d): i for i, d in enumerate(self.support_of_class[y])} for y in self.comparable}
        self.heads = nn.ModuleDict()
        for y in self.comparable:
            k = len(self.support_of_class[y])
            self.heads[str(y)] = (
                nn.Linear(z_dim, k) if hidden <= 0
                else nn.Sequential(nn.Linear(z_dim, hidden), nn.ReLU(), nn.Linear(hidden, k))
            )

    def class_weights(self) -> dict[int, float]:
        """The FIXED p_ref class weights actually used (independent of batch composition)."""
        return dict(self.p_ref)

    def _eligible_mask(self, y: torch.Tensor, d: torch.Tensor, yy: int) -> torch.Tensor:
        m = y == yy
        dm = torch.zeros_like(d, dtype=torch.bool)
        for dom in self.support_of_class[yy]:
            dm |= d == dom
        return m & dm

    def domain_ce(self, Z, y, d, sample_weight=None) -> torch.Tensor:
        """``C_D = Σ_y p_ref(y) · (weighted) mean CE`` over eligible class-``y`` rows. Rows in
        unsupported cells contribute nothing. Returns a 0-d tensor (0 if no comparable class)."""
        y = torch.as_tensor(y, device=Z.device)
        d = torch.as_tensor(d, device=Z.device)
        total = Z.new_zeros(())
        for yy in self.comparable:
            mask = self._eligible_mask(y, d, yy)
            if not bool(mask.any()):
                continue
            Zc = Z[mask]
            labels = torch.tensor([self.dmap[yy][int(dd)] for dd in d[mask].tolist()], device=Z.device)
            ce = F.cross_entropy(self.heads[str(yy)](Zc), labels, reduction="none")
            if sample_weight is not None:
                w = torch.as_tensor(sample_weight, device=Z.device, dtype=ce.dtype)[mask]
                ce_mean = (ce * w).sum() / w.sum().clamp_min(1e-12)
            else:
                ce_mean = ce.mean()
            total = total + self.p_ref[yy] * ce_mean
        return total
