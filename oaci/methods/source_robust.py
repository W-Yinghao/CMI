"""SRC (Source-Robust Calibration / RF-WDC) — a NON-adversarial Stage-2 objective (C11b). From the ERM
checkpoint, directly optimize a smooth worst-domain balanced-CE over the SOURCE_TRAIN domains, under the same
engine-owned risk-feasibility constraint λ(R_src−τ) as OACI. NO domain adversary/critic.

Contract (oaci/train/objective.py MethodObjective): build_critic -> None (engine treats this as encoder-
penalty-only; warmup/critic loops no-op). encoder_penalty returns the differentiable scalar
  A_SRC = τ_lse · logsumexp_d ( R_d^balancedCE / τ_lse )   over source domains d present in the batch,
the smooth-max of per-domain balanced CE. The engine adds λ·R_src and the dual τ handling — SRC supplies ONLY
this term (mirrors A_OACI = −C_D).

Two load-bearing facts (see design notes): (a) encoder_penalty receives z (pre-dropout features), not
logits, so we recompute logits = classifier(z) via a classifier reference stashed during full_surrogate
(which the engine calls first); the deterministic no-dropout endpoint is intentional. (b) logsumexp is NOT
additively decomposable across microbatches, so SRC MUST run on a single full-domain microbatch (the
full_domain_alignment plan) — never a micro-split plan.
"""
from __future__ import annotations

import torch

from ..train.bn import all_eval
from ..train.objective import ActiveStatus, BatchView
from ..train.risk import balanced_ce

SRC_OBJECTIVE = "smooth_worst_domain_balanced_ce"


class SRCObjective:
    name = "SRC"

    def __init__(self, n_classes, n_source_domains, *, smooth_temperature=0.1, min_source_domains=2):
        self.n_classes = int(n_classes)
        self.n_source_domains = int(n_source_domains)
        self.smooth_temperature = float(smooth_temperature)
        self.min_source_domains = int(min_source_domains)
        if self.smooth_temperature <= 0:
            raise ValueError("smooth_temperature must be > 0")
        self._classifier = None                          # stashed in full_surrogate (runs before encoder_penalty)

    # ---- MethodObjective protocol ----
    def active_status(self) -> ActiveStatus:
        if self.n_source_domains < self.min_source_domains:
            return ActiveStatus(False, f"needs >= {self.min_source_domains} source domains for a worst-domain "
                                       f"objective; got {self.n_source_domains}")
        return ActiveStatus(True, None)

    def build_critic(self, feat_dim, device):
        return None                                      # non-adversarial: engine sets opt_adv=None, warmup no-ops

    def critic_loss(self, critic, z_detached, batch):
        raise RuntimeError("SRC has no critic; critic_loss must never be called")

    def _smooth_worst_domain(self, logits, y, d, w) -> torch.Tensor:
        """A_SRC over the domains present in (logits,y,d,w). Differentiable; a single-domain batch reduces to
        that domain's balanced CE."""
        doms = torch.unique(d)
        per = []
        for dom in doms:
            m = d == dom
            per.append(balanced_ce(logits[m], y[m], n_classes=self.n_classes, weight=w[m]))
        stk = torch.stack(per)
        t = self.smooth_temperature
        return t * torch.logsumexp(stk / t, dim=0)

    def encoder_penalty(self, critic, z, batch: BatchView) -> torch.Tensor:
        if self._classifier is None:
            raise RuntimeError("SRC.encoder_penalty called before full_surrogate stashed the classifier")
        if batch.d is None:
            raise ValueError("SRC requires per-row domain ids (batch.d)")
        logits = self._classifier(z)                     # recompute endpoint from pre-dropout z (no dropout)
        return self._smooth_worst_domain(logits, batch.y, batch.d, batch.w)

    def full_surrogate(self, model, data, device, chunk_size) -> float:
        """A_SRC on the FULL population (float; lower is better). Also stashes model.classifier for
        encoder_penalty — the engine calls full_surrogate before the first encoder step."""
        self._classifier = model.classifier
        if data.d is None:
            raise ValueError("SRC full_surrogate requires domain ids on the training data")
        with all_eval(model), torch.inference_mode():
            logits = []
            n = data.X.shape[0]
            for i in range(0, n, int(chunk_size)):
                xb = data.X[i:i + int(chunk_size)].to(device)
                logits.append(model(xb).logits.detach().to("cpu"))
            logits = torch.cat(logits, dim=0)
        y = data.y.to("cpu")
        d = data.d.to("cpu")
        w = data.sample_mass.to("cpu")
        return float(self._smooth_worst_domain(logits.double(), y, d, w.double()).item())

    def diagnostics(self) -> dict:
        return {"objective": SRC_OBJECTIVE, "smooth_temperature": self.smooth_temperature,
                "n_source_domains": self.n_source_domains, "n_classes": self.n_classes,
                "adversarial": False}
