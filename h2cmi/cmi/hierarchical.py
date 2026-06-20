"""Hierarchical neural conditional-entropy CMI estimator + primal-dual leakage budgets.

Key objects:
  reference_conditional_entropy(d, y, pa, n_levels)  -> H(D_j | Y, Pa) in nats (empirical)
  ConditionalDomainCritic                            -> q_psi(d_j | z_c, y, Pa(d_j))
  HierarchicalCMI                                    -> per-factor signed CMI estimates
  DualBudget                                         -> lambda_j <- [lambda_j + eta(I_j-eps_j)]_+

Why signed estimates are kept (review): truncating negative MI estimates to zero induces a
systematic upward bias, so ``estimate`` returns the SIGNED H_ref - CE per factor; the
dual variable lambda_j >= 0 (not a clamp on the estimate) is what gates the penalty.

The encoder term added to the main loss is  sum_j lambda_j * I_hat_j  (the -lambda_j*eps_j
is constant in theta).  Because lambda_j >= 0 this MINIMISES I_hat_j, i.e. MAXIMISES the
critic's conditional cross-entropy -- the conditional-GRL min-max.  Step A fits the critics
on detached z (so they approximate psi*(theta)); Step B updates the encoder through the
frozen critics.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from h2cmi.config import CMIConfig
from h2cmi.domains import DomainDAG, DomainLabels


def reference_conditional_entropy(d: np.ndarray, y: np.ndarray, pa: np.ndarray,
                                  n_levels: int, n_cls: int, n_pa: int,
                                  alpha: float = 1.0) -> float:
    """Empirical H(D | Y, Pa) in nats with Laplace smoothing.

    H = sum_{y,pa} p(y,pa) * [ -sum_d p(d|y,pa) log p(d|y,pa) ].
    Encoder-independent; this is the H(D|Y) side of the corrected estimator (here
    additionally conditioned on the factor's DAG parents for the chain-rule term).
    """
    d = np.asarray(d); y = np.asarray(y); pa = np.asarray(pa)
    ctx = y.astype(np.int64) * n_pa + pa.astype(np.int64)            # joint (y,pa) context id
    n_ctx = n_cls * n_pa
    counts = np.zeros((n_ctx, n_levels), dtype=np.float64)
    np.add.at(counts, (ctx, d), 1.0)
    ctx_tot = counts.sum(1)
    p_ctx = ctx_tot / max(ctx_tot.sum(), 1.0)                        # p(y,pa)
    probs = (counts + alpha) / (ctx_tot[:, None] + alpha * n_levels)  # p(d|y,pa) smoothed
    ent = -(probs * np.log(probs)).sum(1)                            # H(D|y,pa) per context
    return float((p_ctx * ent).sum())


def grad_reverse(x: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """Gradient reversal: identity forward, negated*scale gradient backward."""
    return _GradReverse.apply(x, scale)


class _GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, scale):
        ctx.scale = scale
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        return -ctx.scale * g, None


class ConditionalDomainCritic(nn.Module):
    """q_psi(d_j | z_c, y, Pa(d_j)) -- an MLP over [z_c, onehot(y), embed(parent_key)]."""

    def __init__(self, z_dim, n_levels, n_cls, n_pa, hidden=128):
        super().__init__()
        self.n_cls, self.n_pa, self.n_levels = n_cls, n_pa, n_levels
        emb = max(1, min(8, n_pa))
        self.pa_emb = nn.Embedding(n_pa, emb)
        self.net = nn.Sequential(
            nn.Linear(z_dim + n_cls + emb, hidden), nn.ELU(),
            nn.Linear(hidden, hidden), nn.ELU(),
            nn.Linear(hidden, n_levels))

    def forward(self, z, y, pkey):
        feat = torch.cat([z, F.one_hot(y, self.n_cls).float(), self.pa_emb(pkey)], dim=1)
        return self.net(feat)


class HierarchicalCMI(nn.Module):
    """One conditional critic per penalised factor + the chain-rule CMI decomposition."""

    def __init__(self, z_dim: int, n_classes: int, dag: DomainDAG,
                 train_domains: DomainLabels, train_y: np.ndarray, cfg: CMIConfig):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes
        self.factors = [f.name for f in dag.penalised_factors()]
        self.critics = nn.ModuleDict()
        self._npa: dict[str, int] = {}
        self._nlev: dict[str, int] = {}
        self.budgets: dict[str, float] = {}
        href = {}
        for f in dag.penalised_factors():
            pidx = dag.parent_indices(f.name)
            n_pa = int(np.prod([dag.factors[j].n_levels for j in pidx])) if pidx else 1
            self._npa[f.name] = n_pa
            self._nlev[f.name] = f.n_levels
            self.budgets[f.name] = f.budget
            self.critics[f.name] = ConditionalDomainCritic(
                z_dim, f.n_levels, n_classes, n_pa, cfg.critic_hidden)
            d_j = train_domains.factor(f.name)
            pkey = train_domains.parent_key(f.name)
            href[f.name] = reference_conditional_entropy(
                d_j, train_y, pkey, f.n_levels, n_classes, n_pa)
        # store reference entropies as buffers
        for name, h in href.items():
            self.register_buffer(f"href_{name}", torch.tensor(float(h)))

    # -- batch context helpers ---------------------------------------------------
    def batch_context(self, domains: DomainLabels, idx: np.ndarray):
        """Return dicts {factor: level[B]} and {factor: parent_key[B]} for a batch."""
        sub = domains.subset(idx)
        lev = {f: torch.as_tensor(sub.factor(f), dtype=torch.long) for f in self.factors}
        pk = {f: torch.as_tensor(sub.parent_key(f), dtype=torch.long) for f in self.factors}
        return lev, pk

    def href(self, name) -> torch.Tensor:
        return getattr(self, f"href_{name}")

    # -- Step A: fit critics on detached z ---------------------------------------
    def critic_loss(self, z_det, y, lev, pk):
        loss = z_det.new_zeros(())
        for f in self.factors:
            logits = self.critics[f](z_det, y, pk[f].to(z_det.device))
            loss = loss + F.cross_entropy(logits, lev[f].to(z_det.device))
        return loss

    # -- Step B: signed per-factor CMI estimate (grad to z) ----------------------
    def estimate(self, z, y, lev, pk, grl: bool | None = None):
        """Return (total_penalty_input, dict name->I_hat_j tensor).

        I_hat_j = H_ref_j - CE_j(z).  Critic params are frozen here (Step B steps only
        the encoder optimiser); optionally route z through a GRL so a single backward on
        +CE_j gives the encoder the adversarial (leakage-reducing) gradient.
        """
        terms = {}
        for f in self.factors:
            zf = grad_reverse(z, 1.0) if (grl if grl is not None else self.cfg.grl) else z
            ce = F.cross_entropy(self.critics[f](zf, y, pk[f].to(z.device)),
                                 lev[f].to(z.device))
            # with GRL the forward value is still H_ref - CE; the reversed grad makes the
            # encoder MAXIMISE CE when this term is MINIMISED in the loss.
            terms[f] = self.href(f) - ce
        return terms

    def total_penalty(self, terms: dict, lambdas: dict) -> torch.Tensor:
        out = None
        for f, v in terms.items():
            t = lambdas[f] * v
            out = t if out is None else out + t
        return out if out is not None else torch.zeros(())


class DualBudget:
    """Primal-dual leakage budget (review 5.5): lambda_j <- [lambda_j + eta(I_j-eps_j)]_+.

    One nonnegative dual variable per penalised factor.  Held as plain python floats (not
    autograd) -- they are updated from DETACHED CMI estimates, then used as fixed
    coefficients in the encoder loss.
    """

    def __init__(self, budgets: dict[str, float], cfg: CMIConfig):
        self.budgets = dict(budgets)
        self.cfg = cfg
        self.lmbda = {f: float(cfg.lambda_init) for f in budgets}
        self._t = 0

    def step(self, i_hat: dict[str, float]):
        self._t += 1
        if self._t <= self.cfg.warmup:
            return
        for f, eps in self.budgets.items():
            g = i_hat[f] - eps
            new = self.lmbda[f] + self.cfg.dual_lr * g
            self.lmbda[f] = float(min(max(new, 0.0), self.cfg.lambda_max))

    def as_tensors(self, device):
        return {f: torch.tensor(v, device=device) for f, v in self.lmbda.items()}

    def state(self):
        return dict(self.lmbda)


if __name__ == "__main__":
    from h2cmi.data.eeg_simulator import EEGSimulator, ShiftSpec
    sim = EEGSimulator(n_classes=3, n_chans=8, n_times=128,
                       shift=ShiftSpec(cov=1.0, prior=0.4), seed=0).sample(
        n_sites=3, subjects_per_site=2, sessions_per_subject=2, trials_per_session=20)
    cfg = CMIConfig()
    hcmi = HierarchicalCMI(16, sim.n_classes, sim.dag, sim.domains, sim.y, cfg)
    dual = DualBudget(hcmi.budgets, cfg)
    idx = np.arange(64)
    z = torch.randn(64, 16, requires_grad=True)
    y = torch.as_tensor(sim.y[idx], dtype=torch.long)
    lev, pk = hcmi.batch_context(sim.domains, idx)
    la = hcmi.critic_loss(z.detach(), y, lev, pk); la.backward()
    terms = hcmi.estimate(z, y, lev, pk)
    pen = hcmi.total_penalty(terms, dual.as_tensors(z.device))
    pen.backward()
    print("factors:", hcmi.factors)
    print("H_ref:", {f: round(float(hcmi.href(f)), 3) for f in hcmi.factors})
    print("I_hat:", {f: round(float(v.detach()), 3) for f, v in terms.items()})
    dual.cfg.warmup = 0
    dual.step({f: float(v.detach()) for f, v in terms.items()})
    print("lambda after dual step:", {f: round(v, 3) for f, v in dual.state().items()})
    print("grad to z ok:", bool(torch.isfinite(z.grad).all()))
