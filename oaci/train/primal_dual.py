"""Risk-feasible primal–dual trainer.

    τ = R̂_ERM + ε ,    min_θ  L̃_train(θ)   s.t.   R̂_src(θ) ≤ τ .

* **R̂_ERM** is the *realised* empirical risk of the frozen Stage-1 (ERM) checkpoint — not an
  "ERM lower bound".
* The training inner game uses the PyTorch ``ConditionalDomainAdversary`` (``C_D``), NOT the
  sklearn ``extractable_LQ_ov`` estimator. The critic minimises ``C_D``; the encoder primal is
  ``H_ref − C_D + λ(R_src − τ)`` whose gradient (dropping the constants ``H_ref`` and ``λτ``)
  is ``-domain_ce + λ·task_risk``.
* **λ is the dual multiplier of the RISK CONSTRAINT**, not a weight on the leakage term.
  Dual ascent on the full source guard set: ``λ ← Π_[0,λmax]( λ + η_λ (R̂_guard − τ) )``.
* Critic **warmup freezes the encoder**; an anti-collapse floor ``λ_floor>0`` keeps the task
  anchored so the encoder never maximises domain CE unconstrained at λ=0 (the dual λ itself is
  reported pure). Stage-2 warm-starts from the feasible ERM checkpoint.
* Unsupported cells still enter the **task risk** but never the adversary.
* The ERM checkpoint is deep-copied once and never mutated (byte-exact fallback target).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn

from ..support_graph import SupportGraph
from .adversary import ConditionalDomainAdversary, reference_entropy_bar
from .risk import assert_differentiable_primal, balanced_error, source_risk


# --------------------------------------------------------------------------------------
# models
# --------------------------------------------------------------------------------------
class Encoder(nn.Module):
    def __init__(self, in_dim: int, z_dim: int, hidden: int = 0):
        super().__init__()
        self.net = (
            nn.Linear(in_dim, z_dim) if hidden <= 0
            else nn.Sequential(nn.Linear(in_dim, hidden), nn.ReLU(), nn.Linear(hidden, z_dim))
        )

    def forward(self, x):
        return self.net(x)


class TaskHead(nn.Module):
    def __init__(self, z_dim: int, n_classes: int):
        super().__init__()
        self.lin = nn.Linear(z_dim, n_classes)

    def forward(self, z):
        return self.lin(z)


# --------------------------------------------------------------------------------------
# config / records
# --------------------------------------------------------------------------------------
@dataclass
class TrainConfig:
    z_dim: int = 8
    enc_hidden: int = 32
    adv_hidden: int = 16                 # nonlinear critic: matches the outer probe family
    n_classes: int = 2
    metric: str = "balanced_ce"          # primal risk metric (differentiable: ce | balanced_ce)
    epsilon: float = 0.03                # τ = R̂_ERM + ε
    numerical_tol: float = 1e-4          # selector feasibility slack
    stage1_epochs: int = 150
    stage2_epochs: int = 150
    warmup_steps: int = 60               # critic-only steps (encoder frozen) before the game
    critic_steps: int = 5                # critic updates per encoder update
    lr_enc: float = 1e-2
    lr_task: float = 5e-3
    lr_critic: float = 1e-2
    dual_lr: float = 0.5                 # η_λ
    lambda_init: float = 0.3             # start modest so leakage can fall; dual raises it on violation
    lambda_max: float = 20.0
    lambda_floor: float = 0.02           # anti-collapse floor (keeps the task anchored at λ→0)
    seed: int = 0


@dataclass
class CheckpointRecord:
    epoch: int
    enc_state: dict
    head_state: dict
    R_src: float                          # realised guard-set source risk (primal metric)
    balanced_err: float                   # guard/report metric
    leakage_surrogate: float              # H_ref_bar − C_D on the full set
    lam: float


@dataclass
class TrainResult:
    erm_ckpt: dict                        # {'enc': state, 'head': state} — immutable fallback target
    R_ERM_hat: float
    tau: float
    H_ref_bar: float
    trajectory: list[CheckpointRecord] = field(default_factory=list)
    in_dim: int = 0
    cfg: TrainConfig | None = None


# --------------------------------------------------------------------------------------
# dual update
# --------------------------------------------------------------------------------------
def dual_update(lam: float, R_guard: float, tau: float, eta: float, lam_max: float) -> float:
    """``λ ← clip(λ + η (R_guard − τ), 0, λmax)`` — rises on violation, falls on slack."""
    return float(min(max(lam + eta * (R_guard - tau), 0.0), lam_max))


def _clone_state(module: nn.Module) -> dict:
    return copy.deepcopy({k: v.detach().clone() for k, v in module.state_dict().items()})


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------
def train_risk_feasible(
    X, y, d, group, support_graph: SupportGraph, cfg: TrainConfig, sample_weight=None
) -> TrainResult:
    """Two-stage trainer. Returns the ERM checkpoint, τ, and the Stage-2 trajectory of
    checkpoints (each with its realised guard risk + leakage surrogate). Selection is a
    separate step (``selector.select_checkpoint``)."""
    assert_differentiable_primal(cfg.metric)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    d = torch.as_tensor(np.asarray(d), dtype=torch.long)
    in_dim = X.shape[1]
    nc = cfg.n_classes
    H_ref_bar = reference_entropy_bar(support_graph)

    enc = Encoder(in_dim, cfg.z_dim, cfg.enc_hidden)
    head = TaskHead(cfg.z_dim, nc)

    def guard_risk() -> float:
        with torch.no_grad():
            return float(source_risk(head(enc(X)), y, cfg.metric, nc).item())

    def guard_balerr() -> float:
        with torch.no_grad():
            return balanced_error(head(enc(X)), y, nc)

    # ---- Stage 1: ERM ----
    opt1 = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr_task)
    for _ in range(cfg.stage1_epochs):
        opt1.zero_grad()
        source_risk(head(enc(X)), y, cfg.metric, nc).backward()
        opt1.step()
    R_ERM_hat = guard_risk()
    tau = R_ERM_hat + cfg.epsilon
    erm_ckpt = {"enc": _clone_state(enc), "head": _clone_state(head)}   # immutable

    # ---- Stage 2: adversarial invariance under the risk constraint (warm start from ERM) ----
    adv = ConditionalDomainAdversary(cfg.z_dim, support_graph, hidden=cfg.adv_hidden)
    has_adv = any(True for _ in adv.parameters())          # False iff no comparable class (no-op)
    opt_enc = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr_enc)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=cfg.lr_critic) if has_adv else None

    # critic warmup: encoder FROZEN, train only the critic
    if has_adv:
        for _ in range(cfg.warmup_steps):
            opt_adv.zero_grad()
            with torch.no_grad():
                Z = enc(X)
            adv.domain_ce(Z, y, d, sample_weight).backward()
            opt_adv.step()

    lam = cfg.lambda_init
    trajectory: list[CheckpointRecord] = []
    for epoch in range(cfg.stage2_epochs):
        # critic step(s): minimise C_D with the encoder detached
        if has_adv:
            for _ in range(cfg.critic_steps):
                opt_adv.zero_grad()
                Z = enc(X).detach()
                adv.domain_ce(Z, y, d, sample_weight).backward()
                opt_adv.step()
        # encoder+head step: minimise -C_D + max(λ, floor)·R_src  (constants dropped)
        opt_enc.zero_grad()
        Z = enc(X)
        risk_weight = max(lam, cfg.lambda_floor)
        loss = -adv.domain_ce(Z, y, d, sample_weight) + risk_weight * source_risk(head(Z), y, cfg.metric, nc)
        loss.backward()
        opt_enc.step()
        # dual ascent on the full source guard set
        R_guard = guard_risk()
        lam = dual_update(lam, R_guard, tau, cfg.dual_lr, cfg.lambda_max)
        with torch.no_grad():
            surrogate = H_ref_bar - float(adv.domain_ce(enc(X), y, d, sample_weight).item())
        trajectory.append(
            CheckpointRecord(
                epoch=epoch, enc_state=_clone_state(enc), head_state=_clone_state(head),
                R_src=R_guard, balanced_err=guard_balerr(), leakage_surrogate=surrogate, lam=lam,
            )
        )

    return TrainResult(
        erm_ckpt=erm_ckpt, R_ERM_hat=R_ERM_hat, tau=tau, H_ref_bar=H_ref_bar,
        trajectory=trajectory, in_dim=in_dim, cfg=cfg,
    )
