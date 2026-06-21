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
    # MAIN PATH: lambda_floor = 0 -> the encoder's risk coefficient is EXACTLY the dual λ
    # (the exact Lagrangian). A nonzero floor adds a fixed risk regulariser so the effective
    # primal coefficient no longer equals λ; keep it ONLY as a clearly-labelled stabilisation
    # ablation (it makes the problem a relaxation, not the constrained Lagrangian).
    lambda_floor: float = 0.0            # 0 = exact Lagrangian; >0 = stabilisation ablation only
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


def effective_risk_weight(lam: float, lambda_floor: float) -> float:
    """The encoder's primal coefficient on ``R_src``. With ``lambda_floor == 0`` this is EXACTLY
    the dual ``λ`` (the exact Lagrangian); a nonzero floor is a stabilisation ablation that adds
    a fixed risk regulariser (so the coefficient no longer equals λ)."""
    return max(lam, lambda_floor)


def _clone_state(module: nn.Module) -> dict:
    return copy.deepcopy({k: v.detach().clone() for k, v in module.state_dict().items()})


# --------------------------------------------------------------------------------------
# trainer
# --------------------------------------------------------------------------------------
def train_risk_feasible(
    X, y, d, group, support_graph: SupportGraph, cfg: TrainConfig, sampler=None
) -> TrainResult:
    """Two-stage trainer. Returns the ERM checkpoint, τ, and the Stage-2 trajectory.

    If ``sampler`` (a ``RareCellSampler``) is given, Stage-2 runs on its paired streams
    (task stream incl. ineligible cells; adversary stream over eligible cells only, with
    microbatch accumulation normalised by the fixed ``N_ov``); otherwise it is full-batch.

    If the support graph has **no comparable class** the adversary is undefined, so Stage-2 is
    a TRUE byte-exact no-op: the trainer returns the frozen ERM checkpoint with an empty
    trajectory (the selector then restores ERM exactly), running NO Stage-2 task updates.
    """
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

    # ---- Stage 1: ERM (realised empirical risk of the frozen checkpoint) ----
    opt1 = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr_task)
    for _ in range(cfg.stage1_epochs):
        opt1.zero_grad()
        source_risk(head(enc(X)), y, cfg.metric, nc).backward()
        opt1.step()
    R_ERM_hat = guard_risk()
    tau = R_ERM_hat + cfg.epsilon
    erm_ckpt = {"enc": _clone_state(enc), "head": _clone_state(head)}   # immutable fallback target

    # ---- no comparable class -> Stage-2 is a byte-exact no-op (return ERM, empty trajectory) ----
    if not support_graph.comparable_classes:
        return TrainResult(erm_ckpt=erm_ckpt, R_ERM_hat=R_ERM_hat, tau=tau, H_ref_bar=H_ref_bar,
                           trajectory=[], in_dim=in_dim, cfg=cfg)

    # ---- Stage 2: adversarial invariance under the risk constraint (warm start from ERM) ----
    adv = ConditionalDomainAdversary(cfg.z_dim, support_graph, hidden=cfg.adv_hidden)
    opt_enc = torch.optim.Adam(list(enc.parameters()) + list(head.parameters()), lr=cfg.lr_enc)
    opt_adv = torch.optim.Adam(adv.parameters(), lr=cfg.lr_critic)

    def critic_loss_full():
        return adv.domain_ce(enc(X).detach(), y, d)

    def critic_step_stream():
        opt_adv.zero_grad()
        for mb in sampler.adv_logical_batch().microbatches:
            adv.domain_ce_contribution(enc(X[mb.idx]).detach(), y[mb.idx], d[mb.idx], mb.weight).backward()
        opt_adv.step()

    # critic warmup: encoder FROZEN
    for _ in range(cfg.warmup_steps):
        if sampler is None:
            opt_adv.zero_grad(); critic_loss_full().backward(); opt_adv.step()
        else:
            critic_step_stream()

    lam = cfg.lambda_init
    trajectory: list[CheckpointRecord] = []
    for epoch in range(cfg.stage2_epochs):
        n_inner = 1 if sampler is None else sampler.cfg.steps_per_epoch
        for _ in range(n_inner):
            # critic step(s)
            for _ in range(cfg.critic_steps):
                if sampler is None:
                    opt_adv.zero_grad(); critic_loss_full().backward(); opt_adv.step()
                else:
                    critic_step_stream()
            # encoder+head step: -C_D + max(λ, floor)·R_src  (floor=0 -> coefficient == λ exactly)
            risk_weight = effective_risk_weight(lam, cfg.lambda_floor)
            opt_enc.zero_grad()
            if sampler is None:
                Z = enc(X)
                loss = -adv.domain_ce(Z, y, d) + risk_weight * source_risk(head(Z), y, cfg.metric, nc)
                loss.backward()
            else:
                for mb in sampler.adv_logical_batch().microbatches:          # -C_D (accumulated)
                    (-adv.domain_ce_contribution(enc(X[mb.idx]), y[mb.idx], d[mb.idx], mb.weight)).backward()
                tb = sampler.task_batch()                                    # weighted task risk
                task = source_risk(head(enc(X[tb.idx])), y[tb.idx], cfg.metric, nc, weight=tb.weight)
                (risk_weight * task).backward()
            opt_enc.step()
            R_guard = guard_risk()                                          # full source guard set
            lam = dual_update(lam, R_guard, tau, cfg.dual_lr, cfg.lambda_max)
        with torch.no_grad():
            surrogate = H_ref_bar - float(adv.domain_ce(enc(X), y, d).item())
        trajectory.append(
            CheckpointRecord(
                epoch=epoch, enc_state=_clone_state(enc), head_state=_clone_state(head),
                R_src=guard_risk(), balanced_err=guard_balerr(), leakage_surrogate=surrogate, lam=lam,
            )
        )

    return TrainResult(
        erm_ckpt=erm_ckpt, R_ERM_hat=R_ERM_hat, tau=tau, H_ref_bar=H_ref_bar,
        trajectory=trajectory, in_dim=in_dim, cfg=cfg,
    )
