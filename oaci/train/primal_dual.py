"""Compatibility wrapper around the model-agnostic engine.

``train_risk_feasible`` keeps its public signature but no longer defines its own 2-D
``Encoder``/``TaskHead`` or a second training loop: it builds an MLP backbone via ``build_model``,
materialises the (optional) ``RareCellSampler`` into immutable ID-based plans, and drives
``engine.train_stage1`` / ``train_stage2`` with the OACI adversarial objective. There is exactly one
checkpoint semantics now (``model_state``), exported from ``checkpoint.py``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from ..models import build_model
from ..support_graph import SupportGraph
from .adversary import ConditionalDomainAdversary, reference_entropy_bar
from .batch_plan import (build_full_batch_alignment_plan, build_full_batch_task_plan,
                         build_sampler_plans)
from .bn import all_eval
from .checkpoint import CheckpointRecord, ERMStage, TrainResult  # re-export (new ABI)
from .data import TrainingData
from .engine import EngineConfig, InvocationRegistry, dual_update, effective_risk_weight, train_stage1, train_stage2
from .objective import ActiveStatus
from .rng import derive_seed, forked_rng


@dataclass
class TrainConfig:
    z_dim: int = 8
    enc_hidden: int = 32
    adv_hidden: int = 16
    n_classes: int = 2
    metric: str = "balanced_ce"
    epsilon: float = 0.03
    numerical_tol: float = 1e-4
    stage1_epochs: int = 150
    stage2_epochs: int = 150
    warmup_steps: int = 60
    critic_steps: int = 5
    lr_enc: float = 1e-2
    lr_task: float = 5e-3
    lr_critic: float = 1e-2
    dual_lr: float = 0.5
    lambda_init: float = 0.3
    lambda_max: float = 20.0
    lambda_floor: float = 0.0
    seed: int = 0


class OaciObjective:
    """Conditional-domain adversarial objective (``A = -C_D``); active iff a comparable class
    exists. The formal package adapter (``methods/oaci.py``) replaces this in A2."""
    name = "OACI"

    def __init__(self, support_graph: SupportGraph, adv_hidden: int = 16):
        self.sg = support_graph
        self.adv_hidden = adv_hidden
        self.H_ref = reference_entropy_bar(support_graph)
        self._critic = None

    def active_status(self) -> ActiveStatus:
        if self.sg.comparable_classes:
            return ActiveStatus(True, None)
        return ActiveStatus(False, "no_comparable_class")

    def build_critic(self, feat_dim, device):
        self._critic = ConditionalDomainAdversary(int(feat_dim), self.sg, hidden=self.adv_hidden).to(device)
        return self._critic

    def critic_loss(self, critic, z_detached, batch):                 # minimise C_D
        return critic.domain_ce_contribution(z_detached, batch.y, batch.d, batch.w)

    def encoder_penalty(self, critic, z, batch):                      # encoder maximises C_D -> -C_D
        return -critic.domain_ce_contribution(z, batch.y, batch.d, batch.w)

    def full_surrogate(self, model, data, device, chunk_size):
        with all_eval(model), torch.no_grad():
            z = model(data.X.to(device)).z
            cd = float(self._critic.domain_ce(z, data.y.to(device), data.d.to(device),
                                              importance_weight=data.sample_mass.to(device)).item())
        return self.H_ref - cd

    def diagnostics(self):
        return {"H_ref_bar": self.H_ref}


def _engine_cfg(cfg: TrainConfig, steps_per_epoch: int) -> EngineConfig:
    return EngineConfig(
        metric=cfg.metric, epsilon=cfg.epsilon, numerical_tol=cfg.numerical_tol,
        stage1_epochs=cfg.stage1_epochs, stage1_steps_per_epoch=1, stage2_epochs=cfg.stage2_epochs,
        steps_per_epoch=steps_per_epoch, warmup_steps=cfg.warmup_steps, critic_steps=cfg.critic_steps,
        checkpoint_every=1, lr_stage1=cfg.lr_task, lr_encoder=cfg.lr_enc, lr_critic=cfg.lr_critic,
        dual_lr=cfg.dual_lr, lambda_init=cfg.lambda_init, lambda_max=cfg.lambda_max,
        lambda_floor=cfg.lambda_floor, gradient_clip=0.0, base_seed=cfg.seed)


def make_training_data(X, y, d, group, n_classes, sample_mass=None) -> TrainingData:
    X = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    y = torch.as_tensor(np.asarray(y), dtype=torch.long)
    d = None if d is None else torch.as_tensor(np.asarray(d), dtype=torch.long)
    sm = (torch.ones(X.shape[0]) if sample_mass is None
          else torch.as_tensor(np.asarray(sample_mass), dtype=torch.float32))
    sid = tuple(f"syn{i}" for i in range(X.shape[0]))
    grp = None if group is None else tuple(str(g) for g in np.asarray(group).tolist())
    return TrainingData(X=X, y=y, sample_id=sid, sample_mass=sm, n_classes=int(n_classes),
                        d=d, group=grp).validate()


def train_risk_feasible(X, y, d, group, support_graph: SupportGraph, cfg: TrainConfig,
                        sampler=None, sample_mass=None, device=None,
                        registry: InvocationRegistry | None = None, invocation_id="default") -> TrainResult:
    device = device or torch.device("cpu")
    data = make_training_data(X, y, d, group, cfg.n_classes, sample_mass=sample_mass)
    in_dim = int(data.X.shape[1])

    def factory():
        return build_model("mlp", in_dim=in_dim, n_classes=cfg.n_classes, z_dim=cfg.z_dim, hidden=cfg.enc_hidden)

    with forked_rng(derive_seed(cfg.seed, "model_init"), device):
        model = factory()

    steps_per_epoch = 1 if sampler is None else int(sampler.cfg.steps_per_epoch)
    ecfg = _engine_cfg(cfg, steps_per_epoch)
    stage1_plan = build_full_batch_task_plan(data, "stage1_task", cfg.stage1_epochs, 1, cfg.seed,
                                             "stage1_task_dropout")
    erm_stage = train_stage1(model, data, stage1_plan, ecfg, device, registry, invocation_id)

    if sampler is None:
        task_plan = build_full_batch_task_plan(data, "stage2_task", cfg.stage2_epochs, 1, cfg.seed,
                                               "stage2_task_dropout")
        align = build_full_batch_alignment_plan(data, cfg.warmup_steps, cfg.stage2_epochs,
                                                cfg.critic_steps, cfg.seed)
    else:
        task_plan, align = build_sampler_plans(data, sampler, cfg.stage2_epochs, steps_per_epoch,
                                               cfg.warmup_steps, cfg.critic_steps, cfg.seed)

    objective = OaciObjective(support_graph, adv_hidden=cfg.adv_hidden)
    return train_stage2(factory, erm_stage, data, objective, task_plan, align, ecfg, device)
