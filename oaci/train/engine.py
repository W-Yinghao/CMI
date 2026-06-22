"""Model-agnostic two-stage training engine.

Stage-1 (ERM): normal ``train()`` (BatchNorm stats update). Stage-2: warm-start from the byte-exact
ERM checkpoint; the parent stays in TRAIN (dropout on) while every BatchNorm is forced to EVAL so
its running stats are FROZEN at the ERM values; affine BN parameters remain trainable. Every guard /
checkpoint / feature extraction runs under per-submodule-restored ``eval()`` + ``inference_mode``.
RNG is forked per stochastic forward so a critic call cannot perturb the task-dropout stream.

The engine never references a concrete backbone — it consumes any model returning
``ModelOutput(logits, z)`` and any ``MethodObjective``.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

from .batch_plan import ResolvedBatch, assert_population, resolve
from .bn import bn_buffer_hash, freeze_bn_running_stats
from .checkpoint import CheckpointRecord, ERMStage, TrainResult, clone_state_cpu, model_state_hash
from .evaluate import evaluate_guard
from .objective import BatchView
from .rng import derive_seed, forked_rng
from .risk import assert_differentiable_primal, source_risk


@dataclass(frozen=True)
class EngineConfig:
    metric: str = "balanced_ce"
    epsilon: float = 0.03
    numerical_tol: float = 1e-4
    stage1_epochs: int = 150
    stage1_steps_per_epoch: int = 1
    stage2_epochs: int = 150
    steps_per_epoch: int = 1
    warmup_steps: int = 60
    critic_steps: int = 5
    checkpoint_every: int = 1
    guard_chunk_size: int | None = None
    lr_stage1: float = 5e-3
    lr_encoder: float = 1e-2
    lr_critic: float = 1e-2
    dual_lr: float = 0.5
    lambda_init: float = 0.3
    lambda_max: float = 20.0
    lambda_floor: float = 0.0
    gradient_clip: float = 0.0
    base_seed: int = 0


def dual_update(lam: float, R_guard: float, tau: float, eta: float, lam_max: float) -> float:
    """``λ ← clip(λ + η (R_guard − τ), 0, λmax)`` — rises on violation, falls on slack."""
    return float(min(max(lam + eta * (R_guard - tau), 0.0), lam_max))


def effective_risk_weight(lam: float, lambda_floor: float) -> float:
    """Encoder's primal coefficient on ``R_src``. ``lambda_floor == 0`` -> EXACTLY the dual λ."""
    return max(lam, lambda_floor)


class InvocationRegistry:
    """Enforces 'Stage-1 trained once per (run-key, deletion level)'."""

    def __init__(self):
        self._seen: set = set()

    def claim(self, invocation_id: str) -> None:
        if invocation_id in self._seen:
            raise ValueError(f"Stage-1 already trained for invocation {invocation_id!r}")
        self._seen.add(invocation_id)


def _clip(params, clip):
    if clip and clip > 0:
        torch.nn.utils.clip_grad_norm_(params, clip)


def _gather(step_ids, step_w, index, data, device):
    rb: ResolvedBatch = resolve(step_ids, step_w, index)
    idx = rb.idx
    return idx, rb.weight.to(device), data.X[idx].to(device), data.y[idx].to(device), \
        (None if data.d is None else data.d[idx].to(device))


# -------------------------------- Stage 1 --------------------------------
def train_stage1(model, data, task_plan, cfg: EngineConfig, device=None,
                 registry: InvocationRegistry | None = None, invocation_id: str = "default") -> ERMStage:
    assert_differentiable_primal(cfg.metric)
    assert_population(task_plan.population_signature_hash, data)
    if registry is not None:
        registry.claim(invocation_id)
    device = device or torch.device("cpu")
    model.to(device)
    index = data.index()
    nc = int(data.n_classes)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr_stage1)

    step = 0
    model.train()
    for epoch in task_plan.epochs:
        for bstep in epoch:
            idx, w, xb, yb, _ = _gather(bstep.sample_ids, bstep.importance_weights, index, data, device)
            opt.zero_grad()
            with forked_rng(bstep.step_seed, device):
                logits = model(xb).logits
            source_risk(logits, yb, cfg.metric, nc, weight=w).backward()
            _clip(model.parameters(), cfg.gradient_clip)
            opt.step()
            step += 1

    g = evaluate_guard(model, data, cfg.metric, cfg.guard_chunk_size, device)
    R_erm, tau = g.risk, g.risk + cfg.epsilon
    ckpt = CheckpointRecord(epoch=len(task_plan.epochs) - 1, optimizer_step=step,
                            model_state=clone_state_cpu(model), model_hash=model_state_hash(model),
                            R_src=R_erm, balanced_err=g.balanced_err, train_surrogate=0.0, lam=0.0)
    return ERMStage(checkpoint=ckpt, R_ERM_hat=R_erm, tau=tau,
                    task_plan_hash=task_plan.plan_hash, stage1_invocation_id=invocation_id)


# -------------------------------- Stage 2 --------------------------------
def _critic_update(objective, critic, opt_adv, model, lb, index, data, device):
    """One critic optimizer step over a logical batch. Backbone stays EVAL (no dropout) and z is
    detached -> the backbone and its BN buffers are not touched."""
    if critic is None:
        return
    from .bn import all_eval
    opt_adv.zero_grad()
    for mb in lb.microbatches:
        idx, w, xb, yb, db = _gather(mb.sample_ids, mb.importance_weights, index, data, device)
        with all_eval(model), torch.no_grad():
            z = model(xb).z
        objective.critic_loss(critic, z.detach(), BatchView(yb, db, w)).backward()
    opt_adv.step()


def train_stage2(model_factory, erm_stage: ERMStage, data, objective, task_plan,
                 alignment_plan, cfg: EngineConfig, device=None) -> TrainResult:
    device = device or torch.device("cpu")
    erm_ckpt = erm_stage.checkpoint
    status = objective.active_status()

    def _erm_record(surrogate):
        return CheckpointRecord(epoch=-1, optimizer_step=0, model_state=erm_ckpt.model_state,
                                model_hash=erm_ckpt.model_hash, R_src=erm_stage.R_ERM_hat,
                                balanced_err=erm_ckpt.balanced_err, train_surrogate=surrogate, lam=0.0)

    if not status.active:
        # byte-exact ERM: NO factory call, NO optimizer, NO forward-training step
        return TrainResult(method_name=objective.name, active=False, inactive_reason=status.reason,
                           erm_stage=erm_stage, erm_record=_erm_record(0.0), trajectory=[],
                           initial_model_hash=erm_ckpt.model_hash, task_plan_hash=task_plan.plan_hash,
                           alignment_plan_hash=None)

    assert_population(task_plan.population_signature_hash, data)
    if alignment_plan is not None:
        assert_population(alignment_plan.population_signature_hash, data)
    index = data.index()
    nc = int(data.n_classes)

    model = model_factory().to(device)
    model.load_state_dict(erm_ckpt.model_state)
    initial_hash = model_state_hash(model)
    if initial_hash != erm_ckpt.model_hash:
        raise ValueError("Stage-2 model does not byte-match the ERM checkpoint after load")
    erm_bn = bn_buffer_hash(model)

    with forked_rng(derive_seed(cfg.base_seed, "critic_init"), device):    # critic init is seeded too
        critic = objective.build_critic(getattr(model, "feat_dim", None), device)
    opt_adv = torch.optim.Adam(critic.parameters(), lr=cfg.lr_critic) if critic is not None else None
    opt_enc = torch.optim.Adam(model.parameters(), lr=cfg.lr_encoder)

    # Stage-2 mode: parent TRAIN, BatchNorm EVAL (running stats frozen at ERM)
    model.train()
    freeze_bn_running_stats(model)

    for lb in alignment_plan.warmup_batches:
        _critic_update(objective, critic, opt_adv, model, lb, index, data, device)
    if bn_buffer_hash(model) != erm_bn:
        raise ValueError("critic warmup mutated BatchNorm running stats")

    erm_record = _erm_record(objective.full_surrogate(model, data, device, cfg.guard_chunk_size))

    lam, trajectory, step_idx, opt_step = cfg.lambda_init, [], 0, 0
    for epoch in range(cfg.stage2_epochs):
        for s in range(cfg.steps_per_epoch):
            game = alignment_plan.game_steps[step_idx]
            task_step = task_plan.epochs[epoch][s]
            for cb in game.critic_batches:
                _critic_update(objective, critic, opt_adv, model, cb, index, data, device)

            opt_enc.zero_grad()
            penalty = torch.zeros((), device=device)
            with forked_rng(game.encoder_batch.step_seed, device):
                for mb in game.encoder_batch.microbatches:
                    idx, w, xb, yb, db = _gather(mb.sample_ids, mb.importance_weights, index, data, device)
                    penalty = penalty + objective.encoder_penalty(critic, model(xb).z, BatchView(yb, db, w))
            with forked_rng(task_step.step_seed, device):
                idx, w, xb, yb, _ = _gather(task_step.sample_ids, task_step.importance_weights, index, data, device)
                task = source_risk(model(xb).logits, yb, cfg.metric, nc, weight=w)
            risk_weight = effective_risk_weight(lam, cfg.lambda_floor)
            (penalty + risk_weight * task).backward()
            _clip(model.parameters(), cfg.gradient_clip)
            opt_enc.step()
            opt_step += 1

            if bn_buffer_hash(model) != erm_bn:
                raise ValueError("Stage-2 encoder step mutated BatchNorm running stats")
            g = evaluate_guard(model, data, cfg.metric, cfg.guard_chunk_size, device)
            lam = dual_update(lam, g.risk, erm_stage.tau, cfg.dual_lr, cfg.lambda_max)
            step_idx += 1

        if epoch % cfg.checkpoint_every == cfg.checkpoint_every - 1 or epoch == cfg.stage2_epochs - 1:
            g = evaluate_guard(model, data, cfg.metric, cfg.guard_chunk_size, device)
            if bn_buffer_hash(model) != erm_bn:
                raise ValueError("Stage-2 BatchNorm running stats drifted by checkpoint time")
            surrogate = objective.full_surrogate(model, data, device, cfg.guard_chunk_size)
            trajectory.append(CheckpointRecord(
                epoch=epoch, optimizer_step=opt_step, model_state=clone_state_cpu(model),
                model_hash=model_state_hash(model), R_src=g.risk, balanced_err=g.balanced_err,
                train_surrogate=surrogate, lam=lam))

    return TrainResult(method_name=objective.name, active=True, inactive_reason=None,
                       erm_stage=erm_stage, erm_record=erm_record, trajectory=trajectory,
                       initial_model_hash=initial_hash, task_plan_hash=task_plan.plan_hash,
                       alignment_plan_hash=alignment_plan.plan_hash)
