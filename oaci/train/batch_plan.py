"""ID-based immutable batch plans.

A plan stores STABLE ``sample_id``s and the FINAL (proposal-corrected) ``importance_weights`` —
never raw row indices and never an extra ``sample_mass`` factor (the engine must NOT re-multiply
mass). Resolving a plan validates the population signature and that every id is known; reordering
the input rows leaves the resolved batches identical. Duplicate ids WITHIN a step are legal
(sampling with replacement).
"""
from __future__ import annotations

import hashlib
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .data import TrainingData, population_signature_hash
from .rng import derive_seed


# -------------------------------- plan records --------------------------------
@dataclass(frozen=True)
class BatchStep:
    sample_ids: tuple
    importance_weights: tuple
    step_seed: int


@dataclass(frozen=True)
class TaskBatchPlan:
    role: str                                  # stage1_task | stage2_task
    population_signature_hash: str
    epochs: tuple                              # tuple[tuple[BatchStep, ...], ...]
    plan_hash: str


@dataclass(frozen=True)
class MicrobatchPlan:
    sample_ids: tuple
    importance_weights: tuple


@dataclass(frozen=True)
class LogicalBatchPlan:
    microbatches: tuple                        # tuple[MicrobatchPlan, ...]
    step_seed: int


@dataclass(frozen=True)
class AlignmentGameStep:
    critic_batches: tuple                      # len == critic_steps
    encoder_batch: LogicalBatchPlan


@dataclass(frozen=True)
class AlignmentPlan:
    warmup_batches: tuple                      # tuple[LogicalBatchPlan, ...]
    game_steps: tuple                          # tuple[AlignmentGameStep, ...]
    population_signature_hash: str
    plan_hash: str


# -------------------------------- hashing --------------------------------
def _feed_ids(h, ids, weights):
    for sid, w in zip(ids, weights):
        s = str(sid).encode()
        h.update(len(s).to_bytes(8, "little")); h.update(s)
        h.update(struct.pack("<d", float(w)))
    h.update(b"#")


def _feed_logical(h, lb: LogicalBatchPlan):
    h.update(b"L"); h.update(int(lb.step_seed).to_bytes(8, "little"))
    for mb in lb.microbatches:
        _feed_ids(h, mb.sample_ids, mb.importance_weights)


def _task_plan_hash(role, pop_sig, epochs) -> str:
    h = hashlib.sha256(); h.update(role.encode()); h.update(pop_sig.encode())
    for ep in epochs:
        h.update(b"E")
        for st in ep:
            h.update(b"S"); h.update(int(st.step_seed).to_bytes(8, "little"))
            _feed_ids(h, st.sample_ids, st.importance_weights)
    return h.hexdigest()


def _alignment_plan_hash(pop_sig, warmup, game_steps) -> str:
    h = hashlib.sha256(); h.update(pop_sig.encode())
    h.update(b"W")
    for lb in warmup:
        _feed_logical(h, lb)
    for gs in game_steps:
        h.update(b"G")
        for cb in gs.critic_batches:
            _feed_logical(h, cb)
        h.update(b"|")
        _feed_logical(h, gs.encoder_batch)
    return h.hexdigest()


# -------------------------------- resolving --------------------------------
@dataclass(frozen=True)
class ResolvedBatch:
    idx: torch.Tensor
    weight: torch.Tensor


def assert_population(plan_pop_sig: str, data: TrainingData) -> None:
    if plan_pop_sig != population_signature_hash(data):
        raise ValueError("batch plan population signature does not match the training data")


def resolve(ids, weights, index: dict) -> ResolvedBatch:
    rows = []
    for s in ids:
        if s not in index:
            raise ValueError(f"unknown sample_id in plan: {s!r}")
        rows.append(index[s])
    return ResolvedBatch(idx=torch.as_tensor(rows, dtype=torch.long),
                         weight=torch.as_tensor(np.asarray(weights), dtype=torch.float32))


# -------------------------------- builders (full batch) --------------------------------
def build_full_batch_task_plan(data: TrainingData, role: str, n_epochs: int, steps_per_epoch: int,
                               base_seed: int, namespace: str) -> TaskBatchPlan:
    """Each step is the FULL population, weighted by the base sample mass."""
    ids = tuple(data.sample_id)
    w = tuple(float(x) for x in data.sample_mass.detach().cpu().tolist())
    pop = population_signature_hash(data)
    epochs = tuple(
        tuple(BatchStep(ids, w, derive_seed(base_seed, namespace, e, s)) for s in range(steps_per_epoch))
        for e in range(n_epochs)
    )
    return TaskBatchPlan(role, pop, epochs, _task_plan_hash(role, pop, epochs))


def _full_logical(data: TrainingData, seed: int) -> LogicalBatchPlan:
    ids = tuple(data.sample_id)
    w = tuple(float(x) for x in data.sample_mass.detach().cpu().tolist())
    return LogicalBatchPlan((MicrobatchPlan(ids, w),), seed)


def build_full_batch_alignment_plan(data: TrainingData, warmup_steps: int, total_inner_steps: int,
                                    critic_steps: int, base_seed: int) -> AlignmentPlan:
    pop = population_signature_hash(data)
    warmup = tuple(_full_logical(data, derive_seed(base_seed, "critic_update", "warmup", k))
                   for k in range(warmup_steps))
    game = []
    for t in range(total_inner_steps):
        cbs = tuple(_full_logical(data, derive_seed(base_seed, "critic_update", t, c)) for c in range(critic_steps))
        enc = _full_logical(data, derive_seed(base_seed, "stage2_alignment_dropout", t))
        game.append(AlignmentGameStep(cbs, enc))
    game = tuple(game)
    return AlignmentPlan(warmup, game, pop, _alignment_plan_hash(pop, warmup, game))


# -------------------------------- builders (from a pre-seeded sampler) --------------------------------
def _logical_from_sampler(sampler, sample_id, seed: int) -> LogicalBatchPlan:
    lb = sampler.adv_logical_batch()
    mbs = tuple(MicrobatchPlan(tuple(sample_id[i] for i in mb.idx),
                               tuple(float(x) for x in np.asarray(mb.weight).tolist()))
                for mb in lb.microbatches)
    return LogicalBatchPlan(mbs, seed)


# -------------------------------- split materialisers (independent streams) --------------------------------
# These are used by the formal runner: the task plan and each method's alignment plan are drawn from
# SEPARATE RNG namespaces, so adding critic steps or changing method order never perturbs the shared
# task plan. (The legacy interleaved ``build_sampler_plans`` stays only for the compat wrapper.)
def materialize_task_plan(data: TrainingData, n_epochs: int, steps_per_epoch: int, base_seed: int,
                          role: str = "stage2_task", namespace: str = "stage2_task_dropout") -> TaskBatchPlan:
    return build_full_batch_task_plan(data, role, n_epochs, steps_per_epoch, base_seed, namespace)


def _logical_from_rows(data: TrainingData, rows, seed: int) -> LogicalBatchPlan:
    ids = tuple(data.sample_id[i] for i in rows)
    w = tuple(float(data.sample_mass[i]) for i in rows)
    return LogicalBatchPlan((MicrobatchPlan(ids, w),), seed)


def _alignment_over_rows(data: TrainingData, rows, warmup_steps, total_inner, critic_steps,
                         base_seed, critic_ns, enc_ns) -> AlignmentPlan:
    pop = population_signature_hash(data)
    warmup = tuple(_logical_from_rows(data, rows, derive_seed(base_seed, critic_ns, "warmup", k))
                   for k in range(warmup_steps))
    game = tuple(AlignmentGameStep(
        tuple(_logical_from_rows(data, rows, derive_seed(base_seed, critic_ns, t, c)) for c in range(critic_steps)),
        _logical_from_rows(data, rows, derive_seed(base_seed, enc_ns, t))) for t in range(total_inner))
    return AlignmentPlan(warmup, game, pop, _alignment_plan_hash(pop, warmup, game))


def eligible_rows(data: TrainingData, support_graph) -> list:
    """Rows with ``d ∈ S_y`` for a COMPARABLE class y — the only rows OACI may align over."""
    out = []
    y = data.y.detach().cpu().numpy(); d = data.d.detach().cpu().numpy()
    for i in range(len(data)):
        yy = int(y[i])
        if yy in support_graph.comparable_classes and int(d[i]) in support_graph.support_of_class[yy]:
            out.append(i)
    return out


def materialize_oaci_alignment_plan(data: TrainingData, support_graph, warmup_steps, total_inner,
                                    critic_steps, base_seed) -> AlignmentPlan:
    """OACI alignment over ELIGIBLE rows only (independent ``oaci_*`` RNG namespaces)."""
    return _alignment_over_rows(data, eligible_rows(data, support_graph), warmup_steps, total_inner,
                                critic_steps, base_seed, "oaci_critic", "oaci_align_dropout")


def materialize_full_domain_alignment_plan(data: TrainingData, warmup_steps, total_inner,
                                           critic_steps, base_seed) -> AlignmentPlan:
    """Full-domain alignment over ALL observed rows (global_lpc & uniform SHARE this plan)."""
    return _alignment_over_rows(data, list(range(len(data))), warmup_steps, total_inner,
                                critic_steps, base_seed, "fulldomain_critic", "fulldomain_align_dropout")


def build_sampler_plans(data: TrainingData, sampler, n_epochs: int, steps_per_epoch: int,
                        warmup_steps: int, critic_steps: int, base_seed: int):
    """Pre-materialise a (pre-seeded) ``RareCellSampler`` into immutable ID-based plans, in a FIXED
    draw order, so the engine is plan-driven and reproducible. Returns (task_plan, alignment_plan)."""
    sid = list(data.sample_id)
    pop = population_signature_hash(data)
    # warmup critic draws first (mirrors the streamed trainer's order)
    warmup = tuple(_logical_from_sampler(sampler, sid, derive_seed(base_seed, "critic_update", "warmup", k))
                   for k in range(warmup_steps))
    task_epochs, game = [], []
    t = 0
    for e in range(n_epochs):
        steps = []
        for s in range(steps_per_epoch):
            cbs = tuple(_logical_from_sampler(sampler, sid, derive_seed(base_seed, "critic_update", t, c))
                        for c in range(critic_steps))
            enc = _logical_from_sampler(sampler, sid, derive_seed(base_seed, "stage2_alignment_dropout", t))
            tb = sampler.task_batch()
            steps.append(BatchStep(tuple(sid[i] for i in tb.idx),
                                   tuple(float(x) for x in np.asarray(tb.weight).tolist()),
                                   derive_seed(base_seed, "stage2_task_dropout", e, s)))
            game.append(AlignmentGameStep(cbs, enc))
            t += 1
        task_epochs.append(tuple(steps))
    task_epochs = tuple(task_epochs)
    game = tuple(game)
    task_plan = TaskBatchPlan("stage2_task", pop, task_epochs, _task_plan_hash("stage2_task", pop, task_epochs))
    align = AlignmentPlan(warmup, game, pop, _alignment_plan_hash(pop, warmup, game))
    return task_plan, align
