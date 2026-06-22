"""Sampler-driven, manifest-configured plan materialisers (the runner's plan layer).

Unlike the full-batch builders in ``train/batch_plan.py`` (used only by the compat wrapper), these
consume the manifest's task-batch size / per-cell coverage / microbatch / replacement config via the
string-preserving samplers in ``data/plan_sampler.py``. Stage-1 and Stage-2 task streams use
independent RNG namespaces; the three Stage-2 methods share ONE Stage-2 task plan; global-LPC and
uniform share ONE full-domain alignment plan; critic count / method order never advance a task draw.
"""
from __future__ import annotations

import hashlib  # noqa: F401

from ..train.batch_plan import (AlignmentGameStep, AlignmentPlan, BatchStep, LogicalBatchPlan,
                                MicrobatchPlan, TaskBatchPlan, _alignment_plan_hash, _task_plan_hash)
from ..train.rng import derive_seed
from .plan_sampler import MassUnitTaskSampler, ObservedCellSampler, RareEligibleCellSampler, UnitIndex


def _microbatches(sids, ws, micro_size: int) -> tuple:
    if micro_size is None or micro_size <= 0 or micro_size >= len(sids):
        return (MicrobatchPlan(tuple(sids), tuple(ws)),)
    out = []
    for a in range(0, len(sids), micro_size):
        out.append(MicrobatchPlan(tuple(sids[a:a + micro_size]), tuple(ws[a:a + micro_size])))
    return tuple(out)


def _logical_from_step(sampler, step_index, micro_size) -> LogicalBatchPlan:
    sids, ws, seed = sampler.step(step_index)
    return LogicalBatchPlan(_microbatches(sids, ws, micro_size), seed)


def _sampling_design_hash(pop_sig, role, structure_hash, cells, per_cell, micro_size,
                          accumulation_steps, replacement_mode, idx) -> str:
    h = hashlib.sha256()
    h.update(pop_sig.encode()); h.update(role.encode()); h.update(structure_hash.encode())
    h.update(idx.design_hash().encode())                      # ACTUAL mass-unit -> id/cell/group/mass map
    for c in cells:
        h.update(str(c).encode()); h.update(f":{len(idx.cell_units[c])};".encode())
    h.update(f"|k={per_cell}|mb={micro_size}|acc={accumulation_steps}|rep={replacement_mode}".encode())
    return h.hexdigest()


# -------------------------------- task plans --------------------------------
def _materialize_task_plan(idx: UnitIndex, pop_sig, role, namespace, n_epochs, steps_per_epoch,
                           task_batch_size, base_seed, replacement_mode) -> TaskBatchPlan:
    sampler = MassUnitTaskSampler(idx, task_batch_size, derive_seed(base_seed, namespace),
                                  replacement_mode=replacement_mode)
    epochs = []
    t = 0
    for e in range(n_epochs):
        steps = []
        for s in range(steps_per_epoch):
            sids, ws, seed = sampler.step(t)
            steps.append(BatchStep(tuple(sids), tuple(ws), seed))
            t += 1
        epochs.append(tuple(steps))
    epochs = tuple(epochs)
    return TaskBatchPlan(role, pop_sig, epochs, _task_plan_hash(role, pop_sig, epochs))


def materialize_stage1_task_plan(idx, pop_sig, n_epochs, steps_per_epoch, task_batch_size, base_seed,
                                 replacement_mode="auto") -> TaskBatchPlan:
    return _materialize_task_plan(idx, pop_sig, "stage1_task", "stage1_task_sampler", n_epochs,
                                  steps_per_epoch, task_batch_size, base_seed, replacement_mode)


def materialize_stage2_task_plan(idx, pop_sig, n_epochs, steps_per_epoch, task_batch_size, base_seed,
                                 replacement_mode="auto") -> TaskBatchPlan:
    return _materialize_task_plan(idx, pop_sig, "stage2_task", "stage2_task_sampler", n_epochs,
                                  steps_per_epoch, task_batch_size, base_seed, replacement_mode)


# -------------------------------- alignment plans --------------------------------
def _aligned_plan_hash(pop_sig, role, sdh, warmup, game) -> str:
    h = hashlib.sha256()
    h.update(_alignment_plan_hash(pop_sig, warmup, game).encode())
    h.update(b"|"); h.update(role.encode()); h.update(b"|"); h.update(sdh.encode())   # bind role + design
    return h.hexdigest()


def _check_accumulation(lb, accumulation_steps, role):
    if accumulation_steps is not None and len(lb.microbatches) > accumulation_steps:
        raise ValueError(f"{role}: a logical batch needs {len(lb.microbatches)} microbatches > "
                         f"adv_accumulation_steps={accumulation_steps}; raise capacity")


def _materialize_alignment(sampler, idx, pop_sig, role, structure_hash, cells, per_cell, micro_size,
                           accumulation_steps, replacement_mode, warmup_steps, total_inner, critic_steps) -> AlignmentPlan:
    t = 0
    warmup = []
    for _ in range(warmup_steps):
        lb = _logical_from_step(sampler, t, micro_size); _check_accumulation(lb, accumulation_steps, role)
        warmup.append(lb); t += 1
    game = []
    for _ in range(total_inner):
        cbs = []
        for c in range(critic_steps):
            lb = _logical_from_step(sampler, t + c, micro_size); _check_accumulation(lb, accumulation_steps, role)
            cbs.append(lb)
        t += critic_steps
        enc = _logical_from_step(sampler, t, micro_size); _check_accumulation(enc, accumulation_steps, role); t += 1
        game.append(AlignmentGameStep(tuple(cbs), enc))
    warmup = tuple(warmup); game = tuple(game)
    sdh = _sampling_design_hash(pop_sig, role, structure_hash, cells, per_cell, micro_size,
                                accumulation_steps, replacement_mode, idx)
    return AlignmentPlan(warmup, game, pop_sig, _aligned_plan_hash(pop_sig, role, sdh, warmup, game),
                         role=role, sampling_design_hash=sdh)


def materialize_oaci_alignment_plan(idx: UnitIndex, support_graph, pop_sig, warmup_steps, total_inner,
                                    critic_steps, per_cell, micro_size, base_seed, accumulation_steps=None,
                                    replacement_mode="auto") -> AlignmentPlan:
    sampler = RareEligibleCellSampler(idx, support_graph, per_cell, derive_seed(base_seed, "oaci_alignment_sampler"),
                                      replacement_mode=replacement_mode)
    return _materialize_alignment(sampler, idx, pop_sig, "oaci_alignment", support_graph.support_hash(),
                                  sampler.cells, per_cell, micro_size, accumulation_steps, replacement_mode,
                                  warmup_steps, total_inner, critic_steps)


def materialize_full_domain_alignment_plan(idx: UnitIndex, pop_sig, warmup_steps, total_inner,
                                           critic_steps, per_cell, micro_size, base_seed,
                                           accumulation_steps=None, replacement_mode="auto") -> AlignmentPlan:
    sampler = ObservedCellSampler(idx, per_cell, derive_seed(base_seed, "full_domain_alignment_sampler"),
                                  replacement_mode=replacement_mode)
    cells = idx.observed_cells()
    struct = hashlib.sha256(("observed:" + ",".join(str(c) for c in cells)).encode()).hexdigest()
    return _materialize_alignment(sampler, idx, pop_sig, "full_domain_alignment", struct,
                                  cells, per_cell, micro_size, accumulation_steps, replacement_mode,
                                  warmup_steps, total_inner, critic_steps)
