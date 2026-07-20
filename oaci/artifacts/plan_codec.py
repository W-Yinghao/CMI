"""Exact codecs for leakage design + task / alignment / fold / bootstrap plans.

Each codec round-trips the REAL object (not just its hash) and re-validates the object's native hash
via the public validators. ``$`` -tagged arrays go to deterministic NPZ; structure/offsets and hashes
go to canonical JSON.
"""
from __future__ import annotations

import numpy as np

from ..leakage.crossfit import FoldPlan, validate_fold_plan
from ..leakage.design import LeakageDesign, population_hash as _design_population_hash
from ..leakage.plan import BootstrapDraw, LeakageBootstrapPlan, validate_bootstrap_plan
from ..train.batch_plan import (AlignmentGameStep, AlignmentPlan, BatchStep, LogicalBatchPlan,
                                MicrobatchPlan, TaskBatchPlan, validate_alignment_plan, validate_task_plan)
from .deterministic_npz import to_unicode_array

DESIGN_KIND, TASK_KIND, ALIGN_KIND, FOLD_KIND, BOOTSTRAP_KIND = (
    "leakage_design", "task_plan", "alignment_plan", "fold_plan", "bootstrap_plan")


def _i64(seq):
    return np.ascontiguousarray(np.asarray(list(seq), dtype=np.int64))


def _f64(seq):
    return np.ascontiguousarray(np.asarray(list(seq), dtype=np.float64))


# ---------------------------------- leakage design ----------------------------------
def encode_design(d: LeakageDesign) -> tuple:
    body = {"population_hash": d.population_hash, "support_hash": d.support_hash}
    arrays = {"sample_id": to_unicode_array(d.sample_id), "y": _i64(d.y.tolist()), "d": _i64(d.d.tolist()),
              "group_id": to_unicode_array(d.group_id), "sample_mass": _f64(d.sample_mass.tolist())}
    return d.population_hash, body, arrays


def decode_design(body, arrays) -> LeakageDesign:
    sid = tuple(str(s) for s in arrays["sample_id"].tolist())
    grp = tuple(str(g) for g in arrays["group_id"].tolist())
    d = LeakageDesign(sample_id=sid, y=np.asarray(arrays["y"]), d=np.asarray(arrays["d"]), group_id=grp,
                      sample_mass=np.asarray(arrays["sample_mass"]), population_hash=body["population_hash"],
                      support_hash=body["support_hash"])
    if _design_population_hash(sid, d.y, d.d, grp, d.sample_mass) != d.population_hash:
        raise ValueError("decoded leakage design population hash does not recompute")
    return d


# ---------------------------------- task plan ----------------------------------
def encode_task_plan(p: TaskBatchPlan) -> tuple:
    steps = [st for ep in p.epochs for st in ep]
    body = {"role": p.role, "population_signature_hash": p.population_signature_hash,
            "n_epochs": len(p.epochs), "n_steps": len(steps), "plan_hash": p.plan_hash}
    arrays = {"flat_sample_id": to_unicode_array([s for st in steps for s in st.sample_ids]),
              "flat_weight": _f64([w for st in steps for w in st.importance_weights]),
              "step_seed": _i64([st.step_seed for st in steps]),
              "step_len": _i64([len(st.sample_ids) for st in steps]),
              "epoch_len": _i64([len(ep) for ep in p.epochs])}
    return p.plan_hash, body, arrays


def decode_task_plan(body, arrays) -> TaskBatchPlan:
    sid = [str(s) for s in arrays["flat_sample_id"].tolist()]
    w = arrays["flat_weight"].tolist()
    seeds = arrays["step_seed"].tolist(); slen = arrays["step_len"].tolist(); elen = arrays["epoch_len"].tolist()
    steps, off = [], 0
    for k, n in enumerate(slen):
        steps.append(BatchStep(sample_ids=tuple(sid[off:off + n]),
                               importance_weights=tuple(float(x) for x in w[off:off + n]), step_seed=int(seeds[k])))
        off += n
    epochs, j = [], 0
    for ns in elen:
        epochs.append(tuple(steps[j:j + ns])); j += ns
    p = TaskBatchPlan(role=body["role"], population_signature_hash=body["population_signature_hash"],
                      epochs=tuple(epochs), plan_hash=body["plan_hash"])
    validate_task_plan(p)
    return p


# ---------------------------------- alignment plan ----------------------------------
def encode_alignment_plan(p: AlignmentPlan) -> tuple:
    logical = list(p.warmup_batches)
    critic_counts = [len(gs.critic_batches) for gs in p.game_steps]
    for gs in p.game_steps:
        logical.extend(gs.critic_batches); logical.append(gs.encoder_batch)
    mbs = [mb for lb in logical for mb in lb.microbatches]
    body = {"role": p.role, "sampling_design_hash": p.sampling_design_hash,
            "population_signature_hash": p.population_signature_hash, "plan_hash": p.plan_hash,
            "n_warmup": len(p.warmup_batches)}
    arrays = {"critic_counts": _i64(critic_counts), "lb_step_seed": _i64([lb.step_seed for lb in logical]),
              "lb_n_microbatches": _i64([len(lb.microbatches) for lb in logical]),
              "mb_len": _i64([len(mb.sample_ids) for mb in mbs]),
              "flat_sample_id": to_unicode_array([s for mb in mbs for s in mb.sample_ids]),
              "flat_weight": _f64([w for mb in mbs for w in mb.importance_weights])}
    return p.plan_hash, body, arrays


def _rebuild_logical(lb_seeds, lb_nmb, mb_len, sid, w):
    mbs, off = [], 0
    for n in mb_len:
        mbs.append(MicrobatchPlan(sample_ids=tuple(sid[off:off + n]),
                                  importance_weights=tuple(float(x) for x in w[off:off + n]))); off += n
    logical, j = [], 0
    for i, nmb in enumerate(lb_nmb):
        logical.append(LogicalBatchPlan(microbatches=tuple(mbs[j:j + nmb]), step_seed=int(lb_seeds[i]))); j += nmb
    return logical


def decode_alignment_plan(body, arrays) -> AlignmentPlan:
    sid = [str(s) for s in arrays["flat_sample_id"].tolist()]
    logical = _rebuild_logical(arrays["lb_step_seed"].tolist(), arrays["lb_n_microbatches"].tolist(),
                               arrays["mb_len"].tolist(), sid, arrays["flat_weight"].tolist())
    nw = int(body["n_warmup"])
    warmup = tuple(logical[:nw]); rest = logical[nw:]
    game, c = [], 0
    for nc in arrays["critic_counts"].tolist():
        crit = tuple(rest[c:c + nc]); enc = rest[c + nc]; c += nc + 1
        game.append(AlignmentGameStep(critic_batches=crit, encoder_batch=enc))
    p = AlignmentPlan(warmup_batches=warmup, game_steps=tuple(game),
                      population_signature_hash=body["population_signature_hash"], plan_hash=body["plan_hash"],
                      role=body["role"], sampling_design_hash=body["sampling_design_hash"])
    validate_alignment_plan(p)
    return p


# ---------------------------------- fold plan ----------------------------------
def encode_fold_plan(p: FoldPlan) -> tuple:
    groups = sorted(p.fold_of_group)
    body = {"n_folds": int(p.n_folds), "n_folds_requested": int(p.n_folds_requested),
            "notes": [str(n) for n in p.notes], "population_hash": p.population_hash,
            "support_hash": p.support_hash, "plan_hash": p.plan_hash}
    arrays = {"group": to_unicode_array(groups), "fold": _i64([p.fold_of_group[g] for g in groups]),
              "domain": _i64([p.domain_of_group[g] for g in groups])}
    return p.plan_hash, body, arrays


def decode_fold_plan(body, arrays) -> FoldPlan:
    groups = [str(g) for g in arrays["group"].tolist()]
    fold_of = {g: int(f) for g, f in zip(groups, arrays["fold"].tolist())}
    dom_of = {g: int(d) for g, d in zip(groups, arrays["domain"].tolist())}
    p = FoldPlan(fold_of_group=fold_of, n_folds=int(body["n_folds"]), n_folds_requested=int(body["n_folds_requested"]),
                 domain_of_group=dom_of, notes=list(body["notes"]), population_hash=body["population_hash"],
                 support_hash=body["support_hash"], plan_hash=body["plan_hash"])
    validate_fold_plan(p)
    return p


# ---------------------------------- bootstrap plan ----------------------------------
def encode_bootstrap_plan(p: LeakageBootstrapPlan) -> tuple:
    draws = list(p.candidate_draws)
    groups = [g for g, _ in draws[0].group_multiplicities] if draws else []
    mult = np.zeros((len(draws), len(groups)), dtype=np.int64)
    for i, dr in enumerate(draws):
        mult[i] = [m for _, m in dr.group_multiplicities]
    body = {"alpha": float(p.alpha), "requested_replicates": int(p.requested_replicates),
            "invalid_draw_rate": float(p.invalid_draw_rate), "population_hash": p.population_hash,
            "support_hash": p.support_hash, "fold_plan_hash": p.fold_plan_hash, "plan_hash": p.plan_hash,
            "n_draws": len(draws), "n_groups": len(groups)}
    arrays = {"group": to_unicode_array(groups), "candidate_ids": _i64([dr.candidate_id for dr in draws]),
              "mult_matrix": np.ascontiguousarray(mult),
              "accepted_ids": _i64(list(p.accepted_candidate_ids))}
    return p.plan_hash, body, arrays


def decode_bootstrap_plan(body, arrays) -> LeakageBootstrapPlan:
    groups = [str(g) for g in arrays["group"].tolist()]
    cand = arrays["candidate_ids"].tolist()
    mult = arrays["mult_matrix"]
    if mult.size == 0:
        mult = mult.reshape(len(cand), len(groups))
    draws = tuple(BootstrapDraw(candidate_id=int(cand[i]),
                                group_multiplicities=tuple((groups[j], int(mult[i, j])) for j in range(len(groups))))
                  for i in range(len(cand)))
    p = LeakageBootstrapPlan(population_hash=body["population_hash"], support_hash=body["support_hash"],
                             fold_plan_hash=body["fold_plan_hash"], alpha=float(body["alpha"]),
                             requested_replicates=int(body["requested_replicates"]), candidate_draws=draws,
                             accepted_candidate_ids=tuple(int(x) for x in arrays["accepted_ids"].tolist()),
                             invalid_draw_rate=float(body["invalid_draw_rate"]), plan_hash=body["plan_hash"])
    validate_bootstrap_plan(p)
    return p
