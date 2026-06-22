"""Fold scope: the level-invariant identity (maps, level-0 prior, deletion schedule, source-audit
scope, target signatures) + the per-level source-train population (TrainingData + LeakageDesign +
UnitIndex, with a byte-exact shared population hash). Nothing here depends on the model seed.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass

import numpy as np
import torch

from ..data.plan_sampler import UnitIndex
from ..leakage.design import LeakageDesign, make_leakage_design
from ..leakage.errors import LeakageNonEstimableError, nonestimable_status
from ..leakage.crossfit import FoldPlan, make_fold_plan_from_design
from ..leakage.plan import LeakageBootstrapPlan, make_leakage_bootstrap_plan
from ..support_graph import build_support_graph
from ..train.data import TrainingData, population_signature_hash
from .keys import canonical_json_hash, feed_float64, feed_int64, feed_string
from .support import level0_reference_prior


def _tensor_hash(pop_hash, X, order) -> str:
    h = hashlib.sha256(); h.update(pop_hash.encode())
    h.update(str(X.dtype).encode()); h.update(str(tuple(X.shape[1:])).encode())
    for i in order:
        h.update(np.ascontiguousarray(X[i].detach().cpu().numpy()).tobytes())
    return h.hexdigest()


def _ro(a):
    a = np.array(a, copy=True); a.setflags(write=False); return a


# -------------------------------- per-level population --------------------------------
@dataclass(frozen=True)
class LevelPopulation:
    training_data: TrainingData
    leakage_design: LeakageDesign
    unit_index: UnitIndex
    population_hash: str
    tensor_hash: str
    population_config_hash: str


def build_level_population(fold_data, maps, support_state) -> LevelPopulation:
    rows = support_state.source_train_idx.tolist()                  # canonical (build_level_support sorted)
    sids = tuple(fold_data.sample_id[i] for i in rows)
    y = np.array([int(fold_data.y[i]) for i in rows], dtype=np.int64)
    d = np.array([maps.source_domain_to_index[fold_data.domain_id[i]] for i in rows], dtype=np.int64)
    grp = tuple(fold_data.group_id[i] for i in rows)
    mu = tuple(fold_data.mass_unit_id[i] for i in rows)
    mass = np.array([float(fold_data.sample_mass[i]) for i in rows], dtype=np.float64)
    X = fold_data.X[torch.as_tensor(rows, dtype=torch.long)]
    td = TrainingData(X=X, y=torch.as_tensor(y), sample_id=sids,
                      sample_mass=torch.as_tensor(mass), n_classes=len(maps.class_names),
                      d=torch.as_tensor(d), group=grp).validate()
    design = make_leakage_design(sids, y, d, grp, mass, support_state.support_graph)
    idx = UnitIndex(sids, y, d, grp, mu, mass)
    pop = population_signature_hash(td)
    if pop != design.population_hash:
        raise ValueError("TrainingData and LeakageDesign population hashes disagree")
    order = sorted(range(len(sids)), key=lambda i: sids[i])
    th = _tensor_hash(pop, X, order)
    cfg = hashlib.sha256()
    feed_string(cfg, pop); feed_string(cfg, maps.maps_hash); feed_string(cfg, support_state.support_hash)
    feed_string(cfg, idx.design_hash())
    return LevelPopulation(td, design, idx, pop, th, cfg.hexdigest())


# -------------------------------- source-audit scope --------------------------------
@dataclass(frozen=True)
class AuditScope:
    support_graph: object
    design: LeakageDesign
    reference_prior: np.ndarray
    fold_plan: FoldPlan | None
    bootstrap_plan: LeakageBootstrapPlan | None
    status: str
    domain_ids: tuple
    domain_to_index_items: tuple
    data_population_hash: str
    leakage_population_hash: str
    tensor_hash: str
    audit_scope_hash: str


def build_audit_scope(fold_data, maps, cfg, fold_key) -> AuditScope:
    rows = sorted(fold_data.source_audit_idx.tolist(), key=lambda i: fold_data.sample_id[i])
    sids = tuple(fold_data.sample_id[i] for i in rows)
    audit_doms = tuple(sorted({fold_data.domain_id[i] for i in rows}))
    dmap = {dd: i for i, dd in enumerate(audit_doms)}
    y = np.array([int(fold_data.y[i]) for i in rows], dtype=np.int64)
    d = np.array([dmap[fold_data.domain_id[i]] for i in rows], dtype=np.int64)
    grp = tuple(fold_data.group_id[i] for i in rows)
    mass = np.array([float(fold_data.sample_mass[i]) for i in rows], dtype=np.float64)
    nd, nc = len(audit_doms), len(maps.class_names)
    counts = np.zeros((nd, nc), dtype=np.int64); cmass = np.zeros((nd, nc), dtype=np.float64)
    units: dict = {}
    for k, i in enumerate(rows):
        units.setdefault((int(d[k]), int(y[k])), set()).add(fold_data.support_unit_id[i])
        cmass[int(d[k]), int(y[k])] += float(mass[k])
    for (dd, yy), us in units.items():
        counts[dd, yy] = len(us)
    prior = _ro(cmass.sum(axis=0) / cmass.sum())
    sg = build_support_graph(eligibility_counts=counts, cell_mass=cmass, m=int(cfg.support_m),
                             reference_prior=prior, domain_names=list(audit_doms),
                             class_names=list(maps.class_names))
    design = make_leakage_design(sids, y, d, grp, mass, sg)          # ALWAYS built
    fold = boot = None; status = "estimable"
    from ..train.rng import derive_seed
    try:
        fold = make_fold_plan_from_design(design, sg, n_folds=cfg.probe_folds,
                                          seed=derive_seed(cfg.audit_seed, "audit_fold", fold_key.fold_key_hash))
        boot = make_leakage_bootstrap_plan(design, sg, fold, alpha=cfg.leakage_alpha,
                                           requested_replicates=cfg.audit_bootstrap_replicates,
                                           seed=derive_seed(cfg.audit_seed, "audit_bootstrap", fold_key.fold_key_hash),
                                           max_candidate_multiplier=cfg.max_candidate_multiplier,
                                           max_invalid_draw_rate=cfg.max_invalid_draw_rate)
    except LeakageNonEstimableError as e:
        status = nonestimable_status(e); boot = None
    # reference FoldData's frozen source-audit identity — never invent a second one
    if set(design.sample_id) != set(fold_data.role_ids("source_audit")):
        raise ValueError("audit design sample ids != FoldData source_audit ids")
    dp = fold_data.source_audit_population_hash
    th = fold_data.source_audit_tensor_hash
    h = hashlib.sha256()
    for v in (dp, design.population_hash, th, sg.support_hash(), status):
        feed_string(h, v)
    h.update(np.ascontiguousarray(prior).tobytes())
    return AuditScope(sg, design, prior, fold, boot, status, audit_doms,
                      tuple(dmap.items()), dp, design.population_hash, th, h.hexdigest())


# -------------------------------- scope config --------------------------------
@dataclass(frozen=True)
class ScopePlanConfig:
    support_m: int
    leakage_alpha: float
    probe_folds: int
    probe_capacities: tuple
    l2_C: float
    max_iter: int
    prob_floor: float
    feature_seed_base: int
    selection_bootstrap_replicates: int
    audit_bootstrap_replicates: int
    max_candidate_multiplier: int
    max_invalid_draw_rate: float
    stage1_epochs: int
    stage2_epochs: int
    stage1_steps_per_epoch: int
    stage2_steps_per_epoch: int
    task_batch_size: int
    warmup_steps: int
    critic_steps: int
    min_per_eligible_cell: int
    min_per_observed_cell: int
    adv_microbatch_size: int
    adv_accumulation_steps: int
    replacement_mode: str
    selection_seed: int
    audit_seed: int

    @property
    def config_hash(self) -> str:
        return canonical_json_hash(asdict(self))

    @staticmethod
    def from_manifest(m, support_m) -> "ScopePlanConfig":
        o, t, p, s, e, r, sd = m.optimizer, m.training, m.probe, m.sampler, m.evaluation, m.risk, m.seeds
        return ScopePlanConfig(
            support_m=int(support_m), leakage_alpha=float(e.alpha), probe_folds=int(p.folds),
            probe_capacities=tuple(int(c) for c in p.capacities), l2_C=float(p.l2_C), max_iter=int(p.max_iter),
            prob_floor=float(p.prob_floor), feature_seed_base=int(p.feature_seed_base),
            selection_bootstrap_replicates=int(p.selection_bootstrap),
            audit_bootstrap_replicates=int(p.audit_bootstrap),
            max_candidate_multiplier=int(p.max_candidate_multiplier),
            max_invalid_draw_rate=float(p.max_invalid_draw_rate),
            stage1_epochs=int(t.stage1_epochs), stage2_epochs=int(t.stage2_epochs),
            stage1_steps_per_epoch=int(t.stage1_steps_per_epoch), stage2_steps_per_epoch=int(t.stage2_steps_per_epoch),
            task_batch_size=int(t.task_batch_size), warmup_steps=int(t.warmup_steps), critic_steps=int(t.critic_steps),
            min_per_eligible_cell=int(s.min_per_eligible_cell), min_per_observed_cell=int(s.min_per_observed_cell),
            adv_microbatch_size=int(s.adv_microbatch_size), adv_accumulation_steps=int(s.adv_accumulation_steps),
            replacement_mode=str(s.replacement_mode), selection_seed=int(sd.selection_bootstrap),
            audit_seed=int(sd.audit_bootstrap))


# -------------------------------- fold scope --------------------------------
@dataclass(frozen=True)
class FoldScope:
    fold_key: object
    maps: object
    level0_reference_prior: np.ndarray
    deletion_schedule: object
    source_audit: AuditScope
    target_population_hash: str
    target_tensor_hash: str
    data_contract_hash: str
    scope_config_hash: str
    fold_scope_hash: str


def build_fold_scope(fold_key, maps, fold_data, schedule, cfg: ScopePlanConfig) -> FoldScope:
    ref = _ro(level0_reference_prior(fold_data, maps))
    audit = build_audit_scope(fold_data, maps, cfg, fold_key)
    h = hashlib.sha256()
    for v in (fold_key.fold_key_hash, maps.maps_hash, schedule.schedule_hash, audit.audit_scope_hash,
              fold_data.target_population_hash, fold_data.target_tensor_hash, fold_data.data_contract_hash,
              cfg.config_hash):
        feed_string(h, v)
    h.update(np.ascontiguousarray(ref).tobytes())
    return FoldScope(fold_key, maps, ref, schedule, audit, fold_data.target_population_hash,
                     fold_data.target_tensor_hash, fold_data.data_contract_hash, cfg.config_hash, h.hexdigest())
