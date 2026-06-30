"""Per-level training + selection plans.

Training plans (Stage-1 / Stage-2 task + OACI / full-domain alignment) depend on the model seed via
independent sampler namespaces. Selection plans (design / fold / bootstrap) depend ONLY on the fold
key, the deletion level, the current support and the manifest selection seed — never the model seed.
A structural selection failure sets a typed status and None fold/bootstrap plans; the design is
always kept.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

from ..data.plan_materialize import (materialize_full_domain_alignment_plan,
                                     materialize_oaci_alignment_plan, materialize_stage1_task_plan,
                                     materialize_stage2_task_plan)
from ..leakage.crossfit import make_fold_plan_from_design
from ..leakage.errors import LeakageNonEstimableError, nonestimable_status
from ..leakage.plan import make_leakage_bootstrap_plan
from ..train.rng import derive_seed
from .keys import feed_int64, feed_string


@dataclass(frozen=True)
class LevelPlans:
    level_population: object
    stage1_task: object
    stage2_task: object
    oaci_alignment: object | None
    full_domain_alignment: object | None
    selection_design: object
    selection_fold_plan: object | None
    selection_bootstrap_plan: object | None
    selection_status: str
    training_plans_hash: str
    selection_plans_hash: str
    level_plans_hash: str


def build_level_plans(fold_scope, level, support_state, level_population, cfg, model_seed) -> LevelPlans:
    idx = level_population.unit_index
    pop = level_population.population_hash
    fkh = fold_scope.fold_key.fold_key_hash
    ofkh = fold_scope.fold_key.optimization_fold_hash               # bootstrap/eval-independent training id
    status = {n: st for n, st in support_state.method_status_items}
    total_inner = cfg.stage2_epochs * cfg.stage2_steps_per_epoch

    tb = derive_seed(model_seed, "training", ofkh, level)           # model-seed dependent; NOT bootstrap-dependent
    s1 = materialize_stage1_task_plan(idx, pop, cfg.stage1_epochs, cfg.stage1_steps_per_epoch,
                                      cfg.task_batch_size, tb, replacement_mode=cfg.replacement_mode)
    s2 = materialize_stage2_task_plan(idx, pop, cfg.stage2_epochs, cfg.stage2_steps_per_epoch,
                                      cfg.task_batch_size, tb, replacement_mode=cfg.replacement_mode)
    for plan in (s1, s2):
        if plan.population_signature_hash != pop:
            raise ValueError("task plan population hash != level population hash")

    oaci = None
    if status["OACI"].active:
        oaci = materialize_oaci_alignment_plan(idx, support_state.support_graph, pop, cfg.warmup_steps,
                                               total_inner, cfg.critic_steps, cfg.min_per_eligible_cell,
                                               cfg.adv_microbatch_size, tb,
                                               accumulation_steps=cfg.adv_accumulation_steps,
                                               replacement_mode=cfg.replacement_mode)
    full = None
    if status["global_lpc"].active:                                 # global_lpc & uniform share this object
        full = materialize_full_domain_alignment_plan(idx, pop, cfg.warmup_steps, total_inner,
                                                      cfg.critic_steps, cfg.min_per_observed_cell,
                                                      cfg.adv_microbatch_size, tb,
                                                      accumulation_steps=cfg.adv_accumulation_steps,
                                                      replacement_mode=cfg.replacement_mode)

    design = level_population.leakage_design
    sel_fold = sel_boot = None; sel_status = "estimable"
    try:
        sel_fold = make_fold_plan_from_design(design, support_state.support_graph, n_folds=cfg.probe_folds,
                                              seed=derive_seed(cfg.selection_seed, "selection_fold", fkh, level))
        sel_boot = make_leakage_bootstrap_plan(design, support_state.support_graph, sel_fold,
                                               alpha=cfg.leakage_alpha,
                                               requested_replicates=cfg.selection_bootstrap_replicates,
                                               seed=derive_seed(cfg.selection_seed, "selection_bootstrap", fkh, level),
                                               max_candidate_multiplier=cfg.max_candidate_multiplier,
                                               max_invalid_draw_rate=cfg.max_invalid_draw_rate)
    except LeakageNonEstimableError as ex:
        sel_status = nonestimable_status(ex); sel_boot = None

    th = hashlib.sha256()
    for v in (s1.plan_hash, s2.plan_hash, oaci.plan_hash if oaci else "none",
              full.plan_hash if full else "none"):
        feed_string(th, v)
    feed_int64(th, int(model_seed))                                 # bind model seed explicitly
    training_hash = th.hexdigest()
    sh = hashlib.sha256()
    for v in (design.population_hash, sel_fold.plan_hash if sel_fold else "none",
              sel_boot.plan_hash if sel_boot else "none", sel_status):
        feed_string(sh, v)
    feed_int64(sh, int(level))
    selection_hash = sh.hexdigest()
    lh = hashlib.sha256(); feed_string(lh, pop); feed_string(lh, training_hash); feed_string(lh, selection_hash)
    return LevelPlans(level_population, s1, s2, oaci, full, design, sel_fold, sel_boot, sel_status,
                      training_hash, selection_hash, lh.hexdigest())
