"""Train the four methods from ONE shared ERM stage: ERM (no Stage-2), OACI (oaci_alignment),
global-LPC and uniform (the SAME full_domain_alignment object). Any method order yields the same
per-method results. Inactive methods take the engine's byte-exact ERM path (no factory)."""
from __future__ import annotations

import hashlib

import numpy as np

from ..methods import erm_result
from ..train.engine import train_stage2
from .objectives import make_objective
from .results import TrainedMethod

_STAGE2 = ("OACI", "global_lpc", "uniform")


def _alignment_unique_rows(plan) -> int:
    if plan is None:
        return 0
    return len({s for lb in plan.warmup_batches for mb in lb.microbatches for s in mb.sample_ids})


def _training_diagnostics(name, obj, spec, tr, level_plans, support_state):
    if name == "ERM":
        return {}
    if name == "OACI":
        if tr.active:
            return dict(obj.diagnostics())
        return {"active": False, "support_hash": support_state.support_hash,
                "eligible_alignment_rows": 0, "unique_eligible_alignment_rows": 0, "rejected_ineligible_rows": 0}
    pm = "-" if spec.prior_matrix is None else hashlib.sha256(
        np.ascontiguousarray(spec.prior_matrix).tobytes()).hexdigest()
    return {"active": tr.active, "observed_alignment_rows": _alignment_unique_rows(level_plans.full_domain_alignment),
            "domain_universe": list(range(len(support_state.cell_mass))), "prior_matrix_hash": pm}


def train_four_methods(method_order, stage1_run, level_population, level_plans, support_state,
                       fold_scope, execution_cfg, model_factory, engine_cfg, device):
    if sorted(method_order) != sorted(("ERM",) + _STAGE2):
        raise ValueError(f"method_order must be a permutation of the four methods; got {method_order}")
    erm_stage = stage1_run.erm_stage
    shared_erm = erm_stage.checkpoint.model_hash
    shared_tau = erm_stage.tau
    shared_s2 = level_plans.stage2_task.plan_hash
    data = level_population.training_data

    out = {}
    for name in method_order:
        obj, spec = make_objective(name, support_state, fold_scope, execution_cfg)
        if name == "ERM":
            tr = erm_result(erm_stage)
        else:
            align = level_plans.oaci_alignment if name == "OACI" else level_plans.full_domain_alignment
            tr = train_stage2(model_factory, erm_stage, data, obj, level_plans.stage2_task, align,
                              engine_cfg, device)
            if tr.task_plan_hash != shared_s2:
                raise RuntimeError(f"{name}: Stage-2 task plan hash != shared")
            if tr.active and name == "OACI":
                if tr.alignment_plan_hash != level_plans.oaci_alignment.plan_hash:
                    raise RuntimeError("OACI alignment plan hash mismatch")
                diag = obj.diagnostics()
                if diag["rejected_ineligible_rows"] != 0 or diag["support_hash"] != support_state.support_hash:
                    raise RuntimeError("OACI diagnostics violate the contract")
            if tr.active and name in ("global_lpc", "uniform"):
                if tr.alignment_plan_hash != level_plans.full_domain_alignment.plan_hash:
                    raise RuntimeError(f"{name}: full-domain alignment plan hash mismatch")
        if tr.initial_model_hash != shared_erm:
            raise RuntimeError(f"{name}: initial model hash != shared ERM")
        if abs(tr.erm_stage.tau - shared_tau) > 1e-12:
            raise RuntimeError(f"{name}: tau != shared tau")
        diag = _training_diagnostics(name, obj, spec, tr, level_plans, support_state)
        out[name] = TrainedMethod(method_name=name, active=spec.active, inactive_reason=spec.inactive_reason,
                                  shared_erm_hash=shared_erm, shared_tau=shared_tau,
                                  shared_stage2_task_plan_hash=shared_s2, objective_spec=spec, train_result=tr,
                                  training_diagnostics=diag)
    return out
