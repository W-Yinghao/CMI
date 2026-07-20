"""Stage-1 once per level. The InvocationRegistry is OWNED by the caller (level orchestrator), so a
repeat call would be caught; this helper must be invoked exactly once per level."""
from __future__ import annotations

import math

from ..train.checkpoint import state_hash
from ..train.engine import train_stage1
from ..train.rng import derive_seed, forked_rng
from .keys import canonical_json_hash
from .results import Stage1Run


def stage1_invocation_id(run_key, model_spec, stage1_plan, execution_config_hash) -> str:
    return canonical_json_hash({"run_key": run_key.run_key_hash, "model_spec": model_spec.model_spec_hash,
                                "stage1_plan": stage1_plan.plan_hash, "exec_cfg": execution_config_hash})


def run_stage1_once(run_key, level_population, level_plans, model_factory, model_spec, engine_cfg,
                    execution_config_hash, registry, device) -> Stage1Run:
    inv = stage1_invocation_id(run_key, model_spec, level_plans.stage1_task, execution_config_hash)
    with forked_rng(derive_seed(run_key.model_seed, "model_init", run_key.optimization_identity_hash,
                                model_spec.model_spec_hash), device):
        model = model_factory()
    erm_stage = train_stage1(model, level_population.training_data, level_plans.stage1_task, engine_cfg,
                             device, registry, inv)
    if registry.count(inv) != 1 or registry.total_claims < 1:
        raise RuntimeError("Stage-1 invocation count is not 1")
    if state_hash(erm_stage.checkpoint.model_state) != erm_stage.checkpoint.model_hash:
        raise RuntimeError("ERM checkpoint hash does not recompute")
    if erm_stage.task_plan_hash != level_plans.stage1_task.plan_hash:
        raise RuntimeError("ERMStage task plan hash != Stage-1 plan hash")
    if not (math.isfinite(erm_stage.R_ERM_hat) and math.isfinite(erm_stage.tau)):
        raise RuntimeError("R_ERM_hat / tau not finite")
    if abs(erm_stage.tau - (erm_stage.R_ERM_hat + engine_cfg.epsilon)) > 1e-9:
        raise RuntimeError("tau != R_ERM_hat + epsilon")
    return Stage1Run(erm_stage=erm_stage, invocation_id=inv, invocation_count=registry.count(inv),
                     registry_total_claims=registry.total_claims, model_spec_hash=model_spec.model_spec_hash,
                     engine_base_seed=engine_cfg.base_seed)
