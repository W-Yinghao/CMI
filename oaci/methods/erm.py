"""ERM method: the frozen Stage-1 checkpoint as a TrainResult. It does NOT call train_stage2 and
builds neither a critic nor an optimizer — the selector simply restores the shared ERM checkpoint.
"""
from __future__ import annotations

from ..train.checkpoint import CheckpointRecord, TrainResult


def erm_result(erm_stage) -> TrainResult:
    c = erm_stage.checkpoint
    rec = CheckpointRecord(epoch=-1, optimizer_step=0, model_state=c.model_state, model_hash=c.model_hash,
                           R_src=erm_stage.R_ERM_hat, balanced_err=c.balanced_err, train_surrogate=0.0, lam=0.0)
    return TrainResult(method_name="ERM", active=True, inactive_reason=None, erm_stage=erm_stage,
                       erm_record=rec, trajectory=[], initial_model_hash=c.model_hash,
                       task_plan_hash=erm_stage.task_plan_hash, alignment_plan_hash=None)
