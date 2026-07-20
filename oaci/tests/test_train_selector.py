"""Checkpoint selector: only feasible checkpoints are eligible, the lowest-leakage feasible one
is chosen, and an all-infeasible trajectory falls back byte-exactly to the ERM checkpoint.

Standalone (``python -m oaci.tests.test_train_selector``) and pytest-compatible.
"""
from __future__ import annotations

import torch

from oaci.train.checkpoint import CheckpointRecord, ERMStage, TrainResult
from oaci.train.selector import select_checkpoint, state_hash


def _ckpt(epoch, R, leak):
    state = {"w": torch.tensor([float(epoch)])}
    return CheckpointRecord(epoch=epoch, optimizer_step=epoch + 1, model_state=state,
                            model_hash=state_hash(state), R_src=R, balanced_err=0.0,
                            train_surrogate=leak, lam=0.0)


def _result(traj, tau, R_erm, erm_leak=1.0):
    erm_state = {"w": torch.tensor([-1.0])}
    erm_rec = CheckpointRecord(epoch=-1, optimizer_step=0, model_state=erm_state,
                               model_hash=state_hash(erm_state), R_src=R_erm, balanced_err=0.0,
                               train_surrogate=erm_leak, lam=0.0)
    erm_stage = ERMStage(checkpoint=erm_rec, R_ERM_hat=R_erm, tau=tau, task_plan_hash="t",
                         stage1_invocation_id="inv")
    return TrainResult(method_name="OACI", active=True, inactive_reason=None, erm_stage=erm_stage,
                       erm_record=erm_rec, trajectory=traj, initial_model_hash=erm_rec.model_hash,
                       task_plan_hash="t", alignment_plan_hash="a")


def test_only_feasible_checkpoint_is_selected():
    tau = 0.5
    traj = [
        _ckpt(0, R=0.90, leak=-1.0),    # infeasible but lowest leakage -> must NOT be chosen
        _ckpt(1, R=0.40, leak=0.50),    # feasible
        _ckpt(2, R=0.45, leak=0.20),    # feasible, lower leakage -> winner
    ]
    sel = select_checkpoint(_result(traj, tau, 0.45, erm_leak=1.0))   # ERM leakage worse than 0.2
    assert not sel.used_erm_fallback and not sel.selected_erm
    assert sel.selection_reason == "stage2_best"
    assert sel.selected_epoch == 2
    assert sel.n_feasible == 2
    assert sel.R_src <= tau + 1e-4


def test_feasible_but_worse_leakage_selects_erm():
    tau = 0.5
    # every Stage-2 checkpoint is risk-FEASIBLE but has WORSE (higher) leakage than ERM
    traj = [_ckpt(0, R=0.40, leak=0.80), _ckpt(1, R=0.45, leak=0.60)]
    sel = select_checkpoint(_result(traj, tau, 0.40, erm_leak=0.30))   # ERM leakage best
    assert not sel.used_erm_fallback                # Stage-2 WAS feasible...
    assert sel.selected_erm and sel.selection_reason == "erm_best"   # ...but ERM scored best
    assert sel.selected_epoch == -1
    assert sel.model_hash == _result(traj, tau, 0.40).erm_record.model_hash


def test_all_infeasible_candidates_fall_back_exactly_to_erm():
    tau = 0.5
    traj = [_ckpt(0, R=0.90, leak=-1.0), _ckpt(1, R=0.80, leak=-2.0)]  # all infeasible
    res = _result(traj, tau, 0.49)
    sel = select_checkpoint(res)
    assert sel.used_erm_fallback and sel.selected_erm and sel.selected_epoch == -1
    assert sel.selection_reason == "no_stage2_feasible"
    assert sel.n_feasible == 0
    assert sel.model_hash == res.erm_record.model_hash                   # byte-exact ERM
    assert state_hash(sel.model_state) == state_hash(res.erm_record.model_state)
    assert sel.R_src == res.R_ERM_hat


def test_injected_score_fn_overrides_default_ranking():
    tau = 0.5
    traj = [_ckpt(0, R=0.40, leak=0.10), _ckpt(1, R=0.40, leak=0.90)]
    # default ranks by train_surrogate -> epoch 0; an injected score_fn (maximise leakage)
    # flips it to epoch 1 (ERM leak=0.5 stays between, so it does not win either way)
    sel = select_checkpoint(_result(traj, tau, 0.40, erm_leak=0.5), score_fn=lambda c: -c.train_surrogate)
    assert sel.selected_epoch == 1


def test_erm_wins_exact_or_tolerance_ties():
    tau = 0.5
    # exact tie (Stage-2 score == ERM score) -> ERM kept
    sel = select_checkpoint(_result([_ckpt(0, R=0.4, leak=0.50)], tau, 0.4, erm_leak=0.50))
    assert sel.selected_erm and sel.selection_reason == "erm_best"
    # within tolerance (Stage-2 only 0.01 better, tol 0.05) -> ERM kept
    sel = select_checkpoint(_result([_ckpt(0, R=0.4, leak=0.49)], tau, 0.4, erm_leak=0.50),
                            selection_score_tolerance=0.05)
    assert sel.selected_erm
    # beats ERM by MORE than tolerance -> Stage-2 chosen
    sel = select_checkpoint(_result([_ckpt(0, R=0.4, leak=0.40)], tau, 0.4, erm_leak=0.50),
                            selection_score_tolerance=0.05)
    assert not sel.selected_erm and sel.selection_reason == "stage2_best"


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-selector tests")


if __name__ == "__main__":
    _run_all()
