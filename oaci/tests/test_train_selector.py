"""Checkpoint selector: only feasible checkpoints are eligible, the lowest-leakage feasible one
is chosen, and an all-infeasible trajectory falls back byte-exactly to the ERM checkpoint.

Standalone (``python -m oaci.tests.test_train_selector``) and pytest-compatible.
"""
from __future__ import annotations

import torch

from oaci.train.primal_dual import CheckpointRecord, TrainConfig, TrainResult
from oaci.train.selector import select_checkpoint, state_hash


def _ckpt(epoch, R, leak):
    return CheckpointRecord(
        epoch=epoch,
        enc_state={"w": torch.tensor([float(epoch)])},
        head_state={"w": torch.tensor([0.0])},
        R_src=R, balanced_err=0.0, leakage_surrogate=leak, lam=0.0,
    )


def _result(traj, tau, R_erm, erm_leak=1.0):
    erm = {"enc": {"w": torch.tensor([-1.0])}, "head": {"w": torch.tensor([-2.0])}}
    erm_rec = CheckpointRecord(epoch=-1, enc_state=erm["enc"], head_state=erm["head"],
                               R_src=R_erm, balanced_err=0.0, leakage_surrogate=erm_leak, lam=0.0)
    return TrainResult(
        erm_ckpt=erm, R_ERM_hat=R_erm, tau=tau, H_ref_bar=0.0, erm_record=erm_rec,
        trajectory=traj, in_dim=1, cfg=TrainConfig(numerical_tol=1e-4),
    )


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
    assert state_hash(sel.enc_state) == state_hash(_result(traj, tau, 0.40).erm_ckpt["enc"])


def test_all_infeasible_candidates_fall_back_exactly_to_erm():
    tau = 0.5
    traj = [_ckpt(0, R=0.90, leak=-1.0), _ckpt(1, R=0.80, leak=-2.0)]  # all infeasible
    res = _result(traj, tau, 0.49)
    sel = select_checkpoint(res)
    assert sel.used_erm_fallback and sel.selected_erm and sel.selected_epoch == -1
    assert sel.selection_reason == "no_stage2_feasible"
    assert sel.n_feasible == 0
    assert state_hash(sel.enc_state) == state_hash(res.erm_ckpt["enc"])   # byte-exact ERM
    assert state_hash(sel.head_state) == state_hash(res.erm_ckpt["head"])
    assert sel.R_src == res.R_ERM_hat


def test_injected_score_fn_overrides_default_ranking():
    tau = 0.5
    traj = [_ckpt(0, R=0.40, leak=0.10), _ckpt(1, R=0.40, leak=0.90)]
    # default ranks by leakage_surrogate -> epoch 0; an injected score_fn (maximise leakage)
    # flips it to epoch 1 (ERM leak=0.5 stays between, so it does not win either way)
    sel = select_checkpoint(_result(traj, tau, 0.40, erm_leak=0.5), score_fn=lambda c: -c.leakage_surrogate)
    assert sel.selected_epoch == 1


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-selector tests")


if __name__ == "__main__":
    _run_all()
