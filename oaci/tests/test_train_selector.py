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


def _result(traj, tau, R_erm):
    erm = {"enc": {"w": torch.tensor([-1.0])}, "head": {"w": torch.tensor([-2.0])}}
    return TrainResult(
        erm_ckpt=erm, R_ERM_hat=R_erm, tau=tau, H_ref_bar=0.0,
        trajectory=traj, in_dim=1, cfg=TrainConfig(numerical_tol=1e-4),
    )


def test_only_feasible_checkpoint_is_selected():
    tau = 0.5
    traj = [
        _ckpt(0, R=0.90, leak=-1.0),    # infeasible but lowest leakage -> must NOT be chosen
        _ckpt(1, R=0.40, leak=0.50),    # feasible
        _ckpt(2, R=0.45, leak=0.20),    # feasible, lower leakage -> winner
    ]
    sel = select_checkpoint(_result(traj, tau, 0.45))
    assert not sel.used_erm_fallback
    assert sel.selected_epoch == 2
    assert sel.n_feasible == 2
    assert sel.R_src <= tau + 1e-4


def test_all_infeasible_candidates_fall_back_exactly_to_erm():
    tau = 0.5
    traj = [_ckpt(0, R=0.90, leak=-1.0), _ckpt(1, R=0.80, leak=-2.0)]  # all infeasible
    res = _result(traj, tau, 0.49)
    sel = select_checkpoint(res)
    assert sel.used_erm_fallback and sel.selected_epoch == -1
    assert sel.n_feasible == 0
    assert state_hash(sel.enc_state) == state_hash(res.erm_ckpt["enc"])   # byte-exact ERM
    assert state_hash(sel.head_state) == state_hash(res.erm_ckpt["head"])
    assert sel.R_src == res.R_ERM_hat


def test_injected_score_fn_overrides_default_ranking():
    tau = 0.5
    traj = [_ckpt(0, R=0.40, leak=0.10), _ckpt(1, R=0.40, leak=0.90)]
    # default ranks by leakage_surrogate -> epoch 0; an injected score_fn flips the order
    sel = select_checkpoint(_result(traj, tau, 0.40), score_fn=lambda c: -c.leakage_surrogate)
    assert sel.selected_epoch == 1


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-selector tests")


if __name__ == "__main__":
    _run_all()
