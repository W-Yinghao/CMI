"""Primal–dual trainer: dual ascent direction, ERM-checkpoint immutability, seed
reproducibility, and a safe no-op when there is no comparable class.

Standalone (``python -m oaci.tests.test_train_primal_dual``) and pytest-compatible. Uses tiny
epoch counts for speed.
"""
from __future__ import annotations

import numpy as np

from oaci.train.primal_dual import TrainConfig, dual_update, train_risk_feasible
from oaci.train.selector import select_checkpoint, state_hash
from oaci.train.synthetic import make_covariate_shift


def _fast(seed=0, **kw):
    return TrainConfig(seed=seed, stage1_epochs=40, stage2_epochs=12, warmup_steps=10,
                       z_dim=8, enc_hidden=16, adv_hidden=8, **kw)


def test_dual_increases_on_violation_and_decreases_on_slack():
    tau = 0.5
    assert dual_update(1.0, 0.8, tau, 0.5, 20.0) > 1.0          # violation R>τ -> λ up
    assert dual_update(1.0, 0.3, tau, 0.5, 20.0) < 1.0          # slack R<τ -> λ down
    assert dual_update(0.0, 0.0, tau, 10.0, 20.0) == 0.0        # clip at 0 (0 + 10*(-0.5))
    assert dual_update(19.8, 5.0, tau, 1.0, 20.0) == 20.0       # clip at λmax


def test_erm_reference_checkpoint_is_immutable():
    X, y, d, g, sg = make_covariate_shift(seed=0)
    res = train_risk_feasible(X, y, d, g, sg, _fast())
    h_erm = state_hash(res.erm_record.model_state)
    select_checkpoint(res)                                       # selection must not mutate ERM
    assert state_hash(res.erm_record.model_state) == h_erm
    assert res.erm_record.model_hash == h_erm                    # stored hash matches the live state
    # Stage-2 actually moved the live params away from ERM
    assert state_hash(res.trajectory[-1].model_state) != h_erm


def test_seed_reproducibility():
    X, y, d, g, sg = make_covariate_shift(seed=1)
    r1 = train_risk_feasible(X, y, d, g, sg, _fast(seed=3))
    r2 = train_risk_feasible(X, y, d, g, sg, _fast(seed=3))
    assert abs(r1.R_ERM_hat - r2.R_ERM_hat) < 1e-9
    assert np.allclose([c.R_src for c in r1.trajectory], [c.R_src for c in r2.trajectory])
    assert np.allclose([c.lam for c in r1.trajectory], [c.lam for c in r2.trajectory])
    assert [c.model_hash for c in r1.trajectory] == [c.model_hash for c in r2.trajectory]


def test_no_comparable_classes_is_safe_noop():
    # single domain -> no comparable class -> objective INACTIVE; Stage-2 is a byte-exact ERM no-op.
    X, y, d, g, sg = make_covariate_shift(seed=0, n_domains=1)
    assert sg.comparable_classes == []
    res = train_risk_feasible(X, y, d, g, sg, _fast())
    assert res.active is False and res.inactive_reason == "no_comparable_class"
    assert res.trajectory == []                                 # NO Stage-2 updates ran
    sel = select_checkpoint(res)
    assert sel.used_erm_fallback and sel.selected_epoch == -1
    assert sel.model_hash == res.erm_record.model_hash          # byte-exact ERM


def _run_all() -> None:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for fn in fns:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"PASS  {len(fns)} train-primal-dual tests")


if __name__ == "__main__":
    _run_all()
