"""Guard (Stage-2B1): the v2-replay comparator uses acar.regressor.ActionRegressor(seed=0) and produces a finite subject-macro
v2_replay_red per disease. Torch-free (synthetic action provider; sklearn on both Pythons). Synthetic."""
from __future__ import annotations
import numpy as np
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_v2_replay as VR
from acar.v5.tests._util import ok, stage2b_disease_inputs


def test_regressor_is_actionregressor_seed0():
    from acar.regressor import ActionRegressor
    r = VR.default_regressor_factory()
    assert isinstance(r, ActionRegressor) and r.seed == 0
    ok("default v2 regressor is acar.regressor.ActionRegressor(seed=0)")


def test_v2_replay_red_finite():
    di = stage2b_disease_inputs(n_folds=2, D=6, seed=3, n_windows=40)
    for d in ("PD", "SCZ"):
        red = VR.v2_replay_red_by_disease(d, di[d]["folds"], action_provider=AR.synthetic_action_provider)
        assert isinstance(red, float) and np.isfinite(red)
    prov = VR.make_engine_v2_replay_provider(action_provider=AR.synthetic_action_provider)
    r = prov("PD", {"disease_inputs": di})
    assert isinstance(r, float) and np.isfinite(r)
    ok("v2_replay_red_by_disease + engine provider produce a finite subject-macro red")


def main():
    print("ACAR v5 Stage-2B1 guard: v2-replay uses ActionRegressor(seed=0)")
    test_regressor_is_actionregressor_seed0()
    test_v2_replay_red_finite()
    print("ALL V5 STAGE2B1-V2REPLAY-SEED0 GUARDS PASS")


if __name__ == "__main__":
    main()
