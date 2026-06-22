"""Phase 1.3.1b -- unit tests for the task-gate power floor (competence certificate).

Fast, training-free checks of the certificate machinery: the conservative + monotone lookup, the
3-way safety classification, and the matched positive-control effect-size tuning (Bayes-only).
The heavy end-to-end (gate abstains via TASK_POWER_INSUFFICIENT on a low-power injection while
factorized stays SAFE_ACCEPT) lives in the phase-diagram re-run / test_bayes_calibration."""
import numpy as np
import torch

torch.set_num_threads(1)

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eval.power_certificate import lookup_power, make_control, tune_confound


def _fake_table():
    return {"meta": {"delta_Y": 0.03}, "table": [
        {"n_eff": 2000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": False, "mde": None},
        {"n_eff": 6000, "d_base": 23, "d_extra": 1, "n_cls": 3, "power_ok": True, "mde": 0.03},
        {"n_eff": 2000, "d_base": 22, "d_extra": 2, "n_cls": 3, "power_ok": True, "mde": 0.03},
    ]}


def test_lookup_conservative_and_monotone():
    t = _fake_table()
    # below the smallest calibrated n -> uncovered -> abstain
    ok, inf = lookup_power(t, 1000, 23, 1, 3); assert not ok and not inf["covered"], inf
    # at a covered low-n cell that is NOT power_ok -> not ok
    ok, _ = lookup_power(t, 2000, 23, 1, 3); assert not ok
    # between cells: only the n_eff=2000(False) cell qualifies (<=3000) -> not ok
    ok, _ = lookup_power(t, 3000, 23, 1, 3); assert not ok
    # above the power_ok cell: monotone -> ok
    ok, inf = lookup_power(t, 8000, 23, 1, 3); assert ok and inf["covered"], inf
    # different power_ok shape
    ok, _ = lookup_power(t, 4000, 22, 2, 3); assert ok
    # uncovered shape (dim not in table) -> abstain
    ok, inf = lookup_power(t, 8000, 17, 1, 3); assert not ok and not inf["covered"]
    print("test_lookup_conservative_and_monotone: OK")


def test_control_effect_tuning():
    """The matched explaining-away control hits target Bayes Delta* via the confound scale (so a
    competence grid at {delta_Y, 1.3 delta_Y, 2 delta_Y} is buildable)."""
    cfg = ScoreFisherConfig()
    for tgt in [cfg.delta_Y, 2 * cfg.delta_Y]:
        c = tune_confound(tgt, 23, 1, 6000, 3, 6, base_sep=1.5, sigma=1.0, seed=0)
        _, _, _, _, b = make_control(23, 1, 6000, 3, 6, 1.5, c, 1.0, seed=0)
        assert abs(b - tgt) < 0.01, (tgt, c, b)
        # the deleted carrier is conditionally informative but MARGINALLY ~ uninformative by design
    print("test_control_effect_tuning: OK")


if __name__ == "__main__":
    test_lookup_conservative_and_monotone()
    test_control_effect_tuning()
    print("ALL POWER-CERTIFICATE TESTS PASSED")
