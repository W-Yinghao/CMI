"""CIGL R1 — probe calibration: ECE + temperature scaling."""
import numpy as np

from cmi.eval.probe_calibration import (expected_calibration_error, fit_temperature,
                                        calibration_report, _softmax)


def test_ece_zero_for_perfectly_calibrated():
    # perfectly confident + correct -> ECE 0
    N = 200
    probs = np.zeros((N, 3)); probs[np.arange(N), 0] = 1.0
    labels = np.zeros(N, int)
    assert expected_calibration_error(probs, labels) == 0.0


def test_temperature_scaling_reduces_overconfidence():
    rng = np.random.default_rng(0)
    N, K = 600, 4
    labels = rng.integers(0, K, N)
    # over-confident logits: correct class boosted a lot but only ~65% actually correct
    logits = rng.standard_normal((N, K))
    for i in range(N):
        if rng.random() < 0.65:
            logits[i, labels[i]] += 6.0                          # correct & very confident
        else:
            logits[i, rng.integers(0, K)] += 6.0                 # wrong & very confident
    rep = calibration_report(logits, labels)
    assert rep["temperature"] > 1.0                              # over-confident -> T>1 softens
    assert rep["ece_calibrated"] <= rep["ece_raw"] + 1e-9 and rep["improved"]
    assert rep["nll_calibrated"] <= rep["nll_raw"] + 1e-6


def test_fit_temperature_recovers_scale():
    rng = np.random.default_rng(1)
    N, K = 2000, 3
    logits = 1.5 * rng.standard_normal((N, K))
    probs = _softmax(logits, 1.0)
    labels = np.array([rng.choice(K, p=probs[i]) for i in range(N)])   # sampled from logits -> calibrated at T=1
    T1 = fit_temperature(logits, labels)
    T3 = fit_temperature(3.0 * logits, labels)                          # inflate logits 3x
    assert 0.7 < T1 < 1.4                                               # calibrated -> T~1
    assert T3 > 1.8 * T1                                                # 3x-inflated logits -> larger fitted T
