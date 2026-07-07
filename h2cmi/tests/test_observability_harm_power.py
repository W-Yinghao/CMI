"""Project A Step 14 — tests for the harm-prediction power/sensitivity summary.

Run:  python -m h2cmi.tests.test_observability_harm_power
"""
from __future__ import annotations

from h2cmi.observability.harm_power import build_power

_TABLE = {"n_runs": 54}
_PRED = {"n_runs": 54, "n_harmed": 46, "n_minority_class": 8, "minority_fraction": 0.1481,
         "n_groups": 18, "robust_margin": 0.03, "any_predictor_robust_signal": False,
         "feature_sets": {"R1_target_unlabeled": {"balanced_acc_harm_prediction": 0.6522,
                                                  "perm_null_p95": 0.6413}}}


def test_power_summary_reports_minority_fraction():
    d = build_power(_TABLE, _PRED)
    assert d["n_non_harmed"] == 8 and d["minority_fraction"] == 0.1481
    assert d["minimum_detectable_bacc_approx"] == round(0.6413 + 0.03, 4)


def test_power_summary_marks_underpowered_when_minority_n_small():
    assert build_power(_TABLE, _PRED)["underpowered"] is True
    # a well-powered setting (large balanced minority) is not flagged
    big = dict(_PRED, n_runs=400, n_harmed=200, n_minority_class=200, minority_fraction=0.5)
    assert build_power({"n_runs": 400}, big)["underpowered"] is False


def test_power_summary_does_not_convert_retrospective_signal_to_identifiability():
    d = build_power(_TABLE, _PRED)
    cb = d["claim_boundary"].lower()
    assert "not make target gain identifiable" in cb and "duplicating targets is not evidence" in cb
    assert d["robust_signal"] is False


ALL_TESTS = [
    test_power_summary_reports_minority_fraction,
    test_power_summary_marks_underpowered_when_minority_n_small,
    test_power_summary_does_not_convert_retrospective_signal_to_identifiability,
]


def run():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} HARM-POWER TESTS PASSED")


if __name__ == "__main__":
    run()
