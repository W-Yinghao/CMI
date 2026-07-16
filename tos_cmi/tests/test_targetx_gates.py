"""Unit tests for the deterministic five-gate engine (amendment 03 C4). Pure functions on aggregated LCBs."""
from tos_cmi.eval.targetx_gates import (gate1_observability, gate2_actionability, gate3_oracle_recovery,
    gate4_cross_dataset_safety, gate5_specificity, five_gate_verdict, RECOVERY_MIN, HARM_EPS)


def test_gate1_observability():
    assert gate1_observability(0.05) is True
    assert gate1_observability(0.0) is False
    assert gate1_observability(-0.1) is False
    assert gate1_observability(None) is False


def test_gate2_actionability_requires_all_paired_positive():
    assert gate2_actionability(0.02, 0.01, 0.01, 0.01, 0.01) is True
    assert gate2_actionability(0.02, 0.01, -0.001, 0.01, 0.01) is False   # loses to source-greedy
    assert gate2_actionability(-0.01, 0.01, 0.01, 0.01, 0.01) is False    # Δtx itself not > 0
    assert gate2_actionability(0.02, 0.01, 0.01, 0.01, None) is False     # missing control


def test_gate3_oracle_recovery_threshold():
    assert gate3_oracle_recovery(RECOVERY_MIN) is True
    assert gate3_oracle_recovery(RECOVERY_MIN - 1e-6) is False
    assert gate3_oracle_recovery(0.5) is True
    assert gate3_oracle_recovery(None) is False


def test_gate4_cross_dataset_safety():
    assert gate4_cross_dataset_safety({"a": 0.02, "b": 0.01}) is True        # both positive
    assert gate4_cross_dataset_safety({"a": 0.02, "b": -0.02}) is False      # one positive, one clearly harmful
    assert gate4_cross_dataset_safety({"a": 0.02, "b": -HARM_EPS / 2}) is True   # other only mildly negative
    assert gate4_cross_dataset_safety({"a": -0.01, "b": -0.02}) is True      # none positive -> not the unsafe pattern


def test_gate5_specificity():
    assert gate5_specificity(0.01) is True
    assert gate5_specificity(0.0) is False
    assert gate5_specificity(None) is False


def test_five_gate_verdict_routing():
    allpass = five_gate_verdict(True, True, True, True, True)
    assert allpass["all_pass"] and allpass["outcome"] == "GO_LIGHT_TARGETX_SELECTOR_PLAN"
    util_not_spec = five_gate_verdict(True, True, True, True, False)
    assert util_not_spec["outcome"] == "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"
    no_action = five_gate_verdict(True, False, True, True, False)
    assert no_action["outcome"] == "STOP_ACTIONABILITY_FAILED"
    # actionable+specific but UNSAFE cross-dataset (g4 False) -> incomplete/unsafe, not a GO
    unsafe = five_gate_verdict(True, True, True, False, True)
    assert unsafe["outcome"] == "STOP_INCOMPLETE_OR_UNSAFE"
    # actionable+safe but not specific -> the utility-observable-not-CMI-specific outcome (regardless of g1/g3)
    util_only = five_gate_verdict(False, True, False, True, False)
    assert util_only["outcome"] == "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
