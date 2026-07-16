"""Unit tests for the deterministic five-gate engine (amendments 03/04). Pure functions on aggregated LCBs.
F2.1b: Gate 3 = constrained-hindsight paired inequality; Gate 4 = UCB-for-harm."""
from tos_cmi.eval.targetx_gates import (gate1_observability, gate2_actionability, gate3_oracle_recovery,
    gate4_cross_dataset_safety, gate5_specificity, five_gate_verdict, HARM_EPS)


def test_gate1_observability():
    assert gate1_observability(0.05) is True
    assert gate1_observability(0.0) is False and gate1_observability(-0.1) is False and gate1_observability(None) is False


def test_gate2_actionability_requires_all_paired_positive():
    assert gate2_actionability(0.02, 0.01, 0.01, 0.01, 0.01) is True
    assert gate2_actionability(0.02, 0.01, -0.001, 0.01, 0.01) is False   # loses to source-greedy
    assert gate2_actionability(-0.01, 0.01, 0.01, 0.01, 0.01) is False    # dtx not > 0
    assert gate2_actionability(0.02, 0.01, 0.01, 0.01, None) is False     # missing control


def test_gate3_constrained_hindsight_paired():
    # constrained hindsight real (LCB>0) AND dtx recovers >=25% of it (paired LCB>=0)
    assert gate3_oracle_recovery(0.04, 0.001) is True
    assert gate3_oracle_recovery(0.04, -0.001) is False       # recovers < 25%
    assert gate3_oracle_recovery(-0.01, 0.01) is False        # hindsight ceiling not real
    assert gate3_oracle_recovery(None, 0.01) is False


def test_gate4_uses_ucb_for_harm():
    # one dataset LCB>0, other only MILDLY negative (UCB not < -eps) -> safe
    assert gate4_cross_dataset_safety({"a": {"lo": 0.02, "hi": 0.05}, "b": {"lo": -0.02, "hi": 0.03}}) is True
    # one LCB>0, other CONFIRMED harmful (UCB < -eps) -> unsafe
    assert gate4_cross_dataset_safety({"a": {"lo": 0.02, "hi": 0.05}, "b": {"lo": -0.04, "hi": -HARM_EPS - 0.01}}) is False
    # none positive -> not the unsafe pattern
    assert gate4_cross_dataset_safety({"a": {"lo": -0.01, "hi": 0.01}, "b": {"lo": -0.04, "hi": -0.02}}) is True


def test_gate5_specificity():
    assert gate5_specificity(0.01) is True and gate5_specificity(0.0) is False and gate5_specificity(None) is False


def test_five_gate_verdict_routing():
    assert five_gate_verdict(True, True, True, True, True)["outcome"] == "GO_LIGHT_TARGETX_SELECTOR_PLAN"
    assert five_gate_verdict(True, True, True, True, False)["outcome"] == "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"
    assert five_gate_verdict(True, False, True, True, False)["outcome"] == "STOP_ACTIONABILITY_FAILED"
    assert five_gate_verdict(True, True, True, False, True)["outcome"] == "STOP_INCOMPLETE_OR_UNSAFE"
    assert five_gate_verdict(False, True, False, True, False)["outcome"] == "TARGET_X_UTILITY_OBSERVABLE_NOT_CMI_SPECIFIC"


if __name__ == "__main__":
    import sys, pytest
    sys.exit(pytest.main([__file__, "-v"]))
