"""Project A Step 12 — tests for the minimal-paired phase-transition simulator.

Checks that k=0 is the R1 non-identifiability boundary, k>0 is an R2 labeled slice under an iid
sampling contract (never "full target risk identified"), and the summary carries the sampling-contract
warning. Run:

    python -m h2cmi.tests.test_minimal_paired_phase_transition
"""
from __future__ import annotations

from h2cmi.observability.minimal_paired import SHIFTS, run, simulate


def test_k_zero_matches_r1_nonidentifiability_boundary():
    for shift in SHIFTS:
        r = simulate(shift, 0, n_repeats=20, seed=0)
        assert r["identified_status"] == "not_identified_R1"
        assert r["risk_ci_width"] is None
        assert r["harm_sign_accuracy"] == 0.5
        assert r["abstention_rate_needed"] == 1.0


def test_k_positive_marked_r2_labeled_slice_not_full_target():
    r = simulate("concept_shift", 8, n_repeats=20, seed=0)
    assert r["identified_status"] == "labeled_slice_under_iid_sampling_contract"
    assert "sampling contract" in r["claim_boundary"].lower()
    assert "not full target" in r["claim_boundary"].lower()


def test_phase_transition_summary_contains_sampling_contract_warning():
    s = run(n_repeats=30, seed=0)
    assert "sampling contract" in s["claim_boundary"].lower()
    assert s["k0_status"] == "not_identified_R1"
    # a genuine phase transition: harm-sign accuracy rises with k on a harmful shift
    curve = [r for r in s["records"] if r["shift_type"] == "support_failure"]
    acc_by_k = {r["k"]: r["harm_sign_accuracy"] for r in curve}
    assert acc_by_k[max(acc_by_k)] > acc_by_k[1], "harm-sign accuracy should rise with k"


def test_no_claim_full_target_risk_without_sampling_contract():
    s = run(n_repeats=20, seed=0)
    blob = (s["claim_boundary"] + " " + " ".join(r["claim_boundary"] for r in s["records"])).lower()
    assert "full target risk identified" not in blob
    assert "k labels identify full target risk" not in blob
    # every k>0 record attaches the sampling-contract caveat
    for r in s["records"]:
        if r["k"] > 0:
            assert "sampling contract" in r["claim_boundary"].lower()


ALL_TESTS = [
    test_k_zero_matches_r1_nonidentifiability_boundary,
    test_k_positive_marked_r2_labeled_slice_not_full_target,
    test_phase_transition_summary_contains_sampling_contract_warning,
    test_no_claim_full_target_risk_without_sampling_contract,
]


def run_all():
    for t in ALL_TESTS:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nALL {len(ALL_TESTS)} MINIMAL-PAIRED TESTS PASSED")


if __name__ == "__main__":
    run_all()
