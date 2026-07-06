"""Guard (Stage-2B3): a sub-MIN_BATCH forced tail is adaptation-INELIGIBLE — evaluate_candidate_disease must NOT invoke the action
provider for it (matched_coral/spdim/t3a never run on a forced tail). This is the direct regression for the Stage-2B FAIL where a
1-window forced tail was pushed through matched_coral. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_forced_tail_makes_zero_provider_calls():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=1))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 256, 0),
        ("PD/ds002778/sub-fit1", "train", 256, 1),
        ("PD/ds002778/sub-cal0", "cal", 5, 0),                 # n=5 < MIN_BATCH=8 -> a single FORCED tail (only)
    ])
    prov, calls = stage2b_spy_provider()
    fit_keys = ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"]
    fit_batches = ENG._fit_batches(by_subject, fit_keys, lda, prov)
    cand, thr = stage2b_first_evaluable(fit_batches)
    calls.clear()                                              # count ONLY the CAL evaluation of the forced-only subject
    res = PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
    assert calls == [], f"a forced tail must make ZERO action-provider calls, got {calls}"
    assert res["subject_records"][0]["batches"] == [{"adapted": False, "harmful": False, "forced_identity": True}]
    ok("forced (sub-MIN_BATCH) tail: evaluate_candidate_disease makes ZERO action-provider calls; identity-only record")


def main():
    print("ACAR v5 Stage-2B3 guard: forced tail does not call the action provider")
    test_forced_tail_makes_zero_provider_calls()
    print("ALL V5 STAGE2B3-FORCED-TAIL-NO-PROVIDER GUARDS PASS")


if __name__ == "__main__":
    main()
