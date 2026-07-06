"""Guard (Stage-2B3): eligibility is n < STAGE2_MIN_BATCH (=8), NOT n < 32. A PARTIAL but eligible batch (8 <= n < 32) still runs
ALL actions — this patch only makes sub-MIN_BATCH tails identity-only, and does NOT change the batch-size policy. Synthetic,
torch-free."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_partial_eligible_batch_calls_all_actions():
    assert PE.STAGE2_MIN_BATCH == 8 and PE.STAGE2_BATCH_SIZE == 32          # the eligibility boundary this test pins
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=9))
    for n_partial in (8, 10, 31):                                          # >= MIN_BATCH and < BATCH_SIZE -> ELIGIBLE
        by_subject, lv = stage2b_by_subject([
            ("PD/ds002778/sub-fit0", "train", 256, 0),
            ("PD/ds002778/sub-fit1", "train", 256, 1),
            ("PD/ds002778/sub-cal0", "cal", n_partial, 0),
        ])
        prov, calls = stage2b_spy_provider()
        fit_batches = ENG._fit_batches(by_subject, ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"], lda, prov)
        cand, thr = stage2b_first_evaluable(fit_batches)
        calls.clear()
        PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
        assert {c["name"] for c in calls} == {"identity", *P.ACTIONS}, f"n={n_partial} eligible must call all actions, got {calls}"
        assert all(c["n"] == n_partial for c in calls)
    ok("partial but eligible batches (8<=n<32) still call ALL actions; forced-tail rule is n<MIN_BATCH, not n<32")


def main():
    print("ACAR v5 Stage-2B3 guard: partial-but-eligible batch calls all actions")
    test_partial_eligible_batch_calls_all_actions()
    print("ALL V5 STAGE2B3-PARTIAL-ELIGIBLE-ALL-ACTIONS GUARDS PASS")


if __name__ == "__main__":
    main()
