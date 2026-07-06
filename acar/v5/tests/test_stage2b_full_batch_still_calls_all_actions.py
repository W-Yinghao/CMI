"""Guard (Stage-2B3): the forced-tail exclusion is NARROW — a full 32-window (eligible) batch still runs ALL actions
(identity + matched_coral + spdim + t3a). Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_full_batch_calls_all_four_actions():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=8))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 256, 0),
        ("PD/ds002778/sub-fit1", "train", 256, 1),
        ("PD/ds002778/sub-cal0", "cal", 32, 0),                # exactly one full 32-window ELIGIBLE batch
    ])
    prov, calls = stage2b_spy_provider()
    fit_batches = ENG._fit_batches(by_subject, ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"], lda, prov)
    cand, thr = stage2b_first_evaluable(fit_batches)
    calls.clear()
    PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
    assert {c["name"] for c in calls} == {"identity", *P.ACTIONS}, f"a full batch must call ALL actions, got {calls}"
    assert all(c["n"] == 32 for c in calls)
    ok("a full 32-window eligible batch still calls ALL actions (identity + matched_coral + spdim + t3a)")


def main():
    print("ACAR v5 Stage-2B3 guard: full batch still calls all actions")
    test_full_batch_calls_all_four_actions()
    print("ALL V5 STAGE2B3-FULL-BATCH-ALL-ACTIONS GUARDS PASS")


if __name__ == "__main__":
    main()
