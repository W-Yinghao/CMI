"""Guard (Stage-2B3): a forced tail's batch record is exactly {adapted:False, harmful:False, forced_identity:True}, while an
eligible batch carries forced_identity:False. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_forced_tail_record_is_identity_only():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=4))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 256, 0),
        ("PD/ds002778/sub-fit1", "train", 256, 1),
        ("PD/ds002778/sub-cal0", "cal", 33, 1),                # [32 eligible, 1 forced]
    ])
    prov, _ = stage2b_spy_provider()
    fit_batches = ENG._fit_batches(by_subject, ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"], lda, prov)
    cand, thr = stage2b_first_evaluable(fit_batches)
    res = PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
    batches = res["subject_records"][0]["batches"]
    forced = batches[-1]
    assert forced == {"adapted": False, "harmful": False, "forced_identity": True}, forced
    assert batches[0]["forced_identity"] is False, "the eligible batch must carry forced_identity=False"
    assert set(batches[0]) == {"adapted", "harmful", "forced_identity"}
    ok("forced tail record = {adapted:False, harmful:False, forced_identity:True}; eligible batch forced_identity:False")


def main():
    print("ACAR v5 Stage-2B3 guard: forced tail adapted/harmful False")
    test_forced_tail_record_is_identity_only()
    print("ALL V5 STAGE2B3-FORCED-TAIL-ADAPTED-FALSE GUARDS PASS")


if __name__ == "__main__":
    main()
