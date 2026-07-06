"""Guard (Stage-2B3): a forced tail is NOT dropped — it remains present in the all-batch denominator (coverage / L_harm_all use
total = len(batches)), it just never contributes to the numerators. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import metrics as M
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_forced_tail_stays_in_denominator():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=3))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 256, 0),
        ("PD/ds002778/sub-fit1", "train", 256, 1),
        ("PD/ds002778/sub-cal0", "cal", 33, 0),                # 33 -> window_batches [32 eligible, 1 FORCED]
    ])
    prov, _ = stage2b_spy_provider()
    fit_batches = ENG._fit_batches(by_subject, ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"], lda, prov)
    cand, thr = stage2b_first_evaluable(fit_batches)
    res = PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
    batches = res["subject_records"][0]["batches"]
    assert len(batches) == 2, f"denominator must include the forced tail (2 batches: 1 eligible + 1 forced), got {len(batches)}"
    assert batches[-1] == {"adapted": False, "harmful": False, "forced_identity": True}
    # metrics.collect uses total = len(batches): coverage denominator = 2 (forced tail present, numerator excludes it)
    c = M.collect(res["subject_records"])
    cov = c["coverage"][0]
    assert 0.0 <= cov <= 0.5 + 1e-9, f"coverage over 2 batches (>=1 forced, never adapted) must be <= 0.5, got {cov}"
    ok("forced tail stays in the all-batch denominator (total=len(batches)); excluded only from the numerators")


def main():
    print("ACAR v5 Stage-2B3 guard: forced tail counts in the denominator")
    test_forced_tail_stays_in_denominator()
    print("ALL V5 STAGE2B3-FORCED-TAIL-DENOMINATOR GUARDS PASS")


if __name__ == "__main__":
    main()
