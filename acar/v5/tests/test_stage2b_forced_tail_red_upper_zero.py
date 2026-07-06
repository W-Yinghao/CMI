"""Guard (Stage-2B3): a forced tail contributes 0 to the red_upper harmful-oracle (identity-only; min_a ΔR_a is NOT computed for
an ineligible batch). Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import (ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject,
                                 stage2b_first_evaluable)


def test_forced_only_subject_red_upper_is_zero():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=2))
    by_subject, lv = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 256, 0),
        ("PD/ds002778/sub-fit1", "train", 256, 1),
        ("PD/ds002778/sub-cal0", "cal", 6, 1),                 # forced-only (n=6 < 8)
    ])
    prov, _ = stage2b_spy_provider()
    fit_batches = ENG._fit_batches(by_subject, ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1"], lda, prov)
    cand, thr = stage2b_first_evaluable(fit_batches)
    res = PE.evaluate_candidate_disease(cand, thr, ["PD/ds002778/sub-cal0"], by_subject, lda, lv, action_provider=prov)
    # forced-only subject: upper_term = mean(upper_drs) = mean([0.0]) = 0.0 ; red_upper contribution 0
    assert res["upper_terms"] == [0.0], f"forced-only subject red_upper term must be 0.0, got {res['upper_terms']}"
    assert res["red_upper"] == 0.0
    ok("forced tail contributes 0 to red_upper (identity-only oracle; no min_a ΔR_a on an ineligible batch)")


def main():
    print("ACAR v5 Stage-2B3 guard: forced tail red_upper = 0")
    test_forced_only_subject_red_upper_is_zero()
    print("ALL V5 STAGE2B3-FORCED-TAIL-RED-UPPER GUARDS PASS")


if __name__ == "__main__":
    main()
