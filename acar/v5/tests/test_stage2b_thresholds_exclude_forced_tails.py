"""Guard (Stage-2B3): FIT thresholds are fit only on adaptation-ELIGIBLE batches — forced (sub-MIN_BATCH) tails produce no
proposed-action FIT records and no action-provider call during fitting. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_action_records as AR
from acar.v5.tests._util import ok, stage2b_synthetic_source_state, stage2b_spy_provider, stage2b_by_subject


def test_fit_batches_exclude_forced_tails():
    lda = AR.SourceLDA(stage2b_synthetic_source_state(D=8, seed=5))
    by_subject, _ = stage2b_by_subject([
        ("PD/ds002778/sub-fit0", "train", 33, 0),              # [32 eligible, 1 forced] -> 1 FIT record
        ("PD/ds002778/sub-fit1", "train", 64, 1),              # [32, 32]                -> 2 FIT records
        ("PD/ds002778/sub-fit2", "val", 5, 0),                 # forced-only (n=5)       -> 0 FIT records
    ])
    prov, calls = stage2b_spy_provider()
    fit_keys = ["PD/ds002778/sub-fit0", "PD/ds002778/sub-fit1", "PD/ds002778/sub-fit2"]
    fit_batches = ENG._fit_batches(by_subject, fit_keys, lda, prov)
    assert len(fit_batches) == 3, f"FIT must use only the 3 eligible batches (forced tails excluded), got {len(fit_batches)}"
    assert calls and all(c["n"] >= PE.STAGE2_MIN_BATCH for c in calls), "no action-provider call may occur on a forced tail"
    ok("FIT thresholds use only eligible batches; forced tails produce no FIT record and no provider call")


def main():
    print("ACAR v5 Stage-2B3 guard: thresholds exclude forced tails")
    test_fit_batches_exclude_forced_tails()
    print("ALL V5 STAGE2B3-THRESHOLDS-EXCLUDE-FORCED GUARDS PASS")


if __name__ == "__main__":
    main()
