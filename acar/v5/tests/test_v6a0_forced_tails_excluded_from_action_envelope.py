"""Guard (V6-A0a): forced sub-MIN_BATCH tails are adaptation-ineligible — excluded from the action envelope + beneficial-coverage
denominator, and counted separately in accounting. Synthetic, torch-free."""
from __future__ import annotations
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_policy_eval as PE
from acar.v5 import v6_a0_action_viability as AV
from acar.v5.tests._util import ok, v6a0_eval_fold


def test_forced_tail_excluded():
    assert AV.MIN_BATCH == PE.STAGE2_MIN_BATCH == 8
    fold = v6a0_eval_fold([
        ("PD/ds002778/sub-e0", "eval", 33, 0),      # 33 -> window_batches [32 eligible, 1 forced]
        ("PD/ds002778/sub-e1", "eval", 5, 1),        # 5 -> a single FORCED tail (n<8) -> zero eligible records
    ])
    records, acct = AV.collect_eval_records([fold], AR.synthetic_action_provider)
    # sub-e0 contributes exactly 1 eligible record (the 32-window batch); the size-1 tail is excluded; sub-e1 contributes none
    assert len(records) == 1 and records[0]["subject_key"] == "PD/ds002778/sub-e0" and records[0]["n"] == 32
    assert acct["n_eval_forced_tails"] == 2 and acct["n_eval_eligible_batches"] == 1
    env = AV.oracle_envelope(records)
    assert env["n_eligible_batches"] == 1                       # beneficial-coverage denominator = eligible only
    assert env["oracle_conditional_harm"] == 0.0               # oracle no-ops when no action helps -> never harmful (sanity)
    ok("forced sub-MIN_BATCH tails excluded from action envelope + coverage denominator; counted separately in accounting")


def main():
    print("ACAR v5 V6-A0a guard: forced tails excluded from action envelope")
    test_forced_tail_excluded()
    print("ALL V6A0-FORCED-TAIL-EXCLUDED GUARDS PASS")


if __name__ == "__main__":
    main()
