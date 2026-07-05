"""Guard (Stage-2B1): the v2-replay comparator fails closed (V2ReplayNotEvaluable) when a disease, a fold, or a FIT/CAL/EVAL split
is missing. Torch-free (synthetic action provider). Synthetic."""
from __future__ import annotations
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_v2_replay as VR
from acar.v5.tests._util import expect_raises, ok, stage2b_disease_inputs

SYN = AR.synthetic_action_provider


def test_missing_disease_fails():
    di = stage2b_disease_inputs(seed=1)
    prov = VR.make_engine_v2_replay_provider(action_provider=SYN)
    expect_raises(VR.V2ReplayNotEvaluable, lambda: prov("PD", {"disease_inputs": {"PD": di["PD"]}}))   # SCZ missing
    expect_raises(VR.V2ReplayNotEvaluable, lambda: prov("PD", {"disease_inputs": {}}))
    ok("engine v2-replay provider fails closed when a disease's folds are missing")


def test_missing_folds_or_split_fails():
    di = stage2b_disease_inputs(seed=1)
    expect_raises(VR.V2ReplayNotEvaluable, lambda: VR.v2_replay_red_by_disease("PD", [], action_provider=SYN))
    # drop all CAL subjects from a fold → missing split
    fold = dict(di["PD"]["folds"][0])
    fold["by_subject"] = {k: v for k, v in fold["by_subject"].items() if v["split_role"] != "cal"}
    expect_raises(VR.V2ReplayNotEvaluable, lambda: VR.v2_replay_red_by_disease("PD", [fold], action_provider=SYN))
    ok("no folds, or a missing FIT/CAL/EVAL split → V2ReplayNotEvaluable")


def main():
    print("ACAR v5 Stage-2B1 guard: v2-replay missing disease/split fails closed")
    test_missing_disease_fails()
    test_missing_folds_or_split_fails()
    print("ALL V5 STAGE2B1-V2REPLAY-MISSING GUARDS PASS")


if __name__ == "__main__":
    main()
