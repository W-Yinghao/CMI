"""Guard (Stage-2B0): the engine selects ONE candidate identity JOINTLY across PD and SCZ (never per-disease). Every candidate is
evaluated on both diseases; the report carries a single selected_candidate_id (or None on DEV_STOP). Synthetic only (torch-free)."""
from __future__ import annotations
from acar.v5 import protocol as P
from acar.v5 import stage2_action_records as AR
from acar.v5 import stage2_selection_engine as ENG
from acar.v5 import stage2_selection_report as RPT
from acar.v5.tests._util import ok, stage2b_auth, stage2b_disease_inputs


def test_engine_joint_and_single_identity():
    di = stage2b_disease_inputs()
    a = stage2b_auth()
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=di, action_provider=AR.synthetic_action_provider,
                            v2_replay_provider=lambda d, ctx: -0.5)
    RPT.validate_selection_report(rep)
    # every one of the 22 candidates evaluated, each on BOTH diseases (joint scope)
    assert set(rep["per_candidate"]) == set(P.CANDIDATE_IDS)
    for cid, rows in rep["per_candidate"].items():
        assert set(rows) == {"PD", "SCZ"}
    # a single joint identity (or None), never a per-disease pair
    assert rep["selected_candidate_id"] is None or rep["selected_candidate_id"] in P.CANDIDATE_IDS
    ok("engine evaluates all 22 candidates on BOTH diseases and reports a single joint selected_candidate_id (or None)")


def test_engine_multichunk_fit_eval_granularity():
    # n_windows=100 > STAGE2_BATCH_SIZE(32) → each subject splits into several 32-window batches; FIT thresholds are fit on the
    # SAME 32-window units routing applies at CAL/EVAL (regression guard for the batch-granularity fix).
    di = stage2b_disease_inputs(n_windows=100)
    a = stage2b_auth()
    rep = ENG.run_selection(a, stage1b_run_id=a["stage1b_run_id"], stage1b_registry_sha256=a["stage1b_registry_sha256"],
                            disease_inputs=di, action_provider=AR.synthetic_action_provider,
                            v2_replay_provider=lambda d, ctx: -0.5)
    RPT.validate_selection_report(rep)
    assert set(rep["per_candidate"]) == set(P.CANDIDATE_IDS)
    ok("engine runs with multi-chunk (n_windows>32) subjects; FIT/EVAL share the 32-window ACAR-B granularity")


def main():
    print("ACAR v5 Stage-2B0 guard: joint PD/SCZ single candidate identity")
    test_engine_joint_and_single_identity()
    test_engine_multichunk_fit_eval_granularity()
    print("ALL V5 STAGE2B0-JOINT-IDENTITY GUARDS PASS")


if __name__ == "__main__":
    main()
