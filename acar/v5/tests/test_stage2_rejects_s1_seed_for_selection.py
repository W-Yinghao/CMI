"""Guard (Stage-2A, PROP 4): seeds 20260712 and 20260713 are reserved for S1/Stage-4 robustness and can NEVER influence Stage-2
candidate identity. Any attempt to use an S1-seed ref as a selection input fails closed. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5 import protocol as P
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5 import stage2_selection_runner as RUN2
from acar.v5.substrate import plan as PLAN
from acar.v5.tests._util import expect_raises, ok, stage1b_finalized_package

RUN = "run-syn-0001"


def test_s1_seed_refs_rejected_as_selection_input():
    for seed in (20260712, 20260713):
        ref = f"PD/fold0/seed{seed}"
        expect_raises(RUN2.Stage2RunnerError, lambda r=ref: RUN2.assert_selection_input_allowed(r))
        # the plan's seed-role guard also refuses the selection role for 12/13
        expect_raises(ValueError, lambda s=seed: PLAN.assert_seed_role(s, PLAN.SELECTION_ROLE))
    # the canonical selection seed IS allowed
    assert RUN2.assert_selection_input_allowed(f"SCZ/fold3/seed{P.SELECTION_SEED}") is True
    ok("seeds 20260712/20260713 rejected as selection input; selection role refused for them (PROP 4)")


def test_view_excludes_s1_seed_refs():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        view = INTAKE.admit_and_validate_registry(d, RUN)
        for ref in view.robustness_only_refs:
            assert not view.is_selection_ref(ref)
            expect_raises(INTAKE.Stage2IntakeError, lambda r=ref: view.assert_selection_ref(r))
        # every robustness-only ref carries a non-selection seed
        assert all(int(r.rsplit("seed", 1)[1]) in (20260712, 20260713) for r in view.robustness_only_refs)
    ok("package view keeps all seed-12/13 refs OUT of the selection set and refuses to admit them as selection refs (PROP 4)")


def main():
    print("ACAR v5 Stage-2A guard: S1 seeds cannot drive selection (PROP 4)")
    test_s1_seed_refs_rejected_as_selection_input()
    test_view_excludes_s1_seed_refs()
    print("ALL V5 STAGE2A-REJECTS-S1-SEED GUARDS PASS")


if __name__ == "__main__":
    main()
