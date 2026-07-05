"""Guard (Stage-2A, PROP 3): Stage-2 selection consumes ONLY the 10 canonical selection refs — diseases {PD,SCZ}, folds 0..4,
seed 20260711 — and partitions the other 20 (seeds 20260712/20260713) OUT. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5 import protocol as P
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5 import stage2_selection_runner as RUN2
from acar.v5.substrate import plan as PLAN
from acar.v5.tests._util import ok, stage1b_finalized_package

RUN = "run-syn-0001"
EXPECTED_SELECTION = tuple(sorted(r["ref"] for r in PLAN.selection_refs()))


def test_selection_partition():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        view = INTAKE.admit_and_validate_registry(d, RUN)
        assert view.selection_refs == EXPECTED_SELECTION
        assert len(view.selection_refs) == 10
        assert all(r.endswith(f"seed{P.SELECTION_SEED}") for r in view.selection_refs)
        assert len(view.robustness_only_refs) == 20
        assert all(not r.endswith(f"seed{P.SELECTION_SEED}") for r in view.robustness_only_refs)
        # PD + SCZ × folds 0..4 exactly once each
        assert sorted(view.selection_refs) == sorted(
            f"{dis}/fold{f}/seed{P.SELECTION_SEED}" for dis in ("PD", "SCZ") for f in range(P.OUTER_K))
        # the runner enumerator agrees and returns exactly the 10
        assert RUN2.selection_refs(view) == list(view.selection_refs)
    ok("selection consumes exactly the 10 seed-20260711 refs (PD+SCZ × folds 0..4); 20 robustness refs excluded (PROP 3)")


def main():
    print("ACAR v5 Stage-2A guard: selection uses the selection seed only (PROP 3)")
    test_selection_partition()
    print("ALL V5 STAGE2A-SELECTION-SEED-ONLY GUARDS PASS")


if __name__ == "__main__":
    main()
