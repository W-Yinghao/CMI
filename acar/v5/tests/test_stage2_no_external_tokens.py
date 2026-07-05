"""Guard (Stage-2A, PROP 7): no external / provisional / excluded / held-out site token — and no other-disease DEV cohort —
may appear anywhere in the admitted selection package. Injecting one into the registry metadata fails closed. Synthetic only."""
from __future__ import annotations
import tempfile
from acar.v5 import protocol as P
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5.tests._util import expect_raises, ok, stage1b_finalized_package, synthetic_registry

RUN = "run-syn-0001"


def _package_with_meta(d, meta_over):
    reg = synthetic_registry(meta_over=meta_over)
    return stage1b_finalized_package(d, RUN, registry=reg)


def test_forbidden_site_tokens_rejected():
    # each forbidden external/provisional/excluded site token injected into cohort_inclusion_list must reject
    for tok in INTAKE.FORBIDDEN_SITE_TOKENS:
        with tempfile.TemporaryDirectory() as d:
            _package_with_meta(d, lambda ref, dis, f, s, t=tok: {"cohort_inclusion_list": list(P.DEV_COHORTS[dis]) + [t]})
            expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.admit_and_validate_registry(d, RUN))
    ok(f"each forbidden site token {list(INTAKE.FORBIDDEN_SITE_TOKENS)} in the package → Stage2IntakeError (PROP 7)")


def test_other_disease_cohort_rejected():
    with tempfile.TemporaryDirectory() as d:
        # inject an SCZ cohort (ds004000) into PD entries' cohort list
        _package_with_meta(d, lambda ref, dis, f, s: {
            "cohort_inclusion_list": list(P.DEV_COHORTS[dis]) + (["ds004000"] if dis == "PD" else [])})
        expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.admit_and_validate_registry(d, RUN))
    ok("an other-disease DEV cohort (SCZ ds004000 in a PD entry) → Stage2IntakeError (PROP 7)")


def test_clean_package_admits():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)                          # default meta = disease-matched DEV cohorts only
        view = INTAKE.admit_and_validate_registry(d, RUN)
        assert len(view.all_refs) == 30
    ok("a clean disease-matched package admits (no false positive) (PROP 7)")


def main():
    print("ACAR v5 Stage-2A guard: no external/held-out/foreign tokens in the selection package (PROP 7)")
    test_forbidden_site_tokens_rejected()
    test_other_disease_cohort_rejected()
    test_clean_package_admits()
    print("ALL V5 STAGE2A-NO-EXTERNAL-TOKENS GUARDS PASS")


if __name__ == "__main__":
    main()
