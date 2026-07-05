"""Guard (Stage-2A, PROP 1+2): package intake admits a valid Stage-1B package (admit_run succeeds; registry has exactly 30 refs =
the canonical set) and FAILS CLOSED on a missing/tampered registry or marker. Synthetic only (no real data, no labels)."""
from __future__ import annotations
import os
import tempfile
from acar.v5 import stage2_package_intake as INTAKE
from acar.v5.substrate import stage1b_authorization as SA
from acar.v5.substrate import stage1b_registry_io as RIO
from acar.v5.tests._util import expect_raises, ok, stage1b_finalized_package

RUN = "run-syn-0001"


def test_admits_valid_package():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        view = INTAKE.admit_and_validate_registry(d, RUN)
        assert len(view.all_refs) == 30
        assert set(view.all_refs) == set(SA.CANONICAL_FOLD_REFS)
        assert view.output_root == d and view.run_id == RUN
    ok("valid Stage-1B package → admitted; registry is exactly the 30 canonical fold refs (PROP 1+2)")


def test_missing_registry_or_marker_fail_closed():
    for kw in ({"reg": False}, {"marker": False}):
        with tempfile.TemporaryDirectory() as d:
            stage1b_finalized_package(d, RUN, **kw)
            expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.admit_and_validate_registry(d, RUN))
    ok("missing registry.json OR FINALIZED.json → Stage2IntakeError (fail-closed)")


def test_tampered_marker_fail_closed():
    # bad registry_sha256, wrong n_refs, non-FINALIZED status all reject
    for kw in ({"sha": "0" * 64}, {"n_refs": 29}, {"status": "DRAFT"}):
        with tempfile.TemporaryDirectory() as d:
            stage1b_finalized_package(d, RUN, **kw)
            expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.admit_and_validate_registry(d, RUN))
    ok("bad registry_sha256 / n_refs!=30 / status!=FINALIZED → Stage2IntakeError (fail-closed)")


def test_tampered_registry_bytes_fail_closed():
    with tempfile.TemporaryDirectory() as d:
        stage1b_finalized_package(d, RUN)
        with open(os.path.join(d, RUN, RIO.REGISTRY_FILE), "ab") as f:
            f.write(b" ")                                            # sha no longer matches the marker
        expect_raises(INTAKE.Stage2IntakeError, lambda: INTAKE.admit_and_validate_registry(d, RUN))
    ok("appending a byte to registry.json (sha drift) → Stage2IntakeError (fail-closed)")


def main():
    print("ACAR v5 Stage-2A guard: package intake admits Stage-1B package (PROP 1+2)")
    test_admits_valid_package()
    test_missing_registry_or_marker_fail_closed()
    test_tampered_marker_fail_closed()
    test_tampered_registry_bytes_fail_closed()
    print("ALL V5 STAGE2A-INTAKE-ADMITS GUARDS PASS")


if __name__ == "__main__":
    main()
