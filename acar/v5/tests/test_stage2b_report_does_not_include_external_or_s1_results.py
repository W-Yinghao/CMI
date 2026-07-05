"""Guard (Stage-2B0): the Stage-2 selection report is EITHER a SELECTED candidate OR a DEV_STOP, and it may NEVER carry an
S1/S2/S3 robustness or external/lockbox result. Synthetic only."""
from __future__ import annotations
from acar.v5 import stage2_selection_report as RPT
from acar.v5.tests._util import expect_raises, ok


def _stop():
    return RPT.build_selection_report(outcome=RPT.OUTCOME_DEV_STOP, selected_candidate_id=None, per_candidate={},
                                      per_disease={}, macro={}, holm_family_alpha=0.05, objective="obj",
                                      notes={"dev_stop_reason": "none"})


def test_valid_reports():
    RPT.validate_selection_report(_stop())
    RPT.build_selection_report(outcome=RPT.OUTCOME_SELECTED, selected_candidate_id="V5-P3-001", per_candidate={},
                               per_disease={}, macro={}, holm_family_alpha=0.05, objective="obj", notes={})
    ok("a well-formed DEV_STOP and a well-formed SELECTED report validate")


def test_outcome_consistency():
    # SELECTED requires an id; DEV_STOP forbids one
    expect_raises(RPT.Stage2ReportError, lambda: RPT.build_selection_report(
        outcome=RPT.OUTCOME_SELECTED, selected_candidate_id=None, per_candidate={}, per_disease={}, macro={},
        holm_family_alpha=0.05, objective="obj", notes={}))
    bad_stop = _stop()
    bad_stop["selected_candidate_id"] = "V5-P1-001"
    expect_raises(RPT.Stage2ReportError, lambda: RPT.validate_selection_report(bad_stop))
    ok("SELECTED requires an id; DEV_STOP must have selected_candidate_id=None")


def test_forbidden_keys_rejected():
    for inject in ({"external": {"red": 1}}, {"s1_robustness": {}}, {"lockbox": True}):
        bad = _stop()
        bad.update(inject)
        expect_raises(RPT.Stage2ReportError, lambda b=bad: RPT.validate_selection_report(b))
    bad_notes = _stop()
    bad_notes["notes"] = {"s2": "leak"}
    expect_raises(RPT.Stage2ReportError, lambda: RPT.validate_selection_report(bad_notes))
    ok("an S1/S2/S3 or external/lockbox key at the top level or in notes → Stage2ReportError")


def main():
    print("ACAR v5 Stage-2B0 guard: report excludes external/S1/S2/S3 results")
    test_valid_reports()
    test_outcome_consistency()
    test_forbidden_keys_rejected()
    print("ALL V5 STAGE2B0-REPORT-NO-EXTERNAL GUARDS PASS")


if __name__ == "__main__":
    main()
