import csv
from pathlib import Path

from oaci.multidataset import c84fl_reconciliation as reconciliation


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c84fl_tables"


def _rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_target_schemas_contain_no_target_label_field():
    for name in ("target_unlabeled_trial_registry_schema.csv", "target_instrumentation_schema.csv"):
        rows = _rows(name)
        assert rows
        assert all(row["target_label_or_derived"] == "0" for row in rows)
        payload_fields = {row["field"] for row in rows} - {"target_label_field_count"}
        assert not any("label" in field.lower() or field.lower() == "y" for field in payload_fields)


def test_complete_context_arithmetic_is_exact():
    rows = _rows("complete_context_arithmetic.csv")
    total = next(row for row in rows if row["dataset"] == "TOTAL")
    assert int(total["target_contexts"]) == 944
    assert int(total["candidate_context_slices"]) == 76464
    assert int(total["remaining_candidate_context_slices"]) == 76221


def test_canary_target_artifacts_are_registered_as_slices_only():
    rows = _rows("c84c_canary_target_slice_registry.csv")
    assert len(rows) == 243
    assert all(row["historical_scope"] == "one_target_subject_slice" for row in rows)
    assert all(row["complete_target_artifact_reusable"] == "0" for row in rows)
    assert all(row["C84F_all_target_subset_replay_required"] == "1" for row in rows)


def test_resource_projection_remains_inside_hard_envelopes():
    rows = _rows("resource_estimate.csv")
    assert rows
    assert all(row["within_envelope"] == "1" for row in rows)
    combined = next(row for row in rows if row["resource"] == "download_plus_derived_projection")
    assert int(combined["estimate"]) < 2 * 1024**4


def test_risk_and_failure_ledgers_expose_level1_blocker():
    risks = _rows("risk_register.csv")
    level = next(row for row in risks if row["risk"] == "level1_training_intervention_undefined")
    assert level["status"] == "OPEN_BLOCKING"
    assert level["blocking"] == "1"
    failures = _rows("failure_reason_ledger.csv")
    assert any(row["failure_id"] == "C84FL-B001" and row["blocking"] == "1" for row in failures)


def test_synthetic_contract_fails_only_the_unbound_level1_readiness_item():
    rows = _rows("synthetic_calibration.csv")
    failed = [row for row in rows if row["passed"] == "0"]
    assert [row["contract"] for row in failed] == ["level1_intervention_bound"]


def test_readiness_report_uses_only_reconciliation_failure_gate():
    text = (REPORTS / "C84FL_PROTOCOL_READINESS.md").read_text()
    assert reconciliation.FAIL_GATE in text
    assert "C84F_FULL_FIELD_IMPLEMENTATION_AND_EXECUTION_LOCK_READY_FOR_PI_AUTHORIZATION" not in text
    assert "972" in text
