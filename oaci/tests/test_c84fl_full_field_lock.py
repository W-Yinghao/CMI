import csv
import hashlib
import json
from pathlib import Path
import subprocess

from oaci.multidataset import c84fl_protocol as protocol
from oaci.multidataset import c84fl_reconciliation as reconciliation


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c84fl_tables"


def _rows(name):
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_protocol_arithmetic_and_canary_scope_are_exact():
    payload = json.loads((REPORTS / "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.json").read_text())
    assert payload["field_arithmetic"] == {
        "C84C_reused_phases": 9,
        "C84C_reused_units": 243,
        "C84C_reused_zoos": 3,
        "candidate_context_slices": 76464,
        "complete_training_phases": 72,
        "complete_units": 1944,
        "complete_zoos": 24,
        "remaining_phases": 63,
        "remaining_units": 1701,
        "remaining_zoos": 21,
        "target_contexts": 944,
        "target_subjects": 118,
    }
    assert payload["C84C_reuse"]["canary_target_contexts"] == 3
    assert payload["C84C_reuse"]["canary_candidate_context_slices"] == 243
    assert not payload["C84C_reuse"]["complete_target_artifact_reusable"]


def test_complete_and_remaining_unit_registries_cover_exact_scope():
    complete = _rows("complete_unit_registry.csv")
    remaining = _rows("remaining_training_registry.csv")
    reusable = _rows("c84c_reusable_unit_registry.csv")
    assert len(complete) == 1944 == len({row["unit_id"] for row in complete})
    assert len(remaining) == 1701 == len({row["unit_id"] for row in remaining})
    assert len(reusable) == 243 == len({row["unit_id"] for row in reusable})
    assert not ({row["unit_id"] for row in remaining} & {row["unit_id"] for row in reusable})


def test_wave_counts_are_729_243_729_after_reuse():
    rows = _rows("wave_registry.csv")
    counts = {
        wave: sum(int(row["candidate_units"]) for row in rows if row["wave"] == wave)
        for wave in ("C84C_REUSE", "A", "B0", "B1")
    }
    assert counts == {"C84C_REUSE": 243, "A": 729, "B0": 243, "B1": 729}
    assert all(row["target_value_release_evidence_allowed"] == "0" for row in rows)


def test_level1_intervention_is_detected_as_open_blocker():
    rows = reconciliation.audit_rows()
    level = next(row for row in rows if row["check_id"] == "L02")
    assert level["observed"] == "NONE"
    assert level["blocking"] == 1
    assert level["passed"] == 0


def test_remaining_scope_contains_972_undefined_level1_units():
    remaining = _rows("remaining_training_registry.csv")
    assert sum(row["level"] == "1" for row in remaining) == 972
    assert sum(row["level"] == "0" for row in remaining) == 729


def test_historical_C84FL_failure_had_no_lock_and_current_field_lock_is_unexecuted():
    historical = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", "6d6030f17dc2cdf8c8b180a9376632e238d42e75", "oaci/reports"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout.splitlines()
    assert "oaci/reports/C84F_EXECUTION_LOCK.json" not in historical
    lock = json.loads((REPORTS / "C84F_EXECUTION_LOCK.json").read_text())
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert lock["scope"]["real_execution_at_lock"] is False
    assert not any(REPORTS.glob("C84S*EXECUTION_LOCK*.json"))


def test_protocol_hash_sidecar_replays():
    path = REPORTS / "C84F_FULL_FIELD_EXECUTION_AND_MANIFEST_PROTOCOL.json"
    expected = (path.with_suffix(".sha256").read_text().split()[0])
    assert protocol.sha256_file(path) == expected


def test_overall_report_is_machine_readable_and_hash_bound():
    markdown = REPORTS / "C84FL_OVERALL_REPORT.md"
    machine = REPORTS / "C84FL_OVERALL_REPORT.json"
    sidecar = REPORTS / "C84FL_OVERALL_REPORT.sha256"
    assert markdown.exists() and machine.exists() and sidecar.exists()
    payload = json.loads(machine.read_text())
    assert payload["final_gate"] == reconciliation.FAIL_GATE
    assert payload["arithmetic"]["remaining_level1_units"] == 972
    assert not payload["execution_objects"]["c84f_execution_lock_created"]
    expected = {
        row.split()[1]: row.split()[0]
        for row in sidecar.read_text().splitlines()
        if row.strip()
    }
    assert expected == {
        "C84FL_OVERALL_REPORT.md": hashlib.sha256(markdown.read_bytes()).hexdigest(),
        "C84FL_OVERALL_REPORT.json": hashlib.sha256(machine.read_bytes()).hexdigest(),
    }
