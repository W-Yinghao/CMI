from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from oaci.conditioned_ceiling_coverage import c79_seed4_locked_confirmation as c79


def _rows(name: str) -> list[dict[str, str]]:
    with (c79.TABLE_DIR / name).open(newline="") as stream:
        return list(csv.DictReader(stream))


def test_c79_review_is_mode_r_only_and_ends_at_timing_blocker():
    replay = json.loads(c79.REPLAY_JSON.read_text())
    assert replay["mode"] == "R_review_only"
    assert replay["final_gate"] == "C79_PROTOCOL_OR_TIMING_REPAIR_REQUIRED"
    assert replay["stop_rule"] == "Phase_1_failed_no_implementation_lock_no_Mode_E"


def test_c79_final_protocol_hash_replays_exactly():
    assert c79._sha256(c79.C79_PROTOCOL) == c79.C79_PROTOCOL_EXPECTED_SHA
    assert c79.C79_PROTOCOL_SHA.read_text().strip() == c79.C79_PROTOCOL_EXPECTED_SHA


def test_c79_preoutcome_artifact_is_explicitly_nonfinal_skeleton():
    skeleton = json.loads(c79.C79_SKELETON.read_text())
    assert c79._sha256(c79.C79_SKELETON) == c79.C79_SKELETON_EXPECTED_SHA
    assert skeleton["status"] == "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED"
    assert "C79 final protocol committed and hashed" in skeleton["depends_on"]
    assert c79._first_path_commit(c79.C79_SKELETON) == c79.C79_SKELETON_COMMIT


def test_c79_final_protocol_first_appears_with_c78s_result():
    assert c79._first_path_commit(c79.C79_PROTOCOL) == c79.C79_FINAL_PROTOCOL_COMMIT
    assert c79.C79_FINAL_PROTOCOL_COMMIT == c79.C78S_COMMITS["result"]


def test_c79_final_protocol_was_created_after_c78s_outcomes():
    timeline = {row["event"]: row for row in _rows("c79_protocol_timing.csv")}
    created = c79._iso_to_epoch(timeline["C79_final_protocol_created"]["at_utc"])
    assert created > c79._iso_to_epoch(
        timeline["C78S_first_scientific_outcome_H1_complete"]["at_utc"]
    )
    assert created > c79._iso_to_epoch(timeline["C78S_H3_H4_H5_complete"]["at_utc"])
    assert created > c79._iso_to_epoch(timeline["C78S_H2_complete"]["at_utc"])


def test_c79_final_protocol_scope_is_outcome_adaptive():
    protocol = json.loads(c79.C79_PROTOCOL.read_text())
    assert protocol["C78S_active_hypotheses_to_confirm"] == ["H3", "H4", "H5"]
    source = (
        c79.REPO_ROOT
        / "oaci/conditioned_ceiling_coverage/c78s_seed3_scientific_analysis.py"
    ).read_text()
    assert 'if row["active_after_Holm"]' in source
    assert "c79 = _write_c79_protocol(primary_rows" in source


def test_c79_adaptive_generator_was_transparently_precommitted_but_is_not_final_protocol():
    replay = json.loads(c79.REPLAY_JSON.read_text())
    assert replay["protocol"]["adaptive_generator_rule_first_commit"] == c79.C78S_COMMITS["implementation"]
    assert replay["protocol"]["adaptive_generator_rule_predates_first_outcome"]
    assert replay["protocol"]["first_commit"] == c79.C78S_COMMITS["result"]


def test_c79_final_protocol_exact_registry_is_incomplete():
    registry = _rows("c79_protocol_registry.csv")
    assert len(registry) == 16
    assert sum(row["present_in_final_C79_protocol"] == "1" for row in registry) == 2
    missing = {row["component"] for row in registry if row["blocking_if_strict_confirmation"] == "1"}
    assert "construction/evaluation trial-ID hashes" in missing
    assert "H3 exact feature block, kernel, bandwidth, and scaling" in missing
    assert "null RNG streams" in missing


def test_c78s_protocol_and_all_reference_values_replay():
    protocol = _rows("c78s_protocol_replay.csv")[0]
    assert protocol["sha_match"] == "1"
    assert protocol["observed_sha256"] == c79.C78S_PROTOCOL_EXPECTED_SHA
    references = _rows("c78s_result_replay.csv")
    assert len(references) == len(c79.C78S_REFERENCES)
    assert all(row["passed"] == "1" for row in references)


def test_random_expected_regret_replay_uses_standardized_estimand():
    row = next(row for row in _rows("c78s_result_replay.csv") if row["metric"] == "H1_random_expected_regret")
    assert float(row["observed"]) == pytest.approx(0.48198217863506276)
    assert float(row["expected"]) == pytest.approx(0.4820)
    assert row["passed"] == "1"


def test_c78s_provenance_correction_replays_eight_checks():
    rows = _rows("c78s_provenance_repair_replay.csv")
    checks = [row for row in rows if row["record_type"] == "repair_check"]
    files = [row for row in rows if row["record_type"] == "changed_file"]
    assert len(checks) == 8
    assert all(row["passed"] == "1" for row in checks)
    assert len(files) == 5
    assert all(row["code_or_protocol_file"] == "0" for row in files)
    assert all(row["scientific_result_or_primary_table"] == "0" for row in files)


def test_c78s_provenance_correction_hashes_are_before_after_distinct():
    files = [
        row for row in _rows("c78s_provenance_repair_replay.csv")
        if row["record_type"] == "changed_file"
    ]
    assert all(len(row["before_sha256"]) == 64 for row in files)
    assert all(len(row["after_sha256"]) == 64 for row in files)
    assert all(row["before_sha256"] != row["after_sha256"] for row in files)


def test_c78f_and_c78s_commit_chains_replay():
    for name in ("c78f_commit_manifest_replay.csv", "c78s_commit_replay.csv"):
        rows = _rows(name)
        assert rows
        assert all(row["exists"] == "1" for row in rows)
        assert all(row["ancestor_of_review_anchor"] == "1" for row in rows)


def test_c79_has_three_explicit_protocol_blockers():
    risks = _rows("risk_register.csv")
    blockers = {row["risk"] for row in risks if row["blocking_open"] == "1"}
    assert blockers == {
        "C79_protocol_post_C78S_outcome",
        "C79_protocol_outcome_adaptive_hypothesis_filter",
        "C79_exact_scientific_registry_incomplete",
    }


def test_c79_review_touched_no_seed4_or_oracle_path():
    boundary = _rows("c79_mode_r_execution_boundary.csv")[0]
    protected = [
        "seed4_EEG_loads",
        "seed4_Slurm_jobs",
        "training",
        "forward_or_reinference",
        "GPU",
        "seed4_checkpoints",
        "seed4_caches",
        "seed4_label_views",
        "same_label_oracle",
        "BNCI2014_004",
        "manuscript",
    ]
    assert boundary["mode"] == "R"
    assert all(boundary[name] == "0" for name in protected)


def test_c79_stops_before_execution_lock_and_expected_manifest():
    boundary = _rows("c79_mode_r_execution_boundary.csv")[0]
    assert boundary["execution_lock_created"] == "0"
    assert boundary["expected_seed4_manifest_created"] == "0"
    assert not (c79.REPORT_DIR / "C79_EXECUTION_LOCK_LEDGER.json").exists()
    assert not (c79.REPORT_DIR / "C79_EXPECTED_SEED4_MANIFEST.json").exists()


def test_c79_cli_has_no_execution_command():
    with pytest.raises(SystemExit):
        c79.main(["execute"])


def test_c79_review_red_team_passes_the_negative_gate():
    red = json.loads(c79.RED_TEAM_JSON.read_text())
    assert red["passed"] == red["total"]
    assert red["blockers"] == 0
    assert red["review_gate"] == c79.FINAL_GATE
    assert not red["positive_execution_readiness"]


def test_c79_report_does_not_claim_execution_readiness():
    text = c79.REVIEW_REPORT.read_text()
    assert "C79_PROTOCOL_OR_TIMING_REPAIR_REQUIRED" in text
    assert "Mode E remains prohibited" in c79.RED_TEAM_REPORT.read_text()
    assert "C79_PROTOCOL_REVIEW_READY_FOR_PI_AUTHORIZATION" not in text


def test_c79_artifact_manifest_has_no_raw_payload():
    rows = _rows("artifact_manifest.csv")
    assert rows
    assert all(row["raw_EEG_or_weight_payload"] == "0" for row in rows)
    assert max(int(row["size_bytes"]) for row in rows) < 50 * 1024 * 1024
    for row in rows:
        path = c79.REPO_ROOT / row["path"]
        assert path.is_file()
        assert c79._sha256(path) == row["sha256"]


def test_c79_review_reports_are_present_and_nonempty():
    required = [
        c79.REPLAY_JSON,
        c79.TIMING_REPORT,
        c79.REPAIR_REPORT,
        c79.REVIEW_REPORT,
        c79.RED_TEAM_JSON,
        c79.RED_TEAM_REPORT,
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in required)
