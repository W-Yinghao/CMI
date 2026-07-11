from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c79_tables"
RESULT_PATH = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION.json"


def _result():
    return json.loads(RESULT_PATH.read_text())


def _rows(name):
    with (TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


def test_result_binds_authorized_protocol_and_locks():
    result = _result()
    authorization = result["authorization"]
    assert authorization["protocol_commit"] == "ec4834c"
    assert authorization["protocol_sha256"] == "e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587"
    assert authorization["field_lock_commit"] == "35d0c65"
    assert authorization["analysis_lock_commit"] == "7cebf2e"


def test_complete_and_primary_field_counts_are_exact():
    field = _result()["field"]
    assert field["complete_units"] == 1458
    assert field["primary_units"] == 1296
    assert field["target4_engineering_units"] == 162
    assert field["source_rows"] == 6_718_464
    assert field["target_unlabeled_rows"] == 839_808
    assert field["failed_retained_units"] == 0


def test_target_isolation_and_oracle_closure_are_exact():
    field = _result()["field"]
    assert field["training_target_rows"] == 0
    assert field["training_target_label_reads"] == 0
    assert field["target4_primary"] is False
    assert field["same_label_oracle_accessed"] is False
    assert field["construction_evaluation_overlap"] == 0


def test_p1_locked_seed4_values_and_compound_decision():
    p1 = _result()["co_primary"]["P1"]
    assert p1["measurement"]["effect"] == pytest.approx(0.7564555113866)
    assert p1["measurement"]["Holm_p"] == pytest.approx(0.07003891050583658)
    assert p1["measurement"]["pass"] is False
    assert p1["actionability"]["top1_top5_top10"] == [0.125, 0.5, 0.6875]
    assert p1["actionability"]["materiality_pass"] is True
    assert p1["transition_replicates"] is False


def test_p2_locked_seed4_values_and_compound_decision():
    p2 = _result()["co_primary"]["P2"]
    assert p2["local"]["effect"] == pytest.approx(0.21013679454375883)
    assert p2["local"]["worst_control_raw_p"] == pytest.approx(0.092)
    assert p2["local"]["Holm_p"] == pytest.approx(0.368)
    assert p2["local"]["positive_trajectory_cells"] == 32
    assert p2["local"]["pass"] is False
    assert p2["LOTO_incremental_R2"] == pytest.approx(-0.09849749469014357)
    assert p2["LORO_incremental_R2"] == pytest.approx(-0.03294408371628532)
    assert p2["local_nontransport_replicates"] is False


def test_secondary_candidate_decisions_are_exact_nonqualification_only():
    secondary = _result()["secondary"]
    assert secondary["H2R"]["incremental_deviance_reduction"] == pytest.approx(-8.717406284615379)
    assert secondary["H2R"]["qualifies"] is False
    assert secondary["H4R"]["F2_incremental_R2"] == pytest.approx(-0.09628764196416528)
    assert secondary["H4R"]["qualifies"] is False
    assert secondary["H5R"]["F4_incremental_R2"] == pytest.approx(0.010450398396399496)
    assert secondary["H5R"]["qualifies"] is False
    assert secondary["H6R"]["effect"] == pytest.approx(0.41563504917316746)
    assert secondary["H6R"]["familywise_active"] is False


def test_primary_taxonomy_is_seed4_nonreplication_not_provenance_blocker():
    result = _result()
    assert result["primary_taxonomy"] == "C79-E_seed4_does_not_replicate_either_core_pattern"
    assert result["final_gate"] == result["primary_taxonomy"]


def test_cross_seed_synthesis_has_no_pvalue_rescue():
    result = _result()["cross_seed"]
    assert result["all_registered_effect_directions_concordant"] is True
    assert result["combined_p_values_computed"] == 0
    assert result["P2_gate_decision_differs"] is True
    assert result["P2_local_effect_difference_ci"][0] < 0 < result["P2_local_effect_difference_ci"][1]
    assert result["independent_target_population_claim"] is False


def test_gate_concordance_differs_only_for_registered_p2_gates():
    rows = _rows("cross_seed_gate_concordance.csv")
    discordant = {row["gate"] for row in rows if row["gate_concordant"] == "0"}
    assert discordant == {"P2_L", "P2_overall"}
    assert all(row["no_rescue_of_seed4_primary"] == "1" for row in rows)


def test_scientific_red_team_passes_all_checks():
    rows = _rows("scientific_result_red_team.csv")
    assert len(rows) == 17
    assert all(row["passed"] == "1" for row in rows)


def test_all_repairs_are_additive_and_non_scientific():
    rows = _rows("seed4_retry_repair_ledger.csv")
    assert len(rows) == 6
    assert all(row["scientific_registry_changed"] == "0" for row in rows)
    assert all(row["seed4_outcome_access"] == "0" for row in rows)
    assert all(row["outcome_dependent_decision"] == "0" for row in rows)
    assert all(row["locked_implementation_changed"] == "0" for row in rows)


def test_final_report_preserves_claim_and_stop_boundaries():
    report = (REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION.md").read_text()
    assert "training-seed robustness" in report
    assert "not pre-C78S" in report
    assert "same-label oracle remained closed" in report
    assert "does not authorize" in report
    assert "C79-E_seed4_does_not_replicate_either_core_pattern" in report


def test_final_regression_record_has_four_passing_suites():
    rows = _rows("c79e_regression_verification.csv")
    assert len(rows) == 4
    assert {row["suite"] for row in rows} == {"focused", "c65_c79e", "c23_c79e", "full_oaci"}
    assert all(row["status"] == "PASS" for row in rows)
    assert all(row["failed"] == "0" for row in rows)


def test_failed_regression_attempts_are_preserved():
    rows = _rows("c79e_regression_attempt_ledger.csv")
    failures = [row for row in rows if row["status"] == "FAIL_RETAINED"]
    assert {row["job_id"] for row in failures} == {"893710", "893711", "893712"}
    assert all(row["replacement_job"] not in {"", "none"} for row in failures)


def test_c79e_uses_namespaced_risk_and_failure_ledgers():
    risks = _rows("c79e_risk_register.csv")
    failures = _rows("c79e_failure_reason_ledger.csv")
    assert risks and all(row["blocking"] == "0" for row in risks)
    assert len(failures) == 6
    assert all(row["blocking_provenance_failure"] == "0" for row in failures)
