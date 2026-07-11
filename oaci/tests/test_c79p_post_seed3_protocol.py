from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

from oaci.conditioned_ceiling_coverage import c79e_seed4_replication as c79e
from oaci.conditioned_ceiling_coverage import c79p_post_seed3_protocol as c79p


def test_replacement_protocol_hash_and_epistemic_status_replay():
    protocol, digest = c79p.load_protocol()
    assert digest == "e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587"
    status = protocol["epistemic_status"]
    assert status["designed_after_C78S_outcomes"]
    assert status["prospective_to_seed4_checkpoint_outcomes"]
    assert not status["pre_C78S_confirmatory_protocol"]
    assert status["training_seed_robustness_only"]
    assert not status["new_target_population_confirmation"]
    assert not protocol["authorization"]["received"]


def test_historical_protocol_is_retained_but_has_no_execution_authority():
    historical = c79p.REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION_PROTOCOL.json"
    assert historical.is_file()
    assert c79p.sha256_file(historical) == "7732986513793725d58933d487f5bc8f4fc68bfad0857bb4734a450b41ca5dd4"
    ledger = c79p.read_csv(c79p.TABLE_DIR / "protocol_supersession_ledger.csv")[0]
    assert ledger["pre_C78S_confirmation_valid"] == "0"
    assert ledger["seed4_execution_authority"] == "0"
    assert ledger["superseded_for_future_execution"] == "1"
    assert ledger["history_rewritten"] == "0"


def test_scientific_registry_is_complete_and_unconditional():
    audit = c79p.validate_registry()
    assert audit == {
        "paths": 10,
        "categories": 16,
        "bound_cells": 160,
        "blank_cells": 0,
        "implicit_inherited_cells": 0,
        "active_after_Holm_cells": 0,
    }
    rows = c79p.read_csv(c79p.REGISTRY_PATH)
    assert {row["completeness"] for row in rows} == {"16/16"}


def test_expected_seed4_manifest_arithmetic_and_uniqueness():
    rows = c79p.expected_seed4_units()
    assert len(rows) == 1458
    assert len({row["unit_id"] for row in rows}) == 1458
    assert sum(row["primary"] for row in rows) == 1296
    assert sum(row["target"] == 4 for row in rows) == 162
    assert sum(row["target"] == 4 and row["primary"] for row in rows) == 0
    assert {regime: sum(row["regime"] == regime for row in rows) for regime in c79p.REGIMES} == {
        "ERM": 18,
        "OACI": 720,
        "SRC": 720,
    }


def test_expected_manifest_has_fixed_cadence_and_random_baselines():
    rows = c79p.expected_seed4_units()
    for target in c79p.TARGET_ORDER:
        for level in c79p.LEVELS:
            cell = [row for row in rows if row["target"] == target and row["level"] == level]
            assert len(cell) == 81
            assert sum(row["regime"] == "ERM" for row in cell) == 1
            for regime in ("OACI", "SRC"):
                trajectory = [row for row in cell if row["regime"] == regime]
                assert [row["epoch"] for row in trajectory] == list(range(4, 200, 5))
                assert [row["trajectory_order"] for row in trajectory] == list(range(1, 41))
    assert 1 / 81 == pytest.approx(0.012345679012345678)
    assert 5 / 81 == pytest.approx(0.06172839506172839)
    assert 10 / 81 == pytest.approx(0.12345679012345678)


def test_training_phase_registry_is_54_target_blind_cells():
    phases = c79p.expected_training_phases()
    assert len(phases) == 54
    assert len({row["phase_id"] for row in phases}) == 54
    assert all(row["scientific_outcome_gate_allowed"] == 0 for row in phases)
    assert sum(row["regime"] == "ERM" for row in phases) == 18
    assert sum(row["regime"] == "OACI" for row in phases) == 18
    assert sum(row["regime"] == "SRC" for row in phases) == 18


def test_seed3_reference_values_exact_replay():
    rows = c79p.exact_seed3_replay_rows()
    assert len(rows) == len(c79p.EXPECTED_REFERENCE_VALUES) + 7
    assert all(row["passed"] for row in rows)


def test_fixed_label_partition_replays_with_zero_overlap():
    protocol, _ = c79p.load_protocol()
    rows = c79p.read_csv(c79p.LABEL_SPLIT_PATH)
    assert len(rows) == 8
    assert {int(row["target_id"]) for row in rows} == set(c79p.PRIMARY_TARGETS)
    assert all(row["overlap_rows"] == "0" for row in rows)
    assert all(int(row["union_rows"]) == 576 for row in rows)
    assert c79p.sha256_file(c79p.LABEL_SPLIT_PATH) == protocol["label_split"]["source_table_sha256"]


def test_rng_registry_is_fixed_and_nonadaptive():
    rows = c79p.rng_stream_rows()
    assert len(rows) == 9
    assert all(row["adaptive"] == 0 for row in rows)
    assert any(row["formula"] == "7803+3000+100000*scheme_index+replicate" for row in rows)
    assert any(row["formula"] == "7803+4000+replicate" for row in rows)


def test_registered_holm_mapping_replays_seed3_family():
    raw = {
        "P1_M": 0.011673151750972763,
        "H2R": 0.896,
        "P2_L": 0.002,
        "H4R": 1.0,
        "H5R": 1.0,
        "H6R": 0.019455252918287938,
    }
    adjusted = c79e._holm(raw)
    assert adjusted["P2_L"] == pytest.approx(0.012)
    assert adjusted["P1_M"] == pytest.approx(0.058365758754863814)
    assert adjusted["H6R"] == pytest.approx(0.07782101167315175)
    assert adjusted["H2R"] == pytest.approx(1.0)


def test_field_binding_contract_covers_seed_scope_and_oracle_boundary():
    contract = c79e.field_binding_contract()
    assert contract["guard_before_reference_import"]
    assert contract["substitutions"]["SEED"] == 4
    assert contract["substitutions"]["TARGETS"] == list(c79p.TARGET_ORDER)
    assert contract["substitutions"]["REMAINING_UNITS"] == 1458
    assert not contract["same_label_oracle_created"]


def test_analysis_binding_contract_is_seed4_only_and_unconditional():
    contract = c79e.analysis_binding_contract()
    assert contract["guard_before_reference_import"]
    assert contract["substitutions"]["SEED"] == 4
    assert contract["substitutions"]["PRIMARY_TARGETS"] == list(c79p.PRIMARY_TARGETS)
    assert contract["all_registered_paths_unconditional"]
    assert not contract["active_after_Holm_runtime_selection"]
    assert not contract["target4_primary"]
    assert not contract["same_label_oracle_reachable"]


def test_real_execution_fails_closed_without_future_authorization_record():
    assert not c79p.AUTHORIZATION_RECORD_PATH.exists()
    with pytest.raises((FileNotFoundError, PermissionError)):
        c79p.require_c79e_authorization()


def test_show_binding_contract_is_the_only_unauthorized_adapter_command(capsys):
    assert c79e.main(["show-binding-contract"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["field"]["substitutions"]["SEED"] == 4
    with pytest.raises((FileNotFoundError, PermissionError)):
        c79e.main(["train-oaci-erm", "--target", "1"])


def test_unauthorized_command_does_not_import_training_or_EEG_modules(monkeypatch):
    protected = {
        "oaci.conditioned_ceiling_coverage.c78f_train",
        "oaci.conditioned_ceiling_coverage.c78f_instrument",
        "oaci.data.eeg.bnci",
        "torch",
    }
    before = {name for name in protected if name in sys.modules}
    with pytest.raises((FileNotFoundError, PermissionError)):
        c79e.main(["train-src", "--target", "1"])
    after = {name for name in protected if name in sys.modules}
    assert after == before


def test_expected_manifest_artifact_matches_pure_generator():
    assert c79p.EXPECTED_MANIFEST_JSON.is_file()
    assert c79p.EXPECTED_MANIFEST_CSV.is_file()
    payload = json.loads(c79p.EXPECTED_MANIFEST_JSON.read_text())
    rows = c79p.read_csv(c79p.EXPECTED_MANIFEST_CSV)
    assert payload["engineering_units"] == 1458
    assert payload["primary_units"] == 1296
    assert payload["unit_manifest_sha256"] == c79p.sha256_file(c79p.EXPECTED_MANIFEST_CSV)
    assert len(rows) == 1458
    assert all(row["seed"] == "4" for row in rows)


def test_execution_locks_are_distinct_committed_fail_closed_objects():
    field, field_sha = c79p.load_field_lock()
    analysis, analysis_sha = c79p.load_analysis_lock()
    assert len(field_sha) == len(analysis_sha) == 64
    assert field_sha != analysis_sha
    assert not field["authorization"]["received"]
    assert not analysis["authorization"]["received"]
    assert analysis["field_lock_sha256"] == field_sha
    assert analysis["all_paths_unconditional"]
    assert not analysis["same_label_oracle_reachable"]


def test_preexecution_red_team_and_readiness_gate_are_consistent():
    result = json.loads(c79p.READINESS_JSON.read_text())
    assert result["final_gate"] == c79p.FINAL_GATE
    assert result["primary"] == "C79P-A_post_seed3_replication_protocol_locked_complete"
    assert result["pre_execution_red_team"]["failed"] == 0
    assert not result["execution_boundary"]["C79E_authorized"]
    assert all(value == 0 or value is False for value in result["execution_boundary"].values())


def test_c79p_reports_and_required_tables_exist():
    reports = [
        c79p.IMPLEMENTATION_REPLAY_REPORT,
        c79p.PRE_EXECUTION_RED_TEAM,
        c79p.EXPECTED_MANIFEST_JSON,
        c79p.LOCK_LEDGER_PATH,
        c79p.READINESS_REPORT,
        c79p.READINESS_JSON,
    ]
    tables = [
        "implementation_hashes.csv",
        "exact_seed3_replay.csv",
        "RNG_stream_registry.csv",
        "oracle_reachability_test.csv",
        "label_view_access_test.csv",
        "all_paths_unconditional_execution_test.csv",
        "expected_seed4_field_manifest.csv",
        "pre_execution_red_team_checks.csv",
    ]
    assert all(path.is_file() and path.stat().st_size > 0 for path in reports)
    assert all((c79p.TABLE_DIR / name).is_file() for name in tables)


def test_git_contains_no_c79p_raw_payload_or_large_artifact():
    scan = c79p._tracked_payload_scan()
    assert scan["payload_over_50MiB"] == 0
    assert scan["raw_cache_or_weights"] == 0


def test_readiness_does_not_authorize_scope_expansion():
    text = c79p.READINESS_REPORT.read_text()
    assert c79p.FINAL_GATE in text
    assert "authorizes no seed-4 work" in text
    for forbidden in ("C80", "BNCI2014_004", "same-label oracle", "manuscript"):
        assert forbidden in text
