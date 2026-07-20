from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from oaci.conditioned_ceiling_coverage import c78_seed3_instrumented_pilot as c78


def _rows(name: str) -> list[dict[str, str]]:
    with open(c78.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def test_c78_protocol_hash_and_unique_exact_token_replay():
    protocol, digest, token, field = c78.load_protocol()
    assert digest == c78.PROTOCOL_SHA_PATH.read_text().strip()
    assert field == "authorization.exact_token"
    assert token == "C78_SEED3_MULTIREGIME_INSTRUMENTED_PILOT_AUTHORIZED"
    assert protocol["authorization"]["accepted_channel"] == "exact_CLI_argument_only"
    assert protocol["authorization"]["prompt_text_is_authorization"] is False
    assert protocol["authorization"]["environment_is_authorization"] is False


def test_c78_authorization_is_exact_not_trimmed_or_inferred(monkeypatch):
    _, _, token, _ = c78.load_protocol()
    assert c78.authorization_matches(token, token)
    assert not c78.authorization_matches(None, token)
    assert not c78.authorization_matches("", token)
    assert not c78.authorization_matches(f" {token}", token)
    assert not c78.authorization_matches(f"{token}\n", token)
    assert not c78.authorization_matches("我显式授权", token)
    monkeypatch.setenv("C78_AUTHORIZATION_TOKEN", token)
    assert not c78.authorization_matches(None, token)


def test_c78_manifest_is_exact_82_unit_OACI_ERM_canary():
    protocol, _, _, _ = c78.load_protocol()
    rows = c78.pilot_unit_manifest(protocol)
    assert len(rows) == 82
    assert len({row["unit_id"] for row in rows}) == 82
    assert {row["dataset"] for row in rows} == {"BNCI2014_001"}
    assert {row["target"] for row in rows} == {4}
    assert {row["seed"] for row in rows} == {3}
    assert {row["level"] for row in rows} == {0, 1}
    assert {row["regime"] for row in rows} == {"ERM", "OACI"}
    assert sum(row["regime"] == "ERM" for row in rows) == 2
    assert sum(row["regime"] == "OACI" for row in rows) == 80
    assert not any(row["regime"] == "SRC" for row in rows)
    for level in (0, 1):
        epochs = [row["epoch"] for row in rows if row["level"] == level and row["regime"] == "OACI"]
        assert epochs == list(range(4, 200, 5))


def test_c78_module_import_boundary_has_no_eeg_cuda_or_training_import():
    source = Path(c78.__file__).read_text()
    prefix = source.split("def _dummy_abi", 1)[0]
    assert "oaci.data.eeg" not in prefix
    assert "runner.bnci_data" not in prefix
    assert "train.engine" not in prefix
    assert "runtime.cuda" not in prefix
    assert "import torch" not in prefix


def test_c78_dummy_abi_is_cpu_only_and_exact():
    row = c78._dummy_abi()
    assert row["passed"] == 1
    assert row["device"] == "cpu"
    assert row["Wz_plus_b_max_abs"] <= 1e-6
    assert row["repeat_logit_max_abs"] == 0.0
    assert row["repeat_z_max_abs"] == 0.0
    assert row["real_EEG_rows_loaded"] == 0
    assert row["real_training_steps"] == 0
    assert row["CUDA_initialized"] == 0


def test_c78_view_contract_physically_separates_target_information_classes():
    rows = {row["view"]: row for row in c78._view_rows()}
    assert rows["strict_source_trial_view"]["uses_target_rows"] == 0
    assert rows["target_unlabeled_trial_view"]["uses_target_labels"] == 0
    assert rows["target_construction_view"]["uses_evaluation_labels"] == 0
    assert rows["target_evaluation_view"]["uses_evaluation_labels"] == 1
    assert rows["same_label_oracle_view"]["oracle_descriptor_visible"] == 1
    assert all(row["available_to_training"] == 0 for row in rows.values())
    assert all(row["physically_separate"] == 1 for row in rows.values())


def test_c78_valid_token_cannot_silently_run_through_preflight():
    _, _, token, _ = c78.load_protocol()
    with pytest.raises(RuntimeError, match="separate command"):
        c78.run_preflight(token)


def test_c78_state_if_present_proves_zero_real_execution():
    if not c78.STATE_PATH.exists():
        pytest.skip("C78 P0 state not generated yet")
    state = json.loads(c78.STATE_PATH.read_text())
    assert state["final_gate_candidate"] in {"PILOT_READY_BUT_NOT_AUTHORIZED", "TRAINING_OR_INSTRUMENTATION_BLOCKER"}
    assert state["authorization"] == {
        "CLI_argument_present": False,
        "exact_match": False,
        "training_authorized": False,
    }
    assert all(value == 0 for value in state["execution_boundary"].values())
    assert state["scope"]["planned_units"] == 82
    assert state["scope"]["SRC"] == 0


def test_c78_generated_tables_if_present_are_explicit_not_blank():
    path = c78.TABLE_DIR / "execution_attempt_ledger.csv"
    if not path.exists():
        pytest.skip("C78 P0 tables not generated yet")
    attempts = _rows("execution_attempt_ledger.csv")
    if "mode" in attempts[0] and any(row["mode"] == "authorized_training" for row in attempts):
        no_auth = next(row for row in attempts if row["mode"] == "no_auth_P0")
        authorized = next(row for row in attempts if row["mode"] == "authorized_training")
        assert no_auth["training"] == "0"
        assert no_auth["real_forward"] == "0"
        assert no_auth["checkpoint_count"] == "0"
        assert authorized["authorization_exact"] == "1"
        assert authorized["training"] == "1"
        assert authorized["checkpoint_count"] == "82"
        assert authorized["status"] == "completed"
    else:
        assert len(attempts) == 1
        assert attempts[0]["training_attempted"] == "0"
        assert attempts[0]["real_data_load_attempted"] == "0"
        assert attempts[0]["GPU_requested"] == "0"
        assert attempts[0]["target_label_read"] == "0"
        assert attempts[0]["seed4_access"] == "0"
    units = _rows("c78_unit_manifest.csv")
    assert len(units) == 82
    assert all(row["executed"] == "0" for row in units)
    p2 = _rows("P2_expansion_gate.csv")[0]
    assert p2["SRC_engine_exercised"] == "0"
    assert p2["full_seed3_authorized"] == "0"
    assert p2["gate"] == "SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD"


def test_c78_protocol_anchor_predates_analysis():
    commit = c78._protocol_commit()
    assert commit.startswith("23f549d")
    assert c78._git("merge-base", "--is-ancestor", commit, c78.PARENT_RESULT_COMMIT) == ""


def test_c78_report_artifacts_if_present_require_red_team_pass():
    main = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.md"
    if not main.exists():
        pytest.skip("C78 final report not generated yet")
    red_team = (c78.REPORT_DIR / "C78_RED_TEAM_VERIFICATION.md").read_text()
    assert "Final status: `PASS`" in red_team
    result = json.loads((c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json").read_text())
    if result.get("schema_version") == "c78_seed3_instrumented_pilot_authorized_result_v1":
        assert result["execution_boundary"]["training_attempted"] == 1
        assert result["final_gate"] == "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD"
        assert result["dual_mode_provenance"]["no_auth_gate"] == "PILOT_READY_BUT_NOT_AUTHORIZED"
        assert result["scope"]["actual_units"] == 82
    else:
        assert result["execution_boundary"]["training_attempted"] == 0
        assert result["final_gate"] == "PILOT_READY_BUT_NOT_AUTHORIZED"
    assert result["claims"]["multiregime_replication"] is False
    assert result["claims"]["SRC_exercised"] is False
    assert result["claims"]["full_seed3_ready"] is False


def test_c78_protocol_sha_is_full_256_bit_value():
    value = c78.PROTOCOL_SHA_PATH.read_text().strip()
    assert len(value) == 64
    int(value, 16)
    assert hashlib.sha256(c78.PROTOCOL_PATH.read_bytes()).hexdigest() == value
