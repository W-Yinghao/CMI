from __future__ import annotations

import ast
import csv
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c82_post_c81_baseline_recovery as recovery
from oaci.conditioned_ceiling_coverage import c82_synthetic_end_to_end_recovery as synthetic


def _table(name: str) -> list[dict[str, str]]:
    with (recovery.TABLE_DIR / name).open(newline="") as handle:
        return list(csv.DictReader(handle))


@pytest.fixture(scope="module")
def fixture_A():
    return synthetic.synthetic_fixture("same_method_A")


def test_protocol_hash_replays_exactly():
    protocol, observed = recovery.load_protocol()
    assert observed == recovery.PROTOCOL_SHA_PATH.read_text().strip()
    assert observed == "9f58c7a8e6b495a6d8f510c0d72d24ede4485908ef94bc078abe8f124b03a8f3"
    assert protocol["epistemic_status"]["designed_after_C81_evaluation_label_access"] is True


def test_protocol_preserves_C81_E_and_disclaims_confirmation():
    protocol, _ = recovery.load_protocol()
    assert protocol["historical_C81"]["final_gate"].startswith("C81-E")
    assert protocol["epistemic_status"]["overwrites_C81_E"] is False
    assert protocol["epistemic_status"]["independent_replication"] is False
    assert protocol["epistemic_status"]["external_validation"] is False


def test_protocol_binds_exact_frozen_selection_without_recomputation():
    protocol, _ = recovery.load_protocol()
    frozen = protocol["frozen_selection"]
    assert frozen["manifest_self_sha256"] == "4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519"
    assert frozen["payload_sha256"] == "1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257"
    assert frozen["payload_bytes"] == 415284
    assert frozen["selection_recomputation_allowed"] is False


def test_protocol_formally_removes_ambiguous_LORO():
    protocol, _ = recovery.load_protocol()
    assert protocol["LORO"]["decision"] == "REMOVED_FROM_OPERATIVE_C82_INFERENCE"
    assert protocol["LORO"]["cross_regime_selector_transport_claim"] is False


def test_canonical_schema_has_exact_registered_order():
    protocol, _ = recovery.load_protocol()
    assert tuple(protocol["canonical_method_context_schema"]["field_order"]) == recovery.METHOD_CONTEXT_FIELDS
    assert len(recovery.METHOD_CONTEXT_FIELDS) == 16


def test_dictionary_insertion_order_is_not_semantic(fixture_A):
    rows, _, _, _ = fixture_A
    normalized = recovery.validate_method_context_rows(rows)
    assert len(normalized) == 672
    assert all(tuple(row) == recovery.METHOD_CONTEXT_FIELDS for row in normalized)


def test_missing_method_context_field_is_rejected(fixture_A):
    rows, _, _, _ = fixture_A
    damaged = dict(rows[0])
    damaged.pop("top10")
    with pytest.raises(recovery.C82ValidationError, match="missing"):
        recovery.canonicalize_method_context_row(damaged)


def test_unknown_method_context_field_is_rejected(fixture_A):
    rows, _, _, _ = fixture_A
    damaged = dict(rows[0])
    damaged["unknown"] = 1
    with pytest.raises(recovery.C82ValidationError, match="unknown"):
        recovery.canonicalize_method_context_row(damaged)


@pytest.mark.parametrize("field", ["same_label_oracle_accessed", "target4_primary"])
def test_protected_true_boolean_is_rejected(fixture_A, field):
    rows, _, _, _ = fixture_A
    damaged = dict(rows[0])
    damaged[field] = True
    with pytest.raises(recovery.C82ValidationError, match="protected boolean"):
        recovery.canonicalize_method_context_row(damaged)


def test_evaluation_before_selection_freeze_is_rejected(fixture_A):
    rows, _, _, _ = fixture_A
    damaged = dict(rows[0])
    damaged["evaluation_label_access_after_selection_freeze"] = False
    with pytest.raises(recovery.C82ValidationError, match="protected boolean"):
        recovery.canonicalize_method_context_row(damaged)


def test_target4_is_rejected_by_schema(fixture_A):
    rows, _, _, _ = fixture_A
    damaged = dict(rows[0])
    damaged["target"] = 4
    with pytest.raises(recovery.C82ValidationError, match="primary target"):
        recovery.canonicalize_method_context_row(damaged)


def test_context_coverage_is_exact_32_by_21(fixture_A):
    rows, _, _, _ = fixture_A
    normalized = recovery.validate_method_context_rows(rows)
    assert len({(row["seed"], row["target"], row["level"]) for row in normalized}) == 32
    assert len({row["method_id"] for row in normalized}) == 21


def test_frozen_Q0_has_all_224_context_budget_rows(fixture_A):
    _, q0, _, _ = fixture_A
    normalized = recovery.validate_q0_rows(q0)
    assert len(normalized) == 224
    assert {row["budget"] for row in normalized} == set(recovery.Q0_BUDGETS)


def test_metric_applicability_registry_has_34_methods_and_no_manufactured_metrics():
    rows = _table("method_metric_applicability.csv")
    assert len(rows) == 34
    by_id = {row["method_id"]: row for row in rows}
    assert by_id["S2"]["performance_estimation_metrics"] == "0"
    assert by_id["U12"]["performance_estimation_metrics"] == "0"
    assert by_id["U15"]["performance_estimation_metrics"] == "1"
    assert by_id["U16"]["U16_diagnostic"] == "1"


def test_output_registry_and_implementation_match_exactly():
    assert recovery._registered_csv_outputs() == set(recovery.TABLE_FIELDS)
    assert len(recovery.TABLE_FIELDS) == 23


def test_same_method_A_is_eligible():
    q1 = {3: {m: m == "U7" for m in recovery.PRIMARY_ZERO_METHODS}, 4: {m: m == "U7" for m in recovery.PRIMARY_ZERO_METHODS}}
    q2 = {3: dict(q1[3]), 4: dict(q1[4])}
    result = recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16)
    assert result["primary_taxonomy"] == recovery.GATE_A
    assert result["A_intersection"] == ["U7"]


def test_different_A_methods_must_be_D():
    q1 = {3: {m: m == "U7" for m in recovery.PRIMARY_ZERO_METHODS}, 4: {m: m == "U14" for m in recovery.PRIMARY_ZERO_METHODS}}
    q2 = {3: dict(q1[3]), 4: dict(q1[4])}
    result = recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16)
    assert result["primary_taxonomy"] == recovery.GATE_D
    assert result["A_intersection"] == []


def test_same_method_B_is_eligible():
    q1 = {3: {m: m == "U5" for m in recovery.PRIMARY_ZERO_METHODS}, 4: {m: m == "U5" for m in recovery.PRIMARY_ZERO_METHODS}}
    q2 = {seed: {m: False for m in recovery.PRIMARY_ZERO_METHODS} for seed in recovery.SEEDS}
    result = recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16)
    assert result["primary_taxonomy"] == recovery.GATE_B
    assert result["B_intersection"] == ["U5"]


def test_different_B_methods_must_be_D():
    q1 = {3: {m: m == "U5" for m in recovery.PRIMARY_ZERO_METHODS}, 4: {m: m == "U14" for m in recovery.PRIMARY_ZERO_METHODS}}
    q2 = {seed: {m: False for m in recovery.PRIMARY_ZERO_METHODS} for seed in recovery.SEEDS}
    assert recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16)["primary_taxonomy"] == recovery.GATE_D


def test_stable_C_is_eligible():
    q1 = {seed: {m: False for m in recovery.PRIMARY_ZERO_METHODS} for seed in recovery.SEEDS}
    q2 = {seed: dict(q1[seed]) for seed in recovery.SEEDS}
    assert recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16)["primary_taxonomy"] == recovery.GATE_C


def test_LOTO_below_12_has_D_precedence():
    q1 = {seed: {m: m == "U7" for m in recovery.PRIMARY_ZERO_METHODS} for seed in recovery.SEEDS}
    q2 = {seed: dict(q1[seed]) for seed in recovery.SEEDS}
    assert recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=11)["primary_taxonomy"] == recovery.GATE_D


def test_blocker_has_highest_precedence():
    q1 = {seed: {m: m == "U7" for m in recovery.PRIMARY_ZERO_METHODS} for seed in recovery.SEEDS}
    q2 = {seed: dict(q1[seed]) for seed in recovery.SEEDS}
    assert recovery.classify_same_method_taxonomy(q1=q1, q2=q2, loto_preserved=16, blocker=True)["primary_taxonomy"] == recovery.GATE_E


def test_exact_maxT_rejects_consistent_material_family():
    effects = np.full((8, 12), 0.20)
    evidence = recovery.exact_signflip_maxT(effects, margin=0.05)
    assert np.all(evidence["pvalue"] <= 0.05)


def test_measurement_record_coverage_is_32_by_14(fixture_A):
    _, _, records, _ = fixture_A
    rows, u16 = recovery._measurement_outputs(records)
    assert len(rows) == 32 * len(recovery.RANK_METHODS)
    assert len(u16) == 32
    assert all(row["diagnostic_only"] == 1 for row in u16)


def test_inapplicable_measurement_metrics_are_NA(fixture_A):
    _, _, records, _ = fixture_A
    rows, _ = recovery._measurement_outputs(records)
    by_method = {}
    for row in rows:
        by_method.setdefault(row["method_id"], row)
    assert by_method["U12"]["utility_estimation_MAE"] == "NA"
    assert isinstance(by_method["U15"]["utility_estimation_MAE"], float)


def test_atomic_freeze_failure_leaves_no_final_directory(tmp_path, fixture_A):
    rows, q0, measurements, identity = fixture_A
    final = tmp_path / "result"
    with pytest.raises(recovery.C82ValidationError, match="partial"):
        recovery.run_recovery(
            method_context_rows=rows, q0_rows=q0, measurement_records=measurements,
            selection_identity=identity, final_directory=final, synthetic=True,
            inject_partial_write_failure=True,
        )
    assert not final.exists()


def test_post_evaluation_failure_is_terminal_and_consumes_authorization(tmp_path, fixture_A):
    rows, q0, measurements, identity = fixture_A
    with pytest.raises(recovery.C82PostEvaluationFailure) as caught:
        recovery.run_recovery(
            method_context_rows=rows, q0_rows=q0, measurement_records=measurements,
            selection_identity=identity, final_directory=tmp_path / "result", synthetic=True,
            inject_post_evaluation_failure=True,
        )
    assert caught.value.authorization_consumed is True
    assert caught.value.final_gate == recovery.GATE_E
    assert not (tmp_path / "result").exists()


def test_synthetic_taxonomy_calibration_covers_A_through_E():
    rows = _table("synthetic_taxonomy_calibration.csv")
    assert len(rows) == 6
    assert all(row["passed"] == "1" and row["real_field_used"] == "0" for row in rows)
    assert {row["observed_gate"] for row in rows} == {
        recovery.GATE_A, recovery.GATE_B, recovery.GATE_C, recovery.GATE_D, recovery.GATE_E,
    }


def test_synthetic_end_to_end_freezes_all_tables_and_672_rows():
    rows = _table("synthetic_end_to_end_result_manifest.csv")
    assert len(rows) == 6
    assert all(row["table_count"] == "23" for row in rows)
    assert all(row["method_context_rows"] == "672" for row in rows)
    assert all(row["all_required_artifacts_present"] == "1" and row["passed"] == "1" for row in rows)


def test_synthetic_atomic_failure_suite_all_passes():
    rows = _table("atomic_result_freeze_test.csv")
    assert len(rows) == 5
    assert all(row["passed"] == "1" and row["final_directory_visible"] == "0" for row in rows)


def test_synthetic_same_method_identity_suite_all_passes():
    rows = _table("same_method_identity_test.csv")
    assert len(rows) == 5
    assert all(row["status"] == "PASS" for row in rows)
    different = [row for row in rows if row["test_id"].startswith("different")]
    assert all(row["common_methods"] == "NONE" and row["expected_gate"] == recovery.GATE_D for row in different)


def test_synthetic_LOTO_has_16_panels_and_method_identity():
    rows = _table("synthetic_LOTO_calibration.csv")
    assert len(rows) == 6
    assert all(row["panels"] == "16" and row["same_method_column_present"] == "1" for row in rows)
    assert all(row["passed"] == "1" for row in rows)


def test_synthetic_maxT_and_noninferiority_family_is_complete():
    rows = _table("synthetic_maxT_noninferiority.csv")
    assert len(rows) == 6
    assert all(row["Q1_rows"] == "12" and row["Q2_rows"] == "12" for row in rows)
    assert all(row["same_target_sign_vectors"] == "256" and row["passed"] == "1" for row in rows)


def test_real_entrypoint_fails_closed_without_lock_or_authorization(tmp_path, monkeypatch):
    monkeypatch.setattr(recovery, "LOCK_PATH", tmp_path / "missing_lock.json")
    monkeypatch.setattr(recovery, "LOCK_SHA_PATH", tmp_path / "missing_lock.sha256")
    monkeypatch.setattr(recovery, "AUTHORIZATION_PATH", tmp_path / "missing_authorization.json")
    with pytest.raises(recovery.C82ValidationError, match="execution lock is absent"):
        recovery.require_c82e_authorization()


def test_C82P_source_does_not_call_real_adapter_from_schema_dry_run():
    source = recovery.Path(recovery.__file__).read_text()
    tree = ast.parse(source)
    dry_run = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "schema_dry_run")
    names = {node.id for node in ast.walk(dry_run) if isinstance(node, ast.Name)}
    assert "_real_evaluation_bundle" not in names
    assert "_load_preserved_selection" not in names


def test_schema_dry_run_records_zero_protected_access():
    result = recovery.schema_dry_run()
    assert result["selection_payload_opened"] is False
    assert result["evaluation_view_opened"] is False
    assert result["selection_recomputed"] is False
    assert result["expected_method_context_rows"] == 672


def test_analysis_lock_hash_and_all_bound_objects_replay():
    lock, observed = recovery.load_execution_lock()
    assert observed == recovery.LOCK_SHA_PATH.read_text().strip()
    assert lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED"
    assert len(lock["implementation"]) == 3
    assert len(lock["registry_artifacts"]) == 19
    assert len(lock["field_and_view_manifests"]) == 11


def test_analysis_lock_binds_exact_frozen_selection():
    lock, _ = recovery.load_execution_lock()
    frozen = lock["frozen_selection"]
    assert frozen["manifest_self_sha256"] == "4677ed3aba7758ea0008c2093b44d6fb81d425930727e5941950179737ebd519"
    assert frozen["payload_sha256"] == "1ed893acd9190914eb4cb122f3ef26bc1e2355c4103894b816894bd264669257"
    assert frozen["selection_recomputation_allowed"] is False


def test_analysis_lock_protected_scope_is_all_false():
    lock, _ = recovery.load_execution_lock()
    scope = lock["scope"]
    for key in (
        "training", "forward", "reinference", "GPU", "target4_primary", "same_label_oracle",
        "selection_recomputation", "new_method", "seed5", "BNCI2014_004",
    ):
        assert scope[key] is False
    assert lock["runtime"]["construction_label_content_reopened"] is False


def test_risk_register_contains_all_minimum_risks():
    rows = _table("risk_register.csv")
    assert len(rows) == 33
    risks = {row["risk"] for row in rows}
    assert "dict_order_used_as_schema" in risks
    assert "cross_seed_method_identity_lost" in risks
    assert "same_identity_rerun_after_post_evaluation_failure" in risks


def test_no_real_field_was_used_in_synthetic_readiness_artifacts():
    for name in (
        "synthetic_taxonomy_calibration.csv",
        "synthetic_end_to_end_result_manifest.csv",
        "synthetic_maxT_noninferiority.csv",
        "synthetic_LOTO_calibration.csv",
    ):
        assert all(row["real_field_used"] == "0" for row in _table(name))


def test_no_raw_payload_or_model_file_is_part_of_C82P_changes():
    forbidden = {".npz", ".npy", ".pt", ".pth", ".ckpt", ".pkl"}
    paths = [Path(line[3:]) for line in recovery._git("status", "--short").splitlines() if len(line) > 3]
    assert not any(path.suffix.lower() in forbidden for path in paths)
