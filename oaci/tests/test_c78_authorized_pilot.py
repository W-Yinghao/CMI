from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from oaci.conditioned_ceiling_coverage import c78_authorized_common as common
from oaci.conditioned_ceiling_coverage import c78_authorized_instrument as instrument
from oaci.conditioned_ceiling_coverage import c78_authorized_train as train
from oaci.conditioned_ceiling_coverage import c78_seed3_instrumented_pilot as c78


def test_c78_authorized_execution_lock_replays_all_implementation_hashes():
    lock = common.load_execution_lock()
    assert lock["parent_no_auth_result_commit"] == common.NO_AUTH_RESULT_COMMIT
    assert lock["protocol_sha256"] == c78.PROTOCOL_SHA_PATH.read_text().strip()
    assert lock["authorization"]["received"] is True
    assert lock["scope"] == {
        "dataset": "BNCI2014_001", "target": 4, "seed": 3,
        "levels": [0, 1], "ERM_anchors": 2, "OACI_checkpoints": 80,
        "SRC_units": 0, "total_units": 82,
    }


def test_c78_authorized_guard_is_exact_and_rejects_generic_text():
    _, _, token, _ = c78.load_protocol()
    assert c78.authorization_matches(token, token)
    assert not c78.authorization_matches("authorization_token_exact 我授权", token)
    assert not c78.authorization_matches(token + "\n", token)
    with pytest.raises(PermissionError):
        common.require_authorization("authorization_token_exact 我授权")


def test_c78_training_worker_guards_before_real_loader_import():
    source = Path(train.__file__).read_text()
    function = source[source.index("def train_field"):]
    assert function.index("common.require_authorization") < function.index("from oaci.data.eeg.bnci")
    module_prefix = source[:source.index("def train_field")]
    assert "from oaci.data.eeg.bnci" not in module_prefix
    assert "runtime.cuda" not in module_prefix


def test_c78_locked_training_subjects_exclude_target_and_source_audit():
    from oaci.confirmatory.loso_plan import loso_fold_spec

    split = loso_fold_spec(4, dataset_id="BNCI2014_001")
    assert split["source_train_subjects"] == [7, 8, 9, 1, 2, 3]
    assert split["source_audit_subjects"] == [5, 6]
    assert split["target"] == 4
    assert set(split["source_train_subjects"]).isdisjoint({4, 5, 6})


def test_c78_authorized_unit_manifest_is_exact_and_has_no_SRC():
    rows = common.unit_rows()
    assert len(rows) == 82
    assert len({row["unit_id"] for row in rows}) == 82
    assert sum(row["regime"] == "ERM" for row in rows) == 2
    assert sum(row["regime"] == "OACI" for row in rows) == 80
    assert not any(row["regime"] == "SRC" for row in rows)


def test_authorized_manifest_and_model_contract_reconstruct_without_data(tmp_path):
    path, manifest, split = train._materialized_manifest(tmp_path)
    model_spec, execution, factory, geometry = train._model_contract(manifest)
    assert path.is_file()
    assert split["target"] == 4
    assert split["source_train_subjects"] == [7, 8, 9, 1, 2, 3]
    assert manifest.seeds.model == [3]
    assert manifest.training.stage1_epochs == 200
    assert manifest.training.stage2_epochs == 200
    assert manifest.training.stage2_steps_per_epoch == 20
    assert model_spec.input_shape == (22, 385)
    assert model_spec.n_classes == 4
    assert execution.engine_template.checkpoint_every == 5
    assert geometry["feat_dim"] == 800
    assert factory().feat_dim == 800


def test_optimizer_state_hash_is_stable_and_value_sensitive():
    left = {"state": {0: {"step": 1, "x": np.array([1.0, 2.0])}}, "groups": [{"lr": 0.1}]}
    right = {"groups": [{"lr": 0.1}], "state": {0: {"x": np.array([1.0, 2.0]), "step": 1}}}
    changed = {"state": {0: {"step": 1, "x": np.array([1.0, 2.1])}}, "groups": [{"lr": 0.1}]}
    assert train.optimizer_state_hash(left) == train.optimizer_state_hash(right)
    assert train.optimizer_state_hash(left) != train.optimizer_state_hash(changed)


def test_optimizer_capture_records_registered_stage1_and_stage2_orders(tmp_path):
    torch = pytest.importorskip("torch")
    import oaci.train.engine as engine

    parameter = torch.nn.Parameter(torch.tensor([1.0]))
    cfg = engine.EngineConfig(optimizer_name="adam")
    capture = train.OptimizerCapture(tmp_path, stage1_steps=2, stage2_checkpoint_steps=2)
    with capture.patch_engine():
        capture.begin("stage1", 0)
        optimizer = engine.make_optimizer([parameter], 0.1, cfg)
        for _ in range(2):
            optimizer.zero_grad(); parameter.sum().backward(); optimizer.step()
        assert capture.descriptor("stage1", 0, 0)["optimizer_state_hash"]

        capture.begin("stage2", 0)
        critic_parameter = torch.nn.Parameter(torch.tensor([1.0]))
        encoder_parameter = torch.nn.Parameter(torch.tensor([2.0]))
        critic = engine.make_optimizer([critic_parameter], 0.1, cfg)
        encoder = engine.make_optimizer([encoder_parameter], 0.1, cfg)
        for _ in range(4):
            critic.zero_grad(); critic_parameter.sum().backward(); critic.step()
            encoder.zero_grad(); encoder_parameter.sum().backward(); encoder.step()
        assert capture.descriptor("stage2", 0, 1)["optimizer_state_hash"]
        assert capture.descriptor("stage2", 0, 2)["optimizer_state_hash"]


def test_target_unlabeled_schemas_exclude_all_target_label_fields():
    assert not instrument.TARGET_INPUT_FIELDS & instrument.FORBIDDEN_TARGET_FIELDS
    assert not instrument.TARGET_OUTPUT_FIELDS & instrument.FORBIDDEN_TARGET_FIELDS
    assert "target_class_label" not in instrument.TARGET_INPUT_FIELDS
    assert "split_role" not in instrument.TARGET_OUTPUT_FIELDS


def test_class_margin_definition_matches_one_vs_best_other():
    logits = np.array([[3.0, 2.0, -1.0, 0.0], [0.0, 2.0, 1.0, 4.0]], dtype=np.float32)
    margins = instrument._class_margins(logits)
    np.testing.assert_allclose(margins[0], [1.0, -1.0, -4.0, -3.0])
    np.testing.assert_allclose(margins[1], [-4.0, -2.0, -3.0, 2.0])


def test_field_frozen_gate_if_present_has_exact_82_units_and_no_target_training_rows():
    lock = common.load_execution_lock()
    path = common.field_frozen_path(lock)
    if not path.exists():
        pytest.skip("authorized C78 training has not run yet")
    frozen = common.require_field_frozen(lock)
    assert frozen["unit_count"] == 82
    assert frozen["ERM_anchor_count"] == 2
    assert frozen["OACI_trajectory_count"] == 80
    assert frozen["SRC_count"] == 0
    assert frozen["execution"]["target_data_rows_loaded_during_training"] == 0
    assert frozen["execution"]["target_label_reads_during_training"] == 0


def test_instrumentation_gate_if_present_has_exact_rows_and_identity():
    lock = common.load_execution_lock()
    path = common.instrumentation_gate_path(lock)
    if not path.exists():
        pytest.skip("authorized C78 instrumentation has not run yet")
    gate = common.verify_canonical_manifest(path)
    assert gate["unit_count"] == gate["unique_unit_count"] == 82
    assert gate["source_rows"] == 82 * 8 * 576
    assert gate["target_unlabeled_rows"] == 82 * 576
    assert gate["identity"]["failed_units"] == 0
    assert gate["physical_isolation"]["target_unlabeled_contains_labels"] is False
    assert gate["physical_isolation"]["instrumentation_received_oracle_path"] is False


def test_authorized_final_report_if_present_preserves_dual_mode_provenance():
    result_path = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json"
    result = json.loads(result_path.read_text())
    if result.get("schema_version") != "c78_seed3_instrumented_pilot_authorized_result_v1":
        pytest.skip("authorized C78 final report has not replaced the no-auth result yet")
    assert result["dual_mode_provenance"]["no_auth_commit"].startswith("67bca01")
    assert result["execution_boundary"]["training_attempted"] == 1
    assert result["scope"]["actual_units"] == 82
    assert result["claims"]["multiregime_replication"] is False
    assert result["claims"]["full_seed3_ready"] is False
