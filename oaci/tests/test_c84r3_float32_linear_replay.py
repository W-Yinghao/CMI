from __future__ import annotations

import json
import hashlib
from pathlib import Path

import numpy as np
import pytest

from oaci.multidataset import c84c_real_canary_v3 as canary
from oaci.multidataset import c84r3_canary_runtime_repair as runtime


def _target_artifact(path: Path, *, linear_delta: float = 0.0, repeat_delta: float = 0.0) -> None:
    z = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight = np.asarray([[0.5, -0.25], [-0.75, 0.125]], dtype=np.float32)
    bias = np.asarray([0.1, -0.2], dtype=np.float32)
    logits = z @ weight.T + bias
    logits = logits.copy()
    logits[0, 0] += np.float32(linear_delta)
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
    np.savez_compressed(
        path,
        logits=logits,
        probabilities=probabilities,
        z=z,
        Wz_plus_b=z @ weight.T + bias,
        classifier_weight=weight,
        classifier_bias=bias,
        repeat_logits=logits + np.float32(repeat_delta),
        repeat_z=z,
        target_trial_id=np.asarray(["a", "b"]),
        dataset=np.asarray("Lee2019_MI"),
        target_subject_id=np.asarray(19, dtype=np.int64),
        unit_id=np.asarray("unit"),
    )


def _replay(path: Path):
    return runtime.replay_target_unlabeled_artifact(
        path,
        expected_identity={"dataset": "Lee2019_MI", "target_subject_id": 19, "unit_id": "unit"},
        expected_trial_ids=("a", "b"),
        np=np,
    )


def test_observed_float32_linear_error_passes_repaired_tolerance(tmp_path):
    path = tmp_path / "target.npz"
    _target_artifact(path, linear_delta=2.86102294921875e-6)
    replay = _replay(path)
    assert replay["replay_pass"] is True
    assert replay["linear_replay_max_abs_error"] < 1e-5
    assert replay["linear_replay_abs_tolerance"] == 1e-5
    assert replay["strict_identity_abs_tolerance"] == 1e-6


def test_in_memory_split_tolerance_accepts_observed_error_only_for_linear_replay():
    observed = {
        "Wz_plus_b_max_error": 2.86102294921875e-6,
        "softmax_max_error": 0.0,
        "repeat_logits_max_error": 0.0,
        "repeat_z_max_error": 0.0,
    }
    result = runtime.validate_instrumentation_errors(observed)
    assert result["validation_pass"] is True
    assert result["linear_replay_abs_tolerance"] == 1e-5
    assert result["strict_identity_abs_tolerance"] == 1e-6


def test_in_memory_split_tolerance_rejects_linear_and_strict_perturbations():
    baseline = {
        "Wz_plus_b_max_error": 0.0,
        "softmax_max_error": 0.0,
        "repeat_logits_max_error": 0.0,
        "repeat_z_max_error": 0.0,
    }
    with pytest.raises(runtime.C84R3RuntimeError, match="float32 linear instrumentation identity failed"):
        runtime.validate_instrumentation_errors({**baseline, "Wz_plus_b_max_error": 1.0001e-5})
    with pytest.raises(runtime.C84R3RuntimeError, match="strict instrumentation identity failed"):
        runtime.validate_instrumentation_errors({**baseline, "repeat_logits_max_error": 1.0001e-6})


def test_linear_perturbation_above_repaired_tolerance_fails(tmp_path):
    path = tmp_path / "target.npz"
    _target_artifact(path, linear_delta=1.1e-5)
    with pytest.raises(runtime.C84R3RuntimeError, match="saved z/classifier/logits replay failed"):
        _replay(path)


def test_repeat_logits_remain_on_strict_tolerance(tmp_path):
    path = tmp_path / "target.npz"
    _target_artifact(path, repeat_delta=2e-6)
    with pytest.raises(runtime.C84R3RuntimeError, match="saved repeat-logits replay failed"):
        _replay(path)


def test_repair_contract_and_dry_run_are_non_scientific():
    repair = json.loads(runtime.REPAIR_PROTOCOL_PATH.read_text())
    assert repair["repair"]["linear_reconstruction_abs_tolerance"] == 1e-5
    assert repair["repair"]["softmax_repeat_logits_repeat_z_abs_tolerance"] == 1e-6
    assert repair["repair"]["scientific_registry_changed"] is False
    dry = canary.synthetic_schema_dry_run()
    assert dry["canary_units"] == 243
    assert dry["target_y_field_present"] is False
    assert dry["failed_attempt_reused"] is False


def test_replacement_runtime_requires_new_objects_and_fresh_authorization(tmp_path):
    assert runtime.EXECUTION_LOCK_PATH.name == "C84C_EXECUTION_LOCK_V3.json"
    assert runtime.CANARY_PROTOCOL_PATH.name == "C84_CANARY_PROTOCOL_V4.json"
    assert runtime.AUTHORIZATION_RECORD_PATH.name == "C84C_PI_AUTHORIZATION_RECORD_V3.json"
    assert not runtime.AUTHORIZATION_RECORD_PATH.exists()
    assert runtime.DEFAULT_EXTERNAL_ROOT.name == "oaci-c84-canary-v4"
    assert not (tmp_path / "output").exists()


def test_v4_protocols_bind_split_tolerances_and_no_failed_reuse():
    canary_bytes = runtime.CANARY_PROTOCOL_PATH.read_bytes()
    canary = json.loads(canary_bytes)
    canary_sha = hashlib.sha256(canary_bytes).hexdigest()
    assert runtime.CANARY_PROTOCOL_SHA_PATH.read_text().split()[0] == canary_sha
    assert canary["instrumentation"]["linear_z_classifier_logits_abs_tolerance"] == 1e-5
    assert canary["instrumentation"]["softmax_repeat_logits_repeat_z_abs_tolerance"] == 1e-6
    assert canary["engineering_failure_895366"]["target_y_access"] == 0
    assert canary["engineering_failure_895366"]["failed_artifact_reuse_allowed"] is False
    assert canary["retry"]["retrain_all_243_units"] is True
    assert canary["authorization"]["historical_authorization_reusable"] is False

    field_bytes = runtime.FIELD_PROTOCOL_PATH.read_bytes()
    field = json.loads(field_bytes)
    assert runtime.FIELD_PROTOCOL_SHA_PATH.read_text().split()[0] == hashlib.sha256(field_bytes).hexdigest()
    assert field["parent_canary_protocol_v4_sha256"] == canary_sha
    assert "parent_canary_protocol_v3_sha256" not in field
    assert field["canary_reuse"]["failed_attempt_895366_units_reusable"] == 0
    assert field["canary_reuse"]["replacement_units_required"] == 243
