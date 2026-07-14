import json
from pathlib import Path

import pytest

from oaci.multidataset import c84l1_canary_v2 as canary
from oaci.multidataset import c84l1r1_protocols as protocols
from oaci.multidataset import c84l1r1_runtime_repair as runtime


REPO_ROOT = Path(__file__).resolve().parents[2]


def _errors(linear: float = 0.0, strict: float = 0.0):
    return {
        "Wz_plus_b_max_error": linear,
        "softmax_max_error": strict,
        "repeat_logits_max_error": 0.0,
        "repeat_z_max_error": 0.0,
    }


def test_observed_failure_is_inside_repaired_linear_tolerance():
    result = runtime.validate_instrumentation_errors(_errors(1.239776611328125e-5))
    assert result["validation_pass"] is True
    assert result["linear_replay_abs_tolerance"] == 2e-5


def test_linear_perturbation_above_repaired_tolerance_fails():
    with pytest.raises(runtime.C84L1R1RuntimeError, match="float32 linear"):
        runtime.validate_instrumentation_errors(_errors(2.0000001e-5))


def test_strict_tensor_identity_tolerance_remains_unchanged():
    runtime.validate_instrumentation_errors(_errors(strict=1e-6))
    with pytest.raises(runtime.C84L1R1RuntimeError, match="strict instrumentation"):
        runtime.validate_instrumentation_errors(_errors(strict=1.000001e-6))


def test_persisted_target_replay_uses_repaired_linear_tolerance(tmp_path):
    import numpy as np

    logits = np.asarray([[1.239776611328125e-5, 0.0]], dtype=np.float32)
    probabilities = np.exp(logits - logits.max(axis=1, keepdims=True))
    probabilities /= probabilities.sum(axis=1, keepdims=True)
    path = tmp_path / "target.npz"
    np.savez_compressed(
        path,
        logits=logits,
        probabilities=probabilities,
        z=np.zeros((1, 1), dtype=np.float32),
        Wz_plus_b=np.zeros((1, 2), dtype=np.float32),
        classifier_weight=np.zeros((2, 1), dtype=np.float32),
        classifier_bias=np.zeros(2, dtype=np.float32),
        repeat_logits=logits,
        repeat_z=np.zeros((1, 1), dtype=np.float32),
        target_trial_id=np.asarray(["trial-1"]),
        dataset=np.asarray("Lee2019_MI"),
        target_subject_id=np.asarray(19, dtype=np.int64),
        unit_id=np.asarray("unit-1"),
    )
    result = runtime.replay_target_unlabeled_artifact(
        path,
        expected_identity={"dataset": "Lee2019_MI", "target_subject_id": 19, "unit_id": "unit-1"},
        expected_trial_ids=["trial-1"],
        np=np,
    )
    assert result["replay_pass"] is True
    assert result["linear_replay_max_abs_error"] == pytest.approx(1.239776611328125e-5)


def test_nonfinite_and_field_drift_fail_closed():
    with pytest.raises(runtime.C84L1R1RuntimeError):
        runtime.validate_instrumentation_errors(_errors(float("nan")))
    values = _errors()
    values["unknown"] = 0.0
    with pytest.raises(runtime.C84L1R1RuntimeError, match="field-set"):
        runtime.validate_instrumentation_errors(values)


def test_replacement_root_and_authorization_paths_are_additive():
    assert runtime.DEFAULT_EXTERNAL_ROOT.name == "oaci-c84-level1-canary-v2"
    assert runtime.DEFAULT_EXTERNAL_ROOT != runtime.base.DEFAULT_EXTERNAL_ROOT
    assert runtime.AUTHORIZATION_RECORD_PATH.name == "C84L1C_PI_AUTHORIZATION_RECORD_V2.json"
    assert not runtime.AUTHORIZATION_RECORD_PATH.exists()


def test_replacement_protocol_preserves_scope_and_changes_only_tolerance_contract():
    prior = json.loads((REPO_ROOT / "oaci/reports/C84_LEVEL1_CANARY_PROTOCOL_V1.json").read_text())
    replacement = protocols.build_canary_protocol()
    assert replacement["scope"] == prior["scope"]
    assert replacement["intervention"] == prior["intervention"]
    assert replacement["candidate_identity"] == prior["candidate_identity"]
    assert replacement["instrumentation_replay_tolerances"] == {
        "linear_z_classifier_logits_abs_tolerance": 2e-5,
        "softmax_repeat_logits_repeat_z_abs_tolerance": 1e-6,
        "linear_scope": "float32_1040_term_CPU_GPU_classifier_reconstruction_only",
    }
    assert replacement["historical_failed_attempt"]["partial_artifacts_reusable"] is False


def test_replacement_field_protocol_preserves_science_and_rejects_failed_reuse():
    replacement = protocols.build_field_protocol("canary-v2-sha")
    assert replacement["scientific_field_scope_changed"] is False
    assert replacement["instrumentation_replay_tolerances"]["linear_z_classifier_logits_abs_tolerance"] == 2e-5
    assert replacement["canary_reuse"]["C84L1C_failed_job_895928_complete_units_reusable"] == 0
    assert replacement["canary_reuse"]["C84L1C_replacement_must_retrain_all_units"] == 243


def test_replacement_schema_dry_run_remains_243_units_without_real_data():
    result = canary.base.synthetic_schema_dry_run()
    assert result["candidate_units"] == 243
    assert result["training_phases"] == 9
    assert result["real_EEG_access"] == 0
    assert result["target_y_access"] == 0
    assert result["scientific_metrics"] == 0


def test_historical_lock_and_failed_attempt_are_preserved():
    historical_lock = REPO_ROOT / "oaci/reports/C84L1C_EXECUTION_LOCK.json"
    failed = json.loads((REPO_ROOT / "oaci/reports/C84L1C_FAILED_ATTEMPT_895928.json").read_text())
    assert historical_lock.is_file()
    assert failed["frozen_partial_state"]["complete_units"] == 73
    assert failed["protected_state"]["target_y_accesses"] == 0
    assert failed["governance"]["failed_root_reusable"] is False
