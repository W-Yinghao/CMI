import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oaci.multidataset import c84fr2_target_numerical_replay as replay


def _arrays(rows=12, features=16, seed=9):
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((rows, features)).astype(np.float32)
    weight = rng.standard_normal((2, features)).astype(np.float32)
    bias = rng.standard_normal(2).astype(np.float32)
    logits = torch.nn.functional.linear(
        torch.from_numpy(z), torch.from_numpy(weight), torch.from_numpy(bias),
    ).numpy()
    shifted = logits - logits.max(axis=1, keepdims=True)
    probabilities = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    return {
        "unit_id": np.asarray("synthetic_unit"),
        "dataset": np.asarray("SyntheticMI"),
        "panel": np.asarray("A"),
        "training_seed": np.asarray(5, dtype=np.int64),
        "level": np.asarray(0, dtype=np.int64),
        "level_intervention_id": np.asarray("synthetic_level0"),
        "regime": np.asarray("ERM"),
        "epoch": np.asarray(0, dtype=np.int64),
        "trajectory_order": np.asarray(0, dtype=np.int64),
        "target_subject_id": np.arange(rows, dtype=np.int64),
        "target_trial_id": np.asarray([f"trial-{index}" for index in range(rows)]),
        "session": np.asarray(["0"] * rows),
        "run": np.asarray(["0"] * rows),
        "logits": logits,
        "probabilities": probabilities,
        "z": z,
        "Wz_plus_b": logits.copy(),
        "classifier_weight": weight,
        "classifier_bias": bias,
        "repeat_logits": logits.copy(),
        "repeat_z": z.copy(),
    }


def test_same_backend_functional_identity_is_strict_and_device_aware():
    left = torch.tensor([[1.0, 2.0]], dtype=torch.float32)
    right = left.clone()
    assert replay.validate_same_backend_tensors(
        left, right, torch=torch, require_cuda=False,
    ) == 0.0
    right[0, 0] += 2e-6
    with pytest.raises(replay.C84FR2NumericalError, match="exceeds 1e-6"):
        replay.validate_same_backend_tensors(left, right, torch=torch, require_cuda=False)


def test_exact_digest_round_trip_and_saved_outputs(tmp_path):
    arrays = _arrays()
    path = tmp_path / "target.npz"
    result = replay.write_and_replay_artifact(path, arrays=arrays, np=np, torch=torch)
    assert result["artifact_schema_version"] == "c84f_target_unlabeled_v2"
    assert result["rows"] == 12
    assert set(result["array_digests"]) == set(replay.TARGET_ARRAY_FIELDS)
    assert all(value == 0.0 for value in result["saved_output_replay"].values())
    assert {row["backend"] for row in result["cross_backend_diagnostics"]} == {
        "CPU_PyTorch_float32", "NumPy_float32", "NumPy_float64",
    }


@pytest.mark.parametrize("field", ["z", "classifier_weight", "classifier_bias", "Wz_plus_b"])
def test_tampered_persisted_array_digest_fails(tmp_path, field):
    arrays = _arrays()
    expected = replay.build_digest_registry(arrays, np=np)
    changed = {name: np.array(value, copy=True) for name, value in arrays.items()}
    changed[field].flat[0] += 1e-4
    path = tmp_path / "tampered.npz"
    np.savez_compressed(path, **changed)
    with pytest.raises(replay.C84FR2NumericalError, match="digest mismatch"):
        replay.replay_persisted_artifact(
            path, expected_digests=expected, np=np, torch=torch,
        )


@pytest.mark.parametrize("field", ["Wz_plus_b", "probabilities", "repeat_logits", "repeat_z"])
def test_strict_saved_output_gate_fails_before_write(tmp_path, field):
    arrays = _arrays()
    arrays[field] = arrays[field].copy()
    arrays[field].flat[0] += 2e-6
    path = tmp_path / "rejected.npz"
    with pytest.raises(replay.C84FR2NumericalError, match="exceeds 1e-6"):
        replay.write_and_replay_artifact(path, arrays=arrays, np=np, torch=torch)
    assert not path.exists()


def test_missing_or_unknown_target_field_fails():
    arrays = _arrays()
    arrays.pop("repeat_z")
    with pytest.raises(replay.C84FR2NumericalError, match="field-set drift"):
        replay.build_digest_registry(arrays, np=np)
    arrays = _arrays()
    arrays["target_label"] = np.asarray([0])
    with pytest.raises(replay.C84FR2NumericalError, match="field-set drift"):
        replay.build_digest_registry(arrays, np=np)


def test_legacy_exact_cross_backend_fixture_is_diagnostic_only():
    historical_failure = (
        Path(__file__).resolve().parents[2]
        / "oaci/reports/C84FR1_FAILED_ATTEMPT_896550.json"
    )
    failure = json.loads(historical_failure.read_text())
    assert failure["failure"]["observed_linear_persisted_error"] == (
        2.193450927734375e-05
    )
    rng = np.random.default_rng(164)
    z = rng.standard_normal((256, 1040)).astype(np.float32)
    weight = rng.standard_normal((2, 1040)).astype(np.float32)
    bias = rng.standard_normal(2).astype(np.float32)
    logits_tensor = torch.nn.functional.linear(
        torch.from_numpy(z), torch.from_numpy(weight), torch.from_numpy(bias),
    )
    assert replay.validate_same_backend_tensors(
        logits_tensor, logits_tensor.clone(), torch=torch, require_cuda=False,
    ) == 0.0
    arrays = _arrays(rows=256, features=1040)
    arrays.update({
        "z": z,
        "classifier_weight": weight,
        "classifier_bias": bias,
        "logits": logits_tensor.numpy(),
        "Wz_plus_b": logits_tensor.numpy().copy(),
        "repeat_logits": logits_tensor.numpy().copy(),
        "repeat_z": z.copy(),
    })
    shifted = arrays["logits"] - arrays["logits"].max(axis=1, keepdims=True)
    arrays["probabilities"] = np.exp(shifted) / np.exp(shifted).sum(axis=1, keepdims=True)
    diagnostics = replay.cross_backend_diagnostics(arrays, np=np, torch=torch)
    numpy32 = next(row for row in diagnostics if row["backend"] == "NumPy_float32")
    assert np.isfinite(numpy32["max_abs_error"])
    assert numpy32["max_abs_error"] > 2e-05
    assert numpy32["diagnostic_only"] is True
    assert numpy32["finite"] is True


def test_nonfinite_cross_backend_input_fails():
    arrays = _arrays()
    arrays["z"] = arrays["z"].copy()
    arrays["z"][0, 0] = np.inf
    with pytest.raises(replay.C84FR2NumericalError, match="nonfinite"):
        replay.cross_backend_diagnostics(arrays, np=np, torch=torch)
