from __future__ import annotations

import ast
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from oaci.multidataset import c84c_real_canary_v2 as canary
from oaci.multidataset import c84r2_canary_runtime_repair as runtime


class FakeSeries:
    def __init__(self, values):
        self.values = list(values)

    def tolist(self):
        return list(self.values)


class FakeMetadata:
    def __init__(self, subjects, *, extra=None):
        subjects = list(subjects)
        self.columns = ["subject", "session", "run"]
        self.values = {
            "subject": subjects,
            "session": ["0"] * len(subjects),
            "run": ["1"] * len(subjects),
        }
        if extra:
            for key, value in extra.items():
                self.columns.append(key)
                self.values[key] = list(value)

    def __len__(self):
        return len(self.values["subject"])

    def __getitem__(self, key):
        return FakeSeries(self.values[key])


class FakeEpochs:
    def __init__(self, X, *, channels=None, sfreq=160.0, times=None, bads=()):
        self._X = np.asarray(X)
        self.ch_names = list(channels or canary.EXPECTED_CHANNELS)
        self.info = {"sfreq": sfreq, "bads": list(bads)}
        self.times = np.asarray(times if times is not None else np.arange(self._X.shape[2]) / sfreq)

    def get_data(self, copy=True):
        return self._X.copy() if copy else self._X


class ExplodingTargetLabels:
    def __getitem__(self, key):
        raise AssertionError("target labels were indexed")

    def __iter__(self):
        raise AssertionError("target labels were iterated")

    def __array__(self, *args, **kwargs):
        raise AssertionError("target labels were converted")

    def __repr__(self):
        raise AssertionError("target labels were represented")

    def __str__(self):
        raise AssertionError("target labels were stringified")


def _epochs(n=4, n_times=481):
    X = np.arange(n * 20 * n_times, dtype=np.float32).reshape(n, 20, n_times) + 1.0
    return FakeEpochs(X, times=np.arange(n_times) / 160.0)


def test_actual_epoch_interface_persists_order_sfreq_and_half_open_shape():
    X, interface = canary._epochs_array_and_interface(_epochs(), np)
    assert X.shape == (4, 20, 480)
    assert interface.actual_ch_names == canary.EXPECTED_CHANNELS
    assert interface.actual_sfreq_hz == 160.0
    assert interface.pre_half_open_n_times == 481
    assert interface.final_n_times == 480
    assert interface.last_time_s_before_half_open == 3.0
    assert interface.final_last_time_s == 479 / 160


def test_wrong_channel_order_with_correct_count_fails():
    epochs = _epochs()
    epochs.ch_names = list(reversed(epochs.ch_names))
    with pytest.raises(canary.C84CCanaryV2Error, match="channel order"):
        canary._epochs_array_and_interface(epochs, np)


def test_wrong_sfreq_with_correct_sample_count_fails():
    epochs = FakeEpochs(np.ones((2, 20, 480), dtype=np.float32), sfreq=159.0,
                        times=np.arange(480) / 159.0)
    with pytest.raises(canary.C84CCanaryV2Error, match="sampling rate"):
        canary._epochs_array_and_interface(epochs, np)


def test_bad_or_synthesized_channel_marker_fails():
    epochs = FakeEpochs(np.ones((2, 20, 480), dtype=np.float32), bads=("Cz",))
    with pytest.raises(canary.C84CCanaryV2Error, match="bad/synthesized"):
        canary._epochs_array_and_interface(epochs, np)


def test_missing_or_extra_source_subject_fails():
    labels = np.asarray(["left_hand", "right_hand", "left_hand", "right_hand"])
    result = (_epochs(4), labels, FakeMetadata([1, 1, 2, 2]))
    with pytest.raises(canary.C84CCanaryV2Error, match="subject identity mismatch"):
        canary._source_view_from_loader_result(result, "Cho2017", "source_training", (1, 2, 3), np)
    with pytest.raises(canary.C84CCanaryV2Error, match="subject identity mismatch"):
        canary._source_view_from_loader_result(result, "Cho2017", "source_training", (1,), np)


def test_target_structural_y_is_never_consumed():
    result = (_epochs(2), ExplodingTargetLabels(), FakeMetadata([106, 106]))
    view = canary._target_unlabeled_from_loader_result(result, "PhysionetMI", 106, np)
    assert view.X.shape == (2, 20, 480)
    assert "y" not in view.__dataclass_fields__


def test_target_helper_source_contains_no_slot_one_subscript():
    tree = ast.parse(Path(canary.__file__).read_text())
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef) and node.name == "_target_unlabeled_from_loader_result")
    indices = [node.slice.value for node in ast.walk(function)
               if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
               and isinstance(node.slice.value, int)]
    assert 1 not in indices


def test_target_label_like_metadata_fails():
    metadata = FakeMetadata([106, 106], extra={"class_label": [0, 1]})
    with pytest.raises(canary.C84CCanaryV2Error, match="label-like"):
        canary._target_unlabeled_from_loader_result(
            (_epochs(2), ExplodingTargetLabels(), metadata), "PhysionetMI", 106, np,
        )


def _write_source(path: Path, *, probability_shift=0.0):
    logits = np.asarray([[1.0, 2.0], [2.0, 1.0]], dtype=np.float32)
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    probabilities[0, 0] += probability_shift
    np.savez_compressed(
        path, logits=logits, probabilities=probabilities,
        source_class_label=np.asarray([0, 1]), source_domain_id=np.asarray([12, 13]),
        source_trial_id=np.asarray(["a", "b"]), dataset=np.asarray("Cho2017"),
        panel=np.asarray("A"), seed=np.asarray(5), level=np.asarray(0), unit_id=np.asarray("unit"),
    )


def _write_target(path: Path, *, z_shift=0.0):
    z = np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    weight = np.asarray([[0.2, 0.3], [0.5, -0.1]], dtype=np.float32)
    bias = np.asarray([0.1, -0.2], dtype=np.float32)
    logits = z @ weight.T + bias
    probabilities = np.exp(logits - logits.max(1, keepdims=True))
    probabilities /= probabilities.sum(1, keepdims=True)
    saved_z = z.copy()
    saved_z[0, 0] += z_shift
    np.savez_compressed(
        path, logits=logits, probabilities=probabilities, z=saved_z, Wz_plus_b=logits,
        classifier_weight=weight, classifier_bias=bias, repeat_logits=logits, repeat_z=z,
        target_trial_id=np.asarray(["a", "b"]), dataset=np.asarray("Cho2017"),
        target_subject_id=np.asarray(24), unit_id=np.asarray("unit"),
    )


def test_source_and_target_saved_artifact_replay(tmp_path):
    source = tmp_path / "source.npz"
    target = tmp_path / "target.npz"
    _write_source(source)
    _write_target(target)
    assert runtime.replay_source_audit_artifact(
        source, expected_identity={"dataset": "Cho2017", "panel": "A", "seed": 5, "level": 0,
                                          "unit_id": "unit"},
        expected_trial_ids=("a", "b"), expected_labels=(0, 1), expected_domains=(12, 13), np=np,
    )["replay_pass"]
    assert runtime.replay_target_unlabeled_artifact(
        target, expected_identity={"dataset": "Cho2017", "target_subject_id": 24, "unit_id": "unit"},
        expected_trial_ids=("a", "b"), np=np,
    )["replay_pass"]


def test_saved_softmax_and_z_drift_fail(tmp_path):
    source = tmp_path / "source_bad.npz"
    target = tmp_path / "target_bad.npz"
    _write_source(source, probability_shift=0.1)
    _write_target(target, z_shift=0.1)
    with pytest.raises(runtime.C84R2RuntimeError, match="softmax replay"):
        runtime.replay_source_audit_artifact(
            source, expected_identity={"dataset": "Cho2017", "panel": "A", "seed": 5, "level": 0,
                                              "unit_id": "unit"},
            expected_trial_ids=("a", "b"), expected_labels=(0, 1), expected_domains=(12, 13), np=np,
        )
    with pytest.raises(runtime.C84R2RuntimeError, match="z/classifier/logits"):
        runtime.replay_target_unlabeled_artifact(
            target, expected_identity={"dataset": "Cho2017", "target_subject_id": 24, "unit_id": "unit"},
            expected_trial_ids=("a", "b"), np=np,
        )


def test_optimizer_unloadable_fails(tmp_path):
    path = tmp_path / "optimizer.pt"
    path.write_bytes(b"not-a-torch-file")

    class FakeTorch:
        @staticmethod
        def load(*args, **kwargs):
            raise ValueError("corrupt")

    with pytest.raises(runtime.C84R2RuntimeError, match="unloadable"):
        runtime.replay_optimizer_state(
            {"path": str(path), "file_sha256": runtime.sha256_file(path)},
            phase="ERM", trajectory_order=0, torch=FakeTorch,
        )


def test_sidecar_identity_drift_fails(tmp_path):
    path = tmp_path / "sidecar.json"
    path.write_text(json.dumps({"unit_id": "wrong", "dataset": "Cho2017"}), encoding="utf-8")
    with pytest.raises(runtime.C84R2RuntimeError, match="identity drift"):
        runtime.replay_sidecar(
            path, expected_fields={"unit_id", "dataset"},
            expected_identity={"unit_id": "expected", "dataset": "Cho2017"},
        )


def _complete_rows():
    return [{
        "unit_id": f"unit-{index:03d}",
        "checkpoint_replay_pass": True, "optimizer_replay_pass": True,
        "sidecar_replay_pass": True, "source_audit_replay_pass": True,
        "target_unlabeled_replay_pass": True,
    } for index in range(243)]


def test_complete_gate_requires_all_243_source_and_target_artifacts():
    assert runtime.validate_complete_canary_gate(_complete_rows())["complete"]
    rows = _complete_rows()
    rows[5]["source_audit_replay_pass"] = False
    with pytest.raises(runtime.C84R2RuntimeError, match="incomplete"):
        runtime.validate_complete_canary_gate(rows)
    with pytest.raises(runtime.C84R2RuntimeError, match="243 unique"):
        runtime.validate_complete_canary_gate(_complete_rows()[:-1])


def test_post_consumption_CUDA_failure_leaves_attempt_ledger(monkeypatch, tmp_path):
    binding = {"run_root": tmp_path, "lock": {}, "lock_sha256": "b" * 64}
    consumption = {"sha256": "c" * 64}
    monkeypatch.setattr(runtime, "require_authorization_and_lock", lambda **kwargs: binding)
    monkeypatch.setattr(runtime, "consume_authorization", lambda value: consumption)
    monkeypatch.setattr(runtime, "verify_protected_runtime_versions", lambda *args, **kwargs: {})

    fake_torch = types.ModuleType("torch")
    fake_torch.__version__ = "2.6.0+cu124"
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)
    fake_torch.use_deterministic_algorithms = lambda *args, **kwargs: None
    fake_torch.are_deterministic_algorithms_enabled = lambda: True
    fake_mne = types.ModuleType("mne")
    fake_mne.__version__ = "1.11.0"
    fake_moabb = types.ModuleType("moabb")
    fake_moabb.__version__ = "1.5.0"
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mne", fake_mne)
    monkeypatch.setitem(sys.modules, "moabb", fake_moabb)
    monkeypatch.setenv("SLURM_JOB_ID", "synthetic")
    with pytest.raises(canary.C84CCanaryV2Error, match="CUDA allocation"):
        canary.run_real(authorization_path=tmp_path / "auth.json", output_root=tmp_path)
    events = [json.loads(line) for line in (tmp_path / "execution_attempts.jsonl").read_text().splitlines()]
    assert events[0]["event"] == "attempt_started"
    assert events[-1]["event"] == "failed"
    assert json.loads((tmp_path / "partial_artifact_manifest.json").read_text())["stage"] == "CUDA_and_determinism_check"


def test_schema_dry_run_has_zero_real_access():
    payload = canary.synthetic_schema_dry_run()
    assert payload["canary_units"] == 243
    assert payload["target_y_field_present"] is False
    assert payload["real_EEG_arrays_loaded"] == payload["real_labels_read"] == payload["dataset_downloads"] == 0
