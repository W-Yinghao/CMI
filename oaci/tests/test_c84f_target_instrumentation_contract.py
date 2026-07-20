from __future__ import annotations

import ast
import hashlib
from pathlib import Path

import numpy as np
import pytest

from oaci.multidataset import c84f_field_manifest as manifests
from oaci.multidataset import c84f_target_instrumentation as target
from oaci.multidataset import c84fl2_protocol as protocol


class ExplodingLabelSlot:
    def _explode(self, *args, **kwargs):
        raise AssertionError("target structural y slot was accessed")

    __repr__ = _explode
    __str__ = _explode
    __iter__ = _explode
    __array__ = _explode
    __len__ = _explode
    __getitem__ = _explode


class Column:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return list(self.values)


class Metadata:
    columns = ("subject", "session", "run")

    def __init__(self, subject: int, rows: int):
        self.rows = rows
        self.values = {
            "subject": [subject] * rows,
            "session": ["0"] * rows,
            "run": ["1"] * rows,
        }

    def __len__(self):
        return self.rows

    def __getitem__(self, name):
        return Column(self.values[name])


class Epochs:
    ch_names = list(target.EXPECTED_CHANNELS)

    def __init__(self, rows: int):
        self.info = {"sfreq": 160.0, "bads": []}
        self.times = np.arange(481, dtype=np.float64) / 160.0
        rng = np.random.default_rng(7)
        self.values = rng.normal(size=(rows, 20, 481)).astype(np.float32)

    def get_data(self, copy=True):
        return self.values.copy() if copy else self.values


def _raw_identity(tmp_path: Path) -> list[dict[str, object]]:
    path = tmp_path / "raw.edf"
    path.write_bytes(b"synthetic raw identity")
    return [{"path": str(path), "bytes": path.stat().st_size, "sha256": manifests.sha256_file(path)}]


def _registry_row(tmp_path: Path, *, field_name: str | None = None) -> dict[str, object]:
    raw = _raw_identity(tmp_path)[0]
    row = {
        "dataset": "Synthetic", "target_subject_id": 7,
        "target_trial_id": "Synthetic|subject=7|session=0|run=1|trial=00000",
        "session": "0", "run": "1", "interface_id": protocol.INTERFACE_ID,
        "montage_sha256": protocol.HASHES["montage"], "sample_rate_hz": 160,
        "sample_count": 480, "finite_value_flag": 1,
        "raw_input_path": raw["path"], "raw_input_bytes": raw["bytes"],
        "raw_input_sha256": raw["sha256"],
    }
    if field_name:
        row[field_name] = 0
    return row


def test_structural_target_label_slot_is_never_touched(tmp_path: Path) -> None:
    view = target.target_view_from_loader_result(
        (Epochs(3), ExplodingLabelSlot(), Metadata(7, 3)),
        dataset="Synthetic", subject=7, raw_files=_raw_identity(tmp_path), np=np,
    )
    assert view.subject_id == 7
    assert len(view.trial_id) == 3
    assert view.X.shape == (3, 20, 480)


def test_wrong_channel_order_fails_even_with_twenty_channels(tmp_path: Path) -> None:
    epochs = Epochs(2)
    epochs.ch_names = list(reversed(target.EXPECTED_CHANNELS))
    with pytest.raises(target.C84FTargetInstrumentationError, match="channel order"):
        target.target_view_from_loader_result(
            (epochs, ExplodingLabelSlot(), Metadata(7, 2)),
            dataset="Synthetic", subject=7, raw_files=_raw_identity(tmp_path), np=np,
        )


def test_wrong_sfreq_fails_even_with_480_final_samples(tmp_path: Path) -> None:
    epochs = Epochs(2)
    epochs.info["sfreq"] = 159.0
    with pytest.raises(target.C84FTargetInstrumentationError, match="sampling-rate"):
        target.target_view_from_loader_result(
            (epochs, ExplodingLabelSlot(), Metadata(7, 2)),
            dataset="Synthetic", subject=7, raw_files=_raw_identity(tmp_path), np=np,
        )


def test_target_registry_exact_schema_and_label_rejection(tmp_path: Path) -> None:
    row = _registry_row(tmp_path)
    gate = manifests.validate_target_trial_rows([row], expected_subject_map={"Synthetic": [7]})
    assert gate["target_subjects"] == 1
    assert gate["target_label_fields"] == 0
    with pytest.raises(manifests.C84FManifestError, match="field-set drift"):
        manifests.validate_target_trial_rows(
            [_registry_row(tmp_path, field_name="target_label")],
            expected_subject_map={"Synthetic": [7]},
        )


@pytest.mark.parametrize(
    "errors",
    (
        {
            "linear_in_memory_max_abs_error": 2e-5,
            "linear_persisted_max_abs_error": 2e-5,
            "softmax_max_abs_error": 1e-6,
            "repeat_logits_max_abs_error": 1e-6,
            "repeat_z_max_abs_error": 1e-6,
        },
        {
            "linear_in_memory_max_abs_error": 0.0,
            "linear_persisted_max_abs_error": 0.0,
            "softmax_max_abs_error": 0.0,
            "repeat_logits_max_abs_error": 0.0,
            "repeat_z_max_abs_error": 0.0,
        },
    ),
)
def test_locked_numerical_boundary_passes(errors) -> None:
    assert target.validate_numerical_errors(errors)["validation_pass"]


def test_linear_or_strict_numerical_gate_cannot_widen() -> None:
    base = {
        "linear_in_memory_max_abs_error": 0.0,
        "linear_persisted_max_abs_error": 0.0,
        "softmax_max_abs_error": 0.0,
        "repeat_logits_max_abs_error": 0.0,
        "repeat_z_max_abs_error": 0.0,
    }
    for field, value in (("linear_in_memory_max_abs_error", 2.00001e-5),
                         ("repeat_logits_max_abs_error", 1.00001e-6)):
        failing = {**base, field: value}
        with pytest.raises(manifests.C84FManifestError, match="tolerance"):
            target.validate_numerical_errors(failing)


def _write_subset(path: Path, *, unit: str, subject: int, offset: float = 0.0) -> None:
    logits = np.asarray([[1.0 + offset, -1.0]], dtype=np.float32)
    probabilities = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)
    np.savez_compressed(
        path,
        unit_id=np.asarray(unit), target_subject_id=np.asarray([subject]),
        target_trial_id=np.asarray(["trial-1"]), logits=logits,
        probabilities=probabilities, z=np.asarray([[0.25, 0.5]], dtype=np.float32),
    )


def test_canary_subset_replay_passes_and_mismatch_fails(tmp_path: Path) -> None:
    complete = tmp_path / "complete.npz"
    canary = tmp_path / "canary.npz"
    _write_subset(complete, unit="u1", subject=19)
    _write_subset(canary, unit="u1", subject=19)
    assert target.replay_canary_subset(
        complete_path=complete, canary_path=canary, canary_subject=19, np=np,
    )["passed"]
    _write_subset(canary, unit="u1", subject=19, offset=1e-3)
    with pytest.raises(manifests.C84FManifestError, match="tolerance"):
        target.replay_canary_subset(
            complete_path=complete, canary_path=canary, canary_subject=19, np=np,
        )


def test_target_module_has_no_training_import_or_call() -> None:
    tree = ast.parse(Path(target.__file__).read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            assert not node.module.startswith("oaci.train")
        if isinstance(node, ast.Call):
            function = node.func
            name = function.id if isinstance(function, ast.Name) else function.attr if isinstance(function, ast.Attribute) else ""
            assert not name.startswith("train_")


def test_target_access_barrier_fails_without_model_manifest(tmp_path: Path) -> None:
    with pytest.raises(manifests.C84FManifestError, match="sidecar"):
        target.require_model_field_barrier(tmp_path / "missing.json", tmp_path / "missing.sha256")


def test_partial_result_directory_is_not_published(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    final = tmp_path / "final"
    staging.mkdir()
    with pytest.raises(manifests.C84FManifestError, match="empty"):
        manifests.atomic_publish_directory(staging, final)
    assert not final.exists()
