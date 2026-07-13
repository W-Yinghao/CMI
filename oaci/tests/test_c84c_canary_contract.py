from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pytest

from oaci.multidataset import c84c_real_canary as canary


class FakeSeries:
    def __init__(self, values):
        self._values = values

    def tolist(self):
        return list(self._values)


class FakeMetadata:
    def __init__(self, subject: int, n: int):
        self.columns = ("subject", "session", "run")
        self.values = {
            "subject": [subject] * n,
            "session": ["0"] * n,
            "run": ["1"] * n,
        }

    def __len__(self):
        return len(self.values["subject"])

    def __getitem__(self, key):
        return FakeSeries(self.values[key])


class ExplodingTargetLabels:
    def __iter__(self):
        raise AssertionError("target labels were iterated")

    def __array__(self, *args, **kwargs):
        raise AssertionError("target labels were converted")

    def __repr__(self):
        raise AssertionError("target labels were logged")


def test_module_import_does_not_import_real_scientific_stack(monkeypatch):
    before = set(sys.modules)
    importlib.reload(canary)
    newly_loaded = set(sys.modules) - before
    assert not ({"moabb", "mne", "torch", "braindecode", "skorch"} & {name.split(".")[0] for name in newly_loaded})


def test_loader_and_torch_imports_are_function_local():
    tree = ast.parse(Path(canary.__file__).read_text())
    top_imports = {
        alias.name.split(".")[0]
        for node in tree.body if isinstance(node, (ast.Import, ast.ImportFrom))
        for alias in node.names
    }
    assert not top_imports & {"moabb", "mne", "torch", "braindecode", "skorch", "numpy"}


def test_run_real_guard_and_consumption_precede_protected_imports():
    source = ast.parse(Path(canary.__file__).read_text())
    function = next(node for node in source.body if isinstance(node, ast.FunctionDef) and node.name == "run_real")
    calls = [node for node in function.body if isinstance(node, (ast.Assign, ast.Expr, ast.Import, ast.ImportFrom))]
    rendered = [ast.unparse(node) for node in calls]
    assert "require_authorization_and_lock" in rendered[0]
    assert "consume_authorization" in rendered[1]
    first_protected = min(index for index, text in enumerate(rendered) if "import numpy" in text or "import torch" in text)
    assert first_protected > 1


def test_target_unlabeled_helper_never_reads_structural_y():
    X = np.arange(2 * 20 * 481, dtype=np.float32).reshape(2, 20, 481) + 1
    result = (X, ExplodingTargetLabels(), FakeMetadata(106, 2))
    view = canary._target_unlabeled_from_loader_result(result, "PhysionetMI", 106, np)
    assert view.X.shape == (2, 20, 480)
    assert "y" not in view.__dataclass_fields__
    assert view.as_payload_fields() == ("X", "trial_id", "target_subject_id", "session", "run", "dataset_id")


def test_target_helper_source_contains_no_slot_one_subscript():
    source = ast.parse(Path(canary.__file__).read_text())
    function = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                    and node.name == "_target_unlabeled_from_loader_result")
    constants = [node.slice.value for node in ast.walk(function)
                 if isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant)
                 and isinstance(node.slice.value, int)]
    assert 1 not in constants


def test_target_label_like_metadata_is_rejected():
    metadata = FakeMetadata(106, 2)
    metadata.columns = (*metadata.columns, "label")
    metadata.values["label"] = [0, 1]
    X = np.ones((2, 20, 480), dtype=np.float32)
    with pytest.raises(canary.C84CCanaryError, match="label-like"):
        canary._target_unlabeled_from_loader_result((X, ExplodingTargetLabels(), metadata), "PhysionetMI", 106, np)


def test_schema_dry_run_has_no_real_access_or_authorization_consumption():
    result = canary.synthetic_schema_dry_run()
    assert result["canary_units"] == 243
    assert result["canary_training_phases"] == 9
    assert result["target_y_field_present"] is False
    assert result["real_EEG_arrays_loaded"] == result["real_labels_read"] == result["dataset_downloads"] == 0
    assert result["authorization_consumed"] is False


def test_real_entrypoint_fails_before_output_without_lock_or_authorization(tmp_path):
    with pytest.raises(canary.C84CCanaryError):
        canary.require_authorization_and_lock(authorization_path=tmp_path / "missing.json", output_root=tmp_path / "external")
    assert not (tmp_path / "external").exists()


def test_canary_protocol_forbids_all_scientific_outputs():
    payload = json.loads(canary.CANARY_PROTOCOL_PATH.read_text())
    forbidden = set(payload["forbidden_outputs"])
    assert {"target_accuracy", "target_calibration", "target_regret", "selector_scores", "Q1", "Q2",
            "label_budget_frontier", "cross_dataset_science"} <= forbidden
    assert payload["scope"]["engineering_only"] is True
    assert payload["scope"]["total_units"] == 243


def test_no_real_payload_or_authorization_record_exists_in_C84R():
    assert not canary.AUTHORIZATION_RECORD_PATH.exists()
    forbidden = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".fif", ".edf", ".gdf", ".mat"}
    files = [path for path in (canary.REPO_ROOT / "oaci").rglob("*") if path.is_file()]
    assert not any(path.suffix.lower() in forbidden and "__pycache__" not in str(path) for path in files)
