import ast
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from oaci.multidataset import c84fr2_target_stage as target_stage


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"


def test_repair_and_target_v2_protocol_hashes_replay():
    for stem in (
        "C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL",
        "C84_TARGET_INSTRUMENTATION_PROTOCOL_V2",
    ):
        path = REPORTS / f"{stem}.json"
        sidecar = REPORTS / f"{stem}.sha256"
        expected = sidecar.read_text(encoding="ascii").split()[0]
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected
    repair = json.loads((REPORTS / "C84FR2_TARGET_NUMERICAL_REPLAY_REPAIR_PROTOCOL.json").read_text())
    assert repair["numerical_contract"]["historical_functional_tolerance_widened"] is False
    assert repair["forbidden"]["target_X_reload_during_C84FR2"] is True
    target = json.loads((REPORTS / "C84_TARGET_INSTRUMENTATION_PROTOCOL_V2.json").read_text())
    assert target["historical_partial_root"]["reusable"] is False
    assert target["complete_field_contract"]["context_digest_sidecars"] == 1944


def test_target_only_module_has_no_training_or_scientific_import_or_callable():
    path = ROOT / "oaci/multidataset/c84fr2_target_stage.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imported = set()
    functions = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add(node.module or "")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            functions.add(node.name)
    forbidden_import_tokens = ("oaci.train", "training", "selector", "scientific")
    assert not any(any(token in name for token in forbidden_import_tokens) for name in imported)
    assert not any(name.startswith("train") for name in functions)


def test_target_runtime_uses_direct_functional_linear_before_cpu_transfer():
    source = (ROOT / "oaci/multidataset/c84fr2_target_stage.py").read_text(encoding="utf-8")
    function = source[source.index("def _forward_same_backend"):source.index("def _concatenate_views")]
    assert "torch.nn.functional.linear" in function
    assert function.index("torch.nn.functional.linear") < function.index(".detach().cpu()")
    assert "require_cuda=True" in function


def test_checkpoint_classifier_and_artifact_identity_gates_are_mandatory():
    source = (ROOT / "oaci/multidataset/c84fr2_target_stage.py").read_text(encoding="utf-8")
    instrument = source[source.index("def instrument_unit_v2"):source.index("def _replay_canary_subset")]
    assert "_require_checkpoint_classifier_identity" in instrument
    assert "_require_artifact_identity" in instrument
    assert instrument.index("_require_checkpoint_classifier_identity") < instrument.index(
        "_forward_same_backend"
    )
    assert instrument.index("_require_artifact_identity") < instrument.index(
        "write_and_replay_artifact"
    )


def test_checkpoint_classifier_identity_is_exact():
    class Fixture:
        classifier = torch.nn.Linear(3, 2)

    model = Fixture()
    state = {name: value.clone() for name, value in model.classifier.state_dict().items()}
    bound = {
        "classifier.weight": state["weight"],
        "classifier.bias": state["bias"],
    }
    target_stage._require_checkpoint_classifier_identity(model, bound, torch=torch)
    bound["classifier.weight"] = bound["classifier.weight"].clone()
    bound["classifier.weight"][0, 0] += 1e-7
    with pytest.raises(target_stage.C84FR2TargetStageError, match="byte identity"):
        target_stage._require_checkpoint_classifier_identity(model, bound, torch=torch)


def test_artifact_scalar_and_trial_identity_fail_closed():
    unit = {
        "unit_id": "u", "dataset": "D", "panel": "A", "training_seed": 5,
        "level": 0, "level_intervention_id": "L0", "regime": "ERM",
        "epoch": 0, "trajectory_order": 0,
    }
    arrays = {
        **{name: np.asarray(value) for name, value in unit.items()},
        "target_trial_id": np.asarray(["a", "b"]),
        "target_subject_id": np.asarray([1, 1]),
        "session": np.asarray(["0", "0"]), "run": np.asarray(["0", "0"]),
        "logits": np.zeros((2, 2)), "probabilities": np.zeros((2, 2)),
        "z": np.zeros((2, 3)), "Wz_plus_b": np.zeros((2, 2)),
        "repeat_logits": np.zeros((2, 2)), "repeat_z": np.zeros((2, 3)),
    }
    target_stage._require_artifact_identity(arrays, unit, ["a", "b"], np=np)
    arrays["target_trial_id"] = np.asarray(["b", "a"])
    with pytest.raises(target_stage.C84FR2TargetStageError, match="trial order"):
        target_stage._require_artifact_identity(arrays, unit, ["a", "b"], np=np)


def test_target_raw_manifest_order_is_canonical():
    rows = [
        {"dataset": "PhysionetMI", "path": "/z", "bytes": 2, "sha256": "b" * 64},
        {"dataset": "Cho2017", "path": "/a", "bytes": 1, "sha256": "a" * 64},
    ]
    payload = target_stage.target_raw_manifest_payload(rows)
    assert [(row["dataset"], row["path"]) for row in payload["files"]] == [
        ("Cho2017", "/a"), ("PhysionetMI", "/z"),
    ]
    assert payload["target_labels"] == 0


def test_exact_payload_and_copy_replay(tmp_path):
    payload = {"schema_version": "fixture", "rows": [1, 2, 3]}
    source = tmp_path / "source.json"
    source.write_text(json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n")
    replay = target_stage.require_exact_json_payload(payload, source, object_name="fixture")
    assert replay["exact_replay"] is True
    destination = tmp_path / "copy.json"
    digest = target_stage._copy_exact(source, destination, expected_sha256=replay["sha256"])
    assert digest == replay["sha256"]
    assert destination.read_bytes() == source.read_bytes()
    with pytest.raises(target_stage.C84FR2TargetStageError, match="overwrite"):
        target_stage._copy_exact(source, destination, expected_sha256=replay["sha256"])


def test_partial_complete_manifest_cannot_publish(tmp_path):
    with pytest.raises(target_stage.C84FR2TargetStageError, match="1,944 descriptors"):
        target_stage.publish_complete_field_manifest_v2(
            tmp_path,
            [],
            model_manifest_sha256="a" * 64,
            target_raw_manifest_sha256="b" * 64,
            target_trial_registry_sha256="c" * 64,
            instrumentation={},
            execution_identity={},
        )
    assert not (tmp_path / target_stage.COMPLETE_MANIFEST_NAME).exists()


def test_historical_partial_artifacts_are_never_an_input_to_instrumentation():
    source = (ROOT / "oaci/multidataset/c84fr2_target_stage.py").read_text(encoding="utf-8")
    assert "complete_target_unlabeled_v2" in source
    assert "partial_target_artifact_reused\": False" in source
    assert "complete_target_unlabeled/c84l1_" not in source


def test_structural_target_y_slot_remains_unreferenced():
    source = (ROOT / "oaci/multidataset/c84f_target_instrumentation.py").read_text(encoding="utf-8")
    function = source[
        source.index("def target_view_from_loader_result"):source.index("def _flatten_paths")
    ]
    assert "result[0]" in function
    assert "result[2]" in function
    assert "result[1]" not in function
