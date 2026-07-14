from __future__ import annotations

import ast
import importlib
import json
from pathlib import Path
import sys

import pytest

from oaci.multidataset import c84l1_canary as canary
from oaci.multidataset import c84l1_runtime_guard as runtime


ROOT = Path(__file__).resolve().parents[2]


def test_module_import_is_protected_stack_free():
    before = set(sys.modules)
    importlib.reload(canary)
    loaded = {name.split(".")[0] for name in set(sys.modules) - before}
    assert not loaded & {"numpy", "torch", "mne", "moabb", "braindecode", "skorch"}


def test_protected_imports_follow_guard_consumption_and_attempt_ledger():
    tree = ast.parse(Path(canary.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "run_real")
    rendered = [ast.unparse(node) for node in function.body]
    assert "require_authorization_and_lock" in rendered[0]
    assert "consume_authorization" in rendered[1]
    assert "ExecutionAttemptLedger" in rendered[2]
    protected = min(index for index, text in enumerate(rendered) if "import numpy" in text)
    assert protected > 2


def test_training_bundle_is_parameterized_by_all_paired_factors():
    signature = ast.parse(Path(canary.__file__).read_text())
    function = next(
        node for node in signature.body
        if isinstance(node, ast.FunctionDef) and node.name == "materialize_training_bundle"
    )
    names = {argument.arg for argument in (*function.args.args, *function.args.kwonlyargs)}
    assert {"dataset", "panel", "training_seed", "level"} <= names
    assignments = {
        node.targets[0].id: node.lineno
        for node in ast.walk(function)
        if isinstance(node, ast.Assign)
        and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Name)
    }
    assert assignments["applied"] < assignments["support"]
    assert assignments["applied"] < assignments["stage1"]


def test_level_pair_uses_the_same_model_initialization_rule():
    tree = ast.parse(Path(canary.__file__).read_text())
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "_run_dataset_training")
    body = ast.unparse(function)
    assert "level0_init = _model_init_hash(dataset, TRAINING_SEED, torch)" in body
    assert "level1_init = _model_init_hash(dataset, TRAINING_SEED, torch)" in body
    assert "if level0_init != level1_init" in body


def test_schema_dry_run_covers_243_units_and_no_protected_action():
    result = canary.synthetic_schema_dry_run()
    assert result["candidate_units"] == 243
    assert result["training_phases"] == 9
    assert result["datasets"] == 3
    assert all(row["deleted_rows"] == 8 and row["post_cells"] == 23 for row in result["deletion_cells"])
    assert result["target_y_access"] == 0
    assert result["scientific_metrics"] == 0
    assert result["real_EEG_access"] == 0
    assert result["training_forward_GPU"] == 0


def test_real_entrypoint_fails_before_output_without_lock_or_authorization(tmp_path):
    with pytest.raises(runtime.C84L1RuntimeError):
        runtime.require_authorization_and_lock(
            authorization_path=tmp_path / "missing-authorization.json",
            output_root=tmp_path / "external",
            lock_path=tmp_path / "missing-lock.json",
            lock_sha_path=tmp_path / "missing-lock.sha256",
        )
    assert not (tmp_path / "external").exists()


def test_complete_gate_requires_all_243_artifact_families():
    rows = [{
        "unit_id": f"u{index:03d}",
        "checkpoint_replay_pass": True,
        "optimizer_replay_pass": True,
        "source_audit_replay_pass": True,
        "target_unlabeled_replay_pass": True,
        "sidecar_replay_pass": True,
        "support_replay_pass": True,
        "paired_model_init_pass": True,
        "level0_plan_replay_pass": True,
        "target_y_access": 0,
        "target_scientific_metrics": 0,
    } for index in range(243)]
    assert runtime.validate_complete_level1_canary_gate(rows)["complete"] is True
    rows[0]["source_audit_replay_pass"] = False
    with pytest.raises(runtime.C84L1RuntimeError, match="failed complete replay"):
        runtime.validate_complete_level1_canary_gate(rows)


def test_complete_gate_rejects_target_y_or_scientific_metric():
    row = {
        "unit_id": "same",
        "checkpoint_replay_pass": True,
        "optimizer_replay_pass": True,
        "source_audit_replay_pass": True,
        "target_unlabeled_replay_pass": True,
        "sidecar_replay_pass": True,
        "support_replay_pass": True,
        "paired_model_init_pass": True,
        "level0_plan_replay_pass": True,
        "target_y_access": 1,
        "target_scientific_metrics": 0,
    }
    rows = [{**row, "unit_id": str(index)} for index in range(243)]
    with pytest.raises(runtime.C84L1RuntimeError, match="target-label/scientific"):
        runtime.validate_complete_level1_canary_gate(rows)


def test_sidecar_contract_has_no_scientific_output_fields():
    forbidden = {
        "target_accuracy", "target_calibration", "target_regret", "selector_scores",
        "Q1", "Q2", "label_budget_frontier", "cross_dataset_scientific_result",
    }
    assert not forbidden & canary.SIDECAR_FIELDS
    assert {"level_intervention_id", "deleted_source_subject", "deleted_class"} <= canary.SIDECAR_FIELDS


def test_slurm_wrapper_locks_deterministic_environment_and_only_calls_C84L1C():
    text = (ROOT / "oaci/slurm_c84l1c_canary.sh").read_text()
    assert "CUBLAS_WORKSPACE_CONFIG=:4096:8" in text
    assert "PYTHONHASHSEED=0" in text
    assert "oaci.multidataset.c84l1_canary run-real" in text
    assert "C84F" not in text and "C84S" not in text


def test_C84L1P_has_no_authorization_or_real_payload():
    assert not runtime.AUTHORIZATION_RECORD_PATH.exists()
    forbidden = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".fif", ".edf", ".gdf", ".mat"}
    tracked = [Path(line) for line in __import__("subprocess").run(
        ["git", "ls-files", "oaci"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.splitlines()]
    assert not any(path.suffix.lower() in forbidden for path in tracked)
