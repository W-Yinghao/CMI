from __future__ import annotations

import ast
import importlib
import json
from copy import deepcopy
from pathlib import Path
import sys

import pytest

from oaci.multidataset import c84l1_canary as canary
from oaci.multidataset import c84l1_runtime_guard as runtime
from oaci.multidataset import c84l1r1_runtime_repair as replacement_runtime


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


def test_execution_lock_self_bound_objects_protocols_and_identities_replay():
    lock_sha = runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH)
    lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text())
    assert len(lock_sha) == 64
    assert lock["status"] == runtime.LOCK_READY_STATUS
    assert len(runtime.prior.verify_bound_object_registry(lock)) == lock["runtime_bound_object_count"] == 107
    assert len(runtime.prior.verify_protocol_sidecars(lock)) == 5
    assert runtime.verify_intervention_registry(lock)["cells"] == 6
    assert runtime.verify_candidate_identity(lock)["canary_units"] == 243
    assert runtime.verify_c84c_level0_binding(lock)["reusable_units"] == 243


def test_lock_binds_exact_scope_and_accepted_level0_plan_model_registry():
    lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text())
    assert lock["scope"] == {
        "C84F": False,
        "C84S": False,
        "datasets": ["Lee2019_MI", "Cho2017", "PhysionetMI"],
        "engineering_only": True,
        "level": 1,
        "source_panel": "A",
        "targets": {"Cho2017": 24, "Lee2019_MI": 19, "PhysionetMI": 106},
        "total_units": 243,
        "training_phases": 9,
        "training_seed": 5,
        "units_per_dataset": 81,
    }
    accepted = lock["accepted_C84C_level0"]
    assert accepted["manifest_sha256"] == "530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b"
    assert accepted["unit_ID_digest"] == "4ada05be758975e7c28429819d804b4064a1bdcfd99fe7a4752a3bdbded6d396"
    assert accepted["model_unit_registry_sha256"] == (
        "0f455f9a605dc4427f9a8c10c1ff3e8fa0880bedbb383d283a165e6d3107b2cf"
    )
    assert all(len(row["plan_hashes"]) == 4 for row in accepted["datasets"].values())


def test_bound_object_candidate_and_accepted_manifest_tampering_fail_closed():
    lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text())
    bound_drift = deepcopy(lock)
    bound_drift["runtime_bound_objects"][0]["sha256"] = "0" * 64
    with pytest.raises(runtime.prior.base.C84R2RuntimeError, match="runtime-bound SHA-256 drift"):
        runtime.prior.verify_bound_object_registry(bound_drift)

    candidate_drift = deepcopy(lock)
    candidate_drift["candidate_identity"]["canary_unit_ID_digest"] = "0" * 64
    with pytest.raises(runtime.C84L1RuntimeError, match="243-unit candidate identity drift"):
        runtime.verify_candidate_identity(candidate_drift)

    accepted_drift = deepcopy(lock)
    accepted_drift["accepted_C84C_level0"]["datasets"]["Lee2019_MI"]["plan_hashes"][0] = "0" * 64
    with pytest.raises(runtime.C84L1RuntimeError, match="plan/model registry replay failed"):
        runtime.verify_c84c_level0_binding(accepted_drift)


def test_historical_level1_authorization_is_consumed_and_not_reusable():
    lock = json.loads(runtime.EXECUTION_LOCK_PATH.read_text())
    assert lock["authorization"]["record_present_at_lock"] is False
    assert lock["authorization"]["C84C_authorization_reusable"] is False
    assert runtime.AUTHORIZATION_RECORD_PATH.is_file()
    result = runtime.verify_authorization_record(
        lock,
        runtime.verify_lock_self(runtime.EXECUTION_LOCK_PATH, runtime.EXECUTION_LOCK_SHA_PATH),
        runtime.prior.commit_for_path(runtime.EXECUTION_LOCK_PATH),
        runtime.AUTHORIZATION_RECORD_PATH,
    )
    assert result["authorized_stage"] == "C84L1C"
    assert result["C84F"] is False and result["C84S"] is False
    assert not replacement_runtime.AUTHORIZATION_RECORD_PATH.exists()


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


def test_post_attempt_lifecycle_has_no_replacement_authorization_or_tracked_payload():
    assert runtime.AUTHORIZATION_RECORD_PATH.is_file()
    assert not replacement_runtime.AUTHORIZATION_RECORD_PATH.exists()
    assert not replacement_runtime.DEFAULT_EXTERNAL_ROOT.exists()
    forbidden = {".npy", ".npz", ".pt", ".pth", ".ckpt", ".fif", ".edf", ".gdf", ".mat"}
    tracked = [Path(line) for line in __import__("subprocess").run(
        ["git", "ls-files", "oaci"], cwd=ROOT, text=True, capture_output=True, check=True,
    ).stdout.splitlines()]
    assert not any(path.suffix.lower() in forbidden for path in tracked)
