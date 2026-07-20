from __future__ import annotations

import ast
import csv
import hashlib
from pathlib import Path

import pytest

from oaci.multidataset import c84f_dual_level_training as training
from oaci.multidataset import c84f_field_manifest as manifests
from oaci.multidataset import c84fl2_protocol as protocol
from oaci.multidataset import c84l1_intervention as intervention


def _sha(value: str) -> str:
    return hashlib.sha256(value.encode("ascii")).hexdigest()


def _model_row(unit: str, level: int) -> dict[str, object]:
    return {
        "unit_id": unit, "dataset": "Synthetic", "panel": "A",
        "training_seed": 6, "level": level,
        "level_intervention_id": protocol.LEVEL0_ID if level == 0 else protocol.LEVEL1_ID,
        "regime": "ERM", "epoch": 199, "trajectory_order": 0,
        "checkpoint_path": f"/{unit}.pt", "checkpoint_sha256": _sha(unit + "checkpoint"),
        "optimizer_path": f"/{unit}.optimizer.pt", "optimizer_sha256": _sha(unit + "optimizer"),
        "sidecar_path": f"/{unit}.json", "sidecar_sha256": _sha(unit + "sidecar"),
        "source_audit_path": f"/{unit}.npz", "source_audit_sha256": _sha(unit + "source"),
        "model_state_hash": _sha(unit + "model"),
        "parent_ERM_model_state_hash": _sha(unit + "parent"),
        "previous_trajectory_model_state_hash": _sha(unit + "previous"),
        "population_signature_sha256": _sha(unit + "population"),
        "support_graph_sha256": _sha(unit + "support"),
        "plan_hashes": {"stage1": _sha(unit + "plan")},
        "paired_model_init_hash": _sha("paired"), "reuse_provenance": "C84F",
        "checkpoint_replay_pass": 1, "optimizer_replay_pass": 1,
        "sidecar_replay_pass": 1, "source_audit_replay_pass": 1,
        "training_target_rows": 0, "training_target_labels": 0,
        "source_audit_rows_used_in_training": 0, "target_outcome_retention": 0,
        "target_outcome_retry": 0,
    }


def test_complete_registry_and_remaining_wave_arithmetic() -> None:
    rows = training.operative_rows()
    assert len(rows) == len({row["unit_id"] for row in rows}) == 1944
    assert sum(row["level"] == 0 for row in rows) == 972
    assert sum(row["level"] == 1 for row in rows) == 972
    remaining = [row for row in rows if row["train_in_C84F"]]
    assert len(remaining) == 1458
    assert {wave: sum(row["wave"] == wave for row in remaining) for wave in ("A", "B0", "B1")} == {
        "A": 486, "B0": 486, "B1": 486,
    }


@pytest.mark.parametrize("dataset", protocol.DATASETS)
@pytest.mark.parametrize(("panel", "seed"), (("A", 6), ("B", 5), ("B", 6)))
def test_each_remaining_cell_is_paired_162_units(dataset: str, panel: str, seed: int) -> None:
    scope = training.validate_paired_cell_scope(dataset, panel, seed)
    assert scope["level0_units"] == scope["level1_units"] == 81
    assert scope["candidate_units"] == 162
    assert scope["training_phases"] == 6


def test_historical_superseded_level1_ids_are_not_operative() -> None:
    rows = training.operative_rows()
    superseded = [row for row in rows if row["identity_status"] == "SUPERSEDED_LEVEL1"]
    assert len(superseded) == 972
    assert not ({row["unit_id"] for row in rows} & {row["historical_planned_unit_id"] for row in superseded})


@pytest.mark.parametrize(
    ("dataset", "panel", "subject"),
    (
        ("Lee2019_MI", "A", 31), ("Lee2019_MI", "B", 16),
        ("Cho2017", "A", 17), ("Cho2017", "B", 37),
        ("PhysionetMI", "A", 103), ("PhysionetMI", "B", 109),
    ),
)
def test_fixed_level1_deletion_registry_replays(dataset: str, panel: str, subject: int) -> None:
    fixture = intervention.synthetic_source_panel(dataset, panel)
    level0 = intervention.apply_level_intervention(
        dataset=dataset, panel=panel, level=0,
        source_subjects=fixture["subjects"], source_labels=fixture["labels"],
        source_trial_ids=fixture["trial_ids"],
    )
    level1 = intervention.apply_level_intervention(
        dataset=dataset, panel=panel, level=1,
        source_subjects=fixture["subjects"], source_labels=fixture["labels"],
        source_trial_ids=fixture["trial_ids"],
    )
    assert level0.deleted_indices == ()
    assert level1.deleted_source_subject == subject
    assert level1.deleted_class == "left_hand"
    assert len(level1.post_cell_counts) == 23


def test_source_and_training_entrypoints_have_no_target_input() -> None:
    assert "target" not in training.load_source_panel_views.__annotations__
    for function in (training.train_level, training.train_paired_cell, training.materialize_paired_bundles):
        names = function.__code__.co_varnames[:function.__code__.co_argcount + function.__code__.co_kwonlyargcount]
        assert not any(name == "target" or name.startswith("target_") for name in names)


def test_training_module_imports_protected_packages_only_inside_functions() -> None:
    source = Path(training.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)
    forbidden = {"torch", "numpy", "mne", "moabb"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            assert not ({alias.name.split(".")[0] for alias in node.names} & forbidden)
        if isinstance(node, ast.ImportFrom) and node.module:
            assert node.module.split(".")[0] not in forbidden


def test_model_field_schema_rejects_partial_before_write(tmp_path: Path) -> None:
    scope = manifests.FieldScope(
        units=2, reused_units=0, new_units=2, level_units=1, phases=6,
        subjects=1, contexts=2, slices=2, canary_contexts=0,
    )
    with pytest.raises(manifests.C84FManifestError, match="expected 2"):
        manifests.publish_model_field_manifest(
            tmp_path, [_model_row("u0", 0)], execution_identity={"test": True}, scope=scope,
        )
    assert not (tmp_path / manifests.MODEL_MANIFEST_NAME).exists()
    assert not (tmp_path / manifests.MODEL_MANIFEST_SHA_NAME).exists()


def test_model_field_atomic_publish_and_replay(tmp_path: Path) -> None:
    scope = manifests.FieldScope(
        units=2, reused_units=0, new_units=2, level_units=1, phases=6,
        subjects=1, contexts=2, slices=2, canary_contexts=0,
    )
    result = manifests.publish_model_field_manifest(
        tmp_path, [_model_row("u0", 0), _model_row("u1", 1)],
        execution_identity={"test": True}, scope=scope,
    )
    replay = manifests.verify_model_field_freeze(result["path"], result["sha256_path"], scope=scope)
    assert replay["unit_count"] == 2
    assert replay["training_target_rows"] == 0


def test_protected_counters_fail_model_field_gate() -> None:
    row = _model_row("u0", 0)
    row["training_target_rows"] = 1
    scope = manifests.FieldScope(
        units=1, reused_units=0, new_units=1, level_units=1, phases=3,
        subjects=1, contexts=1, slices=1, canary_contexts=0,
    )
    with pytest.raises(manifests.C84FManifestError, match="protected counters"):
        manifests.validate_model_field_rows([row], scope=scope)


def test_synthetic_contract_is_exact() -> None:
    result = training.synthetic_contract()
    assert result == {
        "schema_version": "c84f_dual_level_synthetic_contract_v1",
        "paired_cells": 9, "units": 1458, "training_phases": 54,
        "waves": {"A": 486, "B0": 486, "B1": 486},
        "target_access_before_model_freeze": 0,
    }
