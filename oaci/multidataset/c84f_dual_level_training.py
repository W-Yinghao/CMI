"""Authorized future C84F dual-level training and field orchestration.

The module is import-safe during C84FL2.  NumPy, torch, MNE, MOABB, loaders,
and training code are imported only after the C84F authorization has been
consumed and the execution-attempt ledger exists.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any, Iterable, Mapping, Sequence
import warnings

from . import c84_dataset_registry_v2 as dataset_registry
from . import c84fl2_protocol as protocol
from . import c84f_field_manifest as manifests
from . import c84f_runtime_guard as runtime


AUTHORIZATION_RECORD_PATH = runtime.AUTHORIZATION_RECORD_PATH
DEFAULT_EXTERNAL_ROOT = runtime.DEFAULT_EXTERNAL_ROOT
CHECKPOINT_EPOCHS = tuple(range(4, 200, 5))
CLASS_NAMES = ("left_hand", "right_hand")
WAVE_CELLS = {
    "A": (("A", 6),),
    "B0": (("B", 5),),
    "B1": (("B", 6),),
}
SOURCE_AUDIT_FIELDS = frozenset({
    "logits", "probabilities", "source_class_label", "source_domain_id", "source_trial_id",
    "dataset", "panel", "seed", "level", "unit_id",
})
FIELD_SIDECAR_FIELDS = frozenset({
    "schema_version", "unit_id", "dataset", "panel", "seed", "level",
    "level_intervention_id", "level_intervention_registry_sha256",
    "deleted_source_subject", "deleted_class", "regime", "epoch", "trajectory_order",
    "interface_id", "montage_sha256", "checkpoint", "optimizer", "source_audit",
    "model_state_hash", "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash",
    "genealogy_rule", "population_signature_sha256", "support_graph_sha256", "plan_hashes",
    "paired_model_init_hash", "paired_model_init_pass", "level_support_replay_pass",
    "source_training_subjects", "source_audit_subjects", "source_training_trial_count",
    "source_audit_trial_count", "training_target_rows", "training_target_labels",
    "source_audit_rows_used_in_training", "target_outcome_retention", "target_outcome_retry",
    "target_artifact", "target_fit_ids", "scientific_metrics",
})


class C84FDualLevelTrainingError(runtime.C84FRuntimeError):
    """Raised on any paired-training, source-view, or orchestration failure."""


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def operative_rows() -> list[dict[str, Any]]:
    rows = []
    path = protocol.TABLE_DIR / "operative_complete_unit_registry_replay.csv"
    for raw in read_csv(path):
        row = dict(raw)
        for field in ("training_seed", "level", "epoch", "trajectory_order", "wave_order", "train_in_C84F"):
            row[field] = int(row[field])
        rows.append(row)
    if len(rows) != 1944 or len({row["unit_id"] for row in rows}) != 1944:
        raise C84FDualLevelTrainingError("operative C84F registry is not 1,944 unique units")
    return rows


def cell_candidate_rows(dataset: str, panel: str, training_seed: int, level: int) -> list[dict[str, Any]]:
    selected = [
        row for row in operative_rows()
        if row["dataset"] == dataset and row["panel"] == panel
        and row["training_seed"] == int(training_seed) and row["level"] == int(level)
    ]
    selected.sort(key=lambda row: (0 if row["regime"] == "ERM" else 1 if row["regime"] == "OACI" else 2,
                                   row["trajectory_order"]))
    if len(selected) != 81:
        raise C84FDualLevelTrainingError(
            f"candidate cell does not contain 81 units: {dataset}/{panel}/{training_seed}/L{level}"
        )
    if any(row["identity_status"] == "SUPERSEDED_LEVEL1" and row["unit_id"] == row["historical_planned_unit_id"]
           for row in selected):
        raise C84FDualLevelTrainingError("historical superseded level-1 ID entered an operative cell")
    return selected


def validate_paired_cell_scope(dataset: str, panel: str, training_seed: int) -> dict[str, Any]:
    levels = {level: cell_candidate_rows(dataset, panel, training_seed, level) for level in (0, 1)}
    if {row["level_intervention_id"] for row in levels[0]} != {protocol.LEVEL0_ID}:
        raise C84FDualLevelTrainingError("level-0 intervention identity drift")
    if {row["level_intervention_id"] for row in levels[1]} != {protocol.LEVEL1_ID}:
        raise C84FDualLevelTrainingError("level-1 intervention identity drift")
    return {
        "dataset": dataset, "panel": panel, "training_seed": int(training_seed),
        "level0_units": 81, "level1_units": 81, "candidate_units": 162,
        "training_phases": 6, "paired_model_initialization_required": True,
    }


def _dataset_and_paradigm(dataset_code: str, loader_classes: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
    Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = loader_classes
    factories = {
        "Lee2019_MI": lambda: Lee2019_MI(train_run=True, test_run=False),
        "Cho2017": Cho2017,
        "PhysionetMI": lambda: PhysionetMI(imagined=True, executed=False),
    }
    if dataset_code not in factories:
        raise C84FDualLevelTrainingError(f"dataset outside C84F lock: {dataset_code}")
    from .c84f_target_instrumentation import EXPECTED_CHANNELS

    paradigm = MotorImagery(
        n_classes=2, events=list(CLASS_NAMES), fmin=4.0, fmax=38.0,
        tmin=0.0, tmax=3.0, channels=list(EXPECTED_CHANNELS), resample=160,
    )
    return factories[dataset_code](), paradigm


def _source_subject_contract(dataset: str, panel: str) -> tuple[tuple[int, ...], tuple[int, ...]]:
    spec = dataset_registry.DATASETS[dataset]
    partition = dataset_registry.partition_subjects(spec)
    panel_key = "source_panel_A" if panel == "A" else "source_panel_B"
    split = dataset_registry.source_train_audit_split(dataset, panel, partition[panel_key])
    training = tuple(int(value) for value in split["source_training"])
    audit = tuple(int(value) for value in split["source_audit"])
    if len(training) != 12 or len(audit) != 4 or set(training) & set(audit):
        raise C84FDualLevelTrainingError(f"source subject contract drift: {dataset}/{panel}")
    if set(training) | set(audit) != set(partition[panel_key]):
        raise C84FDualLevelTrainingError(f"source panel split coverage drift: {dataset}/{panel}")
    return training, audit


def load_source_panel_views(
    dataset_code: str,
    panel: str,
    *,
    loader_classes: tuple[Any, Any, Any, Any],
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> tuple[Any, Any, list[dict[str, Any]]]:
    """Load source train/audit only; this function has no target subject input."""
    from . import c84c_real_canary_v2 as base
    from .c84f_target_instrumentation import _flatten_paths, raw_file_identities

    training_subjects, audit_subjects = _source_subject_contract(dataset_code, panel)
    dataset, paradigm = _dataset_and_paradigm(dataset_code, loader_classes)
    raw_rows: dict[str, dict[str, Any]] = {}
    for subject in (*training_subjects, *audit_subjects):
        values = dataset.data_path(subject, force_update=False, update_path=False, verbose=False)
        for row in raw_file_identities(_flatten_paths(values)):
            raw_rows[row["path"]] = {
                "dataset": dataset_code, "panel": panel, "source_subject": subject, **row,
            }

    def get(subjects: Sequence[int], role: str) -> Any:
        ledger.increment("source_get_data_calls")
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter("always")
            result = paradigm.get_data(dataset=dataset, subjects=list(subjects), return_epochs=True)
        return result, [{
            "dataset": dataset_code, "panel": panel, "role": role,
            "category": item.category.__name__, "message": str(item.message),
        } for item in captured]

    training_result, training_warnings = get(training_subjects, "source_training")
    audit_result, audit_warnings = get(audit_subjects, "source_audit")
    source = base._source_view_from_loader_result(
        training_result, dataset_code, "source_training", training_subjects, np,
    )
    audit = base._source_view_from_loader_result(
        audit_result, dataset_code, "source_audit", audit_subjects, np,
    )
    ledger.increment("source_EEG_arrays", 2)
    ledger.increment("source_label_arrays_read", 2)
    if set(source.subject_id) != set(training_subjects) or set(audit.subject_id) != set(audit_subjects):
        raise C84FDualLevelTrainingError(f"loaded source subjects drift: {dataset_code}/{panel}")
    if set(source.subject_id) & set(audit.subject_id):
        raise C84FDualLevelTrainingError(f"source train/audit overlap: {dataset_code}/{panel}")
    if source.interface.actual_ch_names != audit.interface.actual_ch_names:
        raise C84FDualLevelTrainingError(f"source train/audit channel order differs: {dataset_code}/{panel}")
    warning_rows = training_warnings + audit_warnings
    return source, audit, [*sorted(raw_rows.values(), key=lambda row: row["path"]), *warning_rows]


def source_input_freeze_payload(
    views: Mapping[tuple[str, str], tuple[Any, Any]],
    raw_and_warning_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    panels = []
    for dataset in protocol.DATASETS:
        for panel in ("A", "B"):
            source, audit = views[(dataset, panel)]
            expected_source, expected_audit = _source_subject_contract(dataset, panel)
            if tuple(sorted(set(source.subject_id))) != tuple(sorted(expected_source)):
                raise C84FDualLevelTrainingError("source-training identity failed input freeze")
            if tuple(sorted(set(audit.subject_id))) != tuple(sorted(expected_audit)):
                raise C84FDualLevelTrainingError("source-audit identity failed input freeze")
            panels.append({
                "dataset": dataset, "panel": panel,
                "source_training_subjects": list(expected_source),
                "source_audit_subjects": list(expected_audit),
                "source_training_trial_count": len(source.trial_id),
                "source_audit_trial_count": len(audit.trial_id),
                "source_training_trial_id_sha256": hashlib.sha256(
                    manifests.canonical_bytes(list(source.trial_id))
                ).hexdigest(),
                "source_audit_trial_id_sha256": hashlib.sha256(
                    manifests.canonical_bytes(list(audit.trial_id))
                ).hexdigest(),
                "channel_order": list(source.interface.actual_ch_names),
                "sample_rate_hz": source.interface.actual_sfreq_hz,
                "sample_count": source.interface.final_n_times,
                "sets_disjoint": True, "target_subjects_loaded": 0,
            })
    raw_inputs = [dict(row) for row in raw_and_warning_rows if "path" in row]
    warning_rows = [dict(row) for row in raw_and_warning_rows if "message" in row]
    if len(raw_inputs) + len(warning_rows) != len(raw_and_warning_rows):
        raise C84FDualLevelTrainingError("source input record is neither a raw file nor a warning")
    return {
        "schema_version": "c84f_source_input_freeze_v1",
        "panels": panels, "panel_count": 6,
        "raw_inputs": raw_inputs, "raw_input_file_count": len(raw_inputs),
        "warnings": warning_rows, "warning_count": len(warning_rows),
        "target_EEG_arrays": 0, "target_labels": 0,
        "frozen_at_unix_ns": time.time_ns(),
    }


def model_init_hash(dataset: str, training_seed: int, torch: Any) -> str:
    from . import c84c_real_canary_v2 as base
    from oaci.train.checkpoint import state_hash
    from oaci.train.rng import derive_seed, forked_rng

    with forked_rng(derive_seed(training_seed, dataset, "model_init"), torch.device("cuda:0")):
        model = base._model_factory()
    return state_hash({key: value.detach().cpu() for key, value in model.state_dict().items()})


def materialize_paired_bundles(
    source: Any,
    *,
    dataset: str,
    panel: str,
    training_seed: int,
    torch: Any,
    np: Any,
) -> tuple[dict[int, Any], str]:
    from .c84l1_canary import materialize_training_bundle

    bundles = {
        level: materialize_training_bundle(
            source, dataset=dataset, panel=panel, training_seed=training_seed,
            level=level, torch=torch, np=np,
        )
        for level in (0, 1)
    }
    if bundles[0].plan_hashes == bundles[1].plan_hashes:
        raise C84FDualLevelTrainingError("paired levels did not bind different populations/plans")
    first = model_init_hash(dataset, training_seed, torch)
    second = model_init_hash(dataset, training_seed, torch)
    if first != second:
        raise C84FDualLevelTrainingError("paired level model initialization is not identical")
    if bundles[0].application.deleted_indices or not bundles[1].application.deleted_indices:
        raise C84FDualLevelTrainingError("level support intervention identity drift")
    return bundles, first


def _atomic_save_source_artifact(
    path: Path,
    *,
    model_state: Mapping[str, Any],
    source_audit: Any,
    unit: Mapping[str, Any],
    torch: Any,
    np: Any,
) -> dict[str, Any]:
    from . import c84c_real_canary_v2 as base
    from . import c84r2_canary_runtime_repair as replay

    model = base._model_factory()
    model.load_state_dict(model_state)
    model.eval().to("cuda:0")
    logits, _ = base._forward_model(model, source_audit.X, torch)
    with torch.inference_mode():
        probabilities = torch.softmax(logits, dim=1)
    base._atomic_save_npz(
        path, np,
        logits=logits.numpy(), probabilities=probabilities.numpy(),
        source_class_label=np.asarray(source_audit.y, dtype=np.int64),
        source_domain_id=np.asarray(source_audit.subject_id, dtype=np.int64),
        source_trial_id=np.asarray(source_audit.trial_id, dtype=str),
        dataset=np.asarray(unit["dataset"]), panel=np.asarray(unit["panel"]),
        seed=np.asarray(int(unit["training_seed"]), dtype=np.int64),
        level=np.asarray(int(unit["level"]), dtype=np.int64), unit_id=np.asarray(unit["unit_id"]),
    )
    return replay.replay_source_audit_artifact(
        path,
        expected_identity={
            "dataset": unit["dataset"], "panel": unit["panel"],
            "seed": int(unit["training_seed"]), "level": int(unit["level"]),
            "unit_id": unit["unit_id"],
        },
        expected_trial_ids=source_audit.trial_id, expected_labels=source_audit.y,
        expected_domains=source_audit.subject_id, np=np,
    )


def _plan_hash_mapping(bundle: Any) -> dict[str, str]:
    return {
        "stage1": bundle.plan_hashes[0], "stage2": bundle.plan_hashes[1],
        "OACI": bundle.plan_hashes[2], "SRC": bundle.plan_hashes[3],
    }


def _field_sidecar(
    *,
    unit: Mapping[str, Any], bundle: Any, paired_init_hash: str,
    checkpoint: Mapping[str, Any], optimizer: Mapping[str, Any],
    source_descriptor: Mapping[str, Any], record: Any,
    parent_hash: str, previous_hash: str, source: Any, source_audit: Any,
) -> dict[str, Any]:
    application = bundle.application
    payload = {
        "schema_version": "c84f_training_sidecar_v1", "unit_id": unit["unit_id"],
        "dataset": unit["dataset"], "panel": unit["panel"], "seed": int(unit["training_seed"]),
        "level": int(unit["level"]), "level_intervention_id": unit["level_intervention_id"],
        "level_intervention_registry_sha256": unit["level_intervention_registry_sha256"],
        "deleted_source_subject": application.deleted_source_subject,
        "deleted_class": application.deleted_class,
        "regime": unit["regime"], "epoch": int(unit["epoch"]),
        "trajectory_order": int(unit["trajectory_order"]),
        "interface_id": protocol.INTERFACE_ID, "montage_sha256": protocol.HASHES["montage"],
        "checkpoint": dict(checkpoint), "optimizer": dict(optimizer),
        "source_audit": dict(source_descriptor), "model_state_hash": record.model_hash,
        "parent_ERM_model_state_hash": parent_hash,
        "previous_trajectory_model_state_hash": previous_hash,
        "genealogy_rule": "shared_level_specific_ERM_parent_then_fixed_regime_trajectory_order",
        "population_signature_sha256": bundle.population_signature,
        "support_graph_sha256": bundle.support.support_hash(),
        "plan_hashes": _plan_hash_mapping(bundle), "paired_model_init_hash": paired_init_hash,
        "paired_model_init_pass": True, "level_support_replay_pass": True,
        "source_training_subjects": sorted(set(source.subject_id)),
        "source_audit_subjects": sorted(set(source_audit.subject_id)),
        "source_training_trial_count": len(bundle.source.trial_id),
        "source_audit_trial_count": len(source_audit.trial_id),
        "training_target_rows": 0, "training_target_labels": 0,
        "source_audit_rows_used_in_training": 0, "target_outcome_retention": 0,
        "target_outcome_retry": 0, "target_artifact": None, "target_fit_ids": [],
        "scientific_metrics": 0,
    }
    if set(payload) != FIELD_SIDECAR_FIELDS:
        raise C84FDualLevelTrainingError("C84F training sidecar field-set drift")
    return payload


def train_level(
    *,
    dataset: str,
    panel: str,
    training_seed: int,
    level: int,
    source: Any,
    source_audit: Any,
    bundle: Any,
    paired_init_hash: str,
    root: Path,
    torch: Any,
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> list[dict[str, Any]]:
    from . import c84c_real_canary as legacy
    from . import c84c_real_canary_v2 as base
    from . import c84r2_canary_runtime_repair as replay
    from oaci.methods.oaci import OACIObjective
    from oaci.methods.source_robust import SRCObjective
    from oaci.train.checkpoint import state_hash
    from oaci.train.engine import InvocationRegistry, train_stage1, train_stage2
    from oaci.train.rng import derive_seed, forked_rng

    units = cell_candidate_rows(dataset, panel, training_seed, level)
    by_key = {(row["regime"], int(row["epoch"])): row for row in units}
    output = []
    level_root = root / dataset / f"panel_{panel}" / f"seed_{training_seed}" / f"level_{level}"
    ledger.increment("training_phases_started", 3)
    with legacy._OptimizerCapture(level_root / "optimizer_states") as capture:
        capture.begin("ERM")
        with forked_rng(derive_seed(training_seed, dataset, "model_init"), torch.device("cuda:0")):
            model = base._model_factory()
        erm = train_stage1(
            model, bundle.data, bundle.stage1_plan, bundle.engine_config, torch.device("cuda:0"),
            InvocationRegistry(), f"C84F|{dataset}|{panel}|seed{training_seed}|level{level}",
        )
        ledger.increment("training_phases_completed")
        capture.begin("OACI")
        oaci = train_stage2(
            base._model_factory, erm, bundle.data, OACIObjective(bundle.support, adv_hidden=16),
            bundle.stage2_plan, bundle.oaci_plan, bundle.engine_config, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        capture.begin("SRC")
        src = train_stage2(
            base._model_factory, erm, bundle.data,
            SRCObjective(2, bundle.support.n_domains, smooth_temperature=0.1),
            bundle.stage2_plan, bundle.full_plan, bundle.engine_config, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        if tuple(record.epoch for record in oaci.trajectory) != CHECKPOINT_EPOCHS:
            raise C84FDualLevelTrainingError("OACI checkpoint cadence drift")
        if tuple(record.epoch for record in src.trajectory) != CHECKPOINT_EPOCHS:
            raise C84FDualLevelTrainingError("SRC checkpoint cadence drift")
        records = [("ERM", erm.checkpoint, 0, capture.descriptor("ERM", 0))]
        records.extend(("OACI", record, order, capture.descriptor("OACI", order))
                       for order, record in enumerate(oaci.trajectory, start=1))
        records.extend(("SRC", record, order, capture.descriptor("SRC", order))
                       for order, record in enumerate(src.trajectory, start=1))
        previous = {"ERM": erm.checkpoint.model_hash, "OACI": erm.checkpoint.model_hash,
                    "SRC": erm.checkpoint.model_hash}
        for regime, record, order, optimizer in records:
            unit = by_key[(regime, int(record.epoch))]
            checkpoint_path = level_root / "checkpoints" / f"{unit['unit_id']}.pt"
            checkpoint = legacy._save_torch_state(checkpoint_path, record.model_state, torch)
            checkpoint_replay = replay.replay_checkpoint(
                checkpoint_path, expected_file_sha256=checkpoint["sha256"],
                expected_state_hash=record.model_hash, torch=torch, state_hash_fn=state_hash,
            )
            optimizer_replay = replay.replay_optimizer_state(
                optimizer, phase=regime, trajectory_order=order, torch=torch,
            )
            source_path = level_root / "source_audit" / f"{unit['unit_id']}.npz"
            source_descriptor = _atomic_save_source_artifact(
                source_path, model_state=record.model_state, source_audit=source_audit,
                unit=unit, torch=torch, np=np,
            )
            ledger.increment("source_audit_artifacts")
            sidecar = _field_sidecar(
                unit=unit, bundle=bundle, paired_init_hash=paired_init_hash,
                checkpoint={**checkpoint, "replay": checkpoint_replay},
                optimizer={**optimizer, "replay": optimizer_replay},
                source_descriptor=source_descriptor, record=record,
                parent_hash=erm.checkpoint.model_hash, previous_hash=previous[regime],
                source=source, source_audit=source_audit,
            )
            sidecar_path = level_root / "sidecars" / f"{unit['unit_id']}.json"
            sidecar_sha = manifests.write_json_atomic(sidecar_path, sidecar)
            observed_sidecar = manifests.read_json(sidecar_path)
            if set(observed_sidecar) != FIELD_SIDECAR_FIELDS or observed_sidecar["unit_id"] != unit["unit_id"]:
                raise C84FDualLevelTrainingError("persisted C84F sidecar replay failed")
            output.append({
                "unit_id": unit["unit_id"], "dataset": dataset, "panel": panel,
                "training_seed": training_seed, "level": level,
                "level_intervention_id": unit["level_intervention_id"], "regime": regime,
                "epoch": int(record.epoch), "trajectory_order": order,
                "checkpoint_path": str(checkpoint_path), "checkpoint_sha256": checkpoint["sha256"],
                "optimizer_path": optimizer["path"], "optimizer_sha256": optimizer["file_sha256"],
                "sidecar_path": str(sidecar_path), "sidecar_sha256": sidecar_sha,
                "source_audit_path": str(source_path), "source_audit_sha256": source_descriptor["sha256"],
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous[regime],
                "population_signature_sha256": bundle.population_signature,
                "support_graph_sha256": bundle.support.support_hash(),
                "plan_hashes": _plan_hash_mapping(bundle), "paired_model_init_hash": paired_init_hash,
                "reuse_provenance": "C84F", "checkpoint_replay_pass": 1,
                "optimizer_replay_pass": 1, "sidecar_replay_pass": 1,
                "source_audit_replay_pass": 1, "training_target_rows": 0,
                "training_target_labels": 0, "source_audit_rows_used_in_training": 0,
                "target_outcome_retention": 0, "target_outcome_retry": 0,
            })
            ledger.increment("new_training_units")
            previous[regime] = record.model_hash
    if len(output) != 81:
        raise C84FDualLevelTrainingError(f"training level produced {len(output)} units instead of 81")
    return output


def train_paired_cell(
    *,
    dataset: str,
    panel: str,
    training_seed: int,
    source: Any,
    source_audit: Any,
    root: Path,
    torch: Any,
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> list[dict[str, Any]]:
    validate_paired_cell_scope(dataset, panel, training_seed)
    bundles, paired_init = materialize_paired_bundles(
        source, dataset=dataset, panel=panel, training_seed=training_seed,
        torch=torch, np=np,
    )
    rows = []
    for level in (0, 1):
        rows.extend(train_level(
            dataset=dataset, panel=panel, training_seed=training_seed, level=level,
            source=source, source_audit=source_audit, bundle=bundles[level],
            paired_init_hash=paired_init, root=root, torch=torch, np=np, ledger=ledger,
        ))
    if len(rows) != 162 or {row["paired_model_init_hash"] for row in rows} != {paired_init}:
        raise C84FDualLevelTrainingError("paired training cell did not produce 162 units with one init hash")
    return rows


def _canary_manifests() -> tuple[dict[str, Any], dict[str, Any]]:
    return manifests.read_json(protocol.C84C_MANIFEST), manifests.read_json(protocol.C84L1C_MANIFEST)


def validate_canary_plan_replay(
    source_views: Mapping[tuple[str, str], tuple[Any, Any]], *, torch: Any, np: Any,
) -> dict[str, Any]:
    c84c, c84l1c = _canary_manifests()
    level0_by_dataset = {row["dataset"]: row for row in c84c["datasets"]}
    level1_by_dataset = {row["dataset"]: row for row in c84l1c["datasets"]}
    rows = []
    for dataset in protocol.DATASETS:
        source, _ = source_views[(dataset, "A")]
        bundles, init_hash = materialize_paired_bundles(
            source, dataset=dataset, panel="A", training_seed=5, torch=torch, np=np,
        )
        level0_expected = tuple(level0_by_dataset[dataset]["deterministic_prefix"]["plan_hashes"])
        level1_expected = tuple(level1_by_dataset[dataset]["level1_plan_hashes"])
        if bundles[0].plan_hashes != level0_expected or bundles[1].plan_hashes != level1_expected:
            raise C84FDualLevelTrainingError(f"accepted canary plan replay failed: {dataset}")
        if init_hash != level1_by_dataset[dataset]["paired_model_init_hash"]:
            raise C84FDualLevelTrainingError(f"accepted paired model-init replay failed: {dataset}")
        rows.append({
            "dataset": dataset, "level0_plan_replay_pass": True,
            "level1_plan_replay_pass": True, "paired_model_init_replay_pass": True,
            "paired_model_init_hash": init_hash,
        })
    return {"datasets": rows, "replay_pass": True}


def _replay_reused_artifacts(
    row: Mapping[str, str], *, torch: Any, np: Any,
) -> tuple[dict[str, Any], Mapping[str, Any]]:
    from . import c84r2_canary_runtime_repair as replay
    from oaci.train.checkpoint import state_hash

    sidecar = manifests.read_json(row["sidecar_path"])
    checkpoint = replay.replay_checkpoint(
        Path(row["checkpoint_path"]), expected_file_sha256=row["checkpoint_sha256"],
        expected_state_hash=row["model_state_hash"], torch=torch, state_hash_fn=state_hash,
    )
    optimizer = replay.replay_optimizer_state(
        {"path": row["optimizer_path"], "file_sha256": row["optimizer_sha256"]},
        phase=row["regime"], trajectory_order=int(row["trajectory_order"]), torch=torch,
    )
    source = sidecar["source_audit"]
    with np.load(row["source_audit_path"], allow_pickle=False) as archive:
        if set(archive.files) != SOURCE_AUDIT_FIELDS:
            raise C84FDualLevelTrainingError("reused source-audit schema drift")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        error = float(np.max(np.abs(softmax - probabilities)))
        manifests.validate_finite_error(error, tolerance=1e-6, name="reused source softmax")
    if manifests.sha256_file(row["sidecar_path"]) != row["sidecar_sha256"]:
        raise C84FDualLevelTrainingError("reused sidecar hash drift")
    if source["sha256"] != row["source_audit_sha256"]:
        raise C84FDualLevelTrainingError("reused source-audit sidecar identity drift")
    return {"checkpoint": checkpoint, "optimizer": optimizer, "source_softmax_error": error}, sidecar


def reused_model_rows(
    source_views: Mapping[tuple[str, str], tuple[Any, Any]],
    *,
    torch: Any,
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> list[dict[str, Any]]:
    reuse = read_csv(protocol.TABLE_DIR / "dual_canary_reuse_registry.csv")
    bundles_by_key: dict[tuple[str, int], Any] = {}
    init_by_dataset: dict[str, str] = {}
    for dataset in protocol.DATASETS:
        source, _ = source_views[(dataset, "A")]
        bundles, init_hash = materialize_paired_bundles(
            source, dataset=dataset, panel="A", training_seed=5, torch=torch, np=np,
        )
        init_by_dataset[dataset] = init_hash
        for level, bundle in bundles.items():
            bundles_by_key[(dataset, level)] = bundle
    output = []
    for row in reuse:
        replayed, sidecar = _replay_reused_artifacts(row, torch=torch, np=np)
        level = int(row["level"])
        bundle = bundles_by_key[(row["dataset"], level)]
        output.append({
            "unit_id": row["unit_id"], "dataset": row["dataset"], "panel": row["panel"],
            "training_seed": int(row["training_seed"]), "level": level,
            "level_intervention_id": row["level_intervention_id"], "regime": row["regime"],
            "epoch": int(row["epoch"]), "trajectory_order": int(row["trajectory_order"]),
            "checkpoint_path": row["checkpoint_path"], "checkpoint_sha256": row["checkpoint_sha256"],
            "optimizer_path": row["optimizer_path"], "optimizer_sha256": row["optimizer_sha256"],
            "sidecar_path": row["sidecar_path"], "sidecar_sha256": row["sidecar_sha256"],
            "source_audit_path": row["source_audit_path"], "source_audit_sha256": row["source_audit_sha256"],
            "model_state_hash": row["model_state_hash"],
            "parent_ERM_model_state_hash": row["parent_ERM_model_state_hash"],
            "previous_trajectory_model_state_hash": row["previous_trajectory_model_state_hash"],
            "population_signature_sha256": bundle.population_signature,
            "support_graph_sha256": bundle.support.support_hash(),
            "plan_hashes": _plan_hash_mapping(bundle),
            "paired_model_init_hash": init_by_dataset[row["dataset"]],
            "reuse_provenance": row["reuse_source"], "checkpoint_replay_pass": 1,
            "optimizer_replay_pass": 1, "sidecar_replay_pass": 1,
            "source_audit_replay_pass": 1, "training_target_rows": 0,
            "training_target_labels": 0, "source_audit_rows_used_in_training": 0,
            "target_outcome_retention": 0, "target_outcome_retry": 0,
        })
        ledger.increment("reused_training_units")
    if len(output) != 486:
        raise C84FDualLevelTrainingError("reused model field is not 486 units")
    return output


def _protected_loader_objects() -> tuple[dict[str, Any], tuple[Any, Any, Any, Any]]:
    from moabb.datasets import Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    objects = {
        "moabb.datasets.Lee2019_MI": Lee2019_MI,
        "moabb.datasets.Cho2017": Cho2017,
        "moabb.datasets.PhysionetMI": PhysionetMI,
        "moabb.paradigms.MotorImagery": MotorImagery,
    }
    return objects, (Lee2019_MI, Cho2017, PhysionetMI, MotorImagery)


def run_real(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    binding = runtime.require_authorization_and_lock(
        authorization_path=authorization_path, output_root=output_root,
    )
    consumption = runtime.consume_authorization(binding)
    ledger = runtime.ExecutionAttemptLedger(Path(binding["run_root"]), consumption)
    try:
        ledger.stage("protected_package_imports_and_runtime_versions")
        import numpy as np
        import torch
        import mne
        import moabb

        ledger.increment("package_imports", 4)
        runtime.verify_protected_runtime_versions(binding["lock"], torch=torch, mne=mne, moabb=moabb)
        ledger.stage("CUDA_and_determinism")
        ledger.increment("CUDA_checks")
        if not torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID") is None:
            raise C84FDualLevelTrainingError("C84F requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise C84FDualLevelTrainingError("torch deterministic algorithms are not enabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        ledger.increment("loader_source_replays")
        objects, loader_classes = _protected_loader_objects()
        ledger.increment("dataset_loader_imports")
        runtime.verify_loader_runtime_objects(binding["lock"], objects)

        ledger.stage("source_only_input_freeze")
        source_views = {}
        raw_and_warnings = []
        for dataset in protocol.DATASETS:
            for panel in ("A", "B"):
                source, audit, records = load_source_panel_views(
                    dataset, panel, loader_classes=loader_classes, np=np, ledger=ledger,
                )
                source_views[(dataset, panel)] = (source, audit)
                raw_and_warnings.extend(records)
        source_payload = source_input_freeze_payload(source_views, raw_and_warnings)
        source_path = Path(binding["run_root"]) / "C84F_SOURCE_INPUT_FREEZE.json"
        source_sha = manifests.write_json_atomic(source_path, source_payload)
        validate_canary_plan_replay(source_views, torch=torch, np=np)
        if ledger.counters["target_get_data_calls"] or ledger.counters["target_EEG_arrays"]:
            raise C84FDualLevelTrainingError("target access occurred before model-field training")

        ledger.stage("dual_canary_model_state_source_reuse")
        model_rows = reused_model_rows(source_views, torch=torch, np=np, ledger=ledger)
        for wave, panel, seed in (("A", "A", 6), ("B0", "B", 5), ("B1", "B", 6)):
            ledger.stage(f"training_wave_{wave}")
            before_target = (ledger.counters["target_get_data_calls"], ledger.counters["target_EEG_arrays"])
            for dataset in protocol.DATASETS:
                source, audit = source_views[(dataset, panel)]
                model_rows.extend(train_paired_cell(
                    dataset=dataset, panel=panel, training_seed=seed,
                    source=source, source_audit=audit, root=Path(binding["run_root"]) / "new_model_field",
                    torch=torch, np=np, ledger=ledger,
                ))
            after_target = (ledger.counters["target_get_data_calls"], ledger.counters["target_EEG_arrays"])
            if before_target != after_target:
                raise C84FDualLevelTrainingError(f"wave {wave} accessed target data")
            expected_new = {"A": 486, "B0": 972, "B1": 1458}[wave]
            if ledger.counters["new_training_units"] != expected_new:
                raise C84FDualLevelTrainingError(f"wave {wave} unit release gate failed")
            ledger.publish_partial_manifest("IN_PROGRESS")

        ledger.stage("atomic_model_field_freeze")
        model_rows.sort(key=lambda row: row["unit_id"])
        model_manifest = manifests.publish_model_field_manifest(
            binding["run_root"], model_rows,
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "authorization_consumption_sha256": consumption["sha256"],
                "source_input_freeze_sha256": source_sha,
            },
        )
        ledger.increment("model_field_units", 1944)
        ledger.publish_partial_manifest("MODEL_FIELD_FROZEN")

        ledger.stage("complete_target_unlabeled_registry_after_model_freeze")
        from . import c84f_target_instrumentation as target_stage

        target_views, raw_target_files = target_stage.load_complete_target_views(
            model_manifest_path=model_manifest["path"],
            model_manifest_sha_path=model_manifest["sha256_path"],
            loader_classes=loader_classes, np=np, ledger=ledger,
        )
        raw_target_path = Path(binding["run_root"]) / "C84F_TARGET_RAW_INPUT_MANIFEST.json"
        raw_target_sha = manifests.write_json_atomic(raw_target_path, {
            "schema_version": "c84f_target_raw_input_manifest_v1",
            "files": raw_target_files, "file_count": len(raw_target_files), "target_labels": 0,
        })
        trial_rows = target_stage.target_trial_registry_rows(target_views)
        ledger.increment("target_registry_trials", len(trial_rows))
        target_registry = manifests.publish_target_trial_registry(
            binding["run_root"], trial_rows,
            model_manifest_path=model_manifest["path"],
            model_manifest_sha_path=model_manifest["sha256_path"],
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "target_raw_input_manifest_sha256": raw_target_sha,
            },
        )

        ledger.stage("complete_target_unlabeled_instrumentation")
        reuse_map = {
            row["unit_id"]: row
            for row in read_csv(protocol.TABLE_DIR / "dual_canary_reuse_registry.csv")
        }
        descriptors, instrumentation = target_stage.instrument_complete_field(
            model_rows=model_rows, views=target_views, reuse_rows=reuse_map,
            output_root=binding["run_root"], model_manifest_path=model_manifest["path"],
            model_manifest_sha_path=model_manifest["sha256_path"], torch=torch, np=np, ledger=ledger,
        )
        ledger.stage("atomic_complete_field_manifest")
        complete = manifests.publish_complete_field_manifest(
            binding["run_root"], descriptors,
            operative_unit_ids=[row["unit_id"] for row in model_rows],
            model_manifest_path=model_manifest["path"],
            model_manifest_sha_path=model_manifest["sha256_path"],
            target_registry_path=target_registry["path"],
            target_registry_sha_path=target_registry["sha256_path"],
            instrumentation_summary=instrumentation,
            execution_identity={
                "execution_lock_sha256": binding["lock_sha256"],
                "authorization_consumption_sha256": consumption["sha256"],
            },
        )
        ledger.complete(complete["sha256"])
        return {
            "schema_version": "c84f_execution_result_v1",
            "gate": protocol.FIELD_GATE, "complete_manifest_path": complete["path"],
            "complete_manifest_sha256": complete["sha256"], "model_units": 1944,
            "new_training_units": 1458, "reused_units": 486,
            "target_contexts": 944, "candidate_context_slices": 76464,
            "target_labels": 0, "scientific_metrics": 0, "C84S_authorized": False,
        }
    except Exception as exc:
        ledger.fail(exc)
        raise


def synthetic_contract() -> dict[str, Any]:
    scopes = [
        validate_paired_cell_scope(dataset, panel, seed)
        for dataset in protocol.DATASETS
        for panel, seed in (("A", 6), ("B", 5), ("B", 6))
    ]
    return {
        "schema_version": "c84f_dual_level_synthetic_contract_v1",
        "paired_cells": len(scopes), "units": sum(row["candidate_units"] for row in scopes),
        "training_phases": sum(row["training_phases"] for row in scopes),
        "waves": {"A": 486, "B0": 486, "B1": 486},
        "target_access_before_model_freeze": 0,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84F dual-level full-field execution")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-contract")
    subparsers.add_parser("synthetic-contract")
    real = subparsers.add_parser("run-real")
    real.add_argument("--authorization-record", type=Path, default=AUTHORIZATION_RECORD_PATH)
    real.add_argument("--output-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args(argv)
    if args.command == "show-contract":
        print(json.dumps({
            "stage": "C84F", "execution_lock": str(runtime.EXECUTION_LOCK_PATH),
            "fresh_authorization_required": True, "new_training_units": 1458,
            "reused_units": 486, "target_access_before_model_freeze": 0,
            "target_labels": 0, "scientific_metrics": 0, "C84S": False,
        }, sort_keys=True))
        return 0
    if args.command == "synthetic-contract":
        print(json.dumps(synthetic_contract(), sort_keys=True))
        return 0
    print(json.dumps(run_real(
        authorization_path=args.authorization_record, output_root=args.output_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
