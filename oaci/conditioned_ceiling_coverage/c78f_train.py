"""Authorized C78F historical ERM/OACI/SRC training workers.

Authorization and implementation checks run before any EEG, CUDA, or training
imports.  Each invocation handles one remaining target and never loads that
target or the two source-audit subjects during training.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import c74_cache
from . import c78_authorized_train as c78_train
from . import c78r_train
from . import c78f_full_seed3_field as c78f
from . import c78f_runtime as runtime


DEFAULT_DATALAKE_ROOT = "/projects/EEG-foundation-model/datalake/raw"


def _phase_root(lock: dict[str, Any], target: int, phase: str) -> Path:
    if phase not in {"oaci_erm", "src"}:
        raise ValueError(phase)
    return runtime.target_root(lock, target) / "training" / phase


def _materialized_manifest(root: Path, target: int):
    from oaci.confirmatory.loso_plan import explicit_split, loso_fold_spec
    from oaci.confirmatory.materialize import materialize_pilot_manifest
    from oaci.confirmatory.schema import load_confirmatory
    from oaci.protocol.manifest_v2 import load_v2

    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / f"C78F_target{target:03d}_seed3_fullbudget_manifest.yaml"
    spec = loso_fold_spec(target, dataset_id=c78f.DATASET)
    if path.exists():
        manifest = load_v2(str(path))
        manifest.validate_complete()
        return path, manifest, spec
    protocol = load_confirmatory("oaci/protocol/confirmatory_v2.yaml")
    temporary = config_dir / f".C78F_target{target:03d}.provisional.yaml"
    materialize_pilot_manifest(
        protocol,
        c78f.DATASET,
        target_subject=target,
        out_path=str(temporary),
        model_seeds=[c78f.SEED],
        explicit_split=explicit_split(spec),
        deleted_cell=dict(spec["deleted_cell"]),
        bootstrap_override=None,
    )
    os.replace(temporary, path)
    manifest = load_v2(str(path))
    manifest.validate_complete()
    return path, manifest, spec


def _load_training_context(root: Path, target: int, datalake_root: str) -> dict[str, Any]:
    from oaci.data.eeg.bnci import load_moabb_confirmatory
    from oaci.protocol.manifest_v2 import optimization_manifest_hash
    from oaci.runner.keys import FoldKey
    from oaci.runner.maps import build_frozen_maps
    from oaci.runner.scope import ScopePlanConfig
    from oaci.runner.support import DeletionCell, level0_reference_prior, make_deletion_schedule

    manifest_path, manifest, split = _materialized_manifest(root, target)
    expected_train = sorted(int(item) for item in split["source_train_subjects"])
    source_audit = sorted(int(item) for item in split["source_audit_subjects"])
    if target in expected_train or target in source_audit:
        raise RuntimeError("C78F target leaked into a source split")
    if len(expected_train) != 6 or len(source_audit) != 2:
        raise RuntimeError("C78F expected six train and two audit source subjects")
    if set(expected_train) | set(source_audit) | {target} != set(range(1, 10)):
        raise RuntimeError("C78F LOSO subject partition is incomplete")
    dataset = manifest.enabled_datasets()[c78f.DATASET]
    loaded = load_moabb_confirmatory(
        c78f.DATASET,
        expected_train,
        dataset.preprocessing,
        frozen_class_names=dataset.class_names,
        frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq),
        expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    if sorted(loaded.evidence.subjects) != expected_train or int(loaded.bundle.n) != 6 * 576:
        raise RuntimeError("C78F training loader subject/row isolation failed")
    source = c78_train._source_training_data(loaded.bundle)
    source_domains = sorted(set(source.domain_id))
    evaluation_domains = [
        f"{c78f.DATASET}|subject-{subject:03d}"
        for subject in [*source_audit, target]
    ]
    maps = build_frozen_maps(source.class_names, source_domains, evaluation_domains)
    deletion = DeletionCell(split["deleted_cell"]["domain_id"], split["deleted_cell"]["class_name"])
    schedule = make_deletion_schedule([deletion], source, maps)
    reference = level0_reference_prior(source, maps)
    fold_key = FoldKey(
        manifest.freeze()["sha256"],
        c78f.DATASET,
        f"{c78f.DATASET}|target-subject-{target:03d}",
        int(manifest.seeds.split),
        int(manifest.seeds.deletion),
        optimization_manifest_hash(manifest),
    )
    fold_scope = c78_train.TrainingFoldScope(
        fold_key=fold_key,
        maps=maps,
        level0_reference_prior=reference,
    )
    model_spec, execution_cfg, model_factory, geometry = c78_train._model_contract(manifest)
    return {
        "manifest_path": manifest_path,
        "manifest": manifest,
        "split": split,
        "dataset": dataset,
        "loaded": loaded,
        "source": source,
        "maps": maps,
        "schedule": schedule,
        "reference": reference,
        "fold_key": fold_key,
        "fold_scope": fold_scope,
        "scope_cfg": ScopePlanConfig.from_manifest(manifest, support_m=int(dataset.support_m)),
        "model_spec": model_spec,
        "execution_cfg": execution_cfg,
        "model_factory": model_factory,
        "geometry": geometry,
        "source_train_subjects": expected_train,
        "source_audit_subjects": source_audit,
    }


def _sidecar(
    *, root: Path, target: int, unit: dict[str, Any], record, checkpoint: dict[str, Any],
    optimizer: dict[str, Any], parent: dict[str, Any], previous_hash: str | None,
    context: dict[str, Any], run_key, support, population, plans,
    training_seconds: float,
) -> dict[str, Any]:
    regime = unit["regime"]
    payload = {
        "schema_version": "c78f_checkpoint_sidecar_v1",
        "unit_id": unit["unit_id"],
        "dataset": c78f.DATASET,
        "target": target,
        "source_subjects": context["source_train_subjects"],
        "seed": c78f.SEED,
        "regime": regime,
        "level": int(unit["level"]),
        "epoch": int(unit["epoch"]),
        "trajectory_order": int(unit["trajectory_order"]),
        "optimizer_step": int(record.optimizer_step),
        "checkpoint_id": checkpoint["model_hash"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_tensor_schema": checkpoint["tensors"],
        **optimizer,
        "parent_ERM_checkpoint_id": parent["checkpoint_id"],
        "parent_ERM_checkpoint_file_sha256": parent["checkpoint_file_sha256"],
        "parent_ERM_sidecar_sha256": parent.get("sidecar_sha256", "self_for_ERM"),
        "parent_ERM_read_only": regime == "SRC",
        "ERM_retrained_in_SRC_process": False,
        "OACI_weight_access_in_SRC_process": False,
        "previous_trajectory_checkpoint_id": previous_hash,
        "genealogy_rule": "shared_level_ERM_parent_plus_previous_regime_order",
        "historical_SRC_commit": c78f.SRC_HISTORICAL_COMMIT if regime == "SRC" else "not_applicable",
        "SRC_smooth_temperature": c78f.SRC_SMOOTH_TEMPERATURE if regime == "SRC" else "not_applicable",
        "manifest_path": str(context["manifest_path"]),
        "manifest_file_sha256": c78f.sha256_file(context["manifest_path"]),
        "manifest_canonical_sha256": context["manifest"].freeze()["sha256"],
        "run_key_hash": run_key.run_key_hash,
        "optimization_identity_hash": run_key.optimization_identity_hash,
        "execution_config_hash": context["execution_cfg"].execution_config_hash,
        "model_spec_hash": context["model_spec"].model_spec_hash,
        "support_hash": support.support_hash,
        "level_support_hash": support.level_support_hash,
        "population_hash": population.population_hash,
        "tensor_hash": population.tensor_hash,
        "stage1_task_plan_hash": plans.stage1_task.plan_hash,
        "stage2_task_plan_hash": plans.stage2_task.plan_hash,
        "alignment_plan_hash": (
            plans.oaci_alignment.plan_hash if regime == "OACI"
            else plans.full_domain_alignment.plan_hash if regime == "SRC"
            else "not_applicable"
        ),
        "source_loader_evidence_hash": context["loaded"].evidence.evidence_hash,
        "source_raw_fingerprint": context["loaded"].evidence.raw_data_fingerprint,
        "source_preprocessing_signature": context["loaded"].evidence.resolved_preprocess_hash,
        "R_src": float(record.R_src),
        "balanced_err": float(record.balanced_err),
        "train_surrogate": float(record.train_surrogate),
        "lambda": float(record.lam),
        "training_seconds_level": float(training_seconds),
        "target_fit_ids_empty": True,
        "target_labels_available_to_optimizer": False,
        "target_rows_loaded_in_training_process": 0,
        "source_audit_rows_loaded_in_training_process": 0,
        "checkpoint_retention_rule": unit["retention_rule"],
        "target_outcome_used_for_retention": False,
        "target_outcome_used_for_retry": False,
        "selector_artifact": False,
    }
    return runtime.write_manifest(root / "sidecars" / f"{unit['unit_id']}.json", payload)


def _field_execution(
    *, started_wall: float, started_cpu: float, device, expected_train: list[int],
    root: Path,
) -> dict[str, Any]:
    import torch

    return {
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "GPU_name": torch.cuda.get_device_name(device),
        "GPU_count": 1,
        "GPU_wall_hours": (time.time() - started_wall) / 3600,
        "wall_seconds": time.time() - started_wall,
        "process_CPU_seconds": time.process_time() - started_cpu,
        "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "external_storage_bytes_at_freeze": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "retry_or_requeue_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
        "source_training_subjects": expected_train,
        "source_audit_subjects_loaded_during_training": [],
        "target_subject_loaded_during_training": False,
        "target_data_rows_loaded_during_training": 0,
        "target_label_reads_during_training": 0,
        "selector_target_read": False,
        "seed4_access": 0,
        "BNCI2014_004_access": 0,
    }


def _require_wave_permission(lock: dict[str, Any], target: int) -> None:
    if c78f.wave_for_target(target) == "B":
        gate = runtime.verify_manifest(runtime.wave_gate_path(lock, "A"))
        if not gate.get("all_engineering_gates_passed") or gate.get("target_scientific_outcomes_read"):
            raise PermissionError("C78F Wave B requires a clean Wave-A engineering-only gate")


def train_oaci_erm(target: int, datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    target = runtime.require_target(target)
    _require_wave_permission(lock, target)
    git_boundary = runtime.verify_git_boundary()
    root = _phase_root(lock, target, "oaci_erm")
    frozen_path = runtime.oaci_field_path(lock, target)
    if frozen_path.exists():
        return runtime.require_oaci_field(lock, target)

    import torch
    from oaci.runner.keys import RunKey
    from oaci.runner.objectives import make_objective
    from oaci.runner.plans import build_level_plans
    from oaci.runner.scope import build_level_population
    from oaci.runner.stage1 import run_stage1_once
    from oaci.runner.support import build_level_support
    from oaci.train.engine import InvocationRegistry, train_stage2

    started_wall = time.time()
    started_cpu = time.process_time()
    attempt_path = runtime.campaign_root(lock) / "execution" / "execution_attempts.jsonl"
    runtime.append_jsonl(attempt_path, {
        "event": "start", "stage": "oaci_erm_training", "target": target,
        "wave": c78f.wave_for_target(target), "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "time": c78f.utc_now(), "target_outcomes_read": 0,
    })
    device = torch.device("cuda:0")
    gpu_preflight = c78_train._gpu_preflight(device, root)
    torch.cuda.reset_peak_memory_stats(device)
    context = _load_training_context(root, target, datalake_root)
    prospective = runtime.units_for(target, {"ERM", "OACI"})
    by_key = {(int(row["level"]), row["regime"], int(row["epoch"])): row for row in prospective}
    manifest = context["manifest"]
    capture = c78_train.OptimizerCapture(
        root / "optimizer_states",
        stage1_steps=int(manifest.training.stage1_epochs) * int(manifest.training.stage1_steps_per_epoch),
        stage2_checkpoint_steps=int(manifest.training.checkpoint_every) * int(manifest.training.stage2_steps_per_epoch),
    )
    units: list[dict[str, Any]] = []
    levels = []
    with capture.patch_engine():
        for level in c78f.LEVELS:
            level_started = time.time()
            support = build_level_support(
                context["source"], context["maps"], level, context["schedule"],
                context["reference"], support_m=int(context["dataset"].support_m),
            )
            population = build_level_population(context["source"], context["maps"], support)
            plans = build_level_plans(
                context["fold_scope"], level, support, population, context["scope_cfg"], model_seed=c78f.SEED,
            )
            run_key = RunKey(context["fold_key"], level, c78f.SEED)
            engine_cfg = context["execution_cfg"].engine_config_for(run_key)
            capture.begin("stage1", level)
            erm_started = time.time()
            stage1 = run_stage1_once(
                run_key, population, plans, context["model_factory"], context["model_spec"],
                engine_cfg, context["execution_cfg"].execution_config_hash,
                InvocationRegistry(), device,
            )
            erm_seconds = time.time() - erm_started
            erm = stage1.erm_stage.checkpoint
            capture.begin("stage2", level)
            objective, objective_spec = make_objective("OACI", support, context["fold_scope"], context["execution_cfg"])
            if not objective.active_status().active or plans.oaci_alignment is None:
                raise RuntimeError(f"C78F OACI inactive for target {target}, level {level}")
            oaci_started = time.time()
            trained = train_stage2(
                context["model_factory"], stage1.erm_stage, population.training_data,
                objective, plans.stage2_task, plans.oaci_alignment, engine_cfg, device,
            )
            oaci_seconds = time.time() - oaci_started
            if tuple(record.epoch for record in trained.trajectory) != c78f.OACI_EPOCHS:
                raise RuntimeError("C78F OACI fixed cadence drift")
            level_seconds = time.time() - level_started
            level_units = []
            erm_unit = by_key[(level, "ERM", 199)]
            erm_checkpoint = c78_train._save_checkpoint(root, erm.model_hash, erm.model_state)
            parent = {
                "checkpoint_id": erm.model_hash,
                "checkpoint_file_sha256": erm_checkpoint["file_sha256"],
            }
            erm_sidecar = _sidecar(
                root=root, target=target, unit=erm_unit, record=erm,
                checkpoint=erm_checkpoint, optimizer=capture.descriptor("stage1", level, 0),
                parent=parent, previous_hash=None, context=context, run_key=run_key,
                support=support, population=population, plans=plans, training_seconds=level_seconds,
            )
            erm_sidecar_path = root / "sidecars" / f"{erm_unit['unit_id']}.json"
            parent["sidecar_sha256"] = c78f.sha256_file(erm_sidecar_path)
            level_units.append({
                "unit_id": erm_unit["unit_id"], "target": target, "seed": c78f.SEED,
                "level": level, "regime": "ERM", "epoch": 199, "trajectory_order": 0,
                "checkpoint_id": erm.model_hash, "checkpoint_path": erm_checkpoint["path"],
                "checkpoint_file_sha256": erm_checkpoint["file_sha256"],
                "sidecar_path": str(erm_sidecar_path), "sidecar_sha256": parent["sidecar_sha256"],
                "optimizer_state_hash": erm_sidecar["optimizer_state_hash"],
            })
            previous = erm.model_hash
            trace_rows = [{"regime": "ERM", "epoch": 199, "trajectory_order": 0, "checkpoint_id": erm.model_hash, "R_src": erm.R_src, "balanced_err": erm.balanced_err, "train_surrogate": erm.train_surrogate, "lambda": erm.lam}]
            for order, record in enumerate(trained.trajectory, start=1):
                unit = by_key[(level, "OACI", int(record.epoch))]
                checkpoint = c78_train._save_checkpoint(root, record.model_hash, record.model_state)
                sidecar = _sidecar(
                    root=root, target=target, unit=unit, record=record, checkpoint=checkpoint,
                    optimizer=capture.descriptor("stage2", level, order), parent=parent,
                    previous_hash=previous, context=context, run_key=run_key, support=support,
                    population=population, plans=plans, training_seconds=level_seconds,
                )
                sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
                level_units.append({
                    "unit_id": unit["unit_id"], "target": target, "seed": c78f.SEED,
                    "level": level, "regime": "OACI", "epoch": int(record.epoch), "trajectory_order": order,
                    "checkpoint_id": record.model_hash, "checkpoint_path": checkpoint["path"],
                    "checkpoint_file_sha256": checkpoint["file_sha256"],
                    "sidecar_path": str(sidecar_path), "sidecar_sha256": c78f.sha256_file(sidecar_path),
                    "optimizer_state_hash": sidecar["optimizer_state_hash"],
                })
                trace_rows.append({"regime": "OACI", "epoch": record.epoch, "trajectory_order": order, "checkpoint_id": record.model_hash, "R_src": record.R_src, "balanced_err": record.balanced_err, "train_surrogate": record.train_surrogate, "lambda": record.lam})
                previous = record.model_hash
            trace = c74_cache.write_content_addressed_npz(
                root / "trajectory_traces" / f"level-{level}", "OACI_ERM_trajectory_trace",
                {key: np.asarray([row[key] for row in trace_rows]) for key in trace_rows[0]},
            )
            level_manifest = runtime.write_manifest(root / "levels" / f"level-{level}.json", {
                "schema_version": "c78f_oaci_erm_level_manifest_v1",
                "target": target, "level": level, "units": level_units,
                "checkpoint_count": len(level_units), "ERM_parent": erm.model_hash,
                "trajectory_trace": trace, "level_wall_seconds": level_seconds,
                "ERM_wall_seconds": erm_seconds, "OACI_wall_seconds": oaci_seconds,
                "source_subjects_loaded": context["source_train_subjects"],
                "target_subject_loaded": False, "source_audit_subjects_loaded": [],
                "support_hash": support.support_hash, "objective_spec_hash": objective_spec.objective_spec_hash,
            })
            levels.append(level_manifest)
            units.extend(level_units)
    if len(units) != 82 or {row["unit_id"] for row in units} != {row["unit_id"] for row in prospective}:
        raise RuntimeError("C78F OACI/ERM field differs from locked 82-unit target manifest")
    execution = _field_execution(
        started_wall=started_wall, started_cpu=started_cpu, device=device,
        expected_train=context["source_train_subjects"], root=root,
    )
    field = runtime.write_manifest(frozen_path, {
        "schema_version": "c78f_oaci_erm_field_frozen_v1",
        "created_at_utc": c78f.utc_now(),
        "protocol_sha256": protocol_sha, "target": target, "wave": c78f.wave_for_target(target),
        "units": units, "unit_count": 82, "ERM_count": 2, "OACI_count": 80,
        "retention_uses_target_outcomes": False, "retry_selection_uses_target_outcomes": False,
        "all_retention_decisions_frozen": True, "git": git_boundary, "GPU_preflight": gpu_preflight,
        "execution": execution,
        "materialized_manifest": {"path": str(context["manifest_path"]), "sha256": c78f.sha256_file(context["manifest_path"]), "canonical_sha256": context["manifest"].freeze()["sha256"]},
        "source_loader": {"subjects": context["source_train_subjects"], "rows": int(context["loaded"].bundle.n), "evidence_hash": context["loaded"].evidence.evidence_hash, "network_attempt_count": context["loaded"].evidence.network_attempt_count},
        "model_geometry": context["geometry"],
        "level_manifests": [{"level": row["level"], "path": str(root / "levels" / f"level-{row['level']}.json"), "sha256": c78f.sha256_file(root / "levels" / f"level-{row['level']}.json")} for row in levels],
    })
    runtime.append_jsonl(attempt_path, {"event": "complete", "stage": "oaci_erm_training", "target": target, "time": c78f.utc_now(), "job_id": os.environ.get("SLURM_JOB_ID", "unknown"), "units": 82, "target_outcomes_read": 0, "manifest_sha256": field["manifest_sha256"]})
    print(json.dumps({"gate": "C78F_OACI_ERM_FIELD_FROZEN", "target": target, "units": 82, "wall_seconds": execution["wall_seconds"]}, sort_keys=True))
    return field


def _load_erm_anchor(
    *, lock: dict[str, Any], target: int, level: int, context: dict[str, Any],
    support, population, plans, run_key,
):
    import torch
    from oaci.train.checkpoint import CheckpointRecord, ERMStage, state_hash

    field = runtime.require_oaci_field(lock, target)
    candidates = [row for row in field["units"] if row["regime"] == "ERM" and int(row["level"]) == level]
    if len(candidates) != 1:
        raise RuntimeError("C78F SRC expected exactly one frozen ERM anchor per level")
    unit = candidates[0]
    sidecar = runtime.verify_manifest(unit["sidecar_path"])
    checks = {
        "target": target,
        "manifest_canonical_sha256": context["manifest"].freeze()["sha256"],
        "run_key_hash": run_key.run_key_hash,
        "optimization_identity_hash": run_key.optimization_identity_hash,
        "execution_config_hash": context["execution_cfg"].execution_config_hash,
        "model_spec_hash": context["model_spec"].model_spec_hash,
        "support_hash": support.support_hash,
        "level_support_hash": support.level_support_hash,
        "population_hash": population.population_hash,
        "tensor_hash": population.tensor_hash,
        "source_loader_evidence_hash": context["loaded"].evidence.evidence_hash,
        "source_raw_fingerprint": context["loaded"].evidence.raw_data_fingerprint,
        "source_preprocessing_signature": context["loaded"].evidence.resolved_preprocess_hash,
    }
    for key, expected in checks.items():
        if sidecar[key] != expected:
            raise RuntimeError(f"C78F SRC ERM-anchor context mismatch: {key}")
    state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
    if state_hash(state) != unit["checkpoint_id"]:
        raise RuntimeError("C78F SRC ERM-anchor state hash mismatch")
    record = CheckpointRecord(
        epoch=int(sidecar["epoch"]), optimizer_step=int(sidecar["optimizer_step"]),
        model_state=state, model_hash=unit["checkpoint_id"], R_src=float(sidecar["R_src"]),
        balanced_err=float(sidecar["balanced_err"]), train_surrogate=float(sidecar["train_surrogate"]),
        lam=float(sidecar["lambda"]),
    )
    stage = ERMStage(
        checkpoint=record, R_ERM_hat=float(sidecar["R_src"]),
        tau=float(sidecar["R_src"]) + float(context["manifest"].risk.epsilon),
        task_plan_hash=sidecar["stage1_task_plan_hash"],
        stage1_invocation_id=f"C78F_read_only_target{target}_level{level}_{unit['checkpoint_id']}",
    )
    return stage, {
        "unit_id": unit["unit_id"], "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_path": unit["checkpoint_path"], "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
        "sidecar_path": unit["sidecar_path"], "sidecar_sha256": unit["sidecar_sha256"],
        "read_only": True,
    }


def train_src(target: int, datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    target = runtime.require_target(target)
    _require_wave_permission(lock, target)
    runtime.require_oaci_field(lock, target)
    git_boundary = runtime.verify_git_boundary()
    root = _phase_root(lock, target, "src")
    frozen_path = runtime.src_field_path(lock, target)
    if frozen_path.exists():
        return runtime.require_src_field(lock, target)

    import torch
    from oaci.methods.source_robust import SRCObjective
    from oaci.runner.keys import RunKey
    from oaci.runner.plans import build_level_plans
    from oaci.runner.scope import build_level_population
    from oaci.runner.support import build_level_support
    from oaci.train.engine import train_stage2

    started_wall = time.time()
    started_cpu = time.process_time()
    attempt_path = runtime.campaign_root(lock) / "execution" / "execution_attempts.jsonl"
    runtime.append_jsonl(attempt_path, {"event": "start", "stage": "src_training", "target": target, "wave": c78f.wave_for_target(target), "job_id": os.environ.get("SLURM_JOB_ID", "unknown"), "time": c78f.utc_now(), "target_outcomes_read": 0, "ERM_retrained": 0, "OACI_weight_access": 0})
    device = torch.device("cuda:0")
    gpu_preflight = c78_train._gpu_preflight(device, root)
    torch.cuda.reset_peak_memory_stats(device)
    context = _load_training_context(root, target, datalake_root)
    prospective = runtime.units_for(target, {"SRC"})
    by_key = {(int(row["level"]), int(row["epoch"])): row for row in prospective}
    manifest = context["manifest"]
    capture = c78r_train.SRCOptimizerCapture(
        root / "optimizer_states",
        checkpoint_steps=int(manifest.training.checkpoint_every) * int(manifest.training.stage2_steps_per_epoch),
    )
    units: list[dict[str, Any]] = []
    levels = []
    anchors = []
    with capture.patch_engine():
        for level in c78f.LEVELS:
            level_started = time.time()
            support = build_level_support(context["source"], context["maps"], level, context["schedule"], context["reference"], support_m=int(context["dataset"].support_m))
            population = build_level_population(context["source"], context["maps"], support)
            plans = build_level_plans(context["fold_scope"], level, support, population, context["scope_cfg"], model_seed=c78f.SEED)
            if plans.full_domain_alignment is None:
                raise RuntimeError("C78F SRC full-domain alignment inactive")
            run_key = RunKey(context["fold_key"], level, c78f.SEED)
            engine_cfg = context["execution_cfg"].engine_config_for(run_key)
            erm_stage, parent = _load_erm_anchor(lock=lock, target=target, level=level, context=context, support=support, population=population, plans=plans, run_key=run_key)
            anchors.append({"level": level, **parent})
            objective = SRCObjective(len(context["source"].class_names), len(set(context["source"].domain_id)), smooth_temperature=c78f.SRC_SMOOTH_TEMPERATURE)
            if not objective.active_status().active:
                raise RuntimeError("C78F historical SRC unexpectedly inactive")
            capture.begin(level)
            trained = train_stage2(context["model_factory"], erm_stage, population.training_data, objective, plans.stage2_task, plans.full_domain_alignment, engine_cfg, device)
            if tuple(record.epoch for record in trained.trajectory) != c78f.SRC_EPOCHS:
                raise RuntimeError("C78F SRC fixed cadence drift")
            if trained.initial_model_hash != parent["checkpoint_id"]:
                raise RuntimeError("C78F SRC did not initialize from frozen ERM")
            level_seconds = time.time() - level_started
            level_units = []
            previous = parent["checkpoint_id"]
            trace_rows = []
            for order, record in enumerate(trained.trajectory, start=1):
                unit = by_key[(level, int(record.epoch))]
                checkpoint = c78_train._save_checkpoint(root, record.model_hash, record.model_state)
                sidecar = _sidecar(root=root, target=target, unit=unit, record=record, checkpoint=checkpoint, optimizer=capture.descriptor(level, order), parent=parent, previous_hash=previous, context=context, run_key=run_key, support=support, population=population, plans=plans, training_seconds=level_seconds)
                sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
                level_units.append({"unit_id": unit["unit_id"], "target": target, "seed": c78f.SEED, "level": level, "regime": "SRC", "epoch": int(record.epoch), "trajectory_order": order, "checkpoint_id": record.model_hash, "checkpoint_path": checkpoint["path"], "checkpoint_file_sha256": checkpoint["file_sha256"], "sidecar_path": str(sidecar_path), "sidecar_sha256": c78f.sha256_file(sidecar_path), "optimizer_state_hash": sidecar["optimizer_state_hash"]})
                trace_rows.append({"epoch": record.epoch, "trajectory_order": order, "checkpoint_id": record.model_hash, "R_src": record.R_src, "balanced_err": record.balanced_err, "train_surrogate": record.train_surrogate, "lambda": record.lam})
                previous = record.model_hash
            trace = c74_cache.write_content_addressed_npz(root / "trajectory_traces" / f"level-{level}", "SRC_trajectory_trace", {key: np.asarray([row[key] for row in trace_rows]) for key in trace_rows[0]})
            level_manifest = runtime.write_manifest(root / "levels" / f"level-{level}.json", {
                "schema_version": "c78f_SRC_level_manifest_v1",
                "target": target,
                "level": level,
                "units": level_units,
                "read_only_ERM_parent": parent,
                "checkpoint_count": len(level_units),
                "trajectory_trace": trace,
                "level_wall_seconds": level_seconds,
                "SRC_wall_seconds": level_seconds,
                "source_subjects_loaded": context["source_train_subjects"],
                "target_subject_loaded": False,
                "source_audit_subjects_loaded": [],
                "support_hash": support.support_hash,
                "smooth_temperature": c78f.SRC_SMOOTH_TEMPERATURE,
            })
            levels.append(level_manifest)
            units.extend(level_units)
    if len(units) != 80 or {row["unit_id"] for row in units} != {row["unit_id"] for row in prospective}:
        raise RuntimeError("C78F SRC field differs from locked 80-unit target manifest")
    execution = _field_execution(started_wall=started_wall, started_cpu=started_cpu, device=device, expected_train=context["source_train_subjects"], root=root)
    field = runtime.write_manifest(frozen_path, {
        "schema_version": "c78f_SRC_field_frozen_v1",
        "created_at_utc": c78f.utc_now(),
        "protocol_sha256": protocol_sha,
        "target": target,
        "wave": c78f.wave_for_target(target),
        "units": units,
        "unit_count": 80,
        "SRC_count": 80,
        "ERM_retrained_count": 0,
        "OACI_weight_access_count": 0,
        "read_only_ERM_anchor_access": anchors,
        "retention_uses_target_outcomes": False,
        "retry_selection_uses_target_outcomes": False,
        "all_retention_decisions_frozen": True,
        "git": git_boundary,
        "GPU_preflight": gpu_preflight,
        "execution": execution,
        "materialized_manifest": {
            "path": str(context["manifest_path"]),
            "sha256": c78f.sha256_file(context["manifest_path"]),
            "canonical_sha256": context["manifest"].freeze()["sha256"],
        },
        "source_loader": {
            "subjects": context["source_train_subjects"],
            "rows": int(context["loaded"].bundle.n),
            "evidence_hash": context["loaded"].evidence.evidence_hash,
            "network_attempt_count": context["loaded"].evidence.network_attempt_count,
        },
        "model_geometry": context["geometry"],
        "level_manifests": [
            {
                "level": row["level"],
                "path": str(root / "levels" / f"level-{row['level']}.json"),
                "sha256": c78f.sha256_file(root / "levels" / f"level-{row['level']}.json"),
            }
            for row in levels
        ],
    })
    runtime.append_jsonl(attempt_path, {"event": "complete", "stage": "src_training", "target": target, "time": c78f.utc_now(), "job_id": os.environ.get("SLURM_JOB_ID", "unknown"), "units": 80, "target_outcomes_read": 0, "ERM_retrained": 0, "OACI_weight_access": 0, "manifest_sha256": field["manifest_sha256"]})
    print(json.dumps({"gate": "C78F_SRC_FIELD_FROZEN", "target": target, "units": 80, "wall_seconds": execution["wall_seconds"]}, sort_keys=True))
    return field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78f_train")
    sub = parser.add_subparsers(dest="command", required=True)
    for command in ("oaci-erm", "src"):
        child = sub.add_parser(command)
        child.add_argument("--target", type=int, required=True)
        child.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    args = parser.parse_args(argv)
    if args.command == "oaci-erm":
        train_oaci_erm(args.target, args.datalake_root)
    else:
        train_src(args.target, args.datalake_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
