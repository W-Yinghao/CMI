"""Authorized C78R historical SRC stage-2 training worker.

Authorization, protocol, and implementation checks run before importing the
EEG loader, PyTorch CUDA helpers, or training engine. The worker never trains
ERM/OACI: it loads the two hash-locked C78 ERM anchors read-only and trains only
the two registered SRC trajectories.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from . import c74_cache
from . import c78_authorized_common as c78_common
from . import c78_authorized_train as c78_train
from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


DEFAULT_DATALAKE_ROOT = "/projects/EEG-foundation-model/datalake/raw"


class SRCOptimizerCapture:
    """Capture the sole non-adversarial SRC encoder optimizer at fixed cadence."""

    def __init__(self, directory: Path, checkpoint_steps: int):
        self.directory = directory
        self.checkpoint_steps = int(checkpoint_steps)
        self.level = -1
        self.optimizer = None
        self.count = 0
        self.snapshots: dict[tuple[int, int], dict[str, Any]] = {}
        self._original = None

    def begin(self, level: int) -> None:
        self.level = int(level)
        self.optimizer = None
        self.count = 0

    def make_optimizer(self, params, lr, cfg):
        if self.optimizer is not None:
            raise RuntimeError("historical SRC unexpectedly created more than one optimizer")
        optimizer = self._original(params, lr, cfg)
        self.optimizer = optimizer
        original_step = optimizer.step

        def captured_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.count += 1
            if self.count % self.checkpoint_steps == 0:
                self._snapshot(self.count // self.checkpoint_steps)
            return result

        optimizer.step = captured_step
        return optimizer

    def _snapshot(self, order: int) -> None:
        import torch

        payload = {
            "phase": "SRC_stage2", "level": self.level,
            "trajectory_order": int(order), "step_count": self.count,
            "optimizer": c78_train._cpu_clone(self.optimizer.state_dict()),
        }
        identity = c78_train.optimizer_state_hash(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"level-{self.level}_SRC_order-{order:02d}_optimizer_{identity[:16]}.pt"
        if not path.exists():
            fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
            os.close(fd)
            try:
                torch.save(payload, temporary)
                os.replace(temporary, path)
            finally:
                if os.path.exists(temporary):
                    os.unlink(temporary)
        replay = torch.load(path, map_location="cpu", weights_only=True)
        if c78_train.optimizer_state_hash(replay) != identity:
            raise RuntimeError("C78R optimizer snapshot round-trip mismatch")
        self.snapshots[(self.level, int(order))] = {
            "optimizer_state_hash": identity,
            "optimizer_state_path": str(path),
            "optimizer_state_file_sha256": c78r.sha256_file(path),
            "optimizer_step_count": self.count,
        }

    def descriptor(self, level: int, order: int) -> dict[str, Any]:
        key = (int(level), int(order))
        if key not in self.snapshots:
            raise RuntimeError(f"missing C78R SRC optimizer snapshot {key}")
        return dict(self.snapshots[key])

    @contextlib.contextmanager
    def patch_engine(self):
        import oaci.train.engine as engine

        self._original = engine.make_optimizer
        engine.make_optimizer = self.make_optimizer
        try:
            yield
        finally:
            engine.make_optimizer = self._original
            self._original = None


def _c78_anchor(level: int, *, support, population, plans, run_key, execution_cfg, model_spec, manifest, source_evidence):
    import torch
    from oaci.train.checkpoint import CheckpointRecord, ERMStage, state_hash

    c78_lock = c78_common.load_execution_lock()
    c78_field = c78_common.require_field_frozen(c78_lock)
    candidates = [row for row in c78_field["units"] if row["regime"] == "ERM" and int(row["level"]) == int(level)]
    if len(candidates) != 1:
        raise RuntimeError(f"C78R expected one C78 ERM anchor at level {level}")
    unit = candidates[0]
    committed = {
        int(row["level"]): row
        for row in c78r.read_csv(c78r.C78_CHECKPOINTS)
        if row["regime"] == "ERM"
    }[int(level)]
    if unit["checkpoint_id"] != committed["checkpoint_id"]:
        raise RuntimeError("C78R C78 ERM checkpoint ID differs from committed table")
    if c78r.sha256_file(unit["checkpoint_path"]) != committed["checkpoint_path_sha256"]:
        raise RuntimeError("C78R C78 ERM checkpoint file hash drift")
    if c78r.sha256_file(unit["sidecar_path"]) != committed["sidecar_sha256"]:
        raise RuntimeError("C78R C78 ERM sidecar hash drift")
    sidecar = c78_common.verify_canonical_manifest(unit["sidecar_path"])
    checks = {
        "manifest_canonical_sha256": manifest.freeze()["sha256"],
        "run_key_hash": run_key.run_key_hash,
        "optimization_identity_hash": run_key.optimization_identity_hash,
        "execution_config_hash": execution_cfg.execution_config_hash,
        "model_spec_hash": model_spec.model_spec_hash,
        "support_hash": support.support_hash,
        "level_support_hash": support.level_support_hash,
        "population_hash": population.population_hash,
        "tensor_hash": population.tensor_hash,
        "stage1_task_plan_hash": plans.stage1_task.plan_hash,
        "stage2_task_plan_hash": plans.stage2_task.plan_hash,
        "source_loader_evidence_hash": source_evidence.evidence_hash,
        "source_raw_fingerprint": source_evidence.raw_data_fingerprint,
        "source_preprocessing_signature": source_evidence.resolved_preprocess_hash,
    }
    for key, expected in checks.items():
        if sidecar[key] != expected:
            raise RuntimeError(f"C78R C78 ERM anchor context mismatch at {key}")
    state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
    if state_hash(state) != unit["checkpoint_id"]:
        raise RuntimeError("C78R frozen C78 ERM state hash mismatch")
    record = CheckpointRecord(
        epoch=int(sidecar["epoch"]), optimizer_step=int(sidecar["optimizer_step"]),
        model_state=state, model_hash=unit["checkpoint_id"],
        R_src=float(sidecar["R_src"]), balanced_err=float(sidecar["balanced_err"]),
        train_surrogate=float(sidecar["train_surrogate"]), lam=float(sidecar["lambda"]),
    )
    stage = ERMStage(
        checkpoint=record, R_ERM_hat=float(sidecar["R_src"]),
        tau=float(sidecar["R_src"]) + float(manifest.risk.epsilon),
        task_plan_hash=sidecar["stage1_task_plan_hash"],
        stage1_invocation_id=f"C78_read_only_level_{level}_{unit['checkpoint_id']}",
    )
    return stage, {
        "C78_unit_id": unit["unit_id"], "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_path": unit["checkpoint_path"],
        "checkpoint_file_sha256": committed["checkpoint_path_sha256"],
        "sidecar_path": unit["sidecar_path"], "sidecar_sha256": committed["sidecar_sha256"],
        "read_only": True, "ERM_retrained": False, "OACI_weight_access": False,
    }


def _sidecar(
    *, root: Path, unit: dict[str, Any], record, checkpoint: dict[str, Any],
    optimizer: dict[str, Any], parent: dict[str, Any], previous_hash: str,
    manifest_path: Path, manifest, source_evidence, run_key, support,
    population, plans, execution_cfg, model_spec, training_seconds: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": "c78r_SRC_checkpoint_sidecar_v1",
        "unit_id": unit["unit_id"], "dataset": c78r.DATASET,
        "target": c78r.TARGET, "source_subjects": list(source_evidence.subjects),
        "seed": c78r.SEED, "regime": "SRC", "smooth_temperature": c78r.SMOOTH_TEMPERATURE,
        "level": int(unit["level"]), "epoch": int(unit["epoch"]),
        "trajectory_order": int(unit["trajectory_order"]),
        "optimizer_step": int(record.optimizer_step),
        "checkpoint_id": checkpoint["model_hash"], "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_tensor_schema": checkpoint["tensors"],
        **optimizer,
        "parent_ERM_checkpoint_id": parent["checkpoint_id"],
        "parent_ERM_checkpoint_file_sha256": parent["checkpoint_file_sha256"],
        "parent_ERM_sidecar_sha256": parent["sidecar_sha256"],
        "parent_ERM_read_only": True, "ERM_retrained": False, "OACI_weight_access": False,
        "previous_SRC_trajectory_checkpoint_id": previous_hash,
        "genealogy_rule": "read_only_C78_ERM_parent_plus_previous_SRC_order",
        "historical_SRC_commit": c78r.SRC_COMMIT,
        "SRC_objective_file_sha256": c78r.sha256_file("oaci/methods/source_robust.py"),
        "manifest_path": str(manifest_path), "manifest_file_sha256": c78r.sha256_file(manifest_path),
        "manifest_canonical_sha256": manifest.freeze()["sha256"],
        "run_key_hash": run_key.run_key_hash,
        "optimization_identity_hash": run_key.optimization_identity_hash,
        "execution_config_hash": execution_cfg.execution_config_hash,
        "model_spec_hash": model_spec.model_spec_hash,
        "support_hash": support.support_hash, "level_support_hash": support.level_support_hash,
        "population_hash": population.population_hash, "tensor_hash": population.tensor_hash,
        "stage1_task_plan_hash": plans.stage1_task.plan_hash,
        "stage2_task_plan_hash": plans.stage2_task.plan_hash,
        "full_domain_alignment_plan_hash": plans.full_domain_alignment.plan_hash,
        "source_loader_evidence_hash": source_evidence.evidence_hash,
        "source_raw_fingerprint": source_evidence.raw_data_fingerprint,
        "source_preprocessing_signature": source_evidence.resolved_preprocess_hash,
        "R_src": float(record.R_src), "balanced_err": float(record.balanced_err),
        "train_surrogate": float(record.train_surrogate), "lambda": float(record.lam),
        "training_seconds_level": float(training_seconds),
        "target_fit_ids_empty": True, "target_labels_available_to_optimizer": False,
        "target_rows_loaded_in_training_process": 0,
        "source_audit_rows_loaded_in_training_process": 0,
        "checkpoint_retention_rule": unit["retention_rule"],
        "target_outcome_used_for_retention": False,
        "target_outcome_used_for_retry": False,
        "selector_artifact": False,
    }
    return common.write_manifest(root / "sidecars" / f"{unit['unit_id']}.json", payload)


def train_src_canary(*, authorization_token: str, datalake_root: str) -> dict[str, Any]:
    lock, protocol, protocol_sha = common.require_authorization(authorization_token)
    git_boundary = common.verify_git_boundary()
    root = common.campaign_root(lock)
    frozen_path = common.field_frozen_path(lock)
    if frozen_path.exists():
        return common.require_field_frozen(lock)

    import torch
    from oaci.confirmatory.loso_plan import loso_fold_spec
    from oaci.data.eeg.bnci import load_moabb_confirmatory
    from oaci.methods.source_robust import SRCObjective
    from oaci.protocol.manifest_v2 import optimization_manifest_hash
    from oaci.runner.keys import FoldKey, RunKey
    from oaci.runner.maps import build_frozen_maps
    from oaci.runner.plans import build_level_plans
    from oaci.runner.scope import ScopePlanConfig, build_level_population
    from oaci.runner.support import DeletionCell, build_level_support, level0_reference_prior, make_deletion_schedule
    from oaci.train.engine import train_stage2

    attempt_path = root / "execution" / "training_attempts.jsonl"
    started_wall = time.time()
    started_cpu = time.process_time()
    common.append_jsonl(attempt_path, {
        "event": "start", "time": c78r.utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "authorization_token_sha256": common.sha256_text(authorization_token),
        "git": git_boundary, "target_outcomes_read": 0,
        "ERM_retrained": 0, "OACI_access": 0,
    })
    device = torch.device("cuda:0")
    gpu_preflight = c78_train._gpu_preflight(device, root)
    torch.cuda.reset_peak_memory_stats(device)

    manifest_path, manifest, split = c78_train._materialized_manifest(root)
    expected_train = sorted(int(item) for item in split["source_train_subjects"])
    if expected_train != [1, 2, 3, 7, 8, 9] or c78r.TARGET in expected_train:
        raise RuntimeError("C78R source-only subject contract failed")
    dataset = manifest.enabled_datasets()[c78r.DATASET]
    load_result = load_moabb_confirmatory(
        c78r.DATASET, expected_train, dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    if sorted(load_result.evidence.subjects) != expected_train:
        raise RuntimeError("C78R training loader touched a non-source-train subject")
    source = c78_train._source_training_data(load_result.bundle)
    source_domains = sorted(set(source.domain_id))
    evaluation_domains = [
        f"{c78r.DATASET}|subject-{subject:03d}"
        for subject in [*split["source_audit_subjects"], c78r.TARGET]
    ]
    maps = build_frozen_maps(source.class_names, source_domains, evaluation_domains)
    deletion = DeletionCell(split["deleted_cell"]["domain_id"], split["deleted_cell"]["class_name"])
    schedule = make_deletion_schedule([deletion], source, maps)
    reference = level0_reference_prior(source, maps)
    fold_key = FoldKey(
        manifest.freeze()["sha256"], c78r.DATASET,
        f"{c78r.DATASET}|target-subject-{c78r.TARGET:03d}",
        int(manifest.seeds.split), int(manifest.seeds.deletion), optimization_manifest_hash(manifest),
    )
    fold_scope = c78_train.TrainingFoldScope(fold_key=fold_key, maps=maps, level0_reference_prior=reference)
    scope_cfg = ScopePlanConfig.from_manifest(manifest, support_m=int(dataset.support_m))
    model_spec, execution_cfg, model_factory, geometry = c78_train._model_contract(manifest)
    prospective = common.unit_rows()
    by_key = {(int(row["level"]), int(row["epoch"])): row for row in prospective}
    capture = SRCOptimizerCapture(
        root / "optimizer_states",
        checkpoint_steps=int(manifest.training.checkpoint_every) * int(manifest.training.stage2_steps_per_epoch),
    )
    frozen_units: list[dict[str, Any]] = []
    level_manifests = []
    anchor_access = []
    with capture.patch_engine():
        for level in c78r.LEVELS:
            level_started = time.time()
            support = build_level_support(source, maps, level, schedule, reference, support_m=int(dataset.support_m))
            population = build_level_population(source, maps, support)
            plans = build_level_plans(fold_scope, level, support, population, scope_cfg, model_seed=c78r.SEED)
            if plans.full_domain_alignment is None:
                raise RuntimeError(f"C78R full-domain alignment inactive at level {level}")
            run_key = RunKey(fold_key, level, c78r.SEED)
            engine_cfg = execution_cfg.engine_config_for(run_key)
            erm_stage, anchor = _c78_anchor(
                level, support=support, population=population, plans=plans, run_key=run_key,
                execution_cfg=execution_cfg, model_spec=model_spec, manifest=manifest,
                source_evidence=load_result.evidence,
            )
            if anchor["checkpoint_id"] not in {item["checkpoint_id"] for item in protocol["frozen_erm_initialization"]["anchors"]}:
                raise RuntimeError("C78R runtime ERM anchor not in protocol")
            anchor_access.append({"level": level, **anchor})
            objective = SRCObjective(
                len(source.class_names), len(source_domains),
                smooth_temperature=c78r.SMOOTH_TEMPERATURE,
            )
            if not objective.active_status().active:
                raise RuntimeError(f"C78R SRC unexpectedly inactive at level {level}")
            capture.begin(level)
            trained = train_stage2(
                model_factory, erm_stage, population.training_data,
                objective, plans.stage2_task, plans.full_domain_alignment,
                engine_cfg, device,
            )
            if len(trained.trajectory) != 40 or tuple(record.epoch for record in trained.trajectory) != c78r.SRC_EPOCHS:
                raise RuntimeError(f"C78R SRC checkpoint cadence drift at level {level}")
            if trained.initial_model_hash != anchor["checkpoint_id"]:
                raise RuntimeError("C78R SRC did not initialize from frozen C78 ERM")
            level_seconds = time.time() - level_started
            previous = anchor["checkpoint_id"]
            level_units = []
            trace_rows = []
            for order, record in enumerate(trained.trajectory, start=1):
                unit = by_key[(level, int(record.epoch))]
                checkpoint = c78_train._save_checkpoint(root, record.model_hash, record.model_state)
                sidecar = _sidecar(
                    root=root, unit=unit, record=record, checkpoint=checkpoint,
                    optimizer=capture.descriptor(level, order), parent=anchor,
                    previous_hash=previous, manifest_path=manifest_path, manifest=manifest,
                    source_evidence=load_result.evidence, run_key=run_key, support=support,
                    population=population, plans=plans, execution_cfg=execution_cfg,
                    model_spec=model_spec, training_seconds=level_seconds,
                )
                sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
                level_units.append({
                    "unit_id": unit["unit_id"], "level": level, "regime": "SRC",
                    "epoch": int(record.epoch), "trajectory_order": order,
                    "checkpoint_id": record.model_hash,
                    "checkpoint_path": checkpoint["path"],
                    "checkpoint_file_sha256": checkpoint["file_sha256"],
                    "sidecar_path": str(sidecar_path), "sidecar_sha256": c78r.sha256_file(sidecar_path),
                    "optimizer_state_hash": sidecar["optimizer_state_hash"],
                })
                trace_rows.append({
                    "epoch": record.epoch, "trajectory_order": order,
                    "checkpoint_id": record.model_hash, "R_src": record.R_src,
                    "balanced_err": record.balanced_err,
                    "train_surrogate": record.train_surrogate, "lambda": record.lam,
                })
                previous = record.model_hash
            trace = c74_cache.write_content_addressed_npz(
                root / "trajectory_traces" / f"level-{level}", "SRC_trajectory_trace",
                {
                    "epoch": np.asarray([row["epoch"] for row in trace_rows], dtype=np.int16),
                    "trajectory_order": np.asarray([row["trajectory_order"] for row in trace_rows], dtype=np.int16),
                    "checkpoint_id": np.asarray([row["checkpoint_id"] for row in trace_rows], dtype="<U64"),
                    "R_src": np.asarray([row["R_src"] for row in trace_rows], dtype=np.float64),
                    "balanced_err": np.asarray([row["balanced_err"] for row in trace_rows], dtype=np.float64),
                    "train_surrogate": np.asarray([row["train_surrogate"] for row in trace_rows], dtype=np.float64),
                    "lambda": np.asarray([row["lambda"] for row in trace_rows], dtype=np.float64),
                },
            )
            level_manifest = common.write_manifest(root / "levels" / f"level-{level}.json", {
                "schema_version": "c78r_SRC_level_manifest_v1",
                "level": level, "units": level_units,
                "read_only_C78_ERM_parent": anchor,
                "trajectory_trace": trace, "checkpoint_count": len(level_units),
                "level_wall_seconds": level_seconds,
                "source_subjects_loaded": expected_train,
                "target_subject_loaded": False, "source_audit_subjects_loaded": [],
                "support_hash": support.support_hash,
                "smooth_temperature": c78r.SMOOTH_TEMPERATURE,
            })
            level_manifests.append(level_manifest)
            frozen_units.extend(level_units)

    if len(frozen_units) != c78r.EXPECTED_UNITS or {row["unit_id"] for row in frozen_units} != {row["unit_id"] for row in prospective}:
        raise RuntimeError("C78R frozen field differs from prospective 80-unit manifest")
    wall_seconds = time.time() - started_wall
    process_cpu_seconds = time.process_time() - started_cpu
    external_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    field = common.write_manifest(frozen_path, {
        "schema_version": "c78r_SRC_field_frozen_v1",
        "protocol_sha256": protocol_sha,
        "implementation_identity_sha256": lock["implementation_identity_sha256"],
        "authorization_token_sha256": common.sha256_text(authorization_token),
        "git": git_boundary, "GPU_preflight": gpu_preflight,
        "units": frozen_units, "unit_count": len(frozen_units), "SRC_count": len(frozen_units),
        "ERM_retrained_count": 0, "OACI_weight_access_count": 0,
        "read_only_C78_ERM_anchor_access": anchor_access,
        "all_80_retention_decisions_frozen": True,
        "retention_uses_target_outcomes": False,
        "retry_selection_uses_target_outcomes": False,
        "execution": {
            "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
            "GPU_name": torch.cuda.get_device_name(device), "GPU_count": 1,
            "GPU_wall_hours": wall_seconds / 3600, "wall_seconds": wall_seconds,
            "process_CPU_seconds": process_cpu_seconds,
            "peak_GPU_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "external_storage_bytes_at_freeze": external_bytes,
            "retry_or_requeue_count": int(os.environ.get("SLURM_RESTART_COUNT", "0")),
            "source_training_subjects": expected_train,
            "source_audit_subjects_loaded_during_training": [],
            "target_subject_loaded_during_training": False,
            "target_data_rows_loaded_during_training": 0,
            "target_label_reads_during_training": 0,
            "seed4_access": 0, "BNCI2014_004_access": 0,
        },
        "materialized_manifest": {
            "path": str(manifest_path), "sha256": c78r.sha256_file(manifest_path),
            "canonical_sha256": manifest.freeze()["sha256"],
        },
        "source_loader": {
            "subjects": expected_train, "rows": int(load_result.bundle.n),
            "evidence_hash": load_result.evidence.evidence_hash,
            "raw_fingerprint": load_result.evidence.raw_data_fingerprint,
            "network_attempt_count": load_result.evidence.network_attempt_count,
        },
        "model_geometry": geometry,
        "level_manifests": [
            {
                "level": item["level"],
                "path": str(root / "levels" / f"level-{item['level']}.json"),
                "sha256": c78r.sha256_file(root / "levels" / f"level-{item['level']}.json"),
            }
            for item in level_manifests
        ],
    })
    common.append_jsonl(attempt_path, {
        "event": "complete", "time": c78r.utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "field_manifest_sha256": field["manifest_sha256"],
        "unit_count": c78r.EXPECTED_UNITS, "target_outcomes_read": 0,
        "ERM_retrained": 0, "OACI_access": 0,
    })
    print(json.dumps({
        "gate": "SRC_FIELD_FROZEN", "units": c78r.EXPECTED_UNITS,
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "wall_seconds": wall_seconds,
    }, sort_keys=True))
    return field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78r_train")
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    args = parser.parse_args(argv)
    train_src_canary(authorization_token=args.authorization_token, datalake_root=args.datalake_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
