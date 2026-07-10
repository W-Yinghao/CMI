"""Authorized C78 source-only training worker.

Authorization and implementation-lock checks run before importing the EEG loader,
CUDA helper, or training engine.  The real-data process loads only the six
source-training subjects.  Target and source-audit data are not opened until the
82-checkpoint field has been frozen by this worker.
"""
from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from . import c74_cache
from . import c78_authorized_common as common
from . import c78_seed3_instrumented_pilot as c78


DEFAULT_DATALAKE_ROOT = "/projects/EEG-foundation-model/datalake/raw"


def _nested_update(digest: Any, value: Any) -> None:
    import torch

    if torch.is_tensor(value):
        tensor = value.detach().cpu().contiguous()
        digest.update(b"tensor")
        digest.update(str(tensor.dtype).encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(tensor.numpy().tobytes())
    elif isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        digest.update(b"ndarray")
        digest.update(str(array.dtype).encode())
        digest.update(str(tuple(array.shape)).encode())
        digest.update(array.tobytes())
    elif isinstance(value, dict):
        digest.update(b"dict")
        for key in sorted(value, key=lambda item: (type(item).__name__, repr(item))):
            _nested_update(digest, key)
            _nested_update(digest, value[key])
    elif isinstance(value, (list, tuple)):
        digest.update(type(value).__name__.encode())
        for item in value:
            _nested_update(digest, item)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        digest.update(type(value).__name__.encode())
        digest.update(repr(value).encode())
    else:
        raise TypeError(f"unsupported optimizer-state value: {type(value)!r}")


def optimizer_state_hash(value: Any) -> str:
    digest = hashlib.sha256()
    _nested_update(digest, value)
    return digest.hexdigest()


def _cpu_clone(value: Any) -> Any:
    import torch

    if torch.is_tensor(value):
        return value.detach().cpu().clone()
    if isinstance(value, dict):
        return {key: _cpu_clone(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_cpu_clone(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_cpu_clone(item) for item in value)
    return value


class OptimizerCapture:
    """Passive optimizer snapshotter without modifying the historical engine."""

    def __init__(self, directory: Path, *, stage1_steps: int, stage2_checkpoint_steps: int):
        self.directory = directory
        self.stage1_steps = int(stage1_steps)
        self.stage2_checkpoint_steps = int(stage2_checkpoint_steps)
        self.phase = "unconfigured"
        self.level = -1
        self.optimizers: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.snapshots: dict[tuple[str, int, int], dict[str, Any]] = {}
        self._original_make_optimizer = None

    def begin(self, phase: str, level: int) -> None:
        if phase not in {"stage1", "stage2"}:
            raise ValueError(phase)
        self.phase = phase
        self.level = int(level)
        self.optimizers = {}
        self.counts = {}

    def _label(self) -> str:
        if self.phase == "stage1":
            if self.optimizers:
                raise RuntimeError("stage1 created more than one optimizer")
            return "stage1_encoder"
        if "stage2_critic" not in self.optimizers:
            return "stage2_critic"
        if "stage2_encoder" not in self.optimizers:
            return "stage2_encoder"
        raise RuntimeError("stage2 created more than two optimizers")

    def make_optimizer(self, params, lr, cfg):
        optimizer = self._original_make_optimizer(params, lr, cfg)
        label = self._label()
        self.optimizers[label] = optimizer
        self.counts[label] = 0
        original_step = optimizer.step

        def captured_step(*args, **kwargs):
            result = original_step(*args, **kwargs)
            self.counts[label] += 1
            count = self.counts[label]
            if label == "stage1_encoder" and count == self.stage1_steps:
                self._snapshot(order=0)
            elif label == "stage2_encoder" and count % self.stage2_checkpoint_steps == 0:
                self._snapshot(order=count // self.stage2_checkpoint_steps)
            return result

        optimizer.step = captured_step
        return optimizer

    def _snapshot(self, order: int) -> None:
        import torch

        payload = {
            "phase": self.phase,
            "level": self.level,
            "trajectory_order": int(order),
            "step_counts": dict(self.counts),
            "optimizers": {name: _cpu_clone(opt.state_dict()) for name, opt in self.optimizers.items()},
        }
        identity = optimizer_state_hash(payload)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / (
            f"level-{self.level}_{self.phase}_order-{int(order):02d}_optimizer_{identity[:16]}.pt"
        )
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
        if optimizer_state_hash(replay) != identity:
            raise RuntimeError("optimizer snapshot identity changed across round trip")
        self.snapshots[(self.phase, self.level, int(order))] = {
            "optimizer_state_hash": identity,
            "optimizer_state_path": str(path),
            "optimizer_state_file_sha256": c78.sha256_file(path),
            "optimizer_step_counts": dict(self.counts),
        }

    def descriptor(self, phase: str, level: int, order: int) -> dict[str, Any]:
        key = (phase, int(level), int(order))
        if key not in self.snapshots:
            raise RuntimeError(f"missing optimizer snapshot {key}")
        return dict(self.snapshots[key])

    @contextlib.contextmanager
    def patch_engine(self):
        import oaci.train.engine as engine

        if self._original_make_optimizer is not None:
            raise RuntimeError("optimizer capture patch is already active")
        self._original_make_optimizer = engine.make_optimizer
        engine.make_optimizer = self.make_optimizer
        try:
            yield
        finally:
            engine.make_optimizer = self._original_make_optimizer
            self._original_make_optimizer = None


@dataclass(frozen=True)
class SourceTrainingData:
    X: Any
    y: np.ndarray
    sample_id: tuple[str, ...]
    domain_id: tuple[str, ...]
    group_id: tuple[str, ...]
    support_unit_id: tuple[str, ...]
    mass_unit_id: tuple[str, ...]
    sample_mass: np.ndarray
    class_names: tuple[str, ...]
    source_train_idx: np.ndarray
    source_train_population_hash: str


@dataclass(frozen=True)
class TrainingFoldScope:
    fold_key: Any
    maps: Any
    level0_reference_prior: np.ndarray


def _source_training_data(bundle) -> SourceTrainingData:
    import torch

    count = int(bundle.n)
    indices = np.arange(count, dtype=np.int64)
    indices.setflags(write=False)
    mass = np.ones(count, dtype=np.float64)
    mass.setflags(write=False)
    sample_ids = tuple(str(item) for item in bundle.sample_id.tolist())
    domains = tuple(str(item) for item in bundle.subject_id.tolist())
    groups = tuple(str(item) for item in bundle.recording_id.tolist())
    trials = tuple(str(item) for item in bundle.trial_id.tolist())
    population_hash = hashlib.sha256(
        json.dumps({
            "role": "source_train_only", "sample_ids": sample_ids,
            "labels": [int(item) for item in bundle.y.tolist()],
            "domains": domains, "groups": groups,
        }, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return SourceTrainingData(
        X=torch.from_numpy(np.ascontiguousarray(bundle.X)),
        y=np.asarray(bundle.y, dtype=np.int64), sample_id=sample_ids,
        domain_id=domains, group_id=groups, support_unit_id=trials,
        mass_unit_id=trials, sample_mass=mass,
        class_names=tuple(str(item) for item in bundle.class_names),
        source_train_idx=indices, source_train_population_hash=population_hash,
    )


def _materialized_manifest(root: Path):
    from oaci.confirmatory.loso_plan import explicit_split, loso_fold_spec
    from oaci.confirmatory.materialize import materialize_pilot_manifest
    from oaci.confirmatory.schema import load_confirmatory
    from oaci.protocol.manifest_v2 import load_v2

    config_dir = root / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    path = config_dir / "C78_target004_seed3_fullbudget_manifest.yaml"
    spec = loso_fold_spec(c78.TARGET, dataset_id=c78.DATASET)
    if path.exists():
        manifest = load_v2(str(path))
        manifest.validate_complete()
        return path, manifest, spec
    protocol = load_confirmatory("oaci/protocol/confirmatory_v2.yaml")
    temporary = config_dir / ".C78_manifest.provisional.yaml"
    materialize_pilot_manifest(
        protocol, c78.DATASET, target_subject=c78.TARGET,
        out_path=str(temporary), model_seeds=[c78.SEED],
        explicit_split=explicit_split(spec), deleted_cell=dict(spec["deleted_cell"]),
        bootstrap_override=None,
    )
    os.replace(temporary, path)
    manifest = load_v2(str(path))
    manifest.validate_complete()
    return path, manifest, spec


def _model_contract(manifest):
    from oaci.models import build_model
    from oaci.models.shallow import validate_shallow_geometry
    from oaci.runner.config import ModelSpec, RunnerExecutionConfig

    backbone = manifest.backbone
    geometry = validate_shallow_geometry(22, 385, backbone)
    model_spec = ModelSpec.build(
        "shallow_convnet",
        {
            "temporal_filters": int(backbone.temporal_filters),
            "temporal_kernel_samples": int(backbone.temporal_kernel_samples),
            "pool_kernel_samples": int(backbone.pool_kernel_samples),
            "pool_stride_samples": int(backbone.pool_stride_samples),
            "dropout": float(backbone.dropout),
            "safe_log_eps": float(backbone.safe_log_eps),
        },
        (22, 385), 4,
    )

    def factory():
        return build_model(
            "shallow_convnet", in_chans=22, in_times=385, n_classes=4,
            **dict(model_spec.backbone_config),
        )

    return model_spec, RunnerExecutionConfig.from_manifest(manifest), factory, geometry


def _gpu_preflight(device, root: Path) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    from oaci.models.shallow import ShallowConvNet
    from oaci.train.checkpoint import model_state_hash

    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("C78 authorized training requires an assigned CUDA GPU")
    torch.use_deterministic_algorithms(True, warn_only=False)

    def canary():
        torch.manual_seed(78003)
        torch.cuda.manual_seed_all(78003)
        model = ShallowConvNet(
            22, 385, 4, temporal_filters=40, temporal_kernel_samples=25,
            pool_kernel_samples=75, pool_stride_samples=15, dropout=0.5,
            safe_log_eps=1e-6,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
        x = torch.linspace(-1, 1, 4 * 22 * 385, device=device).reshape(4, 22, 385)
        y = torch.tensor([0, 1, 2, 3], device=device)
        initial = model_state_hash(model)
        model.train()
        optimizer.zero_grad()
        loss = functional.cross_entropy(model(x).logits, y)
        loss.backward()
        optimizer.step()
        return initial, float(loss.detach().cpu()), model_state_hash(model)

    first = canary()
    second = canary()
    passed = first == second
    if not passed:
        raise RuntimeError(f"C78 GPU deterministic canary failed: {first} != {second}")
    payload = {
        "schema_version": "c78_gpu_preflight_v1",
        "device": str(device), "GPU_name": torch.cuda.get_device_name(device),
        "CUDA_version": torch.version.cuda, "torch": torch.__version__,
        "deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "initial_hash": first[0], "loss": first[1], "final_hash": first[2],
        "repeat_exact": passed, "real_EEG_rows_loaded": 0,
        "target_label_reads": 0, "training_canary": "synthetic_one_step_only",
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    }
    return common.write_manifest(root / "gates" / "GPU_PREFLIGHT.json", payload)


def _save_checkpoint(root: Path, checkpoint_id: str, state: dict[str, Any]) -> dict[str, Any]:
    from oaci.artifacts.checkpoint import read_checkpoint_file, write_checkpoint_file

    directory = root / "checkpoints"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"checkpoint_{checkpoint_id}.pt"
    if path.exists():
        # A sidecar is needed for the transport hash; reconstruct a minimal descriptor.
        descriptor = {
            "model_hash": checkpoint_id,
            "file_sha256": c78.sha256_file(path),
            "tensors": {key: {"dtype": str(value.dtype), "shape": list(value.shape)} for key, value in state.items()},
        }
        read_checkpoint_file(path, descriptor)
        return {**descriptor, "path": str(path)}
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=directory)
    os.close(fd)
    try:
        descriptor = write_checkpoint_file(temporary, checkpoint_id, state)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)
    return {**descriptor, "path": str(path)}


def _sidecar(
    *, root: Path, unit: dict[str, str], record, checkpoint: dict[str, Any],
    optimizer: dict[str, Any], parent_hash: str, previous_hash: str | None,
    manifest_path: Path, manifest, source_evidence, run_key, support_state,
    level_population, plans, execution_cfg, model_spec, training_seconds: float,
) -> dict[str, Any]:
    payload = {
        "schema_version": "c78_checkpoint_sidecar_v1",
        "unit_id": unit["unit_id"], "dataset": c78.DATASET,
        "target": c78.TARGET, "source_subjects": list(source_evidence.subjects),
        "seed": c78.SEED, "regime": unit["regime"],
        "level": int(unit["level"]), "epoch": int(unit["epoch"]),
        "trajectory_order": int(unit["trajectory_order"]),
        "optimizer_step": int(record.optimizer_step),
        "checkpoint_id": checkpoint["model_hash"],
        "checkpoint_path": checkpoint["path"],
        "checkpoint_file_sha256": checkpoint["file_sha256"],
        "checkpoint_tensor_schema": checkpoint["tensors"],
        "optimizer_state_hash": optimizer["optimizer_state_hash"],
        "optimizer_state_path": optimizer["optimizer_state_path"],
        "optimizer_state_file_sha256": optimizer["optimizer_state_file_sha256"],
        "optimizer_step_counts": optimizer["optimizer_step_counts"],
        "parent_ERM_checkpoint_id": parent_hash,
        "previous_trajectory_checkpoint_id": previous_hash,
        "genealogy_rule": "shared_level_ERM_parent_plus_previous_OACI_order",
        "historical_objective": unit["regime"],
        "manifest_path": str(manifest_path),
        "manifest_file_sha256": c78.sha256_file(manifest_path),
        "manifest_canonical_sha256": manifest.freeze()["sha256"],
        "run_key_hash": run_key.run_key_hash,
        "optimization_identity_hash": run_key.optimization_identity_hash,
        "execution_config_hash": execution_cfg.execution_config_hash,
        "model_spec_hash": model_spec.model_spec_hash,
        "support_hash": support_state.support_hash,
        "level_support_hash": support_state.level_support_hash,
        "population_hash": level_population.population_hash,
        "tensor_hash": level_population.tensor_hash,
        "stage1_task_plan_hash": plans.stage1_task.plan_hash,
        "stage2_task_plan_hash": plans.stage2_task.plan_hash,
        "OACI_alignment_plan_hash": plans.oaci_alignment.plan_hash if plans.oaci_alignment else None,
        "source_loader_evidence_hash": source_evidence.evidence_hash,
        "source_raw_fingerprint": source_evidence.raw_data_fingerprint,
        "source_preprocessing_signature": source_evidence.resolved_preprocess_hash,
        "R_src": float(record.R_src), "balanced_err": float(record.balanced_err),
        "train_surrogate": float(record.train_surrogate), "lambda": float(record.lam),
        "training_seconds_level": float(training_seconds),
        "target_fit_ids_empty": True, "target_labels_available_to_optimizer": False,
        "target_rows_loaded_in_training_process": 0,
        "checkpoint_retention_rule": unit["retention_rule"],
        "target_outcome_used_for_retention": False,
        "selector_artifact": False,
    }
    sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
    return common.write_manifest(sidecar_path, payload)


def train_field(*, authorization_token: str, datalake_root: str) -> dict[str, Any]:
    lock, protocol, protocol_sha = common.require_authorization(authorization_token)
    git_boundary = common.verify_git_execution_boundary()
    root = common.campaign_root(lock)
    frozen_path = common.field_frozen_path(lock)
    if frozen_path.exists():
        return common.require_field_frozen(lock)

    import torch
    from oaci.confirmatory.loso_plan import loso_fold_spec
    from oaci.data.eeg.bnci import load_moabb_confirmatory
    from oaci.protocol.manifest_v2 import optimization_manifest_hash
    from oaci.runner.keys import FoldKey, RunKey
    from oaci.runner.maps import build_frozen_maps
    from oaci.runner.objectives import make_objective
    from oaci.runner.plans import build_level_plans
    from oaci.runner.scope import ScopePlanConfig, build_level_population
    from oaci.runner.stage1 import run_stage1_once
    from oaci.runner.support import (
        DeletionCell, build_level_support, level0_reference_prior,
        make_deletion_schedule,
    )
    from oaci.train.engine import InvocationRegistry, train_stage2

    attempt_path = root / "execution" / "training_attempts.jsonl"
    started_wall = time.time()
    started_cpu = time.process_time()
    common.append_jsonl(attempt_path, {
        "event": "start", "time": c78.utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "git": git_boundary, "authorization_token_sha256": common.sha256_text(authorization_token),
        "target_outcomes_read": 0,
    })
    device = torch.device("cuda:0")
    gpu_preflight = _gpu_preflight(device, root)
    torch.cuda.reset_peak_memory_stats(device)

    manifest_path, manifest, split = _materialized_manifest(root)
    expected_train = sorted(int(item) for item in split["source_train_subjects"])
    if c78.TARGET in expected_train or expected_train != [1, 2, 3, 7, 8, 9]:
        raise RuntimeError(f"C78 source-only training subject contract failed: {expected_train}")
    dataset = manifest.enabled_datasets()[c78.DATASET]
    preprocessing = dataset.preprocessing
    load_result = load_moabb_confirmatory(
        c78.DATASET, expected_train, preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    if list(load_result.evidence.subjects) != expected_train:
        raise RuntimeError("C78 loader touched a subject outside source_train")
    source = _source_training_data(load_result.bundle)
    source_domains = sorted(set(source.domain_id))
    evaluation_domains = [
        f"{c78.DATASET}|subject-{subject:03d}"
        for subject in [*split["source_audit_subjects"], c78.TARGET]
    ]
    maps = build_frozen_maps(source.class_names, source_domains, evaluation_domains)
    deletion = DeletionCell(split["deleted_cell"]["domain_id"], split["deleted_cell"]["class_name"])
    schedule = make_deletion_schedule([deletion], source, maps)
    reference = level0_reference_prior(source, maps)
    fold_key = FoldKey(
        manifest.freeze()["sha256"], c78.DATASET,
        f"{c78.DATASET}|target-subject-{c78.TARGET:03d}",
        int(manifest.seeds.split), int(manifest.seeds.deletion),
        optimization_manifest_hash(manifest),
    )
    fold_scope = TrainingFoldScope(fold_key=fold_key, maps=maps, level0_reference_prior=reference)
    scope_cfg = ScopePlanConfig.from_manifest(manifest, support_m=int(dataset.support_m))
    model_spec, execution_cfg, model_factory, geometry = _model_contract(manifest)
    prospective = common.unit_rows()
    by_key = {
        (int(row["level"]), row["regime"], int(row["epoch"])): row
        for row in prospective
    }
    capture = OptimizerCapture(
        root / "optimizer_states",
        stage1_steps=int(manifest.training.stage1_epochs) * int(manifest.training.stage1_steps_per_epoch),
        stage2_checkpoint_steps=int(manifest.training.checkpoint_every) * int(manifest.training.stage2_steps_per_epoch),
    )
    frozen_units: list[dict[str, Any]] = []
    level_manifests = []
    with capture.patch_engine():
        for level in c78.LEVELS:
            level_started = time.time()
            support = build_level_support(source, maps, level, schedule, reference, support_m=int(dataset.support_m))
            population = build_level_population(source, maps, support)
            plans = build_level_plans(fold_scope, level, support, population, scope_cfg, model_seed=c78.SEED)
            run_key = RunKey(fold_key, level, c78.SEED)
            engine_cfg = execution_cfg.engine_config_for(run_key)
            registry = InvocationRegistry()

            capture.begin("stage1", level)
            stage1 = run_stage1_once(
                run_key, population, plans, model_factory, model_spec, engine_cfg,
                execution_cfg.execution_config_hash, registry, device,
            )
            erm = stage1.erm_stage.checkpoint
            capture.begin("stage2", level)
            objective, objective_spec = make_objective("OACI", support, fold_scope, execution_cfg)
            if not objective.active_status().active or plans.oaci_alignment is None:
                raise RuntimeError(f"C78 OACI unexpectedly inactive at level {level}")
            trained = train_stage2(
                model_factory, stage1.erm_stage, population.training_data,
                objective, plans.stage2_task, plans.oaci_alignment, engine_cfg, device,
            )
            if len(trained.trajectory) != 40:
                raise RuntimeError(f"C78 level {level} trajectory count {len(trained.trajectory)} != 40")
            if [record.epoch for record in trained.trajectory] != list(c78.OACI_EPOCHS):
                raise RuntimeError(f"C78 level {level} checkpoint cadence drift")
            level_seconds = time.time() - level_started

            level_units = []
            erm_unit = by_key[(level, "ERM", 199)]
            erm_checkpoint = _save_checkpoint(root, erm.model_hash, erm.model_state)
            erm_sidecar = _sidecar(
                root=root, unit=erm_unit, record=erm, checkpoint=erm_checkpoint,
                optimizer=capture.descriptor("stage1", level, 0),
                parent_hash=erm.model_hash, previous_hash=None,
                manifest_path=manifest_path, manifest=manifest,
                source_evidence=load_result.evidence, run_key=run_key,
                support_state=support, level_population=population, plans=plans,
                execution_cfg=execution_cfg, model_spec=model_spec,
                training_seconds=level_seconds,
            )
            level_units.append({
                **{key: erm_sidecar[key] for key in (
                    "unit_id", "level", "regime", "epoch", "trajectory_order",
                    "checkpoint_id", "checkpoint_path", "checkpoint_file_sha256",
                )},
                "sidecar_path": str(root / "sidecars" / f"{erm_unit['unit_id']}.json"),
                "sidecar_sha256": c78.sha256_file(root / "sidecars" / f"{erm_unit['unit_id']}.json"),
            })
            previous = erm.model_hash
            trace_rows = [{
                "regime": "ERM", "epoch": erm.epoch, "trajectory_order": 0,
                "checkpoint_id": erm.model_hash, "R_src": erm.R_src,
                "balanced_err": erm.balanced_err, "train_surrogate": erm.train_surrogate,
                "lambda": erm.lam,
            }]
            for order, record in enumerate(trained.trajectory, start=1):
                unit = by_key[(level, "OACI", int(record.epoch))]
                checkpoint = _save_checkpoint(root, record.model_hash, record.model_state)
                sidecar = _sidecar(
                    root=root, unit=unit, record=record, checkpoint=checkpoint,
                    optimizer=capture.descriptor("stage2", level, order),
                    parent_hash=erm.model_hash, previous_hash=previous,
                    manifest_path=manifest_path, manifest=manifest,
                    source_evidence=load_result.evidence, run_key=run_key,
                    support_state=support, level_population=population, plans=plans,
                    execution_cfg=execution_cfg, model_spec=model_spec,
                    training_seconds=level_seconds,
                )
                level_units.append({
                    **{key: sidecar[key] for key in (
                        "unit_id", "level", "regime", "epoch", "trajectory_order",
                        "checkpoint_id", "checkpoint_path", "checkpoint_file_sha256",
                    )},
                    "sidecar_path": str(root / "sidecars" / f"{unit['unit_id']}.json"),
                    "sidecar_sha256": c78.sha256_file(root / "sidecars" / f"{unit['unit_id']}.json"),
                })
                trace_rows.append({
                    "regime": "OACI", "epoch": record.epoch, "trajectory_order": order,
                    "checkpoint_id": record.model_hash, "R_src": record.R_src,
                    "balanced_err": record.balanced_err, "train_surrogate": record.train_surrogate,
                    "lambda": record.lam,
                })
                previous = record.model_hash

            trace_descriptor = c74_cache.write_content_addressed_npz(
                root / "trajectory_traces" / f"level-{level}", "source_trajectory_trace",
                {
                    "regime": np.asarray([row["regime"] for row in trace_rows], dtype="<U4"),
                    "epoch": np.asarray([row["epoch"] for row in trace_rows], dtype=np.int16),
                    "trajectory_order": np.asarray([row["trajectory_order"] for row in trace_rows], dtype=np.int16),
                    "checkpoint_id": np.asarray([row["checkpoint_id"] for row in trace_rows], dtype="<U64"),
                    "R_src": np.asarray([row["R_src"] for row in trace_rows], dtype=np.float64),
                    "balanced_err": np.asarray([row["balanced_err"] for row in trace_rows], dtype=np.float64),
                    "train_surrogate": np.asarray([row["train_surrogate"] for row in trace_rows], dtype=np.float64),
                    "lambda": np.asarray([row["lambda"] for row in trace_rows], dtype=np.float64),
                },
            )
            level_manifest = common.write_manifest(
                root / "levels" / f"level-{level}.json",
                {
                    "schema_version": "c78_training_level_manifest_v1",
                    "level": level, "units": level_units,
                    "ERM_parent": erm.model_hash,
                    "trajectory_trace": trace_descriptor,
                    "source_subjects_loaded": expected_train,
                    "target_subject_loaded": False,
                    "source_audit_subjects_loaded": [],
                    "checkpoint_count": len(level_units),
                    "level_wall_seconds": level_seconds,
                    "support_hash": support.support_hash,
                    "deleted_cells": [
                        {"domain_id": cell.domain_id, "class_name": cell.class_name}
                        for cell in support.deleted_cells
                    ],
                    "objective_spec_hash": objective_spec.objective_spec_hash,
                },
            )
            level_manifests.append(level_manifest)
            frozen_units.extend(level_units)

    if len(frozen_units) != 82 or {row["unit_id"] for row in frozen_units} != {row["unit_id"] for row in prospective}:
        raise RuntimeError("C78 trained field does not match the prospective 82-unit manifest")
    wall_seconds = time.time() - started_wall
    process_cpu_seconds = time.process_time() - started_cpu
    external_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    field = common.write_manifest(
        frozen_path,
        {
            "schema_version": "c78_authorized_field_frozen_v1",
            "protocol_sha256": protocol_sha,
            "implementation_identity_sha256": lock["implementation_identity_sha256"],
            "authorization_token_sha256": common.sha256_text(authorization_token),
            "git": git_boundary, "GPU_preflight": gpu_preflight,
            "units": frozen_units, "unit_count": len(frozen_units),
            "ERM_anchor_count": sum(row["regime"] == "ERM" for row in frozen_units),
            "OACI_trajectory_count": sum(row["regime"] == "OACI" for row in frozen_units),
            "SRC_count": 0,
            "all_82_retention_decisions_frozen": True,
            "retention_uses_target_outcomes": False,
            "execution": {
                "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
                "GPU_name": torch.cuda.get_device_name(device),
                "GPU_count": 1, "GPU_wall_hours": wall_seconds / 3600,
                "wall_seconds": wall_seconds, "process_CPU_seconds": process_cpu_seconds,
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
                "path": str(manifest_path), "sha256": c78.sha256_file(manifest_path),
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
                    "sha256": c78.sha256_file(root / "levels" / f"level-{item['level']}.json"),
                }
                for item in level_manifests
            ],
        },
    )
    common.append_jsonl(attempt_path, {
        "event": "complete", "time": c78.utc_now(),
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "field_manifest_sha256": field["manifest_sha256"],
        "unit_count": 82, "target_outcomes_read": 0,
    })
    print(json.dumps({
        "gate": "FIELD_FROZEN", "units": 82,
        "job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "wall_seconds": wall_seconds,
    }, sort_keys=True))
    return field


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78_authorized_train")
    parser.add_argument("--authorization-token", required=True)
    parser.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    args = parser.parse_args(argv)
    train_field(authorization_token=args.authorization_token, datalake_root=args.datalake_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
