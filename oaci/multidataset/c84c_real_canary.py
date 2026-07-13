"""Fail-closed C84C real-data engineering canary.

Module import is safe in C84R: EEG loaders and the ML stack are imported only
inside :func:`run_real`, after protocol, lock, authorization, Git, environment,
and external-output checks pass and the authorization is consumed.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from typing import Any, Mapping, Sequence

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as protocol
from .c84r_montage_repair import EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
CANARY_PROTOCOL_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V2.json"
CANARY_PROTOCOL_SHA_PATH = REPORT_DIR / "C84_CANARY_PROTOCOL_V2.sha256"
EXECUTION_LOCK_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK.json"
EXECUTION_LOCK_SHA_PATH = REPORT_DIR / "C84C_EXECUTION_LOCK.sha256"
AUTHORIZATION_RECORD_PATH = REPORT_DIR / "C84C_PI_AUTHORIZATION_RECORD.json"
DEFAULT_EXTERNAL_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84-canary-v2")
EXPECTED_CONDA_PREFIX = "/home/infres/yinwang/anaconda3/envs/icml"
EXPECTED_PYTHON = "3.9.25"
CANARY_TARGETS = protocol.CANARY_TARGETS
SOURCE_PANEL = "A"
TRAINING_SEED = 5
LEVEL = 0
CHECKPOINT_EPOCHS = tuple(range(4, 200, 5))
CLASS_NAMES = ("left_hand", "right_hand")
SCIENTIFIC_OUTPUT_KEYS = frozenset({
    "target_accuracy", "target_balanced_accuracy", "target_calibration",
    "target_regret", "target_label_count", "selector_score", "Q1", "Q2",
    "label_budget_frontier", "cross_dataset_result",
})


class C84CCanaryError(RuntimeError):
    """Raised before protected access when a C84C contract is not satisfied."""


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json_atomic(path: str | Path, payload: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(canonical_bytes(payload) + b"\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _sidecar_digest(path: Path) -> str:
    if not path.is_file():
        raise C84CCanaryError(f"missing SHA sidecar: {path}")
    fields = path.read_text(encoding="ascii").split()
    if not fields or len(fields[0]) != 64:
        raise C84CCanaryError(f"malformed SHA sidecar: {path}")
    return fields[0]


def _git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(["git", *args], cwd=REPO_ROOT, text=True, capture_output=True, check=check)


def _commit_for_path(path: Path) -> str:
    relative = str(path.relative_to(REPO_ROOT))
    value = _git("log", "-1", "--format=%H", "--", relative).stdout.strip()
    if len(value) != 40:
        raise C84CCanaryError(f"cannot resolve committed identity for {relative}")
    return value


def _all_relevant_worktrees_clean() -> bool:
    # The active canonical worktree is execution-relevant. Other registered
    # worktrees are audited by the readiness report and cannot alter this tree.
    return not _git("status", "--porcelain").stdout.strip()


def _verify_output_root(root: Path, lock_sha: str) -> Path:
    run_root = root / f"lock_{lock_sha[:20]}"
    if run_root.exists():
        children = list(run_root.iterdir())
        allowed = {"authorization_consumed.json", "execution_attempts.jsonl"}
        if any(child.name not in allowed for child in children):
            raise C84CCanaryError("C84C external output root is neither empty nor a clean authorization-only root")
    else:
        run_root.mkdir(parents=True, exist_ok=False)
    return run_root


def require_authorization_and_lock(
    *,
    authorization_path: Path = AUTHORIZATION_RECORD_PATH,
    output_root: Path = DEFAULT_EXTERNAL_ROOT,
) -> dict[str, Any]:
    """Validate every protected boundary before any loader/ML import."""
    if not CANARY_PROTOCOL_PATH.is_file() or not EXECUTION_LOCK_PATH.is_file():
        raise C84CCanaryError("C84C protocol or execution lock is absent")
    protocol_sha = _sidecar_digest(CANARY_PROTOCOL_SHA_PATH)
    lock_sha = _sidecar_digest(EXECUTION_LOCK_SHA_PATH)
    if sha256_file(CANARY_PROTOCOL_PATH) != protocol_sha:
        raise C84CCanaryError("C84C protocol hash replay failed")
    if sha256_file(EXECUTION_LOCK_PATH) != lock_sha:
        raise C84CCanaryError("C84C execution-lock hash replay failed")
    canary = read_json(CANARY_PROTOCOL_PATH)
    lock = read_json(EXECUTION_LOCK_PATH)
    if lock.get("status") != "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED":
        raise C84CCanaryError("C84C lock status is not the unique readiness state")
    if lock.get("canary_protocol", {}).get("sha256") != protocol_sha:
        raise C84CCanaryError("C84C lock does not bind the current canary protocol")
    if lock.get("interface", {}).get("montage_sha256") != MONTAGE_SHA256:
        raise C84CCanaryError("C84C lock montage identity differs")
    if canary.get("interface", {}).get("id") != INTERFACE_ID:
        raise C84CCanaryError("C84C canary protocol interface differs")
    if not authorization_path.is_file():
        raise C84CCanaryError("direct C84C PI authorization record is absent")
    authorization = read_json(authorization_path)
    lock_commit = _commit_for_path(EXECUTION_LOCK_PATH)
    required = {
        "schema_version": "c84c_direct_pi_authorization_record_v1",
        "direct_explicit_PI_authorization": True,
        "authorized_stage": "C84C",
        "canary_protocol_sha256": protocol_sha,
        "execution_lock_sha256": lock_sha,
        "execution_lock_commit": lock_commit,
        "C84F": False,
        "C84S": False,
        "scientific_metrics": False,
        "same_label_oracle": False,
    }
    mismatch = {key: (authorization.get(key), expected) for key, expected in required.items()
                if authorization.get(key) != expected}
    if mismatch:
        raise C84CCanaryError(f"C84C authorization binding mismatch: {mismatch}")
    head = _git("rev-parse", "HEAD").stdout.strip()
    origin = _git("rev-parse", "origin/oaci").stdout.strip()
    branch = _git("branch", "--show-current").stdout.strip()
    if head != origin or branch != "oaci" or not _all_relevant_worktrees_clean():
        raise C84CCanaryError("C84C requires clean HEAD == origin/oaci on branch oaci")
    if _git("merge-base", "--is-ancestor", lock_commit, head, check=False).returncode != 0:
        raise C84CCanaryError("C84C execution lock is not an ancestor of HEAD")
    if sys.prefix != EXPECTED_CONDA_PREFIX or ".".join(map(str, sys.version_info[:3])) != EXPECTED_PYTHON:
        raise C84CCanaryError("C84C Python/Conda environment differs from the lock")
    run_root = _verify_output_root(Path(output_root), lock_sha)
    return {
        "protocol": canary,
        "protocol_sha256": protocol_sha,
        "lock": lock,
        "lock_sha256": lock_sha,
        "lock_commit": lock_commit,
        "authorization": authorization,
        "authorization_sha256": sha256_file(authorization_path),
        "HEAD": head,
        "origin_oaci": origin,
        "run_root": run_root,
    }


def consume_authorization(binding: Mapping[str, Any]) -> dict[str, Any]:
    """Consume authorization before loader import; failed attempts remain visible."""
    path = Path(binding["run_root"]) / "authorization_consumed.json"
    if path.exists():
        raise C84CCanaryError("C84C authorization was already consumed")
    payload = {
        "schema_version": "c84c_authorization_consumption_v1",
        "stage": "C84C",
        "protocol_sha256": binding["protocol_sha256"],
        "execution_lock_sha256": binding["lock_sha256"],
        "execution_lock_commit": binding["lock_commit"],
        "authorization_record_sha256": binding["authorization_sha256"],
        "consumed_at_unix_ns": time.time_ns(),
        "before_dataset_loader_import": True,
        "before_download_or_real_array_access": True,
        "target_scientific_outcomes_authorized": False,
    }
    write_json_atomic(path, payload)
    return {**payload, "path": str(path), "sha256": sha256_file(path)}


@dataclass(frozen=True)
class SourceView:
    X: Any
    y: Any
    trial_id: tuple[str, ...]
    subject_id: tuple[int, ...]
    session: tuple[str, ...]
    run: tuple[str, ...]
    dataset_id: str
    role: str


@dataclass(frozen=True)
class TargetUnlabeledView:
    X: Any
    trial_id: tuple[str, ...]
    target_subject_id: int
    session: tuple[str, ...]
    run: tuple[str, ...]
    dataset_id: str

    def as_payload_fields(self) -> tuple[str, ...]:
        return ("X", "trial_id", "target_subject_id", "session", "run", "dataset_id")


def _metadata_column(metadata: Any, name: str, default: str) -> list[Any]:
    if hasattr(metadata, "columns") and name in metadata.columns:
        return metadata[name].tolist()
    return [default] * len(metadata)


def _stable_trial_metadata(metadata: Any, dataset: str, expected_subject: int | None = None) -> dict[str, tuple[Any, ...]]:
    forbidden = {"label", "labels", "y", "target", "event", "class", "class_name"}
    columns = {str(column).lower() for column in getattr(metadata, "columns", ())}
    if columns & forbidden:
        raise C84CCanaryError("target-unlabeled metadata contains a label-like descriptor")
    subject = [int(value) for value in _metadata_column(metadata, "subject", str(expected_subject or -1))]
    if expected_subject is not None and set(subject) != {int(expected_subject)}:
        raise C84CCanaryError("target metadata subject differs from the locked canary target")
    session = [str(value) for value in _metadata_column(metadata, "session", "0")]
    run = [str(value) for value in _metadata_column(metadata, "run", "0")]
    trial_ids = tuple(
        f"{dataset}|subject={subject[i]}|session={session[i]}|run={run[i]}|trial={i:05d}"
        for i in range(len(subject))
    )
    if len(set(trial_ids)) != len(trial_ids):
        raise C84CCanaryError("C84C trial IDs are not unique")
    return {"trial_id": trial_ids, "subject": tuple(subject), "session": tuple(session), "run": tuple(run)}


def _normalize_half_open_array(X: Any, np: Any) -> Any:
    array = np.asarray(X)
    if array.ndim != 3 or array.shape[1] != 20:
        raise C84CCanaryError(f"C84C expected [trial,20,time], got {array.shape}")
    if array.shape[2] == 481:
        array = array[:, :, :480]
    if array.shape[2] != 480:
        raise C84CCanaryError(f"C84C expected the half-open 480-sample epoch, got {array.shape[2]}")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise C84CCanaryError("C84C EEG array contains non-finite values")
    mean = array.mean(axis=2, keepdims=True, dtype=np.float64)
    std = array.std(axis=2, keepdims=True, dtype=np.float64)
    if np.any(std <= 1e-8):
        raise C84CCanaryError("C84C EEG channel has near-zero within-trial variance")
    return np.asarray((array - mean) / std, dtype=np.float32)


def _source_view_from_loader_result(result: Any, dataset: str, role: str, np: Any) -> SourceView:
    X = _normalize_half_open_array(result[0], np)
    labels = np.asarray(result[1])
    metadata = result[2]
    keys = _stable_trial_metadata(metadata, dataset)
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    y = np.asarray([mapping[str(value)] for value in labels], dtype=np.int64)
    if len(y) != X.shape[0]:
        raise C84CCanaryError("C84C source labels and arrays differ in length")
    return SourceView(X, y, keys["trial_id"], keys["subject"], keys["session"], keys["run"], dataset, role)


def _target_unlabeled_from_loader_result(result: Any, dataset: str, subject: int, np: Any) -> TargetUnlabeledView:
    """Discard the structural y slot without binding, indexing, hashing, or logging it."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise C84CCanaryError("C84C target loader result must be a three-slot tuple")
    metadata = result[2]
    keys = _stable_trial_metadata(metadata, dataset, subject)
    X = _normalize_half_open_array(result[0], np)
    del result
    return TargetUnlabeledView(X, keys["trial_id"], int(subject), keys["session"], keys["run"], dataset)


def _real_dataset_and_paradigm(dataset_code: str):
    """Called only after authorization consumption."""
    from moabb.datasets import Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    datasets = {
        "Lee2019_MI": lambda: Lee2019_MI(train_run=True, test_run=False),
        "Cho2017": Cho2017,
        "PhysionetMI": lambda: PhysionetMI(imagined=True, executed=False),
    }
    if dataset_code not in datasets:
        raise C84CCanaryError(f"C84C dataset is outside the lock: {dataset_code}")
    paradigm = MotorImagery(
        n_classes=2,
        events=list(CLASS_NAMES),
        fmin=4.0,
        fmax=38.0,
        tmin=0.0,
        tmax=3.0,
        channels=list(registry.PRIMARY_CHANNELS),
        resample=160,
    )
    return datasets[dataset_code](), paradigm


def _load_canary_views(dataset_code: str, np: Any) -> tuple[SourceView, SourceView, TargetUnlabeledView]:
    spec = registry.DATASETS[dataset_code]
    partition = registry.partition_subjects(spec)
    split = registry.source_train_audit_split(dataset_code, "A", partition["source_panel_A"])
    target = CANARY_TARGETS[dataset_code]
    if partition["targets"][0] != target:
        raise C84CCanaryError("C84C canary target no longer matches the locked partition")
    dataset, paradigm = _real_dataset_and_paradigm(dataset_code)
    source_train_result = paradigm.get_data(dataset=dataset, subjects=list(split["source_training"]), return_epochs=False)
    source_audit_result = paradigm.get_data(dataset=dataset, subjects=list(split["source_audit"]), return_epochs=False)
    target_result = paradigm.get_data(dataset=dataset, subjects=[target], return_epochs=False)
    source_train = _source_view_from_loader_result(source_train_result, dataset_code, "source_training", np)
    source_audit = _source_view_from_loader_result(source_audit_result, dataset_code, "source_audit", np)
    target_unlabeled = _target_unlabeled_from_loader_result(target_result, dataset_code, target, np)
    if set(source_train.subject_id) & set(source_audit.subject_id):
        raise C84CCanaryError("C84C source training/audit subjects overlap")
    if target in set(source_train.subject_id) | set(source_audit.subject_id):
        raise C84CCanaryError("C84C target subject entered a source view")
    return source_train, source_audit, target_unlabeled


class _OptimizerCapture:
    """Passive external optimizer-state capture at the fixed checkpoint cadence."""

    def __init__(self, directory: Path):
        self.directory = directory
        self.phase = "none"
        self.optimizers: dict[str, Any] = {}
        self.counts: dict[str, int] = {}
        self.snapshots: dict[tuple[str, int], dict[str, Any]] = {}
        self._original = None

    def begin(self, phase: str) -> None:
        if phase not in {"ERM", "OACI", "SRC"}:
            raise ValueError(phase)
        self.phase = phase
        self.optimizers = {}
        self.counts = {}

    def _next_label(self) -> str:
        expected = {"ERM": ("encoder",), "OACI": ("critic", "encoder"), "SRC": ("encoder",)}[self.phase]
        if len(self.optimizers) >= len(expected):
            raise C84CCanaryError(f"too many optimizers in {self.phase}")
        return expected[len(self.optimizers)]

    def make_optimizer(self, params, lr, cfg):
        optimizer = self._original(params, lr, cfg)
        label = self._next_label()
        self.optimizers[label] = optimizer
        self.counts[label] = 0
        original_step = optimizer.step

        def captured_step(*args, **kwargs):
            value = original_step(*args, **kwargs)
            self.counts[label] += 1
            count = self.counts[label]
            if self.phase == "ERM" and label == "encoder" and count == 200:
                self._snapshot(0)
            elif self.phase in {"OACI", "SRC"} and label == "encoder" and count % 100 == 0:
                self._snapshot(count // 100)
            return value

        optimizer.step = captured_step
        return optimizer

    def _snapshot(self, order: int) -> None:
        import torch

        payload = {
            "phase": self.phase,
            "trajectory_order": int(order),
            "step_counts": dict(self.counts),
            "optimizers": {key: value.state_dict() for key, value in self.optimizers.items()},
        }
        raw_identity = hashlib.sha256(repr((self.phase, order, sorted(self.counts.items()))).encode("ascii")).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self.directory / f"{self.phase.lower()}_{order:02d}_{raw_identity[:16]}.pt"
        temporary = path.with_name(f".{path.name}.tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)
        self.snapshots[(self.phase, int(order))] = {
            "path": str(path), "file_sha256": sha256_file(path), "step_counts": dict(self.counts),
        }

    def descriptor(self, phase: str, order: int) -> dict[str, Any]:
        try:
            return dict(self.snapshots[(phase, int(order))])
        except KeyError as exc:
            raise C84CCanaryError(f"missing optimizer snapshot {phase}/{order}") from exc

    def __enter__(self):
        import oaci.train.engine as engine

        self._original = engine.make_optimizer
        engine.make_optimizer = self.make_optimizer
        return self

    def __exit__(self, exc_type, exc, traceback):
        import oaci.train.engine as engine

        engine.make_optimizer = self._original
        self._original = None


def _save_torch_state(path: Path, state: Any, torch: Any) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    torch.save(state, temporary)
    os.replace(temporary, path)
    return {"path": str(path), "sha256": sha256_file(path), "bytes": path.stat().st_size}


def _training_objects(source: SourceView, torch: Any, np: Any):
    from oaci.config import SamplerConfig
    from oaci.data.plan_materialize import (
        materialize_full_domain_alignment_plan,
        materialize_oaci_alignment_plan,
        materialize_stage1_task_plan,
        materialize_stage2_task_plan,
    )
    from oaci.data.plan_sampler import UnitIndex
    from oaci.support_graph import build_support_graph, counts_from_labels, empirical_class_prior
    from oaci.train.data import TrainingData, population_signature_hash
    from oaci.train.engine import EngineConfig

    domain_names = sorted(set(source.subject_id))
    domain_map = {subject: index for index, subject in enumerate(domain_names)}
    domains = np.asarray([domain_map[value] for value in source.subject_id], dtype=np.int64)
    groups = tuple(f"{source.dataset_id}|subject={source.subject_id[i]}|session={source.session[i]}|run={source.run[i]}"
                   for i in range(len(source.trial_id)))
    mass = np.ones(len(source.trial_id), dtype=np.float64)
    data = TrainingData(
        X=torch.as_tensor(source.X, dtype=torch.float32),
        y=torch.as_tensor(source.y, dtype=torch.long),
        sample_id=tuple(source.trial_id),
        sample_mass=torch.as_tensor(mass, dtype=torch.float32),
        n_classes=2,
        d=torch.as_tensor(domains, dtype=torch.long),
        group=groups,
    ).validate()
    counts = counts_from_labels(domains, source.y, n_domains=len(domain_names), n_classes=2)
    support = build_support_graph(
        counts, 8, cell_mass=counts.astype(float), reference_prior=empirical_class_prior(counts),
        domain_names=[str(value) for value in domain_names], class_names=list(CLASS_NAMES),
    ).validate()
    index = UnitIndex(data.sample_id, source.y, domains, groups, data.sample_id, mass)
    pop = population_signature_hash(data)
    cfg = SamplerConfig(task_batch_size=256, adv_microbatch_size=256, adv_accumulation_steps=4,
                        min_per_eligible_cell=8, steps_per_epoch=20, replacement_mode="auto", seed=TRAINING_SEED)
    stage1_plan = materialize_stage1_task_plan(index, pop, 200, 1, 256, TRAINING_SEED, "auto")
    stage2_plan = materialize_stage2_task_plan(index, pop, 200, 20, 256, TRAINING_SEED, "auto")
    oaci_plan = materialize_oaci_alignment_plan(index, support, pop, 60, 4000, 5, 8, 256,
                                                TRAINING_SEED, accumulation_steps=4, replacement_mode="auto")
    full_plan = materialize_full_domain_alignment_plan(index, pop, 60, 4000, 5, 8, 256,
                                                       TRAINING_SEED, accumulation_steps=4,
                                                       replacement_mode="auto")
    engine_cfg = EngineConfig(
        metric="balanced_ce", epsilon=0.03, numerical_tol=1e-4,
        stage1_epochs=200, stage1_steps_per_epoch=1, stage2_epochs=200,
        steps_per_epoch=20, warmup_steps=60, critic_steps=5, checkpoint_every=5,
        guard_chunk_size=1024, optimizer_name="adam", weight_decay=0.0,
        lr_stage1=0.005, lr_encoder=0.01, lr_critic=0.01, dual_lr=0.5,
        lambda_init=0.3, lambda_max=20.0, lambda_floor=0.0,
        gradient_clip=0.0, critic_gradient_clip=0.0,
        deterministic_algorithms=True, stage2_bn_mode="frozen_erm_running_stats",
        base_seed=TRAINING_SEED,
    )
    return data, support, stage1_plan, stage2_plan, oaci_plan, full_plan, engine_cfg


def _instrument_state(state: Mapping[str, Any], target: TargetUnlabeledView, output: Path, torch: Any, np: Any) -> dict[str, Any]:
    from oaci.models import build_model

    model = build_model("shallow_convnet", in_chans=20, in_times=480, n_classes=2,
                        temporal_filters=40, temporal_kernel_samples=25,
                        pool_kernel_samples=75, pool_stride_samples=15,
                        dropout=0.5, safe_log_eps=1e-6)
    model.load_state_dict(state)
    model.eval().to("cuda:0")
    logits, features = [], []
    with torch.inference_mode():
        for start in range(0, target.X.shape[0], 1024):
            out = model(torch.as_tensor(target.X[start:start + 1024], device="cuda:0"))
            logits.append(out.logits.detach().cpu())
            features.append(out.z.detach().cpu())
    logits = torch.cat(logits)
    features = torch.cat(features)
    with torch.inference_mode():
        reconstructed = model.classifier(features.to("cuda:0")).cpu()
        probabilities = torch.softmax(logits, dim=1)
        repeated_output = model(torch.as_tensor(target.X, device="cuda:0"))
        repeat = repeated_output.logits.cpu()
        repeat_features = repeated_output.z.cpu()
    wz_error = float(torch.max(torch.abs(reconstructed - logits)))
    repeat_error = float(torch.max(torch.abs(repeat - logits)))
    repeat_z_error = float(torch.max(torch.abs(repeat_features - features)))
    probability_error = float(torch.max(torch.abs(torch.softmax(logits, dim=1) - probabilities)))
    if max(wz_error, repeat_error, repeat_z_error, probability_error) > 1e-6:
        raise C84CCanaryError("C84C instrumentation identity failed")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp.npz")
    np.savez_compressed(temporary, logits=logits.numpy(), probabilities=probabilities.numpy(),
                        z=features.numpy(), Wz_plus_b=reconstructed.numpy(),
                        trial_id=np.asarray(target.trial_id, dtype=str))
    os.replace(temporary, output)
    return {
        "path": str(output), "sha256": sha256_file(output), "bytes": output.stat().st_size,
        "rows": len(target.trial_id), "Wz_plus_b_max_error": wz_error,
        "softmax_max_error": probability_error, "repeat_logits_max_error": repeat_error,
        "repeat_z_max_error": repeat_z_error,
        "target_label_fields": 0,
    }


def _run_dataset_training(dataset: str, source: SourceView, source_audit: SourceView,
                          target: TargetUnlabeledView, root: Path, torch: Any, np: Any) -> dict[str, Any]:
    from oaci.methods.oaci import OACIObjective
    from oaci.methods.source_robust import SRCObjective
    from oaci.models import build_model
    from oaci.train.engine import InvocationRegistry, train_stage1, train_stage2
    from oaci.train.rng import derive_seed, forked_rng

    data, support, stage1_plan, stage2_plan, oaci_plan, full_plan, engine_cfg = _training_objects(source, torch, np)

    def model_factory():
        return build_model("shallow_convnet", in_chans=20, in_times=480, n_classes=2,
                           temporal_filters=40, temporal_kernel_samples=25,
                           pool_kernel_samples=75, pool_stride_samples=15,
                           dropout=0.5, safe_log_eps=1e-6)

    units = [row for row in protocol.candidate_units()
             if row["dataset"] == dataset and row["source_panel"] == "A"
             and row["training_seed"] == 5 and row["level"] == 0]
    by_key = {(row["regime"], row["epoch"]): row for row in units}
    output_units = []
    with _OptimizerCapture(root / "optimizer_states") as capture:
        capture.begin("ERM")
        with forked_rng(derive_seed(TRAINING_SEED, dataset, "model_init"), torch.device("cuda:0")):
            model = model_factory()
        erm = train_stage1(model, data, stage1_plan, engine_cfg, torch.device("cuda:0"),
                           InvocationRegistry(), f"C84C|{dataset}|A|seed5|level0")
        capture.begin("OACI")
        oaci = train_stage2(model_factory, erm, data, OACIObjective(support, adv_hidden=16),
                            stage2_plan, oaci_plan, engine_cfg, torch.device("cuda:0"))
        capture.begin("SRC")
        src = train_stage2(model_factory, erm, data,
                           SRCObjective(2, support.n_domains, smooth_temperature=0.1),
                           stage2_plan, full_plan, engine_cfg, torch.device("cuda:0"))
        if tuple(record.epoch for record in oaci.trajectory) != CHECKPOINT_EPOCHS:
            raise C84CCanaryError("C84C OACI checkpoint cadence drift")
        if tuple(record.epoch for record in src.trajectory) != CHECKPOINT_EPOCHS:
            raise C84CCanaryError("C84C SRC checkpoint cadence drift")
        records = [("ERM", erm.checkpoint, 0, capture.descriptor("ERM", 0))]
        records.extend(("OACI", record, order, capture.descriptor("OACI", order))
                       for order, record in enumerate(oaci.trajectory, start=1))
        records.extend(("SRC", record, order, capture.descriptor("SRC", order))
                       for order, record in enumerate(src.trajectory, start=1))
        previous_hash = {"ERM": erm.checkpoint.model_hash, "OACI": erm.checkpoint.model_hash,
                         "SRC": erm.checkpoint.model_hash}
        for regime, record, order, optimizer in records:
            unit = by_key[(regime, int(record.epoch))]
            checkpoint = _save_torch_state(root / "checkpoints" / f"{unit['unit_id']}.pt", record.model_state, torch)
            instrumentation = _instrument_state(
                record.model_state, target, root / "instrumentation" / f"{unit['unit_id']}.npz", torch, np,
            )
            sidecar = {
                "schema_version": "c84c_candidate_sidecar_v1",
                "unit_id": unit["unit_id"], "dataset": dataset, "panel": "A", "seed": 5, "level": 0,
                "regime": regime, "epoch": int(record.epoch), "trajectory_order": order,
                "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
                "epoch_rule": EPOCH_RULE, "checkpoint": checkpoint, "optimizer": optimizer,
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous_hash[regime],
                "genealogy_rule": "shared_ERM_parent_then_fixed_regime_trajectory_order",
                "instrumentation": instrumentation, "support_hash": support.support_hash(),
                "target_subject": target.target_subject_id, "target_fit_ids": [],
                "training_target_rows": 0, "training_target_labels": 0,
                "source_audit_rows_used_in_training": 0, "target_outcome_retention": 0,
                "target_outcome_retry": 0, "target_scientific_metrics": 0,
            }
            if SCIENTIFIC_OUTPUT_KEYS & set(sidecar):
                raise C84CCanaryError("C84C sidecar exposes a forbidden scientific output")
            sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
            write_json_atomic(sidecar_path, sidecar)
            output_units.append({
                "unit_id": unit["unit_id"], "dataset": dataset, "regime": regime,
                "epoch": int(record.epoch), "trajectory_order": order,
                "checkpoint_sha256": checkpoint["sha256"],
                "instrumentation_sha256": instrumentation["sha256"],
                "sidecar_path": str(sidecar_path), "sidecar_sha256": sha256_file(sidecar_path),
            })
            previous_hash[regime] = record.model_hash
    if len(output_units) != 81:
        raise C84CCanaryError(f"C84C {dataset} produced {len(output_units)} units instead of 81")
    return {
        "dataset": dataset, "units": output_units, "unit_count": len(output_units),
        "source_training_subjects": sorted(set(source.subject_id)),
        "source_audit_subjects": sorted(set(source_audit.subject_id)),
        "source_audit_rows_used_in_training": 0,
        "target_subject": target.target_subject_id,
        "target_unlabeled_fields": list(target.as_payload_fields()),
        "target_label_access": 0, "scientific_metrics": 0,
    }


def run_real(*, authorization_path: Path = AUTHORIZATION_RECORD_PATH,
             output_root: Path = DEFAULT_EXTERNAL_ROOT) -> dict[str, Any]:
    binding = require_authorization_and_lock(authorization_path=authorization_path, output_root=output_root)
    consumption = consume_authorization(binding)
    # Protected imports occur only after authorization has been consumed.
    import numpy as np
    import torch

    if not torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID") is None:
        raise C84CCanaryError("C84C requires an authorized Slurm CUDA allocation")
    torch.use_deterministic_algorithms(True, warn_only=False)
    run_root = Path(binding["run_root"])
    attempt_path = run_root / "execution_attempts.jsonl"
    with attempt_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "start", "job_id": os.environ["SLURM_JOB_ID"],
                                 "authorization_consumption_sha256": consumption["sha256"],
                                 "target_scientific_outcomes": 0}, sort_keys=True) + "\n")
    datasets = []
    try:
        for dataset in protocol.DATASET_ORDER:
            source, audit, target = _load_canary_views(dataset, np)
            datasets.append(_run_dataset_training(dataset, source, audit, target, run_root / dataset, torch, np))
        manifest = {
            "schema_version": "c84c_complete_canary_manifest_v1",
            "protocol_sha256": binding["protocol_sha256"],
            "execution_lock_sha256": binding["lock_sha256"],
            "authorization_consumption_sha256": consumption["sha256"],
            "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
            "datasets": datasets, "unit_count": sum(item["unit_count"] for item in datasets),
            "training_phases": 9, "target_label_access": 0, "target_scientific_metrics": 0,
            "C84F_authorized": False, "C84S_authorized": False,
        }
        if manifest["unit_count"] != 243:
            raise C84CCanaryError("C84C complete unit count is not 243")
        manifest_path = run_root / "C84C_COMPLETE_CANARY_MANIFEST.json"
        write_json_atomic(manifest_path, manifest)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": sha256_file(manifest_path)}
    except Exception as exc:
        with attempt_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"event": "failed", "error_type": type(exc).__name__,
                                     "error": str(exc), "target_scientific_outcomes": 0}, sort_keys=True) + "\n")
        raise


def synthetic_schema_dry_run() -> dict[str, Any]:
    """Validate protected schemas without importing arrays, loaders, or ML."""
    units = protocol.candidate_units()
    canary = [row for row in units if row["canary_subset"]]
    payload = {
        "schema_version": "c84c_schema_dry_run_v1",
        "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
        "channels": list(registry.PRIMARY_CHANNELS), "input_shape": [20, 480],
        "canary_units": len(canary), "canary_training_phases": 9,
        "targets": CANARY_TARGETS, "target_unlabeled_fields": list(TargetUnlabeledView.__dataclass_fields__),
        "target_y_field_present": "y" in TargetUnlabeledView.__dataclass_fields__,
        "real_EEG_arrays_loaded": 0, "real_labels_read": 0, "dataset_downloads": 0,
        "authorization_consumed": False,
    }
    if payload["canary_units"] != 243 or payload["target_y_field_present"]:
        raise C84CCanaryError("C84C dry-run contract failed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84C fail-closed real engineering canary")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("show-contract")
    sub.add_parser("schema-dry-run")
    real = sub.add_parser("run-real")
    real.add_argument("--authorization-record", type=Path, default=AUTHORIZATION_RECORD_PATH)
    real.add_argument("--output-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args(argv)
    if args.command == "show-contract":
        print(json.dumps({
            "stage": "C84C", "protocol": str(CANARY_PROTOCOL_PATH), "lock": str(EXECUTION_LOCK_PATH),
            "authorization_required": True, "magic_token_required": False,
            "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256,
            "run_real_imports_loaders_only_after_authorization_consumption": True,
        }, sort_keys=True))
        return 0
    if args.command == "schema-dry-run":
        print(json.dumps(synthetic_schema_dry_run(), sort_keys=True))
        return 0
    print(json.dumps(run_real(authorization_path=args.authorization_record, output_root=args.output_root), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
