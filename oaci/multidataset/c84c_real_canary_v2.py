"""C84C V3 fail-closed real-data engineering canary adapter.

The module is import-safe during C84R2. NumPy, torch, MNE, MOABB, and dataset
loaders are imported only after a fresh authorization has been consumed and an
execution-attempt ledger has been persisted.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

from . import c84_dataset_registry_v2 as registry
from . import c84r_v2_protocols as protocol
from . import c84c_real_canary as legacy
from . import c84r2_canary_runtime_repair as runtime
from .c84r_montage_repair import EPOCH_RULE, INTERFACE_ID, MONTAGE_SHA256


REPO_ROOT = runtime.REPO_ROOT
REPORT_DIR = runtime.REPORT_DIR
CANARY_PROTOCOL_PATH = runtime.CANARY_PROTOCOL_PATH
CANARY_PROTOCOL_SHA_PATH = runtime.CANARY_PROTOCOL_SHA_PATH
EXECUTION_LOCK_PATH = runtime.EXECUTION_LOCK_PATH
EXECUTION_LOCK_SHA_PATH = runtime.EXECUTION_LOCK_SHA_PATH
AUTHORIZATION_RECORD_PATH = runtime.AUTHORIZATION_RECORD_PATH
DEFAULT_EXTERNAL_ROOT = runtime.DEFAULT_EXTERNAL_ROOT
CANARY_TARGETS = protocol.CANARY_TARGETS
SOURCE_PANEL = "A"
TRAINING_SEED = 5
LEVEL = 0
CHECKPOINT_EPOCHS = tuple(range(4, 200, 5))
CLASS_NAMES = ("left_hand", "right_hand")
EXPECTED_CHANNELS = runtime.EXPECTED_CHANNELS
EXPECTED_SFREQ = 160.0
EXPECTED_FINAL_N_TIMES = 480
SCIENTIFIC_OUTPUT_KEYS = legacy.SCIENTIFIC_OUTPUT_KEYS


class C84CCanaryV2Error(runtime.C84R2RuntimeError):
    """Raised when an executable V3 canary engineering contract fails."""


@dataclass(frozen=True)
class EpochInterface:
    actual_ch_names: tuple[str, ...]
    actual_sfreq_hz: float
    pre_half_open_n_times: int
    final_n_times: int
    first_time_s: float
    last_time_s_before_half_open: float
    final_first_time_s: float
    final_last_time_s: float
    input_shape: tuple[int, int, int]
    bad_channels: tuple[str, ...]
    interpolation_or_synthesis: bool

    def as_dict(self) -> dict[str, Any]:
        return {
            "actual_ch_names": list(self.actual_ch_names),
            "actual_sfreq_hz": self.actual_sfreq_hz,
            "pre_half_open_n_times": self.pre_half_open_n_times,
            "final_n_times": self.final_n_times,
            "first_time_s": self.first_time_s,
            "last_time_s_before_half_open": self.last_time_s_before_half_open,
            "final_first_time_s": self.final_first_time_s,
            "final_last_time_s": self.final_last_time_s,
            "input_shape": list(self.input_shape),
            "bad_channels": list(self.bad_channels),
            "interpolation_or_synthesis": self.interpolation_or_synthesis,
        }


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
    interface: EpochInterface


@dataclass(frozen=True)
class TargetUnlabeledView:
    X: Any
    trial_id: tuple[str, ...]
    target_subject_id: int
    session: tuple[str, ...]
    run: tuple[str, ...]
    dataset_id: str
    interface: EpochInterface

    def as_payload_fields(self) -> tuple[str, ...]:
        return ("X", "trial_id", "target_subject_id", "session", "run", "dataset_id", "interface")


def canonical_bytes(value: Any) -> bytes:
    return runtime.canonical_bytes(value)


def _metadata_column(metadata: Any, name: str, default: str) -> list[Any]:
    if hasattr(metadata, "columns") and name in metadata.columns:
        return metadata[name].tolist()
    return [default] * len(metadata)


def _metadata_keys(
    metadata: Any,
    dataset: str,
    *,
    expected_subjects: Sequence[int],
    forbid_label_like: bool,
) -> dict[str, tuple[Any, ...]]:
    columns = {str(column).strip().lower() for column in getattr(metadata, "columns", ())}
    if forbid_label_like:
        label_tokens = {"y", "target", "targets"}
        if any(column in label_tokens or any(token in column for token in ("label", "class", "event"))
               for column in columns):
            raise C84CCanaryV2Error("target-unlabeled metadata contains a label-like field")
    subject = tuple(int(value) for value in _metadata_column(metadata, "subject", "-1"))
    expected = {int(value) for value in expected_subjects}
    if set(subject) != expected:
        raise C84CCanaryV2Error(
            f"loaded subject identity mismatch: observed={sorted(set(subject))} expected={sorted(expected)}"
        )
    session = tuple(str(value) for value in _metadata_column(metadata, "session", "0"))
    run = tuple(str(value) for value in _metadata_column(metadata, "run", "0"))
    trial_ids = tuple(
        f"{dataset}|subject={subject[index]}|session={session[index]}|run={run[index]}|trial={index:05d}"
        for index in range(len(subject))
    )
    if len(set(trial_ids)) != len(trial_ids):
        raise C84CCanaryV2Error("C84C V3 trial IDs are not unique")
    return {"trial_id": trial_ids, "subject": subject, "session": session, "run": run}


def _epochs_array_and_interface(epochs: Any, np: Any) -> tuple[Any, EpochInterface]:
    channels = tuple(str(value) for value in epochs.ch_names)
    if channels != EXPECTED_CHANNELS:
        raise C84CCanaryV2Error(
            f"actual Epochs channel order differs: observed={channels} expected={EXPECTED_CHANNELS}"
        )
    sfreq = float(epochs.info["sfreq"])
    if abs(sfreq - EXPECTED_SFREQ) > 1e-9:
        raise C84CCanaryV2Error(f"actual Epochs sampling rate differs: {sfreq}")
    bads = tuple(str(value) for value in epochs.info.get("bads", ()))
    if bads:
        raise C84CCanaryV2Error(f"actual Epochs contains bad/synthesized channels: {bads}")
    try:
        values = epochs.get_data(copy=True)
    except TypeError:
        values = epochs.get_data()
    array = np.asarray(values)
    times = np.asarray(epochs.times, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != len(EXPECTED_CHANNELS):
        raise C84CCanaryV2Error(f"C84C V3 expected [trial,20,time], got {array.shape}")
    if len(times) != array.shape[2] or len(times) not in {480, 481}:
        raise C84CCanaryV2Error("actual Epochs time axis is not the locked 480/481 sample interface")
    if abs(float(times[0])) > 1e-9:
        raise C84CCanaryV2Error("actual Epochs does not start at registered cue time 0.0")
    pre_n_times = int(array.shape[2])
    pre_last = float(times[-1])
    if pre_n_times == 481:
        if abs(pre_last - 3.0) > 1e-9:
            raise C84CCanaryV2Error("inclusive 481-sample Epochs endpoint is not 3.0 seconds")
        array = array[:, :, :480]
        final_times = times[:480]
    else:
        final_times = times
    if array.shape[2] != EXPECTED_FINAL_N_TIMES:
        raise C84CCanaryV2Error("half-open Epochs conversion did not produce 480 samples")
    expected_last = (EXPECTED_FINAL_N_TIMES - 1) / EXPECTED_SFREQ
    if abs(float(final_times[-1]) - expected_last) > 1e-9:
        raise C84CCanaryV2Error("half-open Epochs final timestamp differs from 479/160")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise C84CCanaryV2Error("C84C V3 EEG array contains non-finite values")
    mean = array.mean(axis=2, keepdims=True, dtype=np.float64)
    std = array.std(axis=2, keepdims=True, dtype=np.float64)
    if np.any(std <= 1e-8):
        raise C84CCanaryV2Error("C84C V3 EEG channel has near-zero within-trial variance")
    normalized = np.asarray((array - mean) / std, dtype=np.float32)
    interface = EpochInterface(
        actual_ch_names=channels,
        actual_sfreq_hz=sfreq,
        pre_half_open_n_times=pre_n_times,
        final_n_times=int(normalized.shape[2]),
        first_time_s=float(times[0]),
        last_time_s_before_half_open=pre_last,
        final_first_time_s=float(final_times[0]),
        final_last_time_s=float(final_times[-1]),
        input_shape=tuple(int(value) for value in normalized.shape),
        bad_channels=bads,
        interpolation_or_synthesis=False,
    )
    return normalized, interface


def _source_view_from_loader_result(
    result: Any,
    dataset: str,
    role: str,
    expected_subjects: Sequence[int],
    np: Any,
) -> SourceView:
    if not isinstance(result, tuple) or len(result) != 3:
        raise C84CCanaryV2Error("source loader result must be a three-slot tuple")
    X, interface = _epochs_array_and_interface(result[0], np)
    labels = np.asarray(result[1])
    keys = _metadata_keys(result[2], dataset, expected_subjects=expected_subjects, forbid_label_like=False)
    mapping = {name: index for index, name in enumerate(CLASS_NAMES)}
    try:
        y = np.asarray([mapping[str(value)] for value in labels], dtype=np.int64)
    except KeyError as exc:
        raise C84CCanaryV2Error(f"source loader returned an unregistered class: {exc}") from exc
    if len(y) != X.shape[0] or len(keys["trial_id"]) != X.shape[0]:
        raise C84CCanaryV2Error("source labels, metadata and arrays differ in length")
    return SourceView(X, y, keys["trial_id"], keys["subject"], keys["session"], keys["run"],
                      dataset, role, interface)


def _target_unlabeled_from_loader_result(
    result: Any,
    dataset: str,
    subject: int,
    np: Any,
) -> TargetUnlabeledView:
    """Never bind, index, hash, convert, summarize, log, or represent slot 1."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise C84CCanaryV2Error("target loader result must be a three-slot tuple")
    keys = _metadata_keys(result[2], dataset, expected_subjects=(subject,), forbid_label_like=True)
    X, interface = _epochs_array_and_interface(result[0], np)
    del result
    if len(keys["trial_id"]) != X.shape[0]:
        raise C84CCanaryV2Error("target metadata and arrays differ in length")
    return TargetUnlabeledView(X, keys["trial_id"], int(subject), keys["session"], keys["run"], dataset, interface)


def _protected_loader_objects() -> tuple[dict[str, Any], Any, Any, Any, Any]:
    from moabb.datasets import Cho2017, Lee2019_MI, PhysionetMI
    from moabb.paradigms import MotorImagery

    objects = {
        "moabb.datasets.Lee2019_MI": Lee2019_MI,
        "moabb.datasets.Cho2017": Cho2017,
        "moabb.datasets.PhysionetMI": PhysionetMI,
        "moabb.paradigms.MotorImagery": MotorImagery,
    }
    return objects, Lee2019_MI, Cho2017, PhysionetMI, MotorImagery


def _real_dataset_and_paradigm(dataset_code: str, loader_classes: tuple[Any, Any, Any, Any]):
    Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = loader_classes
    factories = {
        "Lee2019_MI": lambda: Lee2019_MI(train_run=True, test_run=False),
        "Cho2017": Cho2017,
        "PhysionetMI": lambda: PhysionetMI(imagined=True, executed=False),
    }
    if dataset_code not in factories:
        raise C84CCanaryV2Error(f"dataset is outside the C84C V3 lock: {dataset_code}")
    paradigm = MotorImagery(
        n_classes=2,
        events=list(CLASS_NAMES),
        fmin=4.0,
        fmax=38.0,
        tmin=0.0,
        tmax=3.0,
        channels=list(EXPECTED_CHANNELS),
        resample=160,
    )
    return factories[dataset_code](), paradigm


def _load_canary_views(
    dataset_code: str,
    np: Any,
    loader_classes: tuple[Any, Any, Any, Any],
    ledger: runtime.ExecutionAttemptLedger,
) -> tuple[SourceView, SourceView, TargetUnlabeledView]:
    spec = registry.DATASETS[dataset_code]
    partition = registry.partition_subjects(spec)
    split = registry.source_train_audit_split(dataset_code, "A", partition["source_panel_A"])
    expected_training = tuple(int(value) for value in split["source_training"])
    expected_audit = tuple(int(value) for value in split["source_audit"])
    target = int(CANARY_TARGETS[dataset_code])
    if partition["targets"][0] != target:
        raise C84CCanaryV2Error("canary target no longer matches the locked partition")
    if set(expected_training) & set(expected_audit) or target in set(expected_training) | set(expected_audit):
        raise C84CCanaryV2Error("locked source/target subject sets overlap")
    dataset, paradigm = _real_dataset_and_paradigm(dataset_code, loader_classes)

    def get(subjects: Sequence[int]) -> Any:
        ledger.increment("get_data_calls_started")
        value = paradigm.get_data(dataset=dataset, subjects=list(subjects), return_epochs=True)
        ledger.increment("get_data_calls_completed")
        return value

    source_train_result = get(expected_training)
    source_audit_result = get(expected_audit)
    target_result = get((target,))
    source_train = _source_view_from_loader_result(
        source_train_result, dataset_code, "source_training", expected_training, np,
    )
    ledger.increment("source_label_arrays_read")
    source_audit = _source_view_from_loader_result(
        source_audit_result, dataset_code, "source_audit", expected_audit, np,
    )
    ledger.increment("source_label_arrays_read")
    target_unlabeled = _target_unlabeled_from_loader_result(target_result, dataset_code, target, np)
    ledger.increment("real_EEG_arrays_materialized", 3)
    if set(source_train.subject_id) != set(expected_training):
        raise C84CCanaryV2Error("source-training subject set differs after loader conversion")
    if set(source_audit.subject_id) != set(expected_audit):
        raise C84CCanaryV2Error("source-audit subject set differs after loader conversion")
    if target_unlabeled.target_subject_id != target:
        raise C84CCanaryV2Error("target subject differs after loader conversion")
    return source_train, source_audit, target_unlabeled


def _atomic_save_npz(path: Path, np: Any, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(name, **arrays)
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def _model_factory():
    from oaci.models import build_model

    return build_model(
        "shallow_convnet", in_chans=20, in_times=480, n_classes=2,
        temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15,
        dropout=0.5, safe_log_eps=1e-6,
    )


def _forward_model(model: Any, X: Any, torch: Any) -> tuple[Any, Any]:
    logits = []
    features = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], 1024):
            output = model(torch.as_tensor(X[start:start + 1024], device="cuda:0"))
            logits.append(output.logits.detach().cpu())
            features.append(output.z.detach().cpu())
    return torch.cat(logits), torch.cat(features)


def _instrument_and_replay(
    state: Mapping[str, Any],
    source_audit: SourceView,
    target: TargetUnlabeledView,
    unit: Mapping[str, Any],
    root: Path,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = _model_factory()
    model.load_state_dict(state)
    model.eval().to("cuda:0")
    source_logits, _ = _forward_model(model, source_audit.X, torch)
    target_logits, target_z = _forward_model(model, target.X, torch)
    with torch.inference_mode():
        source_probabilities = torch.softmax(source_logits, dim=1)
        target_probabilities = torch.softmax(target_logits, dim=1)
        classifier_weight = model.classifier.weight.detach().cpu()
        classifier_bias = model.classifier.bias.detach().cpu()
        reconstructed = target_z @ classifier_weight.T + classifier_bias
        repeat_output = model(torch.as_tensor(target.X, device="cuda:0"))
        repeat_logits = repeat_output.logits.detach().cpu()
        repeat_z = repeat_output.z.detach().cpu()
    errors = {
        "Wz_plus_b_max_error": float(torch.max(torch.abs(reconstructed - target_logits))),
        "softmax_max_error": float(torch.max(torch.abs(torch.softmax(target_logits, dim=1) - target_probabilities))),
        "repeat_logits_max_error": float(torch.max(torch.abs(repeat_logits - target_logits))),
        "repeat_z_max_error": float(torch.max(torch.abs(repeat_z - target_z))),
    }
    if max(errors.values()) > 1e-6:
        raise C84CCanaryV2Error(f"in-memory instrumentation identity failed: {errors}")

    identity = {
        "dataset": unit["dataset"], "panel": unit["source_panel"],
        "seed": unit["training_seed"], "level": unit["level"], "unit_id": unit["unit_id"],
    }
    source_path = root / "source_audit" / f"{unit['unit_id']}.npz"
    _atomic_save_npz(
        source_path, np,
        logits=source_logits.numpy(), probabilities=source_probabilities.numpy(),
        source_class_label=np.asarray(source_audit.y, dtype=np.int64),
        source_domain_id=np.asarray(source_audit.subject_id, dtype=np.int64),
        source_trial_id=np.asarray(source_audit.trial_id, dtype=str),
        dataset=np.asarray(identity["dataset"]), panel=np.asarray(identity["panel"]),
        seed=np.asarray(identity["seed"], dtype=np.int64), level=np.asarray(identity["level"], dtype=np.int64),
        unit_id=np.asarray(identity["unit_id"]),
    )
    source_replay = runtime.replay_source_audit_artifact(
        source_path, expected_identity=identity, expected_trial_ids=source_audit.trial_id,
        expected_labels=source_audit.y, expected_domains=source_audit.subject_id, np=np,
    )
    source_descriptor = {
        **source_replay, "bytes": source_path.stat().st_size,
        "fields": sorted(runtime.SOURCE_AUDIT_FIELDS), "target_label_fields": 0,
        "scientific_metrics": 0,
    }

    target_path = root / "target_unlabeled" / f"{unit['unit_id']}.npz"
    _atomic_save_npz(
        target_path, np,
        logits=target_logits.numpy(), probabilities=target_probabilities.numpy(), z=target_z.numpy(),
        Wz_plus_b=reconstructed.numpy(), classifier_weight=classifier_weight.numpy(),
        classifier_bias=classifier_bias.numpy(), repeat_logits=repeat_logits.numpy(), repeat_z=repeat_z.numpy(),
        target_trial_id=np.asarray(target.trial_id, dtype=str), dataset=np.asarray(identity["dataset"]),
        target_subject_id=np.asarray(target.target_subject_id, dtype=np.int64), unit_id=np.asarray(identity["unit_id"]),
    )
    target_identity = {"dataset": identity["dataset"], "unit_id": identity["unit_id"],
                       "target_subject_id": target.target_subject_id}
    target_replay = runtime.replay_target_unlabeled_artifact(
        target_path, expected_identity=target_identity, expected_trial_ids=target.trial_id, np=np,
    )
    target_descriptor = {
        **target_replay, "bytes": target_path.stat().st_size,
        "fields": sorted(runtime.TARGET_UNLABELED_FIELDS), "target_label_fields": 0,
        **errors,
    }
    return source_descriptor, target_descriptor


def _first_batch_ID_digest(plans: Sequence[Any]) -> str:
    stage1, stage2, oaci, full = plans
    values = {
        "stage1": list(stage1.epochs[0][0].sample_ids),
        "stage2": list(stage2.epochs[0][0].sample_ids),
        "oaci": list(oaci.warmup_batches[0].microbatches[0].sample_ids),
        "src": list(full.warmup_batches[0].microbatches[0].sample_ids),
    }
    return hashlib.sha256(canonical_bytes(values)).hexdigest()


def _deterministic_prefix_fingerprint(
    dataset: str,
    data: Any,
    plans: Sequence[Any],
    engine_cfg: Any,
    torch: Any,
) -> dict[str, Any]:
    from oaci.train.checkpoint import state_hash
    from oaci.train.engine import make_optimizer
    from oaci.train.rng import derive_seed, forked_rng

    device = torch.device("cuda:0")
    init_hashes = []
    prefix_hashes = []
    first_ids = tuple(plans[0].epochs[0][0].sample_ids)
    index = {sample_id: row for row, sample_id in enumerate(data.sample_id)}
    rows = [index[sample_id] for sample_id in first_ids]
    X = data.X[rows].to(device)
    y = data.y[rows].to(device)
    for repeat in range(2):
        with forked_rng(derive_seed(TRAINING_SEED, dataset, "deterministic_prefix"), device):
            model = _model_factory().to(device)
            init_hashes.append(state_hash({key: value.detach().cpu() for key, value in model.state_dict().items()}))
            optimizer = make_optimizer(model.parameters(), engine_cfg.lr_stage1, engine_cfg)
            optimizer.zero_grad(set_to_none=True)
            output = model(X)
            loss = torch.nn.functional.cross_entropy(output.logits, y)
            loss.backward()
            optimizer.step()
            prefix_hashes.append(state_hash({key: value.detach().cpu() for key, value in model.state_dict().items()}))
    if len(set(init_hashes)) != 1 or len(set(prefix_hashes)) != 1:
        raise C84CCanaryV2Error("deterministic-prefix repeated fingerprint differs")
    plan_hashes = [str(plan.plan_hash) for plan in plans]
    return {
        "dataset": dataset,
        "model_init_state_sha256": init_hashes[0],
        "materialized_plan_sha256": hashlib.sha256(canonical_bytes(plan_hashes)).hexdigest(),
        "plan_hashes": plan_hashes,
        "first_registered_batch_ID_sha256": _first_batch_ID_digest(plans),
        "short_prefix_state_sha256": prefix_hashes[0],
        "repeat_count": 2,
        "full_duplicate_training": False,
        "deterministic_algorithms": bool(torch.are_deterministic_algorithms_enabled()),
        "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
    }


SIDECAR_FIELDS = {
    "schema_version", "unit_id", "dataset", "panel", "seed", "level", "regime", "epoch",
    "trajectory_order", "interface_id", "montage_sha256", "epoch_rule", "checkpoint",
    "optimizer", "source_audit", "target_unlabeled", "model_state_hash",
    "parent_ERM_model_state_hash", "previous_trajectory_model_state_hash", "genealogy_rule",
    "support_hash", "target_subject", "target_fit_ids", "deterministic_prefix_sha256",
    "training_target_rows", "training_target_labels", "source_audit_rows_used_in_training",
    "target_outcome_retention", "target_outcome_retry", "target_scientific_metrics",
}


def _run_dataset_training(
    dataset: str,
    source: SourceView,
    source_audit: SourceView,
    target: TargetUnlabeledView,
    root: Path,
    torch: Any,
    np: Any,
    ledger: runtime.ExecutionAttemptLedger,
) -> dict[str, Any]:
    from oaci.methods.oaci import OACIObjective
    from oaci.methods.source_robust import SRCObjective
    from oaci.train.checkpoint import state_hash
    from oaci.train.engine import InvocationRegistry, train_stage1, train_stage2
    from oaci.train.rng import derive_seed, forked_rng

    data, support, stage1_plan, stage2_plan, oaci_plan, full_plan, engine_cfg = legacy._training_objects(
        source, torch, np,
    )
    plans = (stage1_plan, stage2_plan, oaci_plan, full_plan)
    fingerprint = _deterministic_prefix_fingerprint(dataset, data, plans, engine_cfg, torch)
    fingerprint_sha = hashlib.sha256(canonical_bytes(fingerprint)).hexdigest()
    units = [
        row for row in protocol.candidate_units()
        if row["dataset"] == dataset and row["source_panel"] == "A"
        and row["training_seed"] == 5 and row["level"] == 0
    ]
    by_key = {(row["regime"], row["epoch"]): row for row in units}
    output_units = []
    ledger.increment("training_phases_started", 3)
    with legacy._OptimizerCapture(root / "optimizer_states") as capture:
        capture.begin("ERM")
        with forked_rng(derive_seed(TRAINING_SEED, dataset, "model_init"), torch.device("cuda:0")):
            model = _model_factory()
        erm = train_stage1(
            model, data, stage1_plan, engine_cfg, torch.device("cuda:0"),
            InvocationRegistry(), f"C84C-V3|{dataset}|A|seed5|level0",
        )
        ledger.increment("training_phases_completed")
        capture.begin("OACI")
        oaci = train_stage2(
            _model_factory, erm, data, OACIObjective(support, adv_hidden=16),
            stage2_plan, oaci_plan, engine_cfg, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        capture.begin("SRC")
        src = train_stage2(
            _model_factory, erm, data, SRCObjective(2, support.n_domains, smooth_temperature=0.1),
            stage2_plan, full_plan, engine_cfg, torch.device("cuda:0"),
        )
        ledger.increment("training_phases_completed")
        if tuple(record.epoch for record in oaci.trajectory) != CHECKPOINT_EPOCHS:
            raise C84CCanaryV2Error("OACI checkpoint cadence drift")
        if tuple(record.epoch for record in src.trajectory) != CHECKPOINT_EPOCHS:
            raise C84CCanaryV2Error("SRC checkpoint cadence drift")
        records = [("ERM", erm.checkpoint, 0, capture.descriptor("ERM", 0))]
        records.extend(("OACI", record, order, capture.descriptor("OACI", order))
                       for order, record in enumerate(oaci.trajectory, start=1))
        records.extend(("SRC", record, order, capture.descriptor("SRC", order))
                       for order, record in enumerate(src.trajectory, start=1))
        previous_hash = {"ERM": erm.checkpoint.model_hash, "OACI": erm.checkpoint.model_hash,
                         "SRC": erm.checkpoint.model_hash}
        for regime, record, order, optimizer_descriptor in records:
            unit = by_key[(regime, int(record.epoch))]
            checkpoint_path = root / "checkpoints" / f"{unit['unit_id']}.pt"
            checkpoint = legacy._save_torch_state(checkpoint_path, record.model_state, torch)
            checkpoint_replay = runtime.replay_checkpoint(
                checkpoint_path, expected_file_sha256=checkpoint["sha256"],
                expected_state_hash=record.model_hash, torch=torch, state_hash_fn=state_hash,
            )
            optimizer_replay = runtime.replay_optimizer_state(
                optimizer_descriptor, phase=regime, trajectory_order=order, torch=torch,
            )
            source_descriptor, target_descriptor = _instrument_and_replay(
                record.model_state, source_audit, target, unit, root, torch, np,
            )
            ledger.increment("source_audit_artifacts")
            ledger.increment("target_unlabeled_artifacts")
            sidecar = {
                "schema_version": "c84c_candidate_sidecar_v2",
                "unit_id": unit["unit_id"], "dataset": dataset, "panel": "A", "seed": 5, "level": 0,
                "regime": regime, "epoch": int(record.epoch), "trajectory_order": order,
                "interface_id": INTERFACE_ID, "montage_sha256": MONTAGE_SHA256, "epoch_rule": EPOCH_RULE,
                "checkpoint": {**checkpoint, "replay": checkpoint_replay},
                "optimizer": {**optimizer_descriptor, "replay": optimizer_replay},
                "source_audit": source_descriptor, "target_unlabeled": target_descriptor,
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous_hash[regime],
                "genealogy_rule": "shared_ERM_parent_then_fixed_regime_trajectory_order",
                "support_hash": support.support_hash(), "target_subject": target.target_subject_id,
                "target_fit_ids": [], "deterministic_prefix_sha256": fingerprint_sha,
                "training_target_rows": 0, "training_target_labels": 0,
                "source_audit_rows_used_in_training": 0, "target_outcome_retention": 0,
                "target_outcome_retry": 0, "target_scientific_metrics": 0,
            }
            if set(sidecar) != SIDECAR_FIELDS or SCIENTIFIC_OUTPUT_KEYS & set(sidecar):
                raise C84CCanaryV2Error("candidate sidecar field contract drift")
            sidecar_path = root / "sidecars" / f"{unit['unit_id']}.json"
            runtime.write_json_atomic(sidecar_path, sidecar)
            sidecar_replay = runtime.replay_sidecar(
                sidecar_path, expected_fields=SIDECAR_FIELDS,
                expected_identity={"unit_id": unit["unit_id"], "dataset": dataset, "regime": regime,
                                   "epoch": int(record.epoch), "trajectory_order": order},
            )
            output_units.append({
                "unit_id": unit["unit_id"], "dataset": dataset, "regime": regime,
                "epoch": int(record.epoch), "trajectory_order": order,
                "model_state_hash": record.model_hash,
                "parent_ERM_model_state_hash": erm.checkpoint.model_hash,
                "previous_trajectory_model_state_hash": previous_hash[regime],
                "checkpoint_sha256": checkpoint["sha256"], "checkpoint_replay_pass": True,
                "optimizer_sha256": optimizer_descriptor["file_sha256"], "optimizer_replay_pass": True,
                "source_audit_sha256": source_descriptor["sha256"], "source_audit_replay_pass": True,
                "target_unlabeled_sha256": target_descriptor["sha256"], "target_unlabeled_replay_pass": True,
                "sidecar_sha256": sidecar_replay["sha256"], "sidecar_replay_pass": True,
            })
            ledger.increment("complete_units")
            previous_hash[regime] = record.model_hash
    if len(output_units) != 81:
        raise C84CCanaryV2Error(f"{dataset} produced {len(output_units)} units instead of 81")
    for regime in ("OACI", "SRC"):
        rows = [row for row in output_units if row["regime"] == regime]
        for previous, current in zip(rows, rows[1:]):
            if current["previous_trajectory_model_state_hash"] != previous["model_state_hash"]:
                raise C84CCanaryV2Error(f"persisted genealogy drift in {dataset}/{regime}")
    return {
        "dataset": dataset,
        "units": output_units,
        "unit_count": len(output_units),
        "source_training_subjects": sorted(set(source.subject_id)),
        "source_audit_subjects": sorted(set(source_audit.subject_id)),
        "target_subjects": [target.target_subject_id],
        "source_training_trial_count": len(source.trial_id),
        "source_audit_trial_count": len(source_audit.trial_id),
        "target_unlabeled_trial_count": len(target.trial_id),
        "source_training_epoch_interface": source.interface.as_dict(),
        "source_audit_epoch_interface": source_audit.interface.as_dict(),
        "target_unlabeled_epoch_interface": target.interface.as_dict(),
        "deterministic_prefix": fingerprint,
        "deterministic_prefix_sha256": fingerprint_sha,
        "source_audit_rows_used_in_training": 0,
        "target_label_access": 0,
        "scientific_metrics": 0,
    }


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
        ledger.stage("package_imports_and_exact_version_replay")
        import numpy as np
        import torch
        import mne
        import moabb

        ledger.increment("package_imports", 4)
        runtime.verify_protected_runtime_versions(binding["lock"], torch=torch, mne=mne, moabb=moabb)
        ledger.stage("CUDA_and_determinism_check")
        ledger.increment("CUDA_checks")
        if not torch.cuda.is_available() or os.environ.get("SLURM_JOB_ID") is None:
            raise C84CCanaryV2Error("C84C V3 requires an authorized Slurm CUDA allocation")
        torch.use_deterministic_algorithms(True, warn_only=False)
        if not torch.are_deterministic_algorithms_enabled():
            raise C84CCanaryV2Error("torch deterministic algorithms are not enabled")

        ledger.stage("loader_source_identity_replay")
        runtime.verify_loader_source_files(binding["lock"])
        ledger.increment("loader_source_replays")
        objects, Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = _protected_loader_objects()
        ledger.increment("dataset_loader_imports")
        runtime.verify_loader_runtime_objects(binding["lock"], objects)
        loader_classes = (Lee2019_MI, Cho2017, PhysionetMI, MotorImagery)

        datasets = []
        for dataset in protocol.DATASET_ORDER:
            ledger.stage(f"dataset_access:{dataset}")
            source, audit, target = _load_canary_views(dataset, np, loader_classes, ledger)
            ledger.stage(f"training_and_instrumentation:{dataset}")
            datasets.append(_run_dataset_training(
                dataset, source, audit, target, Path(binding["run_root"]) / dataset,
                torch, np, ledger,
            ))
            ledger.publish_partial_manifest("IN_PROGRESS")

        unit_rows = [row for dataset in datasets for row in dataset["units"]]
        complete_gate = runtime.validate_complete_canary_gate(unit_rows)
        manifest = {
            "schema_version": "c84c_complete_canary_manifest_v2",
            "execution_lock_v2_sha256": binding["lock_sha256"],
            "canary_protocol_v3_sha256": binding["lock"]["canary_protocol"]["sha256"],
            "repair_protocol_sha256": binding["lock"]["repair_protocol"]["sha256"],
            "authorization_consumption_sha256": consumption["sha256"],
            "interface_id": INTERFACE_ID,
            "montage_sha256": MONTAGE_SHA256,
            "datasets": datasets,
            "complete_gate": complete_gate,
            "unit_count": len(unit_rows),
            "training_phases": 9,
            "source_audit_artifacts": ledger.counters["source_audit_artifacts"],
            "target_unlabeled_artifacts": ledger.counters["target_unlabeled_artifacts"],
            "target_label_access": ledger.counters["target_y_accesses"],
            "target_scientific_metrics": ledger.counters["target_scientific_metrics"],
            "C84F_authorized": False,
            "C84S_authorized": False,
        }
        manifest_path = Path(binding["run_root"]) / "C84C_COMPLETE_CANARY_MANIFEST.json"
        runtime.write_json_atomic(manifest_path, manifest)
        manifest_sha = runtime.sha256_file(manifest_path)
        ledger.stage("manifest_publication")
        ledger.complete(manifest_sha)
        return {**manifest, "manifest_path": str(manifest_path), "manifest_sha256": manifest_sha}
    except Exception as exc:
        ledger.fail(exc)
        raise


def synthetic_schema_dry_run() -> dict[str, Any]:
    units = [row for row in protocol.candidate_units() if row["canary_subset"]]
    payload = {
        "schema_version": "c84c_schema_dry_run_v2",
        "channels": list(EXPECTED_CHANNELS),
        "montage_sha256": MONTAGE_SHA256,
        "input_shape": [20, 480],
        "canary_units": len(units),
        "canary_unit_ids_sha256": runtime.canary_unit_digest(row["unit_id"] for row in units),
        "source_audit_fields": sorted(runtime.SOURCE_AUDIT_FIELDS),
        "target_unlabeled_fields": sorted(runtime.TARGET_UNLABELED_FIELDS),
        "target_view_dataclass_fields": list(TargetUnlabeledView.__dataclass_fields__),
        "target_y_field_present": "y" in TargetUnlabeledView.__dataclass_fields__,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "authorization_consumed": False,
    }
    if payload["canary_units"] != 243 or payload["target_y_field_present"]:
        raise C84CCanaryV2Error("C84C V3 dry-run contract failed")
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="C84C V3 fail-closed engineering canary")
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("show-contract")
    subparsers.add_parser("schema-dry-run")
    real = subparsers.add_parser("run-real")
    real.add_argument("--authorization-record", type=Path, default=AUTHORIZATION_RECORD_PATH)
    real.add_argument("--output-root", type=Path, default=DEFAULT_EXTERNAL_ROOT)
    args = parser.parse_args(argv)
    if args.command == "show-contract":
        print(json.dumps({
            "stage": "C84C", "protocol": str(CANARY_PROTOCOL_PATH), "lock": str(EXECUTION_LOCK_PATH),
            "authorization_required": True, "magic_token_required": False,
            "montage_sha256": MONTAGE_SHA256, "target_y_access": 0,
            "runtime_bound_bytes_replayed_before_output_root": True,
            "attempt_ledger_before_protected_imports": True,
        }, sort_keys=True))
        return 0
    if args.command == "schema-dry-run":
        print(json.dumps(synthetic_schema_dry_run(), sort_keys=True))
        return 0
    print(json.dumps(run_real(
        authorization_path=args.authorization_record, output_root=args.output_root,
    ), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
