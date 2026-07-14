"""Target-unlabeled registry and complete instrumentation for C84F.

This module contains no training entrypoint and imports numerical/loader
packages only inside protected functions.  Every public data-access function
requires a replayed complete model-field manifest first.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable, Mapping, Sequence

from . import c84_dataset_registry_v2 as dataset_registry
from . import c84fl2_protocol as protocol
from . import c84f_field_manifest as manifests


CLASS_NAMES = ("left_hand", "right_hand")
EXPECTED_CHANNELS = (
    "FC5", "FC3", "FC1", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)
TARGET_NPZ_FIELDS = frozenset({
    "unit_id", "dataset", "panel", "training_seed", "level",
    "level_intervention_id", "regime", "epoch", "trajectory_order",
    "target_subject_id", "target_trial_id", "session", "run",
    "logits", "probabilities", "z", "Wz_plus_b", "classifier_weight",
    "classifier_bias", "repeat_logits", "repeat_z",
})


class C84FTargetInstrumentationError(manifests.C84FManifestError):
    """Raised on protected target access or complete-field replay failure."""


@dataclass(frozen=True)
class TargetSubjectView:
    dataset: str
    subject_id: int
    X: Any
    trial_id: tuple[str, ...]
    session: tuple[str, ...]
    run: tuple[str, ...]
    raw_files: tuple[Mapping[str, Any], ...]
    interface: Mapping[str, Any]


def require_model_field_barrier(
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    *,
    scope: manifests.FieldScope = manifests.REAL_SCOPE,
) -> dict[str, Any]:
    return manifests.verify_model_field_freeze(
        model_manifest_path, model_manifest_sha_path, scope=scope,
    )


def _metadata_column(metadata: Any, name: str, default: str) -> list[Any]:
    if hasattr(metadata, "columns") and name in metadata.columns:
        return metadata[name].tolist()
    return [default] * len(metadata)


def _target_metadata(metadata: Any, dataset: str, subject: int) -> dict[str, tuple[Any, ...]]:
    columns = {str(column).strip().lower() for column in getattr(metadata, "columns", ())}
    forbidden = {"y", "target", "targets"}
    if any(column in forbidden or any(token in column for token in ("label", "class", "event"))
           for column in columns):
        raise C84FTargetInstrumentationError("target metadata contains a label-like field")
    subjects = tuple(int(value) for value in _metadata_column(metadata, "subject", str(subject)))
    if set(subjects) != {int(subject)}:
        raise C84FTargetInstrumentationError(
            f"target loader subject drift: observed={sorted(set(subjects))} expected={subject}"
        )
    sessions = tuple(str(value) for value in _metadata_column(metadata, "session", "0"))
    runs = tuple(str(value) for value in _metadata_column(metadata, "run", "0"))
    trial_ids = tuple(
        f"{dataset}|subject={subjects[index]}|session={sessions[index]}|run={runs[index]}|trial={index:05d}"
        for index in range(len(subjects))
    )
    if len(set(trial_ids)) != len(trial_ids):
        raise C84FTargetInstrumentationError("target trial IDs are not unique")
    return {"subject": subjects, "session": sessions, "run": runs, "trial_id": trial_ids}


def _epoch_array(epochs: Any, np: Any) -> tuple[Any, dict[str, Any]]:
    channels = tuple(str(value) for value in epochs.ch_names)
    if channels != EXPECTED_CHANNELS:
        raise C84FTargetInstrumentationError(f"target channel order drift: {channels}")
    sfreq = float(epochs.info["sfreq"])
    if sfreq != 160.0:
        raise C84FTargetInstrumentationError(f"target sampling-rate drift: {sfreq}")
    bads = tuple(str(value) for value in epochs.info.get("bads", ()))
    if bads:
        raise C84FTargetInstrumentationError(f"target Epochs has bad/synthesized channels: {bads}")
    try:
        values = epochs.get_data(copy=True)
    except TypeError:
        values = epochs.get_data()
    array = np.asarray(values)
    times = np.asarray(epochs.times, dtype=np.float64)
    if array.ndim != 3 or array.shape[1] != 20 or array.shape[2] not in {480, 481}:
        raise C84FTargetInstrumentationError(f"target Epochs shape drift: {array.shape}")
    if len(times) != array.shape[2] or abs(float(times[0])) > 1e-9:
        raise C84FTargetInstrumentationError("target Epochs time-axis drift")
    pre_n_times = int(array.shape[2])
    if pre_n_times == 481:
        if abs(float(times[-1]) - 3.0) > 1e-9:
            raise C84FTargetInstrumentationError("target inclusive endpoint is not 3.0 seconds")
        array = array[:, :, :480]
        times = times[:480]
    expected_last = 479.0 / 160.0
    if array.shape[2] != 480 or abs(float(times[-1]) - expected_last) > 1e-9:
        raise C84FTargetInstrumentationError("target half-open [0,3) interface drift")
    array = np.asarray(array, dtype=np.float32)
    if not np.isfinite(array).all():
        raise C84FTargetInstrumentationError("target EEG contains non-finite values")
    mean = array.mean(axis=2, keepdims=True, dtype=np.float64)
    std = array.std(axis=2, keepdims=True, dtype=np.float64)
    if np.any(std <= 1e-8):
        raise C84FTargetInstrumentationError("target EEG has near-zero within-trial variance")
    normalized = np.asarray((array - mean) / std, dtype=np.float32)
    return normalized, {
        "channels": list(channels), "sample_rate_hz": sfreq,
        "pre_half_open_n_times": pre_n_times, "sample_count": 480,
        "first_time_s": float(times[0]), "last_time_s": float(times[-1]),
        "bad_channels": [], "interpolation_or_synthesis": False,
    }


def target_view_from_loader_result(
    result: Any,
    *,
    dataset: str,
    subject: int,
    raw_files: Sequence[Mapping[str, Any]],
    np: Any,
) -> TargetSubjectView:
    """Use tuple slots 0 and 2 only; slot 1 remains structurally untouched."""
    if not isinstance(result, tuple) or len(result) != 3:
        raise C84FTargetInstrumentationError("target loader result must be a three-slot tuple")
    metadata = result[2]
    keys = _target_metadata(metadata, dataset, subject)
    X, interface = _epoch_array(result[0], np)
    del result
    if len(keys["trial_id"]) != X.shape[0]:
        raise C84FTargetInstrumentationError("target metadata and EEG rows differ")
    return TargetSubjectView(
        dataset=dataset, subject_id=int(subject), X=X,
        trial_id=keys["trial_id"], session=keys["session"], run=keys["run"],
        raw_files=tuple(dict(row) for row in raw_files), interface=interface,
    )


def _flatten_paths(value: Any) -> list[Path]:
    if isinstance(value, (str, os.PathLike)):
        return [Path(value)]
    if isinstance(value, Mapping):
        paths = []
        for child in value.values():
            paths.extend(_flatten_paths(child))
        return paths
    if isinstance(value, Iterable):
        paths = []
        for child in value:
            paths.extend(_flatten_paths(child))
        return paths
    return []


def raw_file_identities(paths: Iterable[str | Path]) -> tuple[dict[str, Any], ...]:
    unique = sorted({Path(path).resolve() for path in paths})
    if not unique:
        raise C84FTargetInstrumentationError("loader exposed no raw input paths")
    rows = []
    for path in unique:
        if not path.is_file() or path.stat().st_size <= 0:
            raise C84FTargetInstrumentationError(f"raw input file is absent/empty: {path}")
        rows.append({
            "path": str(path), "bytes": path.stat().st_size,
            "sha256": manifests.sha256_file(path),
        })
    return tuple(rows)


def _dataset_and_paradigm(dataset_code: str, loader_classes: tuple[Any, Any, Any, Any]) -> tuple[Any, Any]:
    Lee2019_MI, Cho2017, PhysionetMI, MotorImagery = loader_classes
    factories = {
        "Lee2019_MI": lambda: Lee2019_MI(train_run=True, test_run=False),
        "Cho2017": Cho2017,
        "PhysionetMI": lambda: PhysionetMI(imagined=True, executed=False),
    }
    if dataset_code not in factories:
        raise C84FTargetInstrumentationError(f"dataset outside C84F lock: {dataset_code}")
    paradigm = MotorImagery(
        n_classes=2, events=list(CLASS_NAMES), fmin=4.0, fmax=38.0,
        tmin=0.0, tmax=3.0, channels=list(EXPECTED_CHANNELS), resample=160,
    )
    return factories[dataset_code](), paradigm


def load_complete_target_views(
    *,
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    loader_classes: tuple[Any, Any, Any, Any],
    np: Any,
    ledger: Any,
) -> tuple[dict[str, tuple[TargetSubjectView, ...]], list[dict[str, Any]]]:
    require_model_field_barrier(model_manifest_path, model_manifest_sha_path)
    output: dict[str, tuple[TargetSubjectView, ...]] = {}
    raw_manifest: dict[tuple[str, str], dict[str, Any]] = {}
    for dataset_code, spec in dataset_registry.DATASETS.items():
        dataset, paradigm = _dataset_and_paradigm(dataset_code, loader_classes)
        subjects = tuple(int(value) for value in dataset_registry.partition_subjects(spec)["targets"])
        views = []
        for subject in subjects:
            ledger.increment("target_get_data_calls")
            raw_value = dataset.data_path(subject, force_update=False, update_path=False, verbose=False)
            raw_rows = raw_file_identities(_flatten_paths(raw_value))
            for row in raw_rows:
                raw_manifest[(dataset_code, row["path"])] = {"dataset": dataset_code, **row}
            result = paradigm.get_data(dataset=dataset, subjects=[subject], return_epochs=True)
            view = target_view_from_loader_result(
                result, dataset=dataset_code, subject=subject, raw_files=raw_rows, np=np,
            )
            ledger.increment("target_EEG_arrays")
            views.append(view)
        if len(views) != protocol.TARGET_COUNTS[dataset_code]:
            raise C84FTargetInstrumentationError(f"target subject count drift for {dataset_code}")
        output[dataset_code] = tuple(views)
    return output, sorted(raw_manifest.values(), key=lambda row: (row["dataset"], row["path"]))


def target_trial_registry_rows(views: Mapping[str, Sequence[TargetSubjectView]]) -> list[dict[str, Any]]:
    rows = []
    for dataset in protocol.DATASETS:
        for view in views.get(dataset, ()):
            raw_list = sorted(dict(row) for row in view.raw_files)
            raw_digest = hashlib.sha256(manifests.canonical_bytes(raw_list)).hexdigest()
            raw_paths = "|".join(row["path"] for row in raw_list)
            raw_bytes = sum(int(row["bytes"]) for row in raw_list)
            for index, trial_id in enumerate(view.trial_id):
                rows.append({
                    "dataset": dataset, "target_subject_id": view.subject_id,
                    "target_trial_id": trial_id, "session": view.session[index], "run": view.run[index],
                    "interface_id": protocol.INTERFACE_ID,
                    "montage_sha256": protocol.HASHES["montage"],
                    "sample_rate_hz": 160, "sample_count": 480, "finite_value_flag": 1,
                    "raw_input_path": raw_paths, "raw_input_bytes": raw_bytes,
                    "raw_input_sha256": raw_digest,
                })
    manifests.validate_target_trial_rows(rows)
    return rows


def _atomic_save_npz(path: Path, np: Any, **arrays: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    os.close(descriptor)
    try:
        np.savez_compressed(temporary, **arrays)
        with open(temporary, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _model_factory() -> Any:
    from oaci.models import build_model

    return build_model(
        "shallow_convnet", in_chans=20, in_times=480, n_classes=2,
        temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15,
        dropout=0.5, safe_log_eps=1e-6,
    )


def _forward(model: Any, X: Any, torch: Any) -> tuple[Any, Any]:
    logits = []
    features = []
    with torch.inference_mode():
        for start in range(0, X.shape[0], 1024):
            output = model(torch.as_tensor(X[start:start + 1024], device="cuda:0"))
            logits.append(output.logits.detach().cpu())
            features.append(output.z.detach().cpu())
    return torch.cat(logits), torch.cat(features)


def validate_numerical_errors(errors: Mapping[str, Any]) -> dict[str, Any]:
    expected = {
        "linear_in_memory_max_abs_error", "linear_persisted_max_abs_error",
        "softmax_max_abs_error", "repeat_logits_max_abs_error", "repeat_z_max_abs_error",
    }
    if set(errors) != expected:
        raise C84FTargetInstrumentationError("target numerical error registry field-set drift")
    observed = {key: float(errors[key]) for key in expected}
    manifests.validate_finite_error(
        observed["linear_in_memory_max_abs_error"], tolerance=protocol.LINEAR_TOLERANCE,
        name="linear in-memory replay",
    )
    manifests.validate_finite_error(
        observed["linear_persisted_max_abs_error"], tolerance=protocol.LINEAR_TOLERANCE,
        name="linear persisted replay",
    )
    for key in ("softmax_max_abs_error", "repeat_logits_max_abs_error", "repeat_z_max_abs_error"):
        manifests.validate_finite_error(observed[key], tolerance=protocol.STRICT_TOLERANCE, name=key)
    return {
        **observed,
        "linear_replay_abs_tolerance": protocol.LINEAR_TOLERANCE,
        "strict_identity_abs_tolerance": protocol.STRICT_TOLERANCE,
        "validation_pass": True,
    }


def replay_target_artifact(
    path: str | Path,
    *,
    unit: Mapping[str, Any],
    trial_ids: Sequence[str],
    np: Any,
) -> dict[str, Any]:
    target = Path(path)
    with np.load(target, allow_pickle=False) as archive:
        if set(archive.files) != TARGET_NPZ_FIELDS:
            raise C84FTargetInstrumentationError("target artifact schema drift or target-label field present")
        logits = np.asarray(archive["logits"])
        probabilities = np.asarray(archive["probabilities"])
        z = np.asarray(archive["z"])
        weight = np.asarray(archive["classifier_weight"])
        bias = np.asarray(archive["classifier_bias"])
        reconstructed = z @ weight.T + bias
        shifted = logits - np.max(logits, axis=1, keepdims=True)
        softmax = np.exp(shifted) / np.sum(np.exp(shifted), axis=1, keepdims=True)
        errors = validate_numerical_errors({
            "linear_in_memory_max_abs_error": float(np.max(np.abs(np.asarray(archive["Wz_plus_b"]) - logits))),
            "linear_persisted_max_abs_error": float(np.max(np.abs(reconstructed - logits))),
            "softmax_max_abs_error": float(np.max(np.abs(softmax - probabilities))),
            "repeat_logits_max_abs_error": float(np.max(np.abs(np.asarray(archive["repeat_logits"]) - logits))),
            "repeat_z_max_abs_error": float(np.max(np.abs(np.asarray(archive["repeat_z"]) - z))),
        })
        if tuple(archive["target_trial_id"].astype(str)) != tuple(map(str, trial_ids)):
            raise C84FTargetInstrumentationError("target artifact trial-ID replay failed")
        scalar_expected = {
            "unit_id": unit["unit_id"], "dataset": unit["dataset"], "panel": unit["panel"],
            "training_seed": int(unit["training_seed"]), "level": int(unit["level"]),
            "level_intervention_id": unit["level_intervention_id"], "regime": unit["regime"],
            "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
        }
        for field, wanted in scalar_expected.items():
            observed = archive[field].item()
            if str(observed) != str(wanted):
                raise C84FTargetInstrumentationError(f"target artifact identity drift: {field}")
        rows = int(logits.shape[0])
    return {"path": str(target), "sha256": manifests.sha256_file(target), "rows": rows, **errors}


def _concatenate_views(views: Sequence[TargetSubjectView], np: Any) -> dict[str, Any]:
    if not views:
        raise C84FTargetInstrumentationError("cannot instrument an empty target population")
    return {
        "X": np.concatenate([view.X for view in views], axis=0),
        "subject": np.concatenate([
            np.full(len(view.trial_id), view.subject_id, dtype=np.int64) for view in views
        ]),
        "trial_id": tuple(value for view in views for value in view.trial_id),
        "session": tuple(value for view in views for value in view.session),
        "run": tuple(value for view in views for value in view.run),
    }


def _load_checkpoint_state(path: Path, torch: Any) -> Mapping[str, Any]:
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    if not isinstance(state, Mapping) or not state:
        raise C84FTargetInstrumentationError(f"checkpoint state is invalid: {path}")
    return state


def instrument_unit(
    *,
    unit: Mapping[str, Any],
    views: Sequence[TargetSubjectView],
    artifact_path: Path,
    context_index_path: Path,
    torch: Any,
    np: Any,
) -> tuple[dict[str, Any], dict[str, Any]]:
    combined = _concatenate_views(views, np)
    model = _model_factory()
    model.load_state_dict(_load_checkpoint_state(Path(unit["checkpoint_path"]), torch))
    model.eval().to("cuda:0")
    logits, z = _forward(model, combined["X"], torch)
    repeat_logits, repeat_z = _forward(model, combined["X"], torch)
    with torch.inference_mode():
        probabilities = torch.softmax(logits, dim=1)
        weight = model.classifier.weight.detach().cpu()
        bias = model.classifier.bias.detach().cpu()
        reconstructed = z @ weight.T + bias
    in_memory = {
        "linear_in_memory_max_abs_error": float(torch.max(torch.abs(reconstructed - logits))),
        "linear_persisted_max_abs_error": float(torch.max(torch.abs(reconstructed - logits))),
        "softmax_max_abs_error": float(torch.max(torch.abs(torch.softmax(logits, dim=1) - probabilities))),
        "repeat_logits_max_abs_error": float(torch.max(torch.abs(repeat_logits - logits))),
        "repeat_z_max_abs_error": float(torch.max(torch.abs(repeat_z - z))),
    }
    validate_numerical_errors(in_memory)
    _atomic_save_npz(
        artifact_path, np,
        unit_id=np.asarray(unit["unit_id"]), dataset=np.asarray(unit["dataset"]),
        panel=np.asarray(unit["panel"]), training_seed=np.asarray(int(unit["training_seed"]), dtype=np.int64),
        level=np.asarray(int(unit["level"]), dtype=np.int64),
        level_intervention_id=np.asarray(unit["level_intervention_id"]), regime=np.asarray(unit["regime"]),
        epoch=np.asarray(int(unit["epoch"]), dtype=np.int64),
        trajectory_order=np.asarray(int(unit["trajectory_order"]), dtype=np.int64),
        target_subject_id=combined["subject"], target_trial_id=np.asarray(combined["trial_id"], dtype=str),
        session=np.asarray(combined["session"], dtype=str), run=np.asarray(combined["run"], dtype=str),
        logits=logits.numpy(), probabilities=probabilities.numpy(), z=z.numpy(),
        Wz_plus_b=reconstructed.numpy(), classifier_weight=weight.numpy(), classifier_bias=bias.numpy(),
        repeat_logits=repeat_logits.numpy(), repeat_z=repeat_z.numpy(),
    )
    replay = replay_target_artifact(
        artifact_path, unit=unit, trial_ids=combined["trial_id"], np=np,
    )
    offsets = []
    start = 0
    for view in views:
        end = start + len(view.trial_id)
        offsets.append({
            "dataset": view.dataset, "target_subject_id": view.subject_id,
            "row_start_inclusive": start, "row_end_exclusive": end,
            "trial_count": end - start,
            "trial_id_sha256": hashlib.sha256(manifests.canonical_bytes(list(view.trial_id))).hexdigest(),
        })
        start = end
    context_payload = {
        "schema_version": "c84f_target_context_index_v1", "unit_id": unit["unit_id"],
        "dataset": unit["dataset"], "contexts": offsets, "context_count": len(offsets),
        "trial_rows": start, "target_label_fields": 0,
    }
    context_sha = manifests.write_json_atomic(context_index_path, context_payload)
    return replay, {"path": str(context_index_path), "sha256": context_sha, **context_payload}


def replay_canary_subset(
    *,
    complete_path: str | Path,
    canary_path: str | Path,
    canary_subject: int,
    np: Any,
) -> dict[str, Any]:
    with np.load(complete_path, allow_pickle=False) as complete, np.load(canary_path, allow_pickle=False) as canary:
        complete_subjects = np.asarray(complete["target_subject_id"], dtype=np.int64)
        indices = np.where(complete_subjects == int(canary_subject))[0]
        if not len(indices):
            raise C84FTargetInstrumentationError("canary subject absent from complete target artifact")
        fields = ("target_trial_id", "logits", "probabilities", "z")
        max_errors = {}
        for field in fields:
            observed = np.asarray(complete[field])[indices]
            expected = np.asarray(canary[field])
            if field == "target_trial_id":
                if tuple(observed.astype(str)) != tuple(expected.astype(str)):
                    raise C84FTargetInstrumentationError("canary target trial-ID replay failed")
                continue
            error = float(np.max(np.abs(observed - expected)))
            manifests.validate_finite_error(error, tolerance=protocol.STRICT_TOLERANCE, name=f"canary {field}")
            max_errors[f"{field}_max_abs_error"] = error
        if str(complete["unit_id"].item()) != str(canary["unit_id"].item()):
            raise C84FTargetInstrumentationError("canary candidate ID replay failed")
    return {
        "required": True, "applicable": True, "passed": True,
        "target_subject_id": int(canary_subject), "strict_tolerance": protocol.STRICT_TOLERANCE,
        **max_errors,
    }


def instrument_complete_field(
    *,
    model_rows: Sequence[Mapping[str, Any]],
    views: Mapping[str, Sequence[TargetSubjectView]],
    reuse_rows: Mapping[str, Mapping[str, Any]],
    output_root: str | Path,
    model_manifest_path: str | Path,
    model_manifest_sha_path: str | Path,
    torch: Any,
    np: Any,
    ledger: Any,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    require_model_field_barrier(model_manifest_path, model_manifest_sha_path)
    if len(model_rows) != protocol.TOTAL_UNITS:
        raise C84FTargetInstrumentationError("complete target instrumentation requires 1,944 models")
    root = Path(output_root)
    descriptors = []
    context_slices = 0
    canary_contexts = 0
    for unit in model_rows:
        dataset_views = tuple(views[unit["dataset"]])
        artifact_path = root / "complete_target_unlabeled" / f"{unit['unit_id']}.npz"
        index_path = root / "target_context_index" / f"{unit['unit_id']}.json"
        target_descriptor, context_descriptor = instrument_unit(
            unit=unit, views=dataset_views, artifact_path=artifact_path,
            context_index_path=index_path, torch=torch, np=np,
        )
        context_slices += len(dataset_views)
        historical = reuse_rows.get(str(unit["unit_id"]))
        if historical is None:
            witness = {"required": True, "applicable": False, "passed": True}
        else:
            witness = replay_canary_subset(
                complete_path=artifact_path, canary_path=historical["canary_target_path"],
                canary_subject=protocol.CANARY_TARGETS[unit["dataset"]], np=np,
            )
            canary_contexts += 1
        descriptors.append({
            "unit_id": unit["unit_id"],
            "checkpoint": {"path": unit["checkpoint_path"], "sha256": unit["checkpoint_sha256"]},
            "optimizer": {"path": unit["optimizer_path"], "sha256": unit["optimizer_sha256"]},
            "training_sidecar": {"path": unit["sidecar_path"], "sha256": unit["sidecar_sha256"]},
            "source_audit": {"path": unit["source_audit_path"], "sha256": unit["source_audit_sha256"]},
            "complete_target_unlabeled": {"path": str(artifact_path), "sha256": target_descriptor["sha256"]},
            "target_context_index": {"path": str(index_path), "sha256": context_descriptor["sha256"]},
            "interface_id": protocol.INTERFACE_ID,
            "protocol_identities": {
                "field_v7": protocol.HASHES.get("field_v7", "bound_by_execution_lock"),
                "full_field_v2": "bound_by_execution_lock",
            },
            "level_intervention_id": unit["level_intervention_id"],
            "model_reuse_provenance": unit["reuse_provenance"],
            "target_artifact_provenance": "C84F",
            "canary_subset_replay": witness,
            "failed_attempt_provenance": {"failed_artifact_reused": False},
        })
        ledger.increment("target_unlabeled_artifacts")
        ledger.increment("target_context_slices", len(dataset_views))
        if historical is not None:
            ledger.increment("canary_contexts_replayed")
        if len(descriptors) % 81 == 0:
            ledger.publish_partial_manifest("IN_PROGRESS")
    if len(descriptors) != 1944 or context_slices != 76464 or canary_contexts != 486:
        raise C84FTargetInstrumentationError(
            f"complete target arithmetic drift: units={len(descriptors)} slices={context_slices} witnesses={canary_contexts}"
        )
    summary = {
        "unit_artifacts": len(descriptors), "target_contexts": context_slices // 81,
        "candidate_context_slices": context_slices, "target_label_fields": 0,
        "target_y_operations": 0, "target_scientific_metrics": 0,
        "training_invocations": 0,
        "linear_replay_abs_tolerance": protocol.LINEAR_TOLERANCE,
        "strict_identity_abs_tolerance": protocol.STRICT_TOLERANCE,
        "canary_contexts_replayed": canary_contexts // 81,
        "canary_unit_artifacts_replayed": canary_contexts,
        "canary_subset_replay_pass": True,
    }
    return descriptors, summary
