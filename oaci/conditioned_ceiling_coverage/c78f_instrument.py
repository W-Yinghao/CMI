"""C78F target-isolated trial instrumentation and physical-view gates."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c74_cache
from . import c78f_full_seed3_field as c78f
from . import c78f_runtime as runtime


DEFAULT_DATALAKE_ROOT = "/projects/EEG-foundation-model/datalake/raw"
SOURCE_INPUT_FIELDS = {"X", "source_trial_id", "source_domain_id", "source_role", "source_class_label"}
TARGET_INPUT_FIELDS = {"X", "target_trial_id", "target_id"}
SOURCE_OUTPUT_FIELDS = {
    "source_trial_id", "source_domain_id", "source_role", "source_class_label",
    "logits", "probabilities", "prediction", "z", "Wz", "Wz_plus_b", "class_margins",
}
TARGET_OUTPUT_FIELDS = {
    "target_trial_id", "target_id", "logits", "probabilities", "prediction",
    "z", "Wz", "Wz_plus_b", "class_margins",
}
TARGET_LABEL_FIELDS = {"target_trial_id", "target_class_label", "split_role"}
FORBIDDEN_TARGET_FIELDS = {
    "target_class_label", "y_true", "correctness", "target_bAcc", "target_NLL",
    "target_ECE", "joint_good", "target_margin", "split_role",
}


def _unicode(values) -> np.ndarray:
    values = [str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _source_role(target: int, domain_id: str) -> str:
    from oaci.confirmatory.loso_plan import loso_fold_spec

    subject = int(str(domain_id).rsplit("-", 1)[-1])
    split = loso_fold_spec(target, dataset_id=c78f.DATASET)
    if subject in split["source_train_subjects"]:
        return "source_train"
    if subject in split["source_audit_subjects"]:
        return "source_audit"
    raise RuntimeError(f"C78F strict-source view contains non-source subject {subject}")


def _load_manifest(field: dict[str, Any]):
    from oaci.protocol.manifest_v2 import load_v2

    item = field["materialized_manifest"]
    if c78f.sha256_file(item["path"]) != item["sha256"]:
        raise RuntimeError("C78F materialized manifest file drift")
    manifest = load_v2(item["path"])
    manifest.validate_complete()
    if manifest.freeze()["sha256"] != item["canonical_sha256"]:
        raise RuntimeError("C78F materialized manifest canonical drift")
    return manifest


def prepare_primary_views(target: int, datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    target = runtime.require_target(target)
    oaci = runtime.require_oaci_field(lock, target)
    runtime.require_src_field(lock, target)
    path = runtime.primary_view_path(lock, target)
    if path.exists():
        return runtime.verify_manifest(path)

    from oaci.data.eeg.bnci import load_moabb_confirmatory

    manifest = _load_manifest(oaci)
    dataset = manifest.enabled_datasets()[c78f.DATASET]
    source_subjects = [subject for subject in range(1, 10) if subject != target]
    source = load_moabb_confirmatory(
        c78f.DATASET, source_subjects, dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    target_bundle = load_moabb_confirmatory(
        c78f.DATASET, [target], dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    if int(source.bundle.n) != 4608 or int(target_bundle.bundle.n) != 576:
        raise RuntimeError("C78F primary-view trial counts drifted")
    if target in source.evidence.subjects or tuple(target_bundle.evidence.subjects) != (target,):
        raise RuntimeError("C78F source/target primary-view subject isolation failed")

    # The MOABB loader bundles labels structurally. This process never indexes,
    # copies, hashes, summarizes, or emits target_bundle.bundle.y. The physically
    # materialized primary view therefore contains X and IDs only.
    root = runtime.target_root(lock, target) / "views" / "primary_inputs"
    source_ids = _unicode(source.bundle.trial_id)
    source_domains = _unicode(source.bundle.subject_id)
    source_descriptor = c74_cache.write_content_addressed_npz(root / "strict_source", "strict_source_input", {
        "X": np.asarray(source.bundle.X, dtype=np.float32),
        "source_trial_id": source_ids,
        "source_domain_id": source_domains,
        "source_role": _unicode([_source_role(target, domain) for domain in source_domains]),
        "source_class_label": np.asarray(source.bundle.y, dtype=np.int16),
    })
    target_descriptor = c74_cache.write_content_addressed_npz(root / "target_unlabeled", "target_unlabeled_input", {
        "X": np.asarray(target_bundle.bundle.X, dtype=np.float32),
        "target_trial_id": _unicode(target_bundle.bundle.trial_id),
        "target_id": np.full(576, target, dtype=np.int16),
    })
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    if set(target_descriptor["fields"]) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78F target-unlabeled input contains forbidden fields")
    gate = runtime.write_manifest(path, {
        "schema_version": "c78f_primary_input_views_v1",
        "created_at_utc": c78f.utc_now(),
        "protocol_sha256": protocol_sha,
        "target": target,
        "created_after_target_checkpoint_field_freeze": True,
        "complete_seed3_field_frozen_at_creation": runtime.full_field_path(lock).exists(),
        "strict_source_input": source_descriptor,
        "target_unlabeled_input": target_descriptor,
        "target_label_values_accessed_by_primary_code": False,
        "target_label_fields_present": False,
        "label_view_path_present": False,
        "same_label_oracle_path_present": False,
        "source_rows": 4608,
        "target_unlabeled_rows": 576,
        "source_loader_evidence_hash": source.evidence.evidence_hash,
        "target_loader_evidence_hash": target_bundle.evidence.evidence_hash,
        "source_network_attempts": source.evidence.network_attempt_count,
        "target_network_attempts": target_bundle.evidence.network_attempt_count,
    })
    return gate


def _class_margins(logits: np.ndarray) -> np.ndarray:
    out = np.empty_like(logits)
    for class_index in range(logits.shape[1]):
        out[:, class_index] = logits[:, class_index] - np.max(np.delete(logits, class_index, axis=1), axis=1)
    return out


def _load_primary(lock: dict[str, Any], target: int) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    gate = runtime.verify_manifest(runtime.primary_view_path(lock, target))
    if gate["target_label_fields_present"] or gate["label_view_path_present"] or gate["same_label_oracle_path_present"]:
        raise RuntimeError("C78F primary input gate exposes target labels")
    source_descriptor = gate["strict_source_input"]
    target_descriptor = gate["target_unlabeled_input"]
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    with np.load(source_descriptor["path"], allow_pickle=False) as source_file:
        source = {name: np.array(source_file[name], copy=True) for name in source_file.files}
    with np.load(target_descriptor["path"], allow_pickle=False) as target_file:
        target_view = {name: np.array(target_file[name], copy=True) for name in target_file.files}
    if set(target_view) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78F target-unlabeled primary arrays contain forbidden fields")
    return source, target_view, gate


def _existing(path: Path, checkpoint_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = runtime.verify_manifest(path)
    if payload["checkpoint_id"] != checkpoint_id or not payload["all_gates_passed"]:
        raise RuntimeError(f"invalid existing C78F unit instrumentation: {path}")
    for descriptor in payload["shards"]:
        c74_cache.verify_shard(descriptor)
    return payload


def _instrument_unit(
    *, lock: dict[str, Any], target: int, unit: dict[str, Any],
    source: dict[str, np.ndarray], target_view: dict[str, np.ndarray], threads: int,
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional
    from oaci.models import build_model
    from oaci.train.checkpoint import state_hash

    torch.set_num_threads(max(1, int(threads)))
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    directory = runtime.target_root(lock, target) / "instrumentation" / "units" / unit["unit_id"]
    manifest_path = directory / "unit_manifest.json"
    existing = _existing(manifest_path, unit["checkpoint_id"])
    if existing is not None:
        return existing
    started_wall = time.time()
    started_cpu = time.process_time()
    state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
    if state_hash(state) != unit["checkpoint_id"]:
        raise RuntimeError(f"C78F checkpoint state mismatch: {unit['unit_id']}")
    model = build_model(
        "shallow_convnet", in_chans=22, in_times=385, n_classes=4,
        temporal_filters=40, temporal_kernel_samples=25, pool_kernel_samples=75,
        pool_stride_samples=15, dropout=0.5, safe_log_eps=1e-6,
    ).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    weight = model.classifier.weight.detach().cpu().to(torch.float32)
    bias = model.classifier.bias.detach().cpu().to(torch.float32)
    hooks: list[torch.Tensor] = []

    def pre_hook(_module, args):
        hooks.append(args[0].detach().cpu())

    handle = model.classifier.register_forward_pre_hook(pre_hook)

    def infer(X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        count = int(X.shape[0])
        logits = np.empty((count, 4), dtype=np.float32)
        probabilities = np.empty((count, 4), dtype=np.float32)
        predictions = np.empty(count, dtype=np.int16)
        z = np.empty((count, 800), dtype=np.float32)
        wz = np.empty((count, 4), dtype=np.float32)
        reconstructed = np.empty((count, 4), dtype=np.float32)
        maxima = {"identity_abs": 0.0, "softmax_abs": 0.0, "hook_abs": 0.0, "repeat_logits": 0.0, "repeat_z": 0.0}
        with torch.no_grad():
            for start in range(0, count, 128):
                stop = min(start + 128, count)
                x = torch.from_numpy(np.ascontiguousarray(X[start:stop])).to(torch.float32)
                hooks.clear()
                output = model(x)
                captured = hooks[0]
                if start == 0:
                    first_logits = output.logits.detach().clone()
                    first_z = output.z.detach().clone()
                    repeated = model(x)
                    maxima["repeat_logits"] = float(torch.max(torch.abs(first_logits - repeated.logits)).item())
                    maxima["repeat_z"] = float(torch.max(torch.abs(first_z - repeated.z)).item())
                projection = functional.linear(output.z, weight, bias=None)
                replay = projection + bias
                probs = torch.softmax(output.logits, dim=1)
                maxima["identity_abs"] = max(maxima["identity_abs"], float(torch.max(torch.abs(replay - output.logits)).item()))
                maxima["softmax_abs"] = max(maxima["softmax_abs"], float(torch.max(torch.abs(torch.softmax(replay, dim=1) - probs)).item()))
                maxima["hook_abs"] = max(maxima["hook_abs"], float(torch.max(torch.abs(captured - output.z)).item()))
                logits[start:stop] = output.logits.numpy()
                probabilities[start:stop] = probs.numpy()
                predictions[start:stop] = torch.argmax(output.logits, dim=1).numpy()
                z[start:stop] = output.z.numpy()
                wz[start:stop] = projection.numpy()
                reconstructed[start:stop] = replay.numpy()
        return {"logits": logits, "probabilities": probabilities, "prediction": predictions, "z": z, "Wz": wz, "Wz_plus_b": reconstructed, "class_margins": _class_margins(logits)}, maxima

    try:
        source_output, source_identity = infer(source["X"])
        target_output, target_identity = infer(target_view["X"])
    finally:
        handle.remove()
    identity = {key: max(source_identity[key], target_identity[key]) for key in source_identity}
    passed = (
        identity["identity_abs"] <= 1e-6 and identity["softmax_abs"] <= 1e-7
        and identity["hook_abs"] <= 1e-6 and identity["repeat_logits"] == 0.0
        and identity["repeat_z"] == 0.0 and state_hash(model.state_dict()) == unit["checkpoint_id"]
    )
    if not passed:
        raise RuntimeError(f"C78F instrumentation identity failed: {unit['unit_id']} {identity}")
    source_arrays = {key: source[key] for key in ("source_trial_id", "source_domain_id", "source_role", "source_class_label")}
    source_arrays.update(source_output)
    target_arrays = {key: target_view[key] for key in ("target_trial_id", "target_id")}
    target_arrays.update(target_output)
    shards = [
        c74_cache.write_content_addressed_npz(directory, "checkpoint_Wb", {"W": weight.numpy(), "b": bias.numpy()}),
        c74_cache.write_content_addressed_npz(directory, "strict_source_trial", source_arrays),
        c74_cache.write_content_addressed_npz(directory, "target_unlabeled_trial", target_arrays),
    ]
    schemas = {"checkpoint_Wb": {"W", "b"}, "strict_source_trial": SOURCE_OUTPUT_FIELDS, "target_unlabeled_trial": TARGET_OUTPUT_FIELDS}
    for descriptor in shards:
        c74_cache.verify_shard(descriptor, required_fields=schemas[descriptor["kind"]])
    target_fields = set(next(item["fields"] for item in shards if item["kind"] == "target_unlabeled_trial"))
    if target_fields & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78F target-unlabeled output contains forbidden target fields")
    return runtime.write_manifest(manifest_path, {
        "schema_version": "c78f_instrumented_unit_manifest_v1",
        "unit_id": unit["unit_id"], "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_file_sha256": unit["checkpoint_file_sha256"], "sidecar_sha256": unit["sidecar_sha256"],
        "target": target, "seed": c78f.SEED, "level": int(unit["level"]), "regime": unit["regime"],
        "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
        "source_rows": 4608, "target_unlabeled_rows": 576,
        "identity": {**identity, "passed": passed},
        "execution": {"CPU_only": True, "model_eval": not model.training, "gradients_enabled_during_forward": False, "training_attempted": False, "parameter_updates": False, "GPU_used": False, "target_labels_visible": False, "same_label_oracle_descriptor_visible": False, "wall_seconds": time.time() - started_wall, "process_CPU_seconds": time.process_time() - started_cpu, "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown")},
        "shards": shards, "all_gates_passed": True,
    })


def _worker(payload: tuple[int, list[dict[str, Any]], int]) -> list[str]:
    target, units, threads = payload
    lock, _, _ = runtime.require_authorization()
    source, target_view, _ = _load_primary(lock, target)
    completed = []
    for unit in units:
        _instrument_unit(lock=lock, target=target, unit=unit, source=source, target_view=target_view, threads=threads)
        completed.append(unit["unit_id"])
    return completed


def instrument_target(target: int, workers: int = 4, threads_per_worker: int = 12, datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    job_started = time.time()
    lock, _, protocol_sha = runtime.require_authorization()
    target = runtime.require_target(target)
    if c78f.wave_for_target(target) == "B":
        wave_a = runtime.verify_manifest(runtime.wave_gate_path(lock, "A"))
        if not wave_a["all_engineering_gates_passed"] or wave_a["target_scientific_outcomes_read"]:
            raise PermissionError("C78F Wave-B instrumentation requires Wave-A engineering gate")
    prepare_primary_views(target, datalake_root)
    units = runtime.checkpoint_units(lock, target)
    workers = max(1, min(int(workers), len(units)))
    chunks = [units[index::workers] for index in range(workers)]
    if workers == 1:
        completed = [_worker((target, chunks[0], threads_per_worker))]
    else:
        context = mp.get_context("spawn")
        with context.Pool(processes=workers) as pool:
            completed = pool.map(_worker, [(target, chunk, threads_per_worker) for chunk in chunks])
    completed_ids = [unit_id for chunk in completed for unit_id in chunk]
    if len(completed_ids) != 162 or set(completed_ids) != {row["unit_id"] for row in units}:
        raise RuntimeError("C78F target instrumentation worker coverage drift")

    root = runtime.target_root(lock, target)
    unit_rows = []
    source_rows = target_rows = 0
    maxima = {"identity_abs": 0.0, "softmax_abs": 0.0, "hook_abs": 0.0, "repeat_logits": 0.0, "repeat_z": 0.0}
    wall = cpu = 0.0
    for unit_id in sorted(completed_ids):
        path = root / "instrumentation" / "units" / unit_id / "unit_manifest.json"
        unit = runtime.verify_manifest(path)
        if not unit["all_gates_passed"] or unit["execution"]["target_labels_visible"]:
            raise RuntimeError(f"C78F failed instrumentation unit: {unit_id}")
        for descriptor in unit["shards"]:
            c74_cache.verify_shard(descriptor)
        source_rows += int(unit["source_rows"])
        target_rows += int(unit["target_unlabeled_rows"])
        for key in maxima:
            maxima[key] = max(maxima[key], float(unit["identity"][key]))
        wall += float(unit["execution"]["wall_seconds"])
        cpu += float(unit["execution"]["process_CPU_seconds"])
        unit_rows.append({"unit_id": unit_id, "path": str(path), "sha256": c78f.sha256_file(path)})
    if source_rows != 162 * 4608 or target_rows != 162 * 576:
        raise RuntimeError("C78F target instrumentation row count drift")
    gate = runtime.write_manifest(runtime.instrumentation_path(lock, target), {
        "schema_version": "c78f_target_instrumentation_complete_v1",
        "created_at_utc": c78f.utc_now(),
        "protocol_sha256": protocol_sha, "target": target, "wave": c78f.wave_for_target(target),
        "unit_count": 162, "source_rows": source_rows, "target_unlabeled_rows": target_rows,
        "identity": {**maxima, "failed_units": 0},
        "physical_isolation": {"target_unlabeled_contains_labels": False, "instrumentation_received_label_path": False, "instrumentation_received_oracle_path": False, "source_and_target_inputs_separate": True},
        "workers": workers, "threads_per_worker": threads_per_worker,
        "units": unit_rows,
        "execution": {"job_wall_seconds": time.time() - job_started, "summed_unit_wall_seconds": wall, "summed_unit_process_CPU_seconds": cpu, "external_storage_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()), "GPU_used_for_instrumentation": False, "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown")},
        "all_gates_passed": True,
    })
    print(json.dumps({"gate": "C78F_TARGET_INSTRUMENTATION_COMPLETE", "target": target, "units": 162, "source_rows": source_rows, "target_rows": target_rows}, sort_keys=True))
    return gate


def validate_wave(wave: str) -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    if wave not in {"A", "B"}:
        raise ValueError(wave)
    targets = c78f.wave_targets()[wave]
    target_rows = []
    for target in targets:
        oaci = runtime.require_oaci_field(lock, target)
        src = runtime.require_src_field(lock, target)
        instrumentation = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        passed = (
            oaci["unit_count"] == 82 and src["unit_count"] == 80
            and instrumentation["unit_count"] == 162 and instrumentation["all_gates_passed"]
            and oaci["execution"]["target_label_reads_during_training"] == 0
            and src["execution"]["target_label_reads_during_training"] == 0
            and instrumentation["identity"]["failed_units"] == 0
            and not instrumentation["physical_isolation"]["target_unlabeled_contains_labels"]
        )
        target_rows.append({"target": target, "oaci_erm_units": oaci["unit_count"], "src_units": src["unit_count"], "instrumented_units": instrumentation["unit_count"], "target_training_label_reads": 0, "target_scientific_outcomes_read": 0, "engineering_passed": passed})
    if not all(row["engineering_passed"] for row in target_rows):
        raise RuntimeError(f"C78F Wave {wave} engineering gate failed")
    gate = runtime.write_manifest(runtime.wave_gate_path(lock, wave), {
        "schema_version": "c78f_wave_engineering_gate_v1", "protocol_sha256": protocol_sha,
        "created_at_utc": c78f.utc_now(),
        "wave": wave, "targets": list(targets), "units": 648,
        "target_scientific_outcomes_read": False, "target_labels_read_for_gate": False,
        "continuation_basis": "engineering_only", "target_rows": target_rows,
        "all_engineering_gates_passed": True,
    })
    print(json.dumps({"gate": f"C78F_WAVE_{wave}_ENGINEERING_VALID", "targets": list(targets), "units": 648}, sort_keys=True))
    return gate


def freeze_full_field() -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    for wave in ("A", "B"):
        gate = runtime.verify_manifest(runtime.wave_gate_path(lock, wave))
        if not gate["all_engineering_gates_passed"] or gate["target_scientific_outcomes_read"]:
            raise RuntimeError("C78F cannot freeze field before both engineering-only waves pass")
    target_manifests = []
    remaining_units = 0
    source_rows = target_rows = 0
    for target in c78f.TARGETS:
        oaci = runtime.require_oaci_field(lock, target)
        src = runtime.require_src_field(lock, target)
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        remaining_units += int(oaci["unit_count"]) + int(src["unit_count"])
        source_rows += int(instrument["source_rows"])
        target_rows += int(instrument["target_unlabeled_rows"])
        target_manifests.append({
            "target": target,
            "wave": c78f.wave_for_target(target),
            "oaci_erm_path": str(runtime.oaci_field_path(lock, target)), "oaci_erm_sha256": c78f.sha256_file(runtime.oaci_field_path(lock, target)),
            "src_path": str(runtime.src_field_path(lock, target)), "src_sha256": c78f.sha256_file(runtime.src_field_path(lock, target)),
            "instrumentation_path": str(runtime.instrumentation_path(lock, target)), "instrumentation_sha256": c78f.sha256_file(runtime.instrumentation_path(lock, target)),
        })
    if (remaining_units, source_rows, target_rows) != (c78f.REMAINING_UNITS, c78f.EXPECTED_SOURCE_ROWS, c78f.EXPECTED_TARGET_ROWS):
        raise RuntimeError("C78F complete remaining field count drift")
    gate = runtime.write_manifest(runtime.full_field_path(lock), {
        "schema_version": "c78f_full_seed3_field_frozen_v1",
        "created_at_utc": c78f.utc_now(),
        "protocol_sha256": protocol_sha,
        "C78S_protocol_sha256": c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(),
        "remaining_units": remaining_units, "target4_parent_units": 162, "full_seed3_units": 1458,
        "remaining_source_rows": source_rows, "remaining_target_unlabeled_rows": target_rows,
        "full_source_rows": c78f.FULL_SOURCE_ROWS, "full_target_unlabeled_rows": c78f.FULL_TARGET_ROWS,
        "target4_excluded_from_primary_science": True,
        "target_scientific_outcomes_read": False, "label_views_created": False,
        "seed4_access": 0, "BNCI2014_004_access": 0,
        "target_manifests": target_manifests,
        "all_engineering_gates_passed": True,
    })
    print(json.dumps({"gate": "C78F_FULL_FIELD_FROZEN", "remaining_units": remaining_units, "full_units": 1458, "analysis_started": False}, sort_keys=True))
    return gate


def prepare_label_views(target: int, datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    lock, _, protocol_sha = runtime.require_authorization()
    target = runtime.require_target(target)
    full = runtime.verify_manifest(runtime.full_field_path(lock))
    if full["full_seed3_units"] != 1458 or full["target_scientific_outcomes_read"]:
        raise RuntimeError("C78F label views require a clean complete field freeze")
    path = runtime.label_view_path(lock, target)
    if path.exists():
        return runtime.verify_manifest(path)
    from oaci.data.eeg.bnci import load_moabb_confirmatory

    manifest = _load_manifest(runtime.require_oaci_field(lock, target))
    dataset = manifest.enabled_datasets()[c78f.DATASET]
    loaded = load_moabb_confirmatory(
        c78f.DATASET, [target], dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    trial_ids = _unicode(loaded.bundle.trial_id)
    roles = _unicode([c66._future_split_role(item) for item in trial_ids])
    construct = roles == "target_construct"
    evaluation = roles == "target_eval"
    if not construct.any() or not evaluation.any():
        raise RuntimeError("C78F target label split has an empty side")
    root = runtime.target_root(lock, target) / "views" / "target_labels"
    descriptors = {
        "construction": c74_cache.write_content_addressed_npz(root / "construction", "target_construction", {"target_trial_id": trial_ids[construct], "target_class_label": np.asarray(loaded.bundle.y[construct], dtype=np.int16), "split_role": roles[construct]}),
        "evaluation": c74_cache.write_content_addressed_npz(root / "evaluation", "target_evaluation", {"target_trial_id": trial_ids[evaluation], "target_class_label": np.asarray(loaded.bundle.y[evaluation], dtype=np.int16), "split_role": roles[evaluation]}),
        "same_label_oracle": c74_cache.write_content_addressed_npz(root / "oracle", "same_label_oracle", {"target_trial_id": trial_ids, "target_class_label": np.asarray(loaded.bundle.y, dtype=np.int16), "split_role": roles}),
    }
    for descriptor in descriptors.values():
        c74_cache.verify_shard(descriptor, required_fields=TARGET_LABEL_FIELDS)
    return runtime.write_manifest(path, {
        "schema_version": "c78f_post_full_freeze_label_views_v1", "protocol_sha256": protocol_sha,
        "created_at_utc": c78f.utc_now(),
        "full_field_manifest_sha256": full["manifest_sha256"], "target": target,
        "created_after_complete_1458_unit_field_freeze": True,
        "target_label_rows_loaded": 576, "target_label_views": descriptors,
        "available_to_generation_or_primary_instrumentation": False,
        "scientific_outcomes_computed": False, "checkpoint_recommendations_computed": False,
        "consumer": "future_C78S_registered_view_router_only",
    })


def prepare_all_label_views(datalake_root: str = DEFAULT_DATALAKE_ROOT) -> dict[str, Any]:
    lock, _, _ = runtime.require_authorization()
    manifests = [prepare_label_views(target, datalake_root) for target in c78f.TARGETS]
    full = runtime.verify_manifest(runtime.full_field_path(lock))
    pre_label_sha = full["manifest_sha256"]
    full["label_views_created"] = True
    full["pre_label_full_field_manifest_sha256"] = pre_label_sha
    full["label_view_manifests"] = [{"target": target, "path": str(runtime.label_view_path(lock, target)), "sha256": c78f.sha256_file(runtime.label_view_path(lock, target))} for target in c78f.TARGETS]
    full["scientific_analysis_started"] = False
    updated = runtime.write_manifest(runtime.full_field_path(lock), {key: value for key, value in full.items() if key != "manifest_sha256"})
    print(json.dumps({"gate": "C78F_LABEL_VIEWS_PHYSICALLY_ISOLATED", "targets": 8, "label_rows": 8 * 576, "analysis_started": False}, sort_keys=True))
    return updated


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78f_instrument")
    sub = parser.add_subparsers(dest="command", required=True)
    instrument = sub.add_parser("instrument-target")
    instrument.add_argument("--target", type=int, required=True)
    instrument.add_argument("--workers", type=int, default=4)
    instrument.add_argument("--threads-per-worker", type=int, default=12)
    instrument.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    wave = sub.add_parser("validate-wave")
    wave.add_argument("--wave", choices=("A", "B"), required=True)
    sub.add_parser("freeze-full-field")
    labels = sub.add_parser("prepare-all-label-views")
    labels.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    args = parser.parse_args(argv)
    if args.command == "instrument-target":
        instrument_target(args.target, args.workers, args.threads_per_worker, args.datalake_root)
    elif args.command == "validate-wave":
        validate_wave(args.wave)
    elif args.command == "freeze-full-field":
        freeze_full_field()
    else:
        prepare_all_label_views(args.datalake_root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
