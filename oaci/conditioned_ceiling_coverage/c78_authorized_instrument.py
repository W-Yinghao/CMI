"""Post-freeze view provisioning and trial instrumentation for authorized C78."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c74_cache
from . import c78_authorized_common as common
from . import c78_seed3_instrumented_pilot as c78


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
WB_FIELDS = {"W", "b"}
TARGET_LABEL_FIELDS = {"target_trial_id", "target_class_label", "split_role"}
FORBIDDEN_TARGET_FIELDS = {
    "target_class_label", "y_true", "correctness", "target_bAcc", "target_NLL",
    "target_ECE", "joint_good", "target_margin", "split_role",
}


def _unicode(values) -> np.ndarray:
    values = [str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _source_role(domain_id: str) -> str:
    from oaci.confirmatory.loso_plan import loso_fold_spec

    subject = int(str(domain_id).rsplit("-", 1)[-1])
    split = loso_fold_spec(c78.TARGET, dataset_id=c78.DATASET)
    if subject in split["source_train_subjects"]:
        return "source_train"
    if subject in split["source_audit_subjects"]:
        return "source_audit"
    raise RuntimeError(f"C78 source view contains non-source subject {subject}")


def _load_manifest_from_field(frozen: dict[str, Any]):
    from oaci.protocol.manifest_v2 import load_v2

    item = frozen["materialized_manifest"]
    if c78.sha256_file(item["path"]) != item["sha256"]:
        raise RuntimeError("C78 materialized manifest drift before post-freeze provisioning")
    manifest = load_v2(item["path"])
    manifest.validate_complete()
    if manifest.freeze()["sha256"] != item["canonical_sha256"]:
        raise RuntimeError("C78 materialized manifest canonical identity drift")
    return manifest


def prepare_views(*, authorization_token: str, datalake_root: str) -> dict[str, Any]:
    lock, _, protocol_sha = common.require_authorization(authorization_token)
    frozen = common.require_field_frozen(lock)
    primary_path = common.primary_input_gate_path(lock)
    label_path = common.label_view_gate_path(lock)
    if primary_path.exists() and label_path.exists():
        return common.verify_canonical_manifest(primary_path)

    from oaci.data.eeg.bnci import load_moabb_confirmatory

    root = common.campaign_root(lock)
    manifest = _load_manifest_from_field(frozen)
    dataset = manifest.enabled_datasets()[c78.DATASET]
    source_subjects = [subject for subject in range(1, 10) if subject != c78.TARGET]
    source = load_moabb_confirmatory(
        c78.DATASET, source_subjects, dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    target = load_moabb_confirmatory(
        c78.DATASET, [c78.TARGET], dataset.preprocessing,
        frozen_class_names=dataset.class_names, frozen_channels=dataset.channels,
        expected_sfreq=float(dataset.expected_sfreq), expected_n_times=int(dataset.expected_n_times),
        datalake_root=datalake_root,
    )
    if int(source.bundle.n) != 8 * 576 or int(target.bundle.n) != 576:
        raise RuntimeError(f"C78 post-freeze trial count mismatch: {source.bundle.n}, {target.bundle.n}")
    if c78.TARGET in source.evidence.subjects or tuple(target.evidence.subjects) != (c78.TARGET,):
        raise RuntimeError("C78 source/target provisioning subject isolation failed")

    source_ids = _unicode(source.bundle.trial_id)
    source_domains = _unicode(source.bundle.subject_id)
    source_roles = _unicode([_source_role(domain) for domain in source_domains])
    target_ids = _unicode(target.bundle.trial_id)
    split_roles = _unicode([c66._future_split_role(trial_id) for trial_id in target_ids])
    construct = split_roles == "target_construct"
    evaluation = split_roles == "target_eval"
    if not construct.any() or not evaluation.any():
        raise RuntimeError("C78 post-freeze target split has an empty side")

    input_root = root / "physical_views" / "primary_inputs"
    source_descriptor = c74_cache.write_content_addressed_npz(input_root / "strict_source", "strict_source_input", {
        "X": np.asarray(source.bundle.X, dtype=np.float32),
        "source_trial_id": source_ids,
        "source_domain_id": source_domains,
        "source_role": source_roles,
        "source_class_label": np.asarray(source.bundle.y, dtype=np.int16),
    })
    target_descriptor = c74_cache.write_content_addressed_npz(input_root / "target_unlabeled", "target_unlabeled_input", {
        "X": np.asarray(target.bundle.X, dtype=np.float32),
        "target_trial_id": target_ids,
        "target_id": np.full(int(target.bundle.n), c78.TARGET, dtype=np.int16),
    })
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    if set(target_descriptor["fields"]) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78 target-unlabeled input contains a forbidden target field")

    label_root = root / "physical_views" / "target_labels"
    construction_descriptor = c74_cache.write_content_addressed_npz(label_root / "construction", "target_construction", {
        "target_trial_id": target_ids[construct],
        "target_class_label": np.asarray(target.bundle.y[construct], dtype=np.int16),
        "split_role": split_roles[construct],
    })
    evaluation_descriptor = c74_cache.write_content_addressed_npz(label_root / "evaluation", "target_evaluation", {
        "target_trial_id": target_ids[evaluation],
        "target_class_label": np.asarray(target.bundle.y[evaluation], dtype=np.int16),
        "split_role": split_roles[evaluation],
    })
    oracle_descriptor = c74_cache.write_content_addressed_npz(label_root / "oracle", "same_label_oracle", {
        "target_trial_id": target_ids,
        "target_class_label": np.asarray(target.bundle.y, dtype=np.int16),
        "split_role": split_roles,
    })
    for descriptor in (construction_descriptor, evaluation_descriptor, oracle_descriptor):
        c74_cache.verify_shard(descriptor, required_fields=TARGET_LABEL_FIELDS)

    label_gate = common.write_manifest(label_path, {
        "schema_version": "c78_postfreeze_label_views_v1",
        "protocol_sha256": protocol_sha,
        "field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "created_after_field_freeze": True,
        "target_label_rows_loaded": int(target.bundle.n),
        "target_label_views": {
            "construction": construction_descriptor,
            "evaluation": evaluation_descriptor,
            "same_label_oracle": oracle_descriptor,
        },
        "available_to_primary_instrumentation": False,
        "source_loader_evidence_hash": source.evidence.evidence_hash,
        "target_loader_evidence_hash": target.evidence.evidence_hash,
        "source_network_attempts": source.evidence.network_attempt_count,
        "target_network_attempts": target.evidence.network_attempt_count,
    })
    primary = common.write_manifest(primary_path, {
        "schema_version": "c78_primary_input_views_v1",
        "protocol_sha256": protocol_sha,
        "field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "created_after_field_freeze": True,
        "strict_source_input": source_descriptor,
        "target_unlabeled_input": target_descriptor,
        "target_label_fields_present": False,
        "label_view_gate_path_in_primary_descriptor": False,
        "same_label_oracle_path_in_primary_descriptor": False,
        "primary_instrumentation_allowed_fields": {
            "source": sorted(SOURCE_INPUT_FIELDS),
            "target_unlabeled": sorted(TARGET_INPUT_FIELDS),
        },
    })
    print(json.dumps({
        "gate": "POSTFREEZE_VIEWS_READY", "source_rows": int(source.bundle.n),
        "target_rows": int(target.bundle.n),
        "construction_rows": int(construct.sum()), "evaluation_rows": int(evaluation.sum()),
        "label_gate_sha256": label_gate["manifest_sha256"],
    }, sort_keys=True))
    return primary


def _class_margins(logits: np.ndarray) -> np.ndarray:
    count, classes = logits.shape
    out = np.empty_like(logits)
    for class_index in range(classes):
        other = np.delete(logits, class_index, axis=1)
        out[:, class_index] = logits[:, class_index] - np.max(other, axis=1)
    return out


def _load_primary_inputs(lock: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    gate = common.verify_canonical_manifest(common.primary_input_gate_path(lock))
    if gate["target_label_fields_present"] or gate["same_label_oracle_path_in_primary_descriptor"]:
        raise RuntimeError("C78 primary input gate exposes target labels or oracle path")
    source_descriptor = gate["strict_source_input"]
    target_descriptor = gate["target_unlabeled_input"]
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    with np.load(source_descriptor["path"], allow_pickle=False) as source_file:
        source = {name: np.array(source_file[name], copy=True) for name in source_file.files}
    with np.load(target_descriptor["path"], allow_pickle=False) as target_file:
        target = {name: np.array(target_file[name], copy=True) for name in target_file.files}
    if set(target) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78 primary target input acquired a forbidden field")
    return source, target, gate


def _existing_unit(path: Path, expected_checkpoint: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = common.verify_canonical_manifest(path)
    if payload["checkpoint_id"] != expected_checkpoint or not payload["all_gates_passed"]:
        raise RuntimeError(f"invalid existing C78 instrumentation unit: {path}")
    for descriptor in payload["shards"]:
        c74_cache.verify_shard(descriptor)
    return payload


def _instrument_unit(
    *, lock: dict[str, Any], unit: dict[str, Any],
    source: dict[str, np.ndarray], target: dict[str, np.ndarray],
) -> dict[str, Any]:
    import torch
    import torch.nn.functional as functional

    from oaci.models import build_model
    from oaci.train.checkpoint import state_hash

    root = common.campaign_root(lock)
    directory = root / "instrumentation" / "units" / unit["unit_id"]
    manifest_path = directory / "unit_manifest.json"
    existing = _existing_unit(manifest_path, unit["checkpoint_id"])
    if existing is not None:
        return existing
    started_wall = time.time()
    started_cpu = time.process_time()
    state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
    if state_hash(state) != unit["checkpoint_id"]:
        raise RuntimeError(f"C78 checkpoint state mismatch: {unit['unit_id']}")
    model = build_model(
        "shallow_convnet", in_chans=22, in_times=385, n_classes=4,
        temporal_filters=40, temporal_kernel_samples=25,
        pool_kernel_samples=75, pool_stride_samples=15,
        dropout=0.5, safe_log_eps=1e-6,
    ).cpu()
    model.load_state_dict(state, strict=True)
    model.eval()
    weight = model.classifier.weight.detach().cpu().to(torch.float32)
    bias = model.classifier.bias.detach().cpu().to(torch.float32)
    hook_batches: list[torch.Tensor] = []

    def pre_hook(_module, args):
        hook_batches.append(args[0].detach().cpu())

    handle = model.classifier.register_forward_pre_hook(pre_hook)

    def infer(X: np.ndarray) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        count = X.shape[0]
        logits = np.empty((count, 4), dtype=np.float32)
        probabilities = np.empty((count, 4), dtype=np.float32)
        predictions = np.empty(count, dtype=np.int16)
        z = np.empty((count, 800), dtype=np.float32)
        wz = np.empty((count, 4), dtype=np.float32)
        wz_plus_b = np.empty((count, 4), dtype=np.float32)
        maxima = {"identity_abs": 0.0, "softmax_abs": 0.0, "hook_abs": 0.0, "repeat_logits": 0.0, "repeat_z": 0.0}
        with torch.no_grad():
            for start in range(0, count, 128):
                stop = min(start + 128, count)
                x = torch.from_numpy(np.ascontiguousarray(X[start:stop])).to(torch.float32)
                hook_batches.clear()
                output = model(x)
                if start == 0:
                    original_logits = output.logits.detach().clone()
                    original_z = output.z.detach().clone()
                    repeated = model(x)
                    maxima["repeat_logits"] = float(torch.max(torch.abs(original_logits - repeated.logits)).item())
                    maxima["repeat_z"] = float(torch.max(torch.abs(original_z - repeated.z)).item())
                captured = hook_batches[0]
                maxima["hook_abs"] = max(maxima["hook_abs"], float(torch.max(torch.abs(captured - output.z)).item()))
                projection = functional.linear(output.z, weight, bias=None)
                reconstructed = projection + bias
                probs = torch.softmax(output.logits, dim=1)
                replay_probs = torch.softmax(reconstructed, dim=1)
                maxima["identity_abs"] = max(maxima["identity_abs"], float(torch.max(torch.abs(reconstructed - output.logits)).item()))
                maxima["softmax_abs"] = max(maxima["softmax_abs"], float(torch.max(torch.abs(probs - replay_probs)).item()))
                logits[start:stop] = output.logits.numpy()
                probabilities[start:stop] = probs.numpy()
                predictions[start:stop] = torch.argmax(output.logits, dim=1).numpy()
                z[start:stop] = output.z.numpy()
                wz[start:stop] = projection.numpy()
                wz_plus_b[start:stop] = reconstructed.numpy()
        return {
            "logits": logits, "probabilities": probabilities,
            "prediction": predictions, "z": z, "Wz": wz,
            "Wz_plus_b": wz_plus_b, "class_margins": _class_margins(logits),
        }, maxima

    try:
        source_output, source_identity = infer(source["X"])
        target_output, target_identity = infer(target["X"])
    finally:
        handle.remove()
    identity = {
        key: max(source_identity[key], target_identity[key])
        for key in source_identity
    }
    identity_pass = (
        identity["identity_abs"] <= 1e-6
        and identity["softmax_abs"] <= 1e-7
        and identity["hook_abs"] <= 1e-6
        and identity["repeat_logits"] == 0.0
        and identity["repeat_z"] == 0.0
        and state_hash(model.state_dict()) == unit["checkpoint_id"]
    )
    if not identity_pass:
        raise RuntimeError(f"C78 instrumentation identity failed for {unit['unit_id']}: {identity}")

    source_arrays = {
        key: source[key]
        for key in ("source_trial_id", "source_domain_id", "source_role", "source_class_label")
    }
    source_arrays.update(source_output)
    target_arrays = {key: target[key] for key in ("target_trial_id", "target_id")}
    target_arrays.update(target_output)
    shards = [
        c74_cache.write_content_addressed_npz(directory, "checkpoint_Wb", {"W": weight.numpy(), "b": bias.numpy()}),
        c74_cache.write_content_addressed_npz(directory, "strict_source_trial", source_arrays),
        c74_cache.write_content_addressed_npz(directory, "target_unlabeled_trial", target_arrays),
    ]
    schemas = {"checkpoint_Wb": WB_FIELDS, "strict_source_trial": SOURCE_OUTPUT_FIELDS, "target_unlabeled_trial": TARGET_OUTPUT_FIELDS}
    for descriptor in shards:
        c74_cache.verify_shard(descriptor, required_fields=schemas[descriptor["kind"]])
    target_fields = set(next(descriptor["fields"] for descriptor in shards if descriptor["kind"] == "target_unlabeled_trial"))
    if target_fields & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78 target-unlabeled output contains a forbidden field")

    payload = common.write_manifest(manifest_path, {
        "schema_version": "c78_instrumented_unit_manifest_v1",
        "unit_id": unit["unit_id"], "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
        "sidecar_sha256": unit["sidecar_sha256"],
        "target": c78.TARGET, "seed": c78.SEED,
        "level": int(unit["level"]), "regime": unit["regime"],
        "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
        "source_rows": int(source["X"].shape[0]),
        "target_unlabeled_rows": int(target["X"].shape[0]),
        "identity": {**identity, "passed": identity_pass},
        "execution": {
            "CPU_only": True, "model_eval": not model.training,
            "gradients_enabled_during_forward": False,
            "training_attempted": False, "parameter_updates": False,
            "GPU_used": False, "target_labels_visible": False,
            "same_label_oracle_descriptor_visible": False,
            "wall_seconds": time.time() - started_wall,
            "process_CPU_seconds": time.process_time() - started_cpu,
            "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        },
        "shards": shards, "all_gates_passed": True,
    })
    return payload


def instrument_shard(*, authorization_token: str, shard_index: int, num_shards: int) -> dict[str, Any]:
    lock, _, _ = common.require_authorization(authorization_token)
    common.require_field_frozen(lock)
    if not (0 <= int(shard_index) < int(num_shards)) or int(num_shards) <= 0:
        raise ValueError("invalid C78 instrumentation shard")
    source, target, primary = _load_primary_inputs(lock)
    units = common.checkpoint_sidecars(lock)
    selected = [unit for ordinal, unit in enumerate(units) if ordinal % int(num_shards) == int(shard_index)]
    if not selected:
        raise RuntimeError("C78 instrumentation shard is empty")
    manifests = [_instrument_unit(lock=lock, unit=unit, source=source, target=target) for unit in selected]
    root = common.campaign_root(lock)
    shard = common.write_manifest(root / "instrumentation" / "shards" / f"shard-{shard_index:02d}-of-{num_shards:02d}.json", {
        "schema_version": "c78_instrumentation_shard_v1",
        "shard_index": int(shard_index), "num_shards": int(num_shards),
        "unit_count": len(manifests), "unit_ids": [item["unit_id"] for item in manifests],
        "all_gates_passed": all(item["all_gates_passed"] for item in manifests),
        "primary_input_manifest_sha256": primary["manifest_sha256"],
        "target_labels_visible": False,
        "same_label_oracle_descriptor_visible": False,
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    })
    print(json.dumps({"gate": "INSTRUMENTATION_SHARD_COMPLETE", "shard": shard_index, "units": len(manifests), "manifest_sha256": shard["manifest_sha256"]}, sort_keys=True))
    return shard


def aggregate_instrumentation(*, authorization_token: str, num_shards: int) -> dict[str, Any]:
    lock, _, protocol_sha = common.require_authorization(authorization_token)
    frozen = common.require_field_frozen(lock)
    primary = common.verify_canonical_manifest(common.primary_input_gate_path(lock))
    labels = common.verify_canonical_manifest(common.label_view_gate_path(lock))
    root = common.campaign_root(lock)
    shard_manifests = []
    unit_ids: list[str] = []
    for index in range(int(num_shards)):
        path = root / "instrumentation" / "shards" / f"shard-{index:02d}-of-{int(num_shards):02d}.json"
        shard = common.verify_canonical_manifest(path)
        if not shard["all_gates_passed"] or shard["target_labels_visible"]:
            raise RuntimeError(f"C78 instrumentation shard gate failed: {path}")
        shard_manifests.append({"path": str(path), "sha256": c78.sha256_file(path), "unit_count": shard["unit_count"]})
        unit_ids.extend(shard["unit_ids"])
    if len(unit_ids) != 82 or len(set(unit_ids)) != 82:
        raise RuntimeError(f"C78 instrumentation coverage is not 82 unique units: {len(unit_ids)}/{len(set(unit_ids))}")
    expected = {row["unit_id"] for row in frozen["units"]}
    if set(unit_ids) != expected:
        raise RuntimeError("C78 instrumentation unit set differs from frozen field")

    unit_manifests = []
    source_rows = target_rows = 0
    max_identity = max_softmax = max_repeat = max_hook = 0.0
    wall = cpu = 0.0
    for unit_id in sorted(unit_ids):
        path = root / "instrumentation" / "units" / unit_id / "unit_manifest.json"
        unit = common.verify_canonical_manifest(path)
        if not unit["all_gates_passed"]:
            raise RuntimeError(f"C78 unit instrumentation gate failed: {unit_id}")
        for descriptor in unit["shards"]:
            c74_cache.verify_shard(descriptor)
        source_rows += int(unit["source_rows"])
        target_rows += int(unit["target_unlabeled_rows"])
        max_identity = max(max_identity, float(unit["identity"]["identity_abs"]))
        max_softmax = max(max_softmax, float(unit["identity"]["softmax_abs"]))
        max_repeat = max(max_repeat, float(unit["identity"]["repeat_logits"]), float(unit["identity"]["repeat_z"]))
        max_hook = max(max_hook, float(unit["identity"]["hook_abs"]))
        wall += float(unit["execution"]["wall_seconds"])
        cpu += float(unit["execution"]["process_CPU_seconds"])
        unit_manifests.append({"unit_id": unit_id, "path": str(path), "sha256": c78.sha256_file(path)})
    external_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    gate = common.write_manifest(common.instrumentation_gate_path(lock), {
        "schema_version": "c78_instrumentation_complete_v1",
        "protocol_sha256": protocol_sha,
        "field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "primary_input_manifest_sha256": primary["manifest_sha256"],
        "label_view_manifest_sha256": labels["manifest_sha256"],
        "unit_count": 82, "unique_unit_count": 82,
        "source_rows": source_rows, "target_unlabeled_rows": target_rows,
        "expected_source_rows": 82 * 8 * 576,
        "expected_target_unlabeled_rows": 82 * 576,
        "identity": {
            "Wz_plus_b_logits_max_abs": max_identity,
            "softmax_max_abs": max_softmax,
            "repeat_max_abs": max_repeat,
            "hook_z_max_abs": max_hook,
            "failed_units": 0,
        },
        "physical_isolation": {
            "target_unlabeled_contains_labels": False,
            "instrumentation_received_label_gate_path": False,
            "instrumentation_received_oracle_path": False,
            "source_and_target_input_views_separate": True,
            "construction_evaluation_oracle_separate": True,
        },
        "shards": shard_manifests, "units": unit_manifests,
        "execution": {
            "summed_unit_wall_seconds": wall, "summed_unit_process_CPU_seconds": cpu,
            "external_storage_bytes": external_bytes, "GPU_used_for_instrumentation": False,
        },
        "all_gates_passed": True,
    })
    print(json.dumps({"gate": "INSTRUMENTATION_COMPLETE", "units": 82, "source_rows": source_rows, "target_rows": target_rows, "max_identity": max_identity}, sort_keys=True))
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78_authorized_instrument")
    sub = parser.add_subparsers(dest="command", required=True)
    prepare = sub.add_parser("prepare-views")
    prepare.add_argument("--authorization-token", required=True)
    prepare.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    instrument = sub.add_parser("instrument")
    instrument.add_argument("--authorization-token", required=True)
    instrument.add_argument("--shard-index", type=int, required=True)
    instrument.add_argument("--num-shards", type=int, required=True)
    aggregate = sub.add_parser("aggregate")
    aggregate.add_argument("--authorization-token", required=True)
    aggregate.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare-views":
        prepare_views(authorization_token=args.authorization_token, datalake_root=args.datalake_root)
    elif args.command == "instrument":
        instrument_shard(authorization_token=args.authorization_token, shard_index=args.shard_index, num_shards=args.num_shards)
    else:
        aggregate_instrumentation(authorization_token=args.authorization_token, num_shards=args.num_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
