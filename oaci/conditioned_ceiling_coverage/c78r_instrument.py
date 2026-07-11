"""Post-freeze C78R view linking and CPU trial instrumentation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np

from . import c74_cache
from . import c78_authorized_common as c78_common
from . import c78_authorized_instrument as c78_instrument
from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


SOURCE_INPUT_FIELDS = c78_instrument.SOURCE_INPUT_FIELDS
TARGET_INPUT_FIELDS = c78_instrument.TARGET_INPUT_FIELDS
SOURCE_OUTPUT_FIELDS = c78_instrument.SOURCE_OUTPUT_FIELDS
TARGET_OUTPUT_FIELDS = c78_instrument.TARGET_OUTPUT_FIELDS
WB_FIELDS = c78_instrument.WB_FIELDS
FORBIDDEN_TARGET_FIELDS = c78_instrument.FORBIDDEN_TARGET_FIELDS


def link_c78_views(*, authorization_token: str) -> dict[str, Any]:
    lock, _, protocol_sha = common.require_authorization(authorization_token)
    frozen = common.require_field_frozen(lock)
    primary_path = common.primary_input_gate_path(lock)
    label_path = common.label_view_gate_path(lock)
    if primary_path.exists() and label_path.exists():
        return common.verify_manifest(primary_path)

    c78_lock = c78_common.load_execution_lock()
    c78_field = c78_common.require_field_frozen(c78_lock)
    c78_primary_path = c78_common.primary_input_gate_path(c78_lock)
    c78_label_path = c78_common.label_view_gate_path(c78_lock)
    c78_primary = c78_common.verify_canonical_manifest(c78_primary_path)
    c78_labels = c78_common.verify_canonical_manifest(c78_label_path)
    source_descriptor = c78_primary["strict_source_input"]
    target_descriptor = c78_primary["target_unlabeled_input"]
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    if set(target_descriptor["fields"]) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78R linked target-unlabeled input exposes labels")
    if int(source_descriptor["row_count"]) != 8 * 576 or int(target_descriptor["row_count"]) != 576:
        raise RuntimeError("C78R linked C78 trial view row counts drift")

    labels = common.write_manifest(label_path, {
        "schema_version": "c78r_linked_label_views_v1",
        "protocol_sha256": protocol_sha,
        "SRC_field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "created_after_SRC_field_freeze": True,
        "C78_field_manifest_sha256": c78_field["manifest_sha256"],
        "C78_label_gate_path": str(c78_label_path),
        "C78_label_gate_file_sha256": c78r.sha256_file(c78_label_path),
        "target_label_views": c78_labels["target_label_views"],
        "read_only_reuse": True,
        "available_to_primary_instrumentation": False,
    })
    primary = common.write_manifest(primary_path, {
        "schema_version": "c78r_linked_primary_views_v1",
        "protocol_sha256": protocol_sha,
        "SRC_field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "created_after_SRC_field_freeze": True,
        "C78_field_manifest_sha256": c78_field["manifest_sha256"],
        "C78_primary_gate_file_sha256": c78r.sha256_file(c78_primary_path),
        "strict_source_input": source_descriptor,
        "target_unlabeled_input": target_descriptor,
        "read_only_reuse": True,
        "target_label_fields_present": False,
        "label_view_gate_path_in_primary_descriptor": False,
        "same_label_oracle_path_in_primary_descriptor": False,
        "primary_instrumentation_allowed_fields": {
            "source": sorted(SOURCE_INPUT_FIELDS),
            "target_unlabeled": sorted(TARGET_INPUT_FIELDS),
        },
    })
    print(json.dumps({
        "gate": "C78R_POSTFREEZE_VIEWS_LINKED",
        "source_rows": source_descriptor["row_count"],
        "target_rows": target_descriptor["row_count"],
        "label_gate_sha256": labels["manifest_sha256"],
    }, sort_keys=True))
    return primary


def _load_inputs(lock: dict[str, Any]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    gate = common.verify_manifest(common.primary_input_gate_path(lock))
    if gate["target_label_fields_present"] or gate["same_label_oracle_path_in_primary_descriptor"]:
        raise RuntimeError("C78R primary input gate exposes target labels or oracle")
    source_descriptor = gate["strict_source_input"]
    target_descriptor = gate["target_unlabeled_input"]
    c74_cache.verify_shard(source_descriptor, required_fields=SOURCE_INPUT_FIELDS)
    c74_cache.verify_shard(target_descriptor, required_fields=TARGET_INPUT_FIELDS)
    with np.load(source_descriptor["path"], allow_pickle=False) as stream:
        source = {name: np.array(stream[name], copy=True) for name in stream.files}
    with np.load(target_descriptor["path"], allow_pickle=False) as stream:
        target = {name: np.array(stream[name], copy=True) for name in stream.files}
    if set(target) & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78R target-unlabeled input acquired a forbidden field")
    return source, target, gate


def _existing(path: Path, checkpoint_id: str) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = common.verify_manifest(path)
    if payload["checkpoint_id"] != checkpoint_id or not payload["all_gates_passed"]:
        raise RuntimeError(f"invalid existing C78R instrumentation unit: {path}")
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
    previous = _existing(manifest_path, unit["checkpoint_id"])
    if previous is not None:
        return previous
    started_wall = time.time()
    started_cpu = time.process_time()
    state = torch.load(unit["checkpoint_path"], map_location="cpu", weights_only=True)
    if state_hash(state) != unit["checkpoint_id"]:
        raise RuntimeError(f"C78R checkpoint state mismatch: {unit['unit_id']}")
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
        count = int(X.shape[0])
        logits = np.empty((count, 4), dtype=np.float32)
        probabilities = np.empty((count, 4), dtype=np.float32)
        prediction = np.empty(count, dtype=np.int16)
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
                captured = hook_batches[0]
                if start == 0:
                    first_logits = output.logits.detach().clone()
                    first_z = output.z.detach().clone()
                    repeated = model(x)
                    maxima["repeat_logits"] = float(torch.max(torch.abs(first_logits - repeated.logits)).item())
                    maxima["repeat_z"] = float(torch.max(torch.abs(first_z - repeated.z)).item())
                maxima["hook_abs"] = max(maxima["hook_abs"], float(torch.max(torch.abs(captured - output.z)).item()))
                projection = functional.linear(output.z, weight, bias=None)
                reconstructed = projection + bias
                probs = torch.softmax(output.logits, dim=1)
                replay_probs = torch.softmax(reconstructed, dim=1)
                maxima["identity_abs"] = max(maxima["identity_abs"], float(torch.max(torch.abs(reconstructed - output.logits)).item()))
                maxima["softmax_abs"] = max(maxima["softmax_abs"], float(torch.max(torch.abs(probs - replay_probs)).item()))
                logits[start:stop] = output.logits.numpy()
                probabilities[start:stop] = probs.numpy()
                prediction[start:stop] = torch.argmax(output.logits, dim=1).numpy()
                z[start:stop] = output.z.numpy()
                wz[start:stop] = projection.numpy()
                wz_plus_b[start:stop] = reconstructed.numpy()
        return {
            "logits": logits, "probabilities": probabilities, "prediction": prediction,
            "z": z, "Wz": wz, "Wz_plus_b": wz_plus_b,
            "class_margins": c78_instrument._class_margins(logits),
        }, maxima

    try:
        source_output, source_identity = infer(source["X"])
        target_output, target_identity = infer(target["X"])
    finally:
        handle.remove()
    identity = {key: max(source_identity[key], target_identity[key]) for key in source_identity}
    passed = (
        identity["identity_abs"] <= 1e-6
        and identity["softmax_abs"] <= 1e-7
        and identity["hook_abs"] <= 1e-6
        and identity["repeat_logits"] == 0.0
        and identity["repeat_z"] == 0.0
        and state_hash(model.state_dict()) == unit["checkpoint_id"]
    )
    if not passed:
        raise RuntimeError(f"C78R instrumentation identity failure: {unit['unit_id']} {identity}")

    source_arrays = {key: source[key] for key in ("source_trial_id", "source_domain_id", "source_role", "source_class_label")}
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
    target_fields = set(next(item["fields"] for item in shards if item["kind"] == "target_unlabeled_trial"))
    if target_fields & FORBIDDEN_TARGET_FIELDS:
        raise RuntimeError("C78R target-unlabeled output contains a forbidden field")
    return common.write_manifest(manifest_path, {
        "schema_version": "c78r_instrumented_unit_v1",
        "unit_id": unit["unit_id"], "checkpoint_id": unit["checkpoint_id"],
        "checkpoint_file_sha256": unit["checkpoint_file_sha256"],
        "sidecar_sha256": unit["sidecar_sha256"],
        "target": c78r.TARGET, "seed": c78r.SEED,
        "level": int(unit["level"]), "regime": "SRC",
        "epoch": int(unit["epoch"]), "trajectory_order": int(unit["trajectory_order"]),
        "source_rows": int(source["X"].shape[0]),
        "target_unlabeled_rows": int(target["X"].shape[0]),
        "identity": {**identity, "passed": passed},
        "execution": {
            "CPU_only": True, "model_eval": not model.training,
            "gradients_enabled": False, "training_attempted": False,
            "parameter_updates": False, "target_labels_visible": False,
            "same_label_oracle_descriptor_visible": False,
            "wall_seconds": time.time() - started_wall,
            "process_CPU_seconds": time.process_time() - started_cpu,
            "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        },
        "shards": shards, "all_gates_passed": True,
    })


def instrument_shard(*, authorization_token: str, shard_index: int, num_shards: int) -> dict[str, Any]:
    lock, _, _ = common.require_authorization(authorization_token)
    common.require_field_frozen(lock)
    if not (0 <= int(shard_index) < int(num_shards)) or int(num_shards) <= 0:
        raise ValueError("invalid C78R instrumentation shard")
    source, target, primary = _load_inputs(lock)
    units = common.checkpoint_units(lock)
    selected = [unit for ordinal, unit in enumerate(units) if ordinal % int(num_shards) == int(shard_index)]
    if not selected:
        raise RuntimeError("C78R instrumentation shard is empty")
    manifests = [_instrument_unit(lock=lock, unit=unit, source=source, target=target) for unit in selected]
    root = common.campaign_root(lock)
    shard = common.write_manifest(root / "instrumentation" / "shards" / f"shard-{shard_index:02d}-of-{num_shards:02d}.json", {
        "schema_version": "c78r_instrumentation_shard_v1",
        "shard_index": int(shard_index), "num_shards": int(num_shards),
        "unit_count": len(manifests), "unit_ids": [item["unit_id"] for item in manifests],
        "all_gates_passed": all(item["all_gates_passed"] for item in manifests),
        "primary_input_manifest_sha256": primary["manifest_sha256"],
        "target_labels_visible": False, "same_label_oracle_descriptor_visible": False,
        "SLURM_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
    })
    print(json.dumps({"gate": "C78R_INSTRUMENTATION_SHARD_COMPLETE", "shard": shard_index, "units": len(manifests)}, sort_keys=True))
    return shard


def aggregate(*, authorization_token: str, num_shards: int) -> dict[str, Any]:
    lock, _, protocol_sha = common.require_authorization(authorization_token)
    frozen = common.require_field_frozen(lock)
    primary = common.verify_manifest(common.primary_input_gate_path(lock))
    labels = common.verify_manifest(common.label_view_gate_path(lock))
    root = common.campaign_root(lock)
    unit_ids: list[str] = []
    shards = []
    for index in range(int(num_shards)):
        path = root / "instrumentation" / "shards" / f"shard-{index:02d}-of-{int(num_shards):02d}.json"
        shard = common.verify_manifest(path)
        if not shard["all_gates_passed"] or shard["target_labels_visible"]:
            raise RuntimeError(f"C78R instrumentation shard failed: {path}")
        unit_ids.extend(shard["unit_ids"])
        shards.append({"path": str(path), "sha256": c78r.sha256_file(path), "unit_count": shard["unit_count"]})
    if len(unit_ids) != c78r.EXPECTED_UNITS or len(set(unit_ids)) != c78r.EXPECTED_UNITS:
        raise RuntimeError("C78R instrumentation does not cover 80 unique units")
    if set(unit_ids) != {row["unit_id"] for row in frozen["units"]}:
        raise RuntimeError("C78R instrumentation unit set differs from frozen field")
    source_rows = target_rows = 0
    maxima = {"Wz_plus_b_logits_max_abs": 0.0, "softmax_max_abs": 0.0, "repeat_max_abs": 0.0, "hook_z_max_abs": 0.0}
    wall = cpu = 0.0
    units = []
    for unit_id in sorted(unit_ids):
        path = root / "instrumentation" / "units" / unit_id / "unit_manifest.json"
        unit = common.verify_manifest(path)
        for descriptor in unit["shards"]:
            c74_cache.verify_shard(descriptor)
        source_rows += int(unit["source_rows"])
        target_rows += int(unit["target_unlabeled_rows"])
        maxima["Wz_plus_b_logits_max_abs"] = max(maxima["Wz_plus_b_logits_max_abs"], float(unit["identity"]["identity_abs"]))
        maxima["softmax_max_abs"] = max(maxima["softmax_max_abs"], float(unit["identity"]["softmax_abs"]))
        maxima["repeat_max_abs"] = max(maxima["repeat_max_abs"], float(unit["identity"]["repeat_logits"]), float(unit["identity"]["repeat_z"]))
        maxima["hook_z_max_abs"] = max(maxima["hook_z_max_abs"], float(unit["identity"]["hook_abs"]))
        wall += float(unit["execution"]["wall_seconds"])
        cpu += float(unit["execution"]["process_CPU_seconds"])
        units.append({"unit_id": unit_id, "path": str(path), "sha256": c78r.sha256_file(path)})
    if source_rows != c78r.EXPECTED_UNITS * 8 * 576 or target_rows != c78r.EXPECTED_UNITS * 576:
        raise RuntimeError("C78R aggregate trial row identity failed")
    gate = common.write_manifest(common.instrumentation_gate_path(lock), {
        "schema_version": "c78r_instrumentation_complete_v1",
        "protocol_sha256": protocol_sha,
        "field_frozen_manifest_sha256": frozen["manifest_sha256"],
        "primary_input_manifest_sha256": primary["manifest_sha256"],
        "label_view_manifest_sha256": labels["manifest_sha256"],
        "unit_count": c78r.EXPECTED_UNITS, "unique_unit_count": len(set(unit_ids)),
        "source_rows": source_rows, "target_unlabeled_rows": target_rows,
        "expected_source_rows": c78r.EXPECTED_UNITS * 8 * 576,
        "expected_target_unlabeled_rows": c78r.EXPECTED_UNITS * 576,
        "identity": {**maxima, "failed_units": 0},
        "physical_isolation": {
            "target_unlabeled_contains_labels": False,
            "instrumentation_received_label_gate_path": False,
            "instrumentation_received_oracle_path": False,
            "source_and_target_input_views_separate": True,
            "construction_evaluation_oracle_separate": True,
            "C78_inputs_reused_read_only_after_SRC_freeze": True,
        },
        "shards": shards, "units": units,
        "execution": {
            "summed_unit_wall_seconds": wall, "summed_unit_process_CPU_seconds": cpu,
            "external_storage_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
            "GPU_used_for_instrumentation": False,
        },
        "all_gates_passed": True,
    })
    print(json.dumps({"gate": "C78R_INSTRUMENTATION_COMPLETE", "units": c78r.EXPECTED_UNITS, "source_rows": source_rows, "target_rows": target_rows}, sort_keys=True))
    return gate


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78r_instrument")
    sub = parser.add_subparsers(dest="command", required=True)
    link = sub.add_parser("link-views")
    link.add_argument("--authorization-token", required=True)
    instrument = sub.add_parser("instrument")
    instrument.add_argument("--authorization-token", required=True)
    instrument.add_argument("--shard-index", type=int, required=True)
    instrument.add_argument("--num-shards", type=int, required=True)
    aggregate_parser = sub.add_parser("aggregate")
    aggregate_parser.add_argument("--authorization-token", required=True)
    aggregate_parser.add_argument("--num-shards", type=int, required=True)
    args = parser.parse_args(argv)
    if args.command == "link-views":
        link_c78_views(authorization_token=args.authorization_token)
    elif args.command == "instrument":
        instrument_shard(authorization_token=args.authorization_token, shard_index=args.shard_index, num_shards=args.num_shards)
    else:
        aggregate(authorization_token=args.authorization_token, num_shards=args.num_shards)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
