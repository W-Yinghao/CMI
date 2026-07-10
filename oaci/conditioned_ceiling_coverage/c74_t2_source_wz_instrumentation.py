"""C74 T2-only frozen source and z/Wz instrumentation campaign.

Real EEG inference is reachable only through the ``instrument`` subcommand and
only after exact CLI authorization, locked-protocol verification, and an
explicit T2 unit-role check.  Validation and analysis subcommands never load a
model or EEG data.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import time

import numpy as np

from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c69_powered_trial_cache_scaleup as c69
from . import c74_cache as cache


MILESTONE = "C74"
DEFAULT_DATALAKE_ROOT = c69.DEFAULT_DATALAKE_ROOT

CHECKPOINT_FIELDS = {"W", "b"}
SOURCE_FIELDS = {
    "source_trial_id", "source_domain_id", "source_role", "source_class_label",
    "logits", "probabilities", "predicted_class", "z", "Wz", "Wz_plus_b",
}
TARGET_UNLABELED_FIELDS = {
    "target_trial_id", "target_id", "logits", "probabilities",
    "predicted_class", "z", "Wz", "Wz_plus_b",
}
CONSTRUCTION_FIELDS = {"target_trial_id", "target_class_label", "split_role"}
EVALUATION_FIELDS = {"target_trial_id", "target_class_label", "split_role"}
ORACLE_FIELDS = {"target_trial_id", "target_class_label", "split_role"}
SHARD_SCHEMAS = {
    "checkpoint_Wb": CHECKPOINT_FIELDS,
    "strict_source_trial": SOURCE_FIELDS,
    "target_unlabeled_representation": TARGET_UNLABELED_FIELDS,
    "target_construction_labels": CONSTRUCTION_FIELDS,
    "target_evaluation_labels": EVALUATION_FIELDS,
    "same_label_oracle": ORACLE_FIELDS,
}
FORBIDDEN_TARGET_UNLABELED = {
    "target_class_label", "y_true", "correctness", "target_bAcc", "target_NLL",
    "target_ECE", "joint_good", "target_margin", "split_role",
}


def _sha256_file(path: str) -> str:
    return cache.sha256_file(path)


def _preprocessing_signature(dataset_contract: dict) -> str:
    return hashlib.sha256(
        json.dumps(dataset_contract["preprocessing"], sort_keys=True).encode("utf-8")
    ).hexdigest()


def _unicode(values) -> np.ndarray:
    values = [str(value) for value in values]
    width = max((len(value) for value in values), default=1)
    return np.asarray(values, dtype=f"<U{width}")


def _relative_error(actual: np.ndarray, expected: np.ndarray) -> float:
    denominator = np.maximum(np.abs(expected), np.finfo(np.float32).tiny)
    return float(np.max(np.abs(actual - expected) / denominator))


def _source_role(subject_id: str, target_id: int) -> str:
    from oaci.confirmatory.loso_plan import loso_fold_spec

    subject = int(str(subject_id).rsplit("-", 1)[-1])
    split = loso_fold_spec(target_id)
    if subject in split["source_train_subjects"]:
        return "source_train"
    if subject in split["source_audit_subjects"]:
        return "source_audit"
    raise RuntimeError(f"subject {subject} is not a source for target {target_id}")


def _existing_unit(path: Path, expected_checkpoint_id: str) -> dict | None:
    if not path.is_file():
        return None
    payload = cache.verify_unit_manifest(path, rehash_payloads=True)
    if payload["checkpoint_id"] != expected_checkpoint_id or not payload["all_gates_passed"]:
        raise RuntimeError(f"invalid existing C74 unit manifest: {path}")
    return payload


def _instrument_unit(row: dict, bundle, protocol: dict, stage: str) -> dict:
    import torch
    import torch.nn.functional as functional
    from oaci.models import build_model
    from oaci.train.checkpoint import state_hash

    target_id = int(row["target"])
    unit_id = row["c74_unit_id"]
    unit_dir = cache.unit_directory(protocol, stage, target_id, unit_id)
    manifest_path = unit_dir / "unit_manifest.json"
    existing = _existing_unit(manifest_path, row["checkpoint_id"])
    if existing is not None:
        return existing

    started = time.time()
    unit_dir.mkdir(parents=True, exist_ok=True)
    file_hash = _sha256_file(row["pt_path"])
    if file_hash != row["pt_file_sha256"]:
        raise RuntimeError(f"checkpoint file SHA mismatch for {row['checkpoint_id']}")
    load = c66._state_load_metadata(row)
    if load["load_status"] != "pass" or not int(load["state_hash_matches_checkpoint_id"]):
        raise RuntimeError(f"checkpoint state identity failed for {row['checkpoint_id']}: {load['error']}")
    if not int(load["sidecar_tensor_schema_matches"]):
        raise RuntimeError(f"checkpoint sidecar schema mismatch for {row['checkpoint_id']}")

    spec = c66._model_spec_for_row(row)
    arch = dict(spec["backbone"])
    in_chans, in_times = [int(value) for value in spec["input_shape"]]
    model = build_model(
        spec["factory"], in_chans=in_chans, in_times=in_times,
        n_classes=int(spec["n_classes"]), **arch,
    )
    model.load_state_dict(load["state"], strict=True)
    model.eval()
    c66._assert_cpu_model_and_tensor(model)
    if not hasattr(model, "classifier"):
        raise RuntimeError(f"C74 classifier hook unavailable for {row['checkpoint_id']}")
    classifier = model.classifier
    weight = classifier.weight.detach().cpu().to(dtype=torch.float32)
    bias = classifier.bias.detach().cpu().to(dtype=torch.float32)
    if list(weight.shape) != [4, 800] or list(bias.shape) != [4]:
        raise RuntimeError(f"unexpected C74 W/b ABI: W={list(weight.shape)}, b={list(bias.shape)}")

    count = len(bundle.y)
    logits_all = np.empty((count, 4), dtype=np.float32)
    probabilities_all = np.empty((count, 4), dtype=np.float32)
    predictions_all = np.empty(count, dtype=np.int16)
    z_all = np.empty((count, 800), dtype=np.float32)
    wz_all = np.empty((count, 4), dtype=np.float32)
    wz_plus_b_all = np.empty((count, 4), dtype=np.float32)
    hook_batches: list[torch.Tensor] = []

    def pre_hook(_module, args):
        hook_batches.append(args[0].detach().cpu())

    handle = classifier.register_forward_pre_hook(pre_hook)
    max_hook_error = 0.0
    max_identity_abs = 0.0
    max_identity_relative = 0.0
    max_softmax_error = 0.0
    max_numpy_softmax_cross_impl_error = 0.0
    repeat_logits_error = 0.0
    repeat_z_error = 0.0
    try:
        with torch.no_grad():
            for start in range(0, count, 128):
                stop = min(start + 128, count)
                x = torch.from_numpy(np.ascontiguousarray(bundle.X[start:stop])).to(dtype=torch.float32)
                c66._assert_cpu_model_and_tensor(model, x)
                hook_batches.clear()
                output = model(x)
                if output.logits.device.type != "cpu" or output.z.device.type != "cpu":
                    raise RuntimeError("C74 CPU-only output guard failed")
                if start == 0:
                    first_logits = output.logits.detach().clone()
                    first_z = output.z.detach().clone()
                    repeated = model(x)
                    repeat_logits_error = float(torch.max(torch.abs(first_logits - repeated.logits)).item())
                    repeat_z_error = float(torch.max(torch.abs(first_z - repeated.z)).item())
                captured = hook_batches[0]
                hook_error = float(torch.max(torch.abs(captured - output.z.detach().cpu())).item())
                max_hook_error = max(max_hook_error, hook_error)
                wz = functional.linear(output.z, weight, bias=None)
                reconstructed = wz + bias
                probs = torch.softmax(output.logits, dim=1)
                numpy_logits = output.logits.detach().cpu().numpy().astype(np.float32, copy=False)
                stored_probs = probs.detach().cpu().numpy().astype(np.float32, copy=False)
                replayed_probs = torch.softmax(torch.from_numpy(numpy_logits), dim=1).numpy()
                shifted = numpy_logits - numpy_logits.max(axis=1, keepdims=True)
                numpy_probs = np.exp(shifted)
                numpy_probs /= numpy_probs.sum(axis=1, keepdims=True)
                reconstructed_np = reconstructed.detach().cpu().numpy().astype(np.float32, copy=False)
                max_identity_abs = max(max_identity_abs, float(np.max(np.abs(reconstructed_np - numpy_logits))))
                max_identity_relative = max(max_identity_relative, _relative_error(reconstructed_np, numpy_logits))
                max_softmax_error = max(
                    max_softmax_error,
                    float(np.max(np.abs(stored_probs - replayed_probs))),
                )
                max_numpy_softmax_cross_impl_error = max(
                    max_numpy_softmax_cross_impl_error,
                    float(np.max(np.abs(stored_probs - numpy_probs))),
                )
                logits_all[start:stop] = numpy_logits
                probabilities_all[start:stop] = stored_probs
                predictions_all[start:stop] = torch.argmax(output.logits, dim=1).detach().cpu().numpy()
                z_all[start:stop] = output.z.detach().cpu().numpy()
                wz_all[start:stop] = wz.detach().cpu().numpy()
                wz_plus_b_all[start:stop] = reconstructed_np
    finally:
        handle.remove()

    state_after = state_hash(model.state_dict())
    tolerances = protocol["identity_tolerances"]
    identity_passed = (
        max_identity_abs <= float(tolerances["Wz_plus_b_logits_max_abs"])
        and max_identity_relative <= float(tolerances["Wz_plus_b_logits_max_relative"])
        and max_softmax_error <= float(tolerances["softmax_probability_max_abs"])
        and max(repeat_logits_error, repeat_z_error) <= float(tolerances["repeat_forward_max_abs"])
        and max_hook_error <= float(tolerances["Wz_plus_b_logits_max_abs"])
        and state_after == row["checkpoint_id"]
    )
    if not identity_passed:
        raise RuntimeError(
            f"C74 numerical identity failed for {row['checkpoint_id']}: "
            f"abs={max_identity_abs}, rel={max_identity_relative}, softmax={max_softmax_error}, "
            f"hook={max_hook_error}, repeat={max(repeat_logits_error, repeat_z_error)}"
        )

    target_domain = f"BNCI2014_001|subject-{target_id:03d}"
    target_mask = np.asarray(bundle.subject_id == target_domain)
    source_mask = ~target_mask
    source_indices = np.where(source_mask)[0]
    target_indices = np.where(target_mask)[0]
    if len(target_indices) != 576 or len(source_indices) != 8 * 576:
        raise RuntimeError(
            f"C74 trial count contract failed for target {target_id}: "
            f"target={len(target_indices)}, source={len(source_indices)}"
        )

    trial_ids = _unicode(bundle.trial_id)
    source_domains = _unicode(bundle.subject_id[source_indices])
    source_roles = _unicode([_source_role(domain, target_id) for domain in source_domains])
    target_trial_ids = trial_ids[target_indices]
    split_roles = _unicode([c69._future_split_role(trial_id) for trial_id in target_trial_ids])
    construct_mask = split_roles == "target_construct"
    evaluation_mask = split_roles == "target_eval"
    if not construct_mask.any() or not evaluation_mask.any():
        raise RuntimeError(f"C74 target label split is empty for target {target_id}")

    shards = [
        cache.write_content_addressed_npz(unit_dir, "checkpoint_Wb", {
            "W": weight.numpy(), "b": bias.numpy(),
        }),
        cache.write_content_addressed_npz(unit_dir, "strict_source_trial", {
            "source_trial_id": trial_ids[source_indices],
            "source_domain_id": source_domains,
            "source_role": source_roles,
            "source_class_label": np.asarray(bundle.y[source_indices], dtype=np.int16),
            "logits": logits_all[source_indices],
            "probabilities": probabilities_all[source_indices],
            "predicted_class": predictions_all[source_indices],
            "z": z_all[source_indices],
            "Wz": wz_all[source_indices],
            "Wz_plus_b": wz_plus_b_all[source_indices],
        }),
        cache.write_content_addressed_npz(unit_dir, "target_unlabeled_representation", {
            "target_trial_id": target_trial_ids,
            "target_id": np.full(len(target_indices), target_id, dtype=np.int16),
            "logits": logits_all[target_indices],
            "probabilities": probabilities_all[target_indices],
            "predicted_class": predictions_all[target_indices],
            "z": z_all[target_indices],
            "Wz": wz_all[target_indices],
            "Wz_plus_b": wz_plus_b_all[target_indices],
        }),
        cache.write_content_addressed_npz(unit_dir, "target_construction_labels", {
            "target_trial_id": target_trial_ids[construct_mask],
            "target_class_label": np.asarray(bundle.y[target_indices][construct_mask], dtype=np.int16),
            "split_role": split_roles[construct_mask],
        }),
        cache.write_content_addressed_npz(unit_dir, "target_evaluation_labels", {
            "target_trial_id": target_trial_ids[evaluation_mask],
            "target_class_label": np.asarray(bundle.y[target_indices][evaluation_mask], dtype=np.int16),
            "split_role": split_roles[evaluation_mask],
        }),
        cache.write_content_addressed_npz(unit_dir, "same_label_oracle", {
            "target_trial_id": target_trial_ids,
            "target_class_label": np.asarray(bundle.y[target_indices], dtype=np.int16),
            "split_role": split_roles,
        }),
    ]
    for shard in shards:
        expected = SHARD_SCHEMAS[shard["kind"]]
        cache.verify_shard(shard, required_fields=expected)
    target_fields = next(set(shard["fields"]) for shard in shards if shard["kind"] == "target_unlabeled_representation")
    if target_fields & FORBIDDEN_TARGET_UNLABELED:
        raise RuntimeError(f"C74 target-unlabeled leakage: {sorted(target_fields & FORBIDDEN_TARGET_UNLABELED)}")

    payload = cache.self_hashed_manifest({
        "schema_version": "c74_t2_source_wz_unit_manifest_v1",
        "milestone": MILESTONE,
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "authorization_token_sha256": cache.sha256_text(cache.AUTH_TOKEN),
        "stage": stage,
        "unit_id": unit_id,
        "checkpoint_id": row["checkpoint_id"],
        "checkpoint_file_sha256": file_hash,
        "checkpoint_sidecar_sha256": _sha256_file(row["json_path"]),
        "target_id": target_id,
        "seed": int(row["seed"]),
        "level": int(row["level"]),
        "regime": row["regime"],
        "trajectory_id": row["trajectory_id"],
        "candidate_order": int(row["candidate_order"]),
        "model_factory": spec["factory"],
        "input_shape": [in_chans, in_times],
        "representation_shape": [800],
        "W_shape": list(weight.shape),
        "b_shape": list(bias.shape),
        "source_rows": int(len(source_indices)),
        "target_unlabeled_rows": int(len(target_indices)),
        "target_construction_rows": int(construct_mask.sum()),
        "target_evaluation_rows": int(evaluation_mask.sum()),
        "same_label_oracle_rows": int(len(target_indices)),
        "identity": {
            "Wz_plus_b_logits_max_abs": max_identity_abs,
            "Wz_plus_b_logits_max_relative": max_identity_relative,
            "softmax_probability_max_abs": max_softmax_error,
            "numpy_softmax_cross_implementation_max_abs": max_numpy_softmax_cross_impl_error,
            "hook_z_max_abs": max_hook_error,
            "repeat_logits_max_abs": repeat_logits_error,
            "repeat_z_max_abs": repeat_z_error,
            "state_hash_after": state_after,
            "passed": identity_passed,
        },
        "execution": {
            "CPU_only": True, "model_eval": not model.training,
            "gradients_enabled": torch.is_grad_enabled(), "training_attempted": False,
            "parameter_updates": False, "GPU_used": False,
            "real_EEG_forward_attempted": True,
            "wall_seconds": time.time() - started,
        },
        "view_isolation": {
            "strict_source_contains_target_rows": False,
            "target_unlabeled_contains_target_labels": False,
            "construction_contains_evaluation_labels": False,
            "oracle_available_to_primary_smoke": False,
            "passed": True,
        },
        "shards": shards,
        "all_gates_passed": True,
    })
    cache.atomic_json(manifest_path, payload)
    return cache.verify_unit_manifest(manifest_path, rehash_payloads=True)


def instrument_stage_target(
    *, stage: str, target_id: int, authorization_token: str,
    datalake_root: str, num_threads: int,
) -> dict:
    """Authorized entry point.  No other function in this module loads EEG."""
    protocol = cache.load_locked_protocol()
    if not cache.authorization_ok(authorization_token):
        raise PermissionError("C74 real forward requires the exact CLI authorization token")
    if not 1 <= int(target_id) <= 9:
        raise ValueError("C74 target-id must be in 1..9")
    rows = cache.stage_rows(stage, target_id)
    if stage == "P1_expansion":
        gate_path = cache.stage_gate_path(protocol, "P0_pilot")
        if not gate_path.is_file():
            raise PermissionError("C74 P1 blocked: aggregate P0 gate is absent")
        gate = json.loads(gate_path.read_text())
        expected_gate_hash = gate.pop("manifest_sha256", "")
        observed_gate_hash = cache.sha256_text(json.dumps(gate, sort_keys=True, separators=(",", ":")))
        if expected_gate_hash != observed_gate_hash:
            raise PermissionError("C74 P1 blocked: aggregate P0 gate self-hash failed")
        if gate.get("final_gate") != "P0_PILOT_ALL_GATES_PASSED" or int(gate.get("validated_units", 0)) != 54:
            raise PermissionError("C74 P1 blocked: aggregate P0 gate did not pass")

    import torch
    from oaci.data.eeg.bnci import load_moabb_confirmatory

    torch.set_num_threads(max(1, int(num_threads)))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    pp, dataset_contract = c66._preprocess_namespace(rows[0])
    observed_signature = _preprocessing_signature(dataset_contract)
    expected_signature = protocol["model_instrumentation"]["preprocessing_signature"]
    if observed_signature != expected_signature:
        raise RuntimeError(
            f"C74 preprocessing signature drift: expected {expected_signature}, observed {observed_signature}"
        )
    loaded = load_moabb_confirmatory(
        "BNCI2014_001", list(range(1, 10)), pp,
        frozen_class_names=dataset_contract["class_names"],
        frozen_channels=dataset_contract["channels"],
        expected_sfreq=float(dataset_contract["expected_sfreq"]),
        expected_n_times=int(dataset_contract["expected_n_times"]),
        datalake_root=datalake_root,
    )
    bundle = loaded.bundle
    if bundle.X.shape != (9 * 576, 22, 385) or len(bundle.class_names) != 4:
        raise RuntimeError(f"C74 loaded EEG ABI mismatch: {bundle.X.shape}, classes={bundle.class_names}")

    unit_manifests = []
    for index, row in enumerate(rows, 1):
        print(
            json.dumps({
                "event": "c74_unit_start", "stage": stage, "target_id": target_id,
                "unit_index": index, "unit_count": len(rows), "unit_id": row["c74_unit_id"],
            }, sort_keys=True),
            flush=True,
        )
        unit_manifests.append(_instrument_unit(row, bundle, protocol, stage))
        print(
            json.dumps({
                "event": "c74_unit_complete", "stage": stage, "target_id": target_id,
                "unit_index": index, "unit_count": len(rows), "unit_id": row["c74_unit_id"],
            }, sort_keys=True),
            flush=True,
        )

    job_payload = {
        "schema_version": "c74_instrumentation_job_manifest_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "stage": stage,
        "target_id": int(target_id),
        "unit_count": len(unit_manifests),
        "unit_manifest_paths": [
            str(cache.unit_directory(protocol, stage, target_id, manifest["unit_id"]) / "unit_manifest.json")
            for manifest in unit_manifests
        ],
        "unit_manifest_sha256s": [manifest["manifest_sha256"] for manifest in unit_manifests],
        "dataset_evidence_hash": loaded.evidence.evidence_hash,
        "raw_data_fingerprint": loaded.evidence.raw_data_fingerprint,
        "resolved_preprocess_hash": bundle.preprocess_hash,
        "network_attempt_count": loaded.evidence.network_attempt_count,
        "CPU_only": True,
        "training_attempted": False,
        "GPU_used": False,
        "all_gates_passed": all(manifest["all_gates_passed"] for manifest in unit_manifests),
    }
    job_path = cache.run_root(protocol) / stage / f"target-{target_id:03d}" / "job_manifest.json"
    cache.atomic_json(job_path, job_payload)
    return {"job_manifest": str(job_path), **job_payload}


def validate_stage(stage: str, *, rehash_payloads: bool = True) -> dict:
    """Aggregate and validate one stage without loading EEG or checkpoints."""
    protocol = cache.load_locked_protocol()
    expected_rows = []
    for target_id in range(1, 10):
        expected_rows.extend(cache.stage_rows(stage, target_id))
    expected_count = 54 if stage == "P0_pilot" else 162
    if len(expected_rows) != expected_count:
        raise RuntimeError(f"C74 {stage} expected-unit count drift")

    manifests = []
    for row in expected_rows:
        path = cache.unit_directory(protocol, stage, int(row["target"]), row["c74_unit_id"]) / "unit_manifest.json"
        manifest = cache.verify_unit_manifest(path, rehash_payloads=rehash_payloads)
        if manifest["unit_id"] != row["c74_unit_id"] or manifest["checkpoint_id"] != row["checkpoint_id"]:
            raise RuntimeError(f"C74 unit identity mismatch: {path}")
        if manifest["stage"] != stage or not manifest["all_gates_passed"]:
            raise RuntimeError(f"C74 unit gate failed: {path}")
        descriptors = {shard["kind"]: shard for shard in manifest["shards"]}
        if set(descriptors) != set(SHARD_SCHEMAS):
            raise RuntimeError(f"C74 view set mismatch: {path}")
        for kind, fields in SHARD_SCHEMAS.items():
            if set(descriptors[kind]["fields"]) != fields:
                raise RuntimeError(f"C74 schema mismatch: {path}:{kind}")
        if set(descriptors["target_unlabeled_representation"]["fields"]) & FORBIDDEN_TARGET_UNLABELED:
            raise RuntimeError(f"C74 target-label leakage: {path}")
        manifests.append(manifest)

    identity_keys = (
        "Wz_plus_b_logits_max_abs", "Wz_plus_b_logits_max_relative",
        "softmax_probability_max_abs", "hook_z_max_abs",
        "repeat_logits_max_abs", "repeat_z_max_abs",
    )
    maxima = {key: max(float(manifest["identity"][key]) for manifest in manifests) for key in identity_keys}
    gate_payload = cache.self_hashed_manifest({
        "schema_version": "c74_stage_gate_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "stage": stage,
        "validated_units": len(manifests),
        "targets": sorted({int(manifest["target_id"]) for manifest in manifests}),
        "unit_manifest_sha256s": sorted(manifest["manifest_sha256"] for manifest in manifests),
        "identity_maxima": maxima,
        "payloads_rehashed": bool(rehash_payloads),
        "source_rows": sum(int(manifest["source_rows"]) for manifest in manifests),
        "target_unlabeled_rows": sum(int(manifest["target_unlabeled_rows"]) for manifest in manifests),
        "failed_units": 0,
        "T3_HO_units_touched": 0,
        "physical_view_isolation_passed": True,
        "final_gate": "P0_PILOT_ALL_GATES_PASSED" if stage == "P0_pilot" else "P1_EXPANSION_ALL_GATES_PASSED",
    })
    gate_path = cache.stage_gate_path(protocol, stage)
    file_sha256 = cache.atomic_json(gate_path, gate_payload)
    return {"gate_path": str(gate_path), "gate_file_sha256": file_sha256, **gate_payload}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    instrument = subparsers.add_parser("instrument", help="authorized T2 real-EEG instrumentation")
    instrument.add_argument("--stage", choices=("P0_pilot", "P1_expansion"), required=True)
    instrument.add_argument("--target-id", type=int, required=True)
    instrument.add_argument("--authorization-token", required=True)
    instrument.add_argument("--datalake-root", default=DEFAULT_DATALAKE_ROOT)
    instrument.add_argument("--num-threads", type=int, default=48)
    validate = subparsers.add_parser("validate", help="no-forward aggregate stage validation")
    validate.add_argument("--stage", choices=("P0_pilot", "P1_expansion"), required=True)
    validate.add_argument("--no-rehash-payloads", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "instrument":
        result = instrument_stage_target(
            stage=args.stage, target_id=args.target_id,
            authorization_token=args.authorization_token,
            datalake_root=args.datalake_root, num_threads=args.num_threads,
        )
    else:
        result = validate_stage(args.stage, rehash_payloads=not args.no_rehash_payloads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
