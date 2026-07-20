"""Authorized cross-node preprocessing replay for the C74 evidence-hash audit."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import socket

import numpy as np

from . import c66_reinference_only_trial_cache_microcampaign as c66
from . import c74_cache as cache


INPUT_MAX_ABS_TOLERANCE = 1e-5
INPUT_MEAN_ABS_TOLERANCE = 1e-7
Z_MAX_ABS_TOLERANCE = 1e-4
LOGIT_MAX_ABS_TOLERANCE = 1e-4
PROBABILITY_MAX_ABS_TOLERANCE = 1e-5
REPLICATES = ("nodecpu01", "nodecpu02")


def _directory(protocol: dict) -> Path:
    return cache.run_root(protocol) / "preprocessing_cross_node_replay"


def _manifest_path(protocol: dict, replicate: str) -> Path:
    return _directory(protocol) / f"{replicate}_manifest.json"


def _capture(replicate: str, token: str, datalake_root: str, num_threads: int) -> dict:
    protocol = cache.load_locked_protocol()
    if replicate not in REPLICATES:
        raise ValueError(f"invalid C74 preprocessing replicate: {replicate}")
    if not cache.authorization_ok(token):
        raise PermissionError("C74 preprocessing replay requires exact CLI authorization")
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname != replicate:
        raise RuntimeError(f"C74 preprocessing replay expected host {replicate}, got {hostname}")

    import torch
    from oaci.data.eeg.bnci import load_moabb_confirmatory
    from oaci.data.eeg.schema import tensor_content_hash

    torch.set_num_threads(max(1, int(num_threads)))
    torch.set_num_interop_threads(1)
    row = cache.stage_rows("P0_pilot", 1)[0]
    pp, dataset_contract = c66._preprocess_namespace(row)
    loaded = load_moabb_confirmatory(
        "BNCI2014_001", list(range(1, 10)), pp,
        frozen_class_names=dataset_contract["class_names"],
        frozen_channels=dataset_contract["channels"],
        expected_sfreq=float(dataset_contract["expected_sfreq"]),
        expected_n_times=int(dataset_contract["expected_n_times"]),
        datalake_root=datalake_root,
    )
    bundle = loaded.bundle
    descriptor = cache.write_content_addressed_npz(_directory(protocol), f"preprocessed_EEG_{replicate}", {
        "X": bundle.X.astype(np.float32, copy=False),
        "y": bundle.y.astype(np.int16, copy=False),
        "trial_id": np.asarray(bundle.trial_id, dtype="<U160"),
    })
    payload = cache.self_hashed_manifest({
        "schema_version": "c74_cross_node_preprocessing_capture_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "replicate": replicate, "hostname": hostname,
        "dataset_evidence_hash": loaded.evidence.evidence_hash,
        "raw_data_fingerprint": loaded.evidence.raw_data_fingerprint,
        "resolved_preprocess_hash": bundle.preprocess_hash,
        "full_tensor_hash": tensor_content_hash(bundle.X),
        "network_attempt_count": loaded.evidence.network_attempt_count,
        "shape": list(bundle.X.shape), "dtype": str(bundle.X.dtype),
        "descriptor": descriptor,
        "real_EEG_loaded": True, "model_forward_attempted": False,
        "training_attempted": False, "GPU_used": False,
    })
    path = _manifest_path(protocol, replicate)
    cache.atomic_json(path, payload)
    return {"manifest_path": str(path), **payload}


def _load_capture(protocol: dict, replicate: str) -> tuple[dict, dict[str, np.ndarray]]:
    path = _manifest_path(protocol, replicate)
    manifest = cache.verify_unit_manifest(path, rehash_payloads=False)
    descriptor = manifest["descriptor"]
    cache.verify_shard(descriptor, required_fields={"X", "y", "trial_id"})
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        arrays = {name: shard[name] for name in shard.files}
    return manifest, arrays


def _compare(token: str, num_threads: int) -> dict:
    protocol = cache.load_locked_protocol()
    if not cache.authorization_ok(token):
        raise PermissionError("C74 preprocessing replay comparison requires exact CLI authorization")
    left_manifest, left = _load_capture(protocol, REPLICATES[0])
    right_manifest, right = _load_capture(protocol, REPLICATES[1])
    if not np.array_equal(left["trial_id"], right["trial_id"]) or not np.array_equal(left["y"], right["y"]):
        raise RuntimeError("C74 cross-node trial-ID/label alignment failed")
    input_difference = np.abs(left["X"].astype(float) - right["X"].astype(float))

    import torch
    from oaci.models import build_model

    torch.set_num_threads(max(1, int(num_threads)))
    torch.set_num_interop_threads(1)
    torch.set_grad_enabled(False)
    row = cache.stage_rows("P0_pilot", 1)[0]
    loaded_state = c66._state_load_metadata(row)
    if loaded_state["load_status"] != "pass" or not int(loaded_state["state_hash_matches_checkpoint_id"]):
        raise RuntimeError("C74 cross-node replay checkpoint identity failed")
    spec = c66._model_spec_for_row(row)
    model = build_model(
        spec["factory"], in_chans=int(spec["input_shape"][0]), in_times=int(spec["input_shape"][1]),
        n_classes=int(spec["n_classes"]), **dict(spec["backbone"]),
    )
    model.load_state_dict(loaded_state["state"], strict=True)
    model.eval()
    c66._assert_cpu_model_and_tensor(model)
    max_z = 0.0
    max_logits = 0.0
    max_probabilities = 0.0
    prediction_disagreements = 0
    with torch.no_grad():
        for start in range(0, len(left["X"]), 128):
            stop = min(start + 128, len(left["X"]))
            left_output = model(torch.from_numpy(np.ascontiguousarray(left["X"][start:stop])))
            right_output = model(torch.from_numpy(np.ascontiguousarray(right["X"][start:stop])))
            max_z = max(max_z, float(torch.max(torch.abs(left_output.z - right_output.z)).item()))
            max_logits = max(max_logits, float(torch.max(torch.abs(left_output.logits - right_output.logits)).item()))
            left_prob = torch.softmax(left_output.logits, dim=1)
            right_prob = torch.softmax(right_output.logits, dim=1)
            max_probabilities = max(max_probabilities, float(torch.max(torch.abs(left_prob - right_prob)).item()))
            prediction_disagreements += int(torch.sum(torch.argmax(left_output.logits, 1) != torch.argmax(right_output.logits, 1)).item())

    values = {
        "input_max_abs": float(np.max(input_difference)),
        "input_mean_abs": float(np.mean(input_difference)),
        "input_p999_abs": float(np.quantile(input_difference, 0.999)),
        "input_nonzero_fraction": float(np.mean(input_difference != 0)),
        "z_max_abs": max_z,
        "logit_max_abs": max_logits,
        "probability_max_abs": max_probabilities,
        "prediction_disagreements": prediction_disagreements,
    }
    passed = (
        values["input_max_abs"] <= INPUT_MAX_ABS_TOLERANCE
        and values["input_mean_abs"] <= INPUT_MEAN_ABS_TOLERANCE
        and values["z_max_abs"] <= Z_MAX_ABS_TOLERANCE
        and values["logit_max_abs"] <= LOGIT_MAX_ABS_TOLERANCE
        and values["probability_max_abs"] <= PROBABILITY_MAX_ABS_TOLERANCE
        and prediction_disagreements == 0
    )
    payload = cache.self_hashed_manifest({
        "schema_version": "c74_cross_node_preprocessing_comparison_v1",
        "protocol_sha256": cache.sha256_file(cache.PROTOCOL_PATH),
        "left_replicate": left_manifest["replicate"],
        "right_replicate": right_manifest["replicate"],
        "evidence_hashes_equal": left_manifest["dataset_evidence_hash"] == right_manifest["dataset_evidence_hash"],
        "raw_fingerprints_equal": left_manifest["raw_data_fingerprint"] == right_manifest["raw_data_fingerprint"],
        "resolved_preprocess_hashes_equal": left_manifest["resolved_preprocess_hash"] == right_manifest["resolved_preprocess_hash"],
        "trial_ids_equal": True, "labels_equal": True,
        **values,
        "tolerances": {
            "input_max_abs": INPUT_MAX_ABS_TOLERANCE,
            "input_mean_abs": INPUT_MEAN_ABS_TOLERANCE,
            "z_max_abs": Z_MAX_ABS_TOLERANCE,
            "logit_max_abs": LOGIT_MAX_ABS_TOLERANCE,
            "probability_max_abs": PROBABILITY_MAX_ABS_TOLERANCE,
            "prediction_disagreements": 0,
        },
        "same_frozen_T2_checkpoint_used": True,
        "model_eval": True, "gradients_disabled": True,
        "training_attempted": False, "GPU_used": False,
        "passed": passed,
        "interpretation": "cross_node_float32_precision_audit_not_scientific_outcome",
    })
    path = _directory(protocol) / "cross_node_preprocessing_comparison.json"
    cache.atomic_json(path, payload)
    if not passed:
        raise RuntimeError(f"C74 cross-node preprocessing drift exceeded tolerance: {values}")
    return {"comparison_path": str(path), **payload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    capture = sub.add_parser("capture")
    capture.add_argument("--replicate", choices=REPLICATES, required=True)
    capture.add_argument("--authorization-token", required=True)
    capture.add_argument("--datalake-root", default="/projects/EEG-foundation-model/datalake/raw")
    capture.add_argument("--num-threads", type=int, default=48)
    compare = sub.add_parser("compare")
    compare.add_argument("--authorization-token", required=True)
    compare.add_argument("--num-threads", type=int, default=48)
    args = parser.parse_args(argv)
    if args.command == "capture":
        result = _capture(args.replicate, args.authorization_token, args.datalake_root, args.num_threads)
    else:
        result = _compare(args.authorization_token, args.num_threads)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
