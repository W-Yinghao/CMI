#!/usr/bin/env python
"""Provenance-only immutable checkpoint closure for S2P Phase B.

No scientific endpoint is computed. Seven mutable checkpoint sources are
copied into content-addressed, no-overwrite payloads. The complete ten-object
set then receives byte, strict-model, parameter, and checksum-feature checks.
"""
import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import shutil
import stat
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import lmdb
import numpy as np
import torch

sys.path.insert(0, os.path.expanduser("~/eeg2025/CBraMod"))
from models.cbramod import CBraMod


JOB_IDS = {
    "H200_s0": "890125_0",
    "H200_s1": "890147_1",
    "H500_s0": "890147_2",
    "H500_s1": "890147_3",
    "H1000_s0": "890147_4",
    "H1000_s1": "890147_5",
    "H2000_s0": "890151_6",
    "H2000_s1": "890151_7",
}
LOW_TAGS = ["H200_s0", "H200_s1", "H500_s0", "H500_s1", "H1000_s0", "H1000_s1"]
HIGH_TAGS = ["H2000_s0", "H2000_s1"]
ALL_TAGS = ["random", "released", *LOW_TAGS, *HIGH_TAGS]
CHECKSUM_KEYS = [
    f"sub{subject:03d}.pkl-{clip}-{class_id % 3}"
    for subject in (0, 79)
    for class_id, clip in enumerate((0, 3, 6, 9, 12, 16, 19, 22, 25))
]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            chunk = fobj.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(obj):
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def tensor_state_sha(state):
    digest = hashlib.sha256()
    for key in sorted(state):
        tensor = state[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(json.dumps(list(tensor.shape), separators=(",", ":")).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def write_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n")


def write_csv(path, rows):
    path = Path(path)
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def make_model(seed=0):
    torch.manual_seed(seed)
    return CBraMod(
        in_dim=200,
        out_dim=200,
        d_model=200,
        dim_feedforward=800,
        seq_len=10,
        n_layer=12,
        nhead=8,
    )


def unwrap_state_dict(obj):
    if not isinstance(obj, dict):
        raise RuntimeError("checkpoint is not a dictionary")
    for key in ("model_state", "model", "state_dict", "model_state_dict"):
        if key in obj and isinstance(obj[key], dict):
            obj = obj[key]
            break
    if not isinstance(obj, dict) or not obj:
        raise RuntimeError("checkpoint does not contain a state dictionary")
    return {(key[7:] if key.startswith("module.") else key): value for key, value in obj.items()}


def strict_model_from_state(state):
    model = make_model()
    result = model.load_state_dict(state, strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError(
            f"strict reload mismatch missing={result.missing_keys} unexpected={result.unexpected_keys}"
        )
    return model


def state_exact(left, right):
    if set(left) != set(right):
        return False, []
    unequal = [key for key in sorted(left) if not torch.equal(left[key], right[key])]
    return not unequal, unequal


def stable_source(path):
    path = Path(path)
    before = path.stat()
    first = sha256_file(path)
    middle = path.stat()
    second = sha256_file(path)
    after = path.stat()
    signatures = [
        (item.st_size, item.st_mtime_ns) for item in (before, middle, after)
    ]
    if len(set(signatures)) != 1 or first != second:
        raise RuntimeError(f"source changed across consecutive SHA checks: {path}")
    return {
        "path": str(path.resolve()),
        "mode": oct(stat.S_IMODE(before.st_mode)),
        "size": before.st_size,
        "mtime_ns": before.st_mtime_ns,
        "sha256_first": first,
        "sha256_second": second,
    }


def content_addressed_copy(source, artifact_root):
    source = Path(source)
    stable = stable_source(source)
    digest = stable["sha256_first"]
    destination = Path(artifact_root) / f"sha256_{digest}.pt"
    reused = destination.exists()
    if reused:
        if destination.is_symlink():
            raise RuntimeError(f"content-addressed destination must not be a symlink: {destination}")
        if destination.stat().st_size != stable["size"] or sha256_file(destination) != digest:
            raise RuntimeError(f"existing content-addressed destination conflicts: {destination}")
    else:
        temporary = Path(artifact_root) / f".{destination.name}.tmp.{os.getpid()}"
        try:
            with source.open("rb") as src, temporary.open("xb") as dst:
                shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
                dst.flush()
                os.fsync(dst.fileno())
            if temporary.stat().st_size != stable["size"] or sha256_file(temporary) != digest:
                raise RuntimeError(f"temporary copy verification failed: {temporary}")
            os.chmod(temporary, 0o444)
            try:
                os.link(temporary, destination)
            except FileExistsError:
                if destination.stat().st_size != stable["size"] or sha256_file(destination) != digest:
                    raise RuntimeError(f"concurrent destination conflict: {destination}")
            temporary.unlink(missing_ok=True)
        finally:
            temporary.unlink(missing_ok=True)
    os.chmod(destination, 0o444)
    if destination.stat().st_size != stable["size"] or sha256_file(destination) != digest:
        raise RuntimeError(f"destination byte-integrity check failed: {destination}")
    source_after = source.stat()
    source_sha_after = sha256_file(source)
    if (
        source_after.st_size != stable["size"]
        or source_after.st_mtime_ns != stable["mtime_ns"]
        or source_sha_after != digest
    ):
        raise RuntimeError(f"source changed during copy: {source}")
    return stable, destination, reused


def load_checksum_batch(lmdb_path):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    samples = []
    with env.begin() as txn:
        for key in CHECKSUM_KEYS:
            raw = txn.get(key.encode())
            if raw is None:
                raise RuntimeError(f"checksum feature key missing: {key}")
            obj = pickle.loads(raw)
            sample = np.asarray(obj["sample"], dtype=np.float32)
            if sample.shape != (32, 10, 200):
                raise RuntimeError(f"checksum sample shape mismatch: {key} {sample.shape}")
            samples.append(sample)
    env.close()
    unnormalized = np.stack(samples).astype(np.float32)
    normalized = (
        unnormalized - unnormalized.mean(-1, keepdims=True)
    ) / (unnormalized.std(-1, keepdims=True) + 1e-6)
    return normalized.astype(np.float32), {
        "keys": CHECKSUM_KEYS,
        "keys_sha256": canonical_sha(CHECKSUM_KEYS),
        "shape": list(normalized.shape),
        "unnormalized_float32_sha256": hashlib.sha256(unnormalized.tobytes()).hexdigest(),
        "normalized_float32_sha256": hashlib.sha256(normalized.astype(np.float32).tobytes()).hexdigest(),
        "target_labels_read": False,
    }


@torch.inference_mode()
def feature(model, batch, device):
    model = model.to(device).eval()
    tensor = torch.from_numpy(batch).to(device)
    patch = model.patch_embedding(tensor, None)
    encoded = model.encoder(patch)
    output = encoded.mean(2).reshape(encoded.shape[0], -1).float().cpu().numpy()
    return np.ascontiguousarray(output)


def verify_model_pair(tag, source_path, immutable_path, batch, device, random_contract=False):
    if random_contract:
        source_model = make_model(seed=0)
        immutable_model = make_model(seed=0)
        source_state = source_model.state_dict()
        immutable_state = immutable_model.state_dict()
        strict_probe = make_model(seed=1)
        strict_probe.load_state_dict(source_state, strict=True)
        source_obj = {}
        immutable_obj = {}
    else:
        source_obj = torch.load(source_path, map_location="cpu", weights_only=False)
        immutable_obj = torch.load(immutable_path, map_location="cpu", weights_only=False)
        source_state = unwrap_state_dict(source_obj)
        immutable_state = unwrap_state_dict(immutable_obj)
        source_model = strict_model_from_state(source_state)
        immutable_model = strict_model_from_state(immutable_state)
    exact, unequal = state_exact(source_state, immutable_state)
    if not exact:
        raise RuntimeError(f"{tag} parameter mismatch: {unequal[:5]}")
    source_feature = feature(source_model, batch, device)
    source_repeat = feature(source_model, batch, device)
    immutable_feature = feature(immutable_model, batch, device)
    repeat_diff = float(np.max(np.abs(source_feature - source_repeat)))
    copy_diff = float(np.max(np.abs(source_feature - immutable_feature)))
    if repeat_diff != 0.0 or copy_diff != 0.0:
        raise RuntimeError(
            f"{tag} functional equivalence is not exact repeat={repeat_diff} copy={copy_diff}"
        )
    source_feature_sha = hashlib.sha256(source_feature.tobytes()).hexdigest()
    immutable_feature_sha = hashlib.sha256(immutable_feature.tobytes()).hexdigest()
    if source_feature_sha != immutable_feature_sha:
        raise RuntimeError(f"{tag} feature hashes differ despite zero max difference")
    source_state_sha = tensor_state_sha(source_state)
    immutable_state_sha = tensor_state_sha(immutable_state)
    if source_state_sha != immutable_state_sha:
        raise RuntimeError(f"{tag} parameter-state hashes differ")
    del source_model, immutable_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return {
        "source_obj": source_obj,
        "immutable_obj": immutable_obj,
        "reload": {
            "tag": tag,
            "strict_reload_pass": True,
            "missing_key_count": 0,
            "unexpected_key_count": 0,
            "parameter_tensor_count": len(source_state),
            "parameter_exact_pass": True,
            "source_parameter_sha256": source_state_sha,
            "immutable_parameter_sha256": immutable_state_sha,
        },
        "feature": {
            "tag": tag,
            "source_path": str(source_path),
            "immutable_path": str(immutable_path),
            "checksum_batch_size": len(batch),
            "feature_shape": "x".join(map(str, source_feature.shape)),
            "source_repeat_max_abs_diff": repeat_diff,
            "source_vs_immutable_max_abs_diff": copy_diff,
            "source_feature_sha256": source_feature_sha,
            "immutable_feature_sha256": immutable_feature_sha,
            "feature_equivalence_pass": True,
        },
    }


def checkpoint_provenance(tag, source_obj, summary):
    if tag in ("random", "released"):
        return {
            "selected_epoch": "NA",
            "selection_metric": "not_applicable",
            "selection_metric_value": "NA",
            "config_sha256": "NA",
            "data_manifest_sha256": "NA",
            "code_commit": "NA",
        }
    config = source_obj.get("config") or {}
    route_manifest = source_obj.get("route_b_manifest") or summary.get("cell_manifest") or {}
    epoch = int(source_obj.get("epoch", -1))
    val_loss = float(source_obj.get("val_loss"))
    if epoch != int(summary["best_epoch"]):
        raise RuntimeError(f"{tag} selected epoch differs from run summary")
    if not math.isclose(val_loss, float(summary["best_val_loss"]), rel_tol=0, abs_tol=1e-12):
        raise RuntimeError(f"{tag} selected val loss differs from run summary")
    if summary.get("target_labels_used") is not False:
        raise RuntimeError(f"{tag} target-label selection firewall failed")
    return {
        "selected_epoch": epoch,
        "selection_metric": "pretrain_val_loss_only",
        "selection_metric_value": val_loss,
        "config_sha256": canonical_sha(config),
        "data_manifest_sha256": canonical_sha(route_manifest),
        "code_commit": str(source_obj.get("git") or "unrecorded_in_checkpoint"),
    }


def run(args):
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    b0 = json.loads(Path(args.b0_go_nogo).read_text())
    expected_blockers = {"released", *LOW_TAGS}
    if set(b0.get("immutable_blocking_tags", [])) != expected_blockers:
        raise RuntimeError("B0 blocker set differs from the approved seven-object closure")
    if b0.get("phase_b1_compute_authorized") is not False:
        raise RuntimeError("B0 must keep B1 scientific compute disabled")

    artifact_root = Path(args.artifact_root)
    artifact_root.mkdir(parents=True, exist_ok=True)
    os.chmod(artifact_root, 0o755)
    copied = {}
    source_root = Path(args.source_root)
    for tag in LOW_TAGS:
        copied[tag] = content_addressed_copy(source_root / tag / "best.pth", artifact_root)
    copied["released"] = content_addressed_copy(args.released_checkpoint, artifact_root)

    h2000_manifest = {
        row["tag"]: row
        for row in json.loads(Path(args.h2000_manifest).read_text())["checkpoints"]
    }
    for tag in HIGH_TAGS:
        source = Path(h2000_manifest[tag]["source_checkpoint"])
        immutable = Path(h2000_manifest[tag]["immutable_checkpoint"])
        stable = stable_source(source)
        if stable["sha256_first"] != h2000_manifest[tag]["sha256"]:
            raise RuntimeError(f"{tag} mutable source no longer matches Phase-A immutable payload")
        if sha256_file(immutable) != h2000_manifest[tag]["sha256"]:
            raise RuntimeError(f"{tag} Phase-A immutable SHA mismatch")

    batch, batch_manifest = load_checksum_batch(args.faced_lmdb)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("closure requested CUDA but no GPU is visible")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    manifest_rows = []
    reload_rows = []
    feature_rows = []
    copy_rows = []

    model_code_files = [
        Path(os.path.expanduser("~/eeg2025/CBraMod/models/cbramod.py")),
        Path(os.path.expanduser("~/eeg2025/CBraMod/models/criss_cross_transformer.py")),
    ]
    random_contract = {
        "architecture": {
            "in_dim": 200,
            "out_dim": 200,
            "d_model": 200,
            "dim_feedforward": 800,
            "seq_len": 10,
            "n_layer": 12,
            "nhead": 8,
        },
        "torch_seed": 0,
        "model_source_sha256": {str(path): sha256_file(path) for path in model_code_files},
        "torch_version": torch.__version__,
    }
    random_contract_sha = canonical_sha(random_contract)
    random_path = f"random_init_contract://sha256_{random_contract_sha}"
    random_check = verify_model_pair(
        "random", random_path, random_path, batch, device, random_contract=True
    )
    reload_rows.append(random_check["reload"])
    feature_rows.append(random_check["feature"])
    manifest_rows.append({
        "tag": "random",
        "role": "random_init_reference",
        "budget_h": "NA",
        "seed": 0,
        "training_run_id": "NA",
        "training_job_id": "NA",
        "source_path": random_path,
        "source_mode": "logical_contract",
        "source_size_bytes": "NA",
        "source_sha256": random_contract_sha,
        "selected_epoch": "NA",
        "selection_metric": "not_applicable",
        "selection_metric_value": "NA",
        "config_sha256": random_contract_sha,
        "data_manifest_sha256": "NA",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "immutable_path": random_path,
        "immutable_size_bytes": "NA",
        "immutable_sha256": random_contract_sha,
        "immutable_mode": "logical_contract",
        "strict_reload_pass": True,
        "parameter_exact_pass": True,
        "feature_hash": random_check["feature"]["immutable_feature_sha256"],
        "feature_max_abs_diff": 0.0,
        "feature_equivalence_pass": True,
        "training_provenance": "deterministic_code_and_seed_contract",
        "provenance_status": "PASS_DETERMINISTIC_CONTRACT",
    })

    for tag in ["released", *LOW_TAGS, *HIGH_TAGS]:
        if tag in copied:
            stable, immutable_path, reused = copied[tag]
            source_path = Path(stable["path"])
            source_sha = stable["sha256_first"]
            copy_rows.append({
                "tag": tag,
                "source_path": str(source_path),
                "source_sha256_first": stable["sha256_first"],
                "source_sha256_second": stable["sha256_second"],
                "source_double_sha_stable": True,
                "source_size_bytes": stable["size"],
                "immutable_path": str(immutable_path.resolve()),
                "immutable_sha256": sha256_file(immutable_path),
                "immutable_size_bytes": immutable_path.stat().st_size,
                "destination_reused_same_hash": reused,
                "destination_no_overwrite_contract": True,
                "byte_integrity_pass": True,
            })
        else:
            row = h2000_manifest[tag]
            source_path = Path(row["source_checkpoint"])
            immutable_path = Path(row["immutable_checkpoint"])
            source_sha = row["sha256"]

        check = verify_model_pair(tag, source_path, immutable_path, batch, device)
        reload_rows.append(check["reload"])
        feature_rows.append(check["feature"])
        summary = {}
        if tag in LOW_TAGS or tag in HIGH_TAGS:
            summary = json.loads((source_root / tag / "run_summary.json").read_text())
        provenance = checkpoint_provenance(tag, check["source_obj"], summary)
        budget = float(tag[1:].split("_s")[0]) if tag.startswith("H") else "NA"
        seed = int(tag.rsplit("_s", 1)[1]) if tag.startswith("H") else "NA"
        training_provenance = (
            "externally_released_locally_unverified" if tag == "released"
            else "route_b_run_summary_and_checkpoint"
        )
        manifest_rows.append({
            "tag": tag,
            "role": "released_reference" if tag == "released" else "route_b_checkpoint",
            "budget_h": budget,
            "seed": seed,
            "training_run_id": tag if tag.startswith("H") else "NA",
            "training_job_id": JOB_IDS.get(tag, "NA"),
            "source_path": str(source_path.resolve()),
            "source_mode": oct(stat.S_IMODE(source_path.stat().st_mode)),
            "source_size_bytes": source_path.stat().st_size,
            "source_sha256": source_sha,
            **provenance,
            "immutable_path": str(Path(immutable_path).resolve()),
            "immutable_size_bytes": Path(immutable_path).stat().st_size,
            "immutable_sha256": sha256_file(immutable_path),
            "immutable_mode": oct(stat.S_IMODE(Path(immutable_path).stat().st_mode)),
            "strict_reload_pass": True,
            "parameter_exact_pass": True,
            "feature_hash": check["feature"]["immutable_feature_sha256"],
            "feature_max_abs_diff": check["feature"]["source_vs_immutable_max_abs_diff"],
            "feature_equivalence_pass": True,
            "training_provenance": training_provenance,
            "provenance_status": (
                "PASS_REFERENCE_ONLY" if tag == "released" else "PASS"
            ),
        })

    expected_tags = set(ALL_TAGS)
    if {row["tag"] for row in manifest_rows} != expected_tags:
        raise RuntimeError("closure manifest does not contain the exact ten-object set")
    if not all(
        row["strict_reload_pass"]
        and row["parameter_exact_pass"]
        and row["feature_equivalence_pass"]
        and float(row["feature_max_abs_diff"]) == 0.0
        for row in manifest_rows
    ):
        raise RuntimeError("complete-set integrity gate failed")
    for row in manifest_rows:
        if row["tag"] == "random":
            continue
        immutable = Path(row["immutable_path"])
        if immutable.is_symlink() or stat.S_IMODE(immutable.stat().st_mode) & 0o222:
            raise RuntimeError(f"B1 immutable path contract failed: {row['tag']}")
        if row["immutable_sha256"] not in immutable.name:
            raise RuntimeError(f"B1 immutable filename is not content-addressed: {row['tag']}")

    os.chmod(artifact_root, 0o555)
    write_csv(out / "phase_b_checkpoint_immutable_manifest.csv", manifest_rows)
    write_csv(out / "phase_b_checkpoint_reload_verification.csv", reload_rows)
    write_csv(out / "phase_b_feature_equivalence_rerun.csv", feature_rows)
    copy_report = {
        "phase": "B_checkpoint_content_addressed_copy",
        "approved_copy_count": 7,
        "copied_or_reused_count": len(copy_rows),
        "artifact_root": str(artifact_root.resolve()),
        "artifact_root_mode": oct(stat.S_IMODE(artifact_root.stat().st_mode)),
        "destination_pattern": "sha256_<FULL_SHA256>.pt",
        "overwrite_allowed": False,
        "immutable_bit_attempted": False,
        "scientific_immutability_basis": "content_addressed_path_plus_runtime_sha_verification",
        "checksum_feature_batch": batch_manifest,
        "copies": copy_rows,
    }
    write_json(out / "phase_b_checkpoint_copy_verification.json", copy_report)
    closure = {
        "phase": "B_checkpoint_provenance_closure",
        "status": "PASS",
        "checkpoint_count_expected": 10,
        "checkpoint_count_immutable": 10,
        "physical_checkpoint_payloads": 9,
        "deterministic_random_contracts": 1,
        "all_checkpoint_sha256_pinned": True,
        "all_checkpoint_strict_reload_pass": True,
        "all_checkpoint_parameter_exact_pass": True,
        "all_checkpoint_feature_equivalence_pass": True,
        "feature_equivalence_required_max_abs_diff": 0.0,
        "mutable_checkpoint_path_used_by_b1": False,
        "checkpoint_selection_pretrain_val_only": True,
        "target_labels_used": False,
        "released_training_provenance_verified": False,
        "released_reference_use": "path_validity_reference_only",
        "training_launched": False,
        "fine_tuning_launched": False,
        "h4000_launched": False,
        "scientific_metrics_computed": False,
        "phase_b1_compute_authorized": False,
        "phase_b1_go_recommended": True,
        "requires_pm_review_before_b1": True,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    write_json(out / "phase_b_provenance_closure.json", closure)
    return closure


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1",
    )
    parser.add_argument(
        "--h2000-manifest",
        default="results/s2p_route_b_h2000_immutable_closure/h2000_immutable_checkpoint_manifest.json",
    )
    parser.add_argument(
        "--released-checkpoint",
        default="/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth",
    )
    parser.add_argument(
        "--artifact-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_phase_b_checkpoints",
    )
    parser.add_argument("--faced-lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    parser.add_argument(
        "--b0-go-nogo",
        default="results/s2p_route_b_representation_emergence_b0/phase_b0_go_nogo.json",
    )
    parser.add_argument(
        "--out-dir",
        default="results/s2p_route_b_phase_b_checkpoint_closure",
    )
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    try:
        result = run(args)
    except Exception as exc:
        artifact_root = Path(args.artifact_root)
        if artifact_root.exists():
            os.chmod(artifact_root, 0o555)
        write_json(out / "phase_b_provenance_closure.json", {
            "phase": "B_checkpoint_provenance_closure",
            "status": "NO_GO",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "phase_b1_compute_authorized": False,
            "training_launched": False,
            "fine_tuning_launched": False,
            "scientific_metrics_computed": False,
        })
        raise
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
