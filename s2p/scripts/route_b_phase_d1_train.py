#!/usr/bin/env python
"""Train one Phase-D1 nested-compute CBraMod trajectory.

Primary snapshots are fixed global updates on one uninterrupted trajectory.
Validation is diagnostic only and no downstream labels are read.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import torch


WINDOW_SAMPLES = 6000
PATCH_SAMPLES = 200
PATCHES = 30
CHANNELS = 33
MODEL_KWARGS = {
    "in_dim": 200,
    "out_dim": 200,
    "d_model": 200,
    "dim_feedforward": 800,
    "seq_len": 30,
    "n_layer": 12,
    "nhead": 8,
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_sha(value) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(payload).hexdigest()


def array_sha(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode() + b"\0")
    digest.update(str(tuple(value.shape)).encode() + b"\0")
    digest.update(value.tobytes())
    return digest.hexdigest()


def state_sha(model: torch.nn.Module) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous().numpy()
        digest.update(name.encode() + b"\0")
        digest.update(str(value.dtype).encode() + b"\0")
        digest.update(str(tuple(value.shape)).encode() + b"\0")
        digest.update(value.tobytes())
    return digest.hexdigest()


def derived_seed(namespace: str, *parts) -> int:
    text = "|".join([namespace, *(str(part) for part in parts)])
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**31 - 1)


def configure_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def append_jsonl(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    print(json.dumps(value, sort_keys=True), flush=True)


def load_schedule_row(path: Path, trajectory_id: str, unique_data_h: int) -> dict:
    rows = [
        row
        for row in read_csv(path)
        if row["trajectory_id"] == trajectory_id
        and int(float(row["unique_data_h"])) == unique_data_h
    ]
    if len(rows) != 2 or {row["snapshot_role"] for row in rows} != {"P_low", "P_high"}:
        raise RuntimeError(f"schedule does not define two snapshots for {trajectory_id}")
    return {row["snapshot_role"]: row for row in rows}


def load_pairing_row(path: Path, trajectory_id: str, unique_data_h: int) -> dict:
    rows = [
        row
        for row in read_csv(path)
        if row["trajectory_id"] == trajectory_id
        and int(float(row["unique_data_h"])) == unique_data_h
    ]
    if len(rows) != 1:
        raise RuntimeError(f"pairing manifest mismatch for {trajectory_id}")
    return rows[0]


def load_manifest(path: Path, subset_seed: int, unique_data_h: int) -> list[dict]:
    rows = [
        row
        for row in read_csv(path)
        if int(row["subset_seed"]) == subset_seed
        and int(float(row["unique_data_h"])) == unique_data_h
    ]
    rows.sort(key=lambda row: (row["group_id"], int(row["allocation_rank"])))
    expected = unique_data_h * 120
    observed = sum(int(row["take_windows"]) for row in rows)
    if observed != expected:
        raise RuntimeError(f"manifest window count {observed} != {expected}")
    return rows


def manifest_identity_sha(rows: list[dict]) -> str:
    digest = hashlib.sha256()
    values = []
    for row in rows:
        values.extend(
            (
                int(row["subject"]),
                int(row["recording_id"]),
                row["filepath_relative"],
                window,
            )
            for window in range(
                int(row["window_start"]), int(row["window_stop_exclusive"])
            )
        )
    for subject, recording, filepath, window in sorted(values):
        digest.update(f"{subject}|{recording}|{filepath}|{window}\n".encode())
    return digest.hexdigest()


class WindowStream:
    def __init__(
        self,
        rows: list[dict],
        metadata_path: Path,
        channel_contract: Path,
        data_root: Path,
        batch_size: int,
        loader_seed_root: int,
        unique_data_h: int,
    ):
        self.rows = rows
        self.data_root = data_root
        self.batch_size = batch_size
        self.loader_seed_root = loader_seed_root
        self.unique_data_h = unique_data_h
        metadata = pd.read_parquet(metadata_path)
        metadata = metadata.set_index("filepath", drop=False)
        spec = json.loads(channel_contract.read_text())
        self.orders = {row["group_id"]: list(row["channels"]) for row in spec["groups"]}
        self.channels = {}
        for row in rows:
            filepath = row["filepath_relative"]
            if filepath not in metadata.index:
                raise RuntimeError(f"manifest path absent from metadata: {filepath}")
            entry = metadata.loc[filepath]
            if isinstance(entry, pd.DataFrame):
                raise RuntimeError(f"non-unique metadata filepath: {filepath}")
            source_channels = json.loads(entry["channels"])
            target_channels = self.orders[row["group_id"]]
            try:
                self.channels[filepath] = [source_channels.index(name) for name in target_channels]
            except ValueError as error:
                raise RuntimeError(f"canonical channel missing for {filepath}") from error

    def batches(self, epoch: int, max_windows: int | None = None):
        seed = derived_seed(
            "S2P-D1-loader-epoch-v1", self.loader_seed_root, self.unique_data_h, epoch
        )
        rng = np.random.default_rng(seed)
        order = np.arange(len(self.rows))
        rng.shuffle(order)
        carry = []
        emitted = 0
        for row_index in order:
            row = self.rows[int(row_index)]
            start = int(row["window_start"])
            stop = int(row["window_stop_exclusive"])
            window_ids = np.arange(start, stop)
            rng.shuffle(window_ids)
            filepath = row["filepath_relative"]
            source = np.load(self.data_root / filepath, mmap_mode="r")
            channel_index = self.channels[filepath]
            for window_id in window_ids:
                if max_windows is not None and emitted + len(carry) >= max_windows:
                    break
                begin = int(window_id) * WINDOW_SAMPLES
                raw = np.asarray(
                    source[begin : begin + WINDOW_SAMPLES, channel_index], dtype=np.float32
                )
                if raw.shape != (WINDOW_SAMPLES, CHANNELS):
                    raise RuntimeError(f"window shape mismatch: {filepath}#{window_id} {raw.shape}")
                value = raw.reshape(PATCHES, PATCH_SAMPLES, CHANNELS).transpose(2, 0, 1)
                value = (value - value.mean(-1, keepdims=True)) / (
                    value.std(-1, keepdims=True) + 1e-6
                )
                carry.append(np.ascontiguousarray(value.astype(np.float32)))
                if len(carry) == self.batch_size:
                    batch = np.stack(carry)
                    carry.clear()
                    emitted += len(batch)
                    yield torch.from_numpy(batch)
            if max_windows is not None and emitted + len(carry) >= max_windows:
                break
        if carry:
            raise RuntimeError(
                f"trajectory requires full batches; epoch left {len(carry)} windows"
            )


def validation_rows(repo_root: Path, contract_dir: Path) -> list[dict]:
    sys.path.insert(0, str(repo_root / "s2p" / "scripts"))
    import route_b_33ch_loader as loader

    return loader.build_route_b_cell(200, 0, contract_dir=str(contract_dir))["val"]


def validation_batches(rows, contract_dir: Path, batch_size: int):
    import route_b_33ch_loader as loader

    carry = []
    for windows, _ in loader.windows_for(rows, contract_dir=str(contract_dir)):
        for value in windows:
            carry.append(value)
            if len(carry) == batch_size:
                yield torch.from_numpy(np.stack(carry))
                carry.clear()
    if carry:
        yield torch.from_numpy(np.stack(carry))


def native_loss(model, batch, mask_ratio: float, device: torch.device, seed: int):
    from utils.util import generate_mask

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    batch = batch.to(device, non_blocking=True)
    mask = generate_mask(
        batch.shape[0], batch.shape[1], batch.shape[2], mask_ratio=mask_ratio, device=device
    )
    output = model(batch, mask=mask)
    selected = mask == 1
    if not selected.any():
        raise RuntimeError("native mask selected zero patches")
    return torch.nn.functional.mse_loss(output[selected], batch[selected])


@torch.inference_mode()
def feature_canary(model, batch: torch.Tensor, device: torch.device) -> np.ndarray:
    model.eval()
    value = batch.to(device)
    patch = model.patch_embedding(value, None)
    encoded = model.encoder(patch)
    return np.ascontiguousarray(
        encoded.mean(dim=2).reshape(encoded.shape[0], -1).float().cpu().numpy()
    )


def recursive_equal(left, right) -> bool:
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return torch.equal(left.cpu(), right.cpu())
    if isinstance(left, dict) and isinstance(right, dict):
        return left.keys() == right.keys() and all(recursive_equal(left[k], right[k]) for k in left)
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        return len(left) == len(right) and all(recursive_equal(a, b) for a, b in zip(left, right))
    return left == right


def close_snapshot(
    payload: dict,
    snapshots_dir: Path,
    model_class,
    source_model: torch.nn.Module,
    device: torch.device,
    canary_batch: torch.Tensor,
) -> dict:
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    temporary = snapshots_dir / f"pending_{uuid.uuid4().hex}.pth"
    if temporary.exists():
        raise RuntimeError("snapshot temporary path collision")
    torch.save(payload, temporary)
    with temporary.open("rb") as handle:
        os.fsync(handle.fileno())
    digest = sha256_file(temporary)
    destination = snapshots_dir / f"sha256_{digest}.pth"
    if destination.exists():
        if sha256_file(destination) != digest:
            raise RuntimeError("content-addressed destination collision")
        temporary.unlink()
    else:
        os.link(temporary, destination)
        temporary.unlink()
    destination.chmod(0o444)

    source_feature = feature_canary(source_model, canary_batch, device)
    loaded = torch.load(destination, map_location="cpu", weights_only=False)
    if loaded["global_update"] != payload["global_update"]:
        raise RuntimeError("snapshot global update changed on reload")
    check = model_class(**MODEL_KWARGS).to(device)
    result = check.load_state_dict(loaded["model_state"], strict=True)
    if result.missing_keys or result.unexpected_keys:
        raise RuntimeError("snapshot strict reload failed")
    if not recursive_equal(payload["model_state"], loaded["model_state"]):
        raise RuntimeError("snapshot model tensors changed on reload")
    if not recursive_equal(payload["optimizer_state"], loaded["optimizer_state"]):
        raise RuntimeError("snapshot optimizer state changed on reload")
    if not recursive_equal(payload["scheduler_state"], loaded["scheduler_state"]):
        raise RuntimeError("snapshot scheduler state changed on reload")

    check_optimizer = torch.optim.AdamW(check.parameters(), lr=1.0)
    check_optimizer.load_state_dict(loaded["optimizer_state"])
    check_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        check_optimizer,
        T_max=int(loaded["scheduler_state"]["T_max"]),
        eta_min=float(loaded["scheduler_state"]["eta_min"]),
    )
    check_scheduler.load_state_dict(loaded["scheduler_state"])
    if not recursive_equal(check_optimizer.state_dict(), loaded["optimizer_state"]):
        raise RuntimeError("snapshot optimizer load_state_dict verification failed")
    if not recursive_equal(check_scheduler.state_dict(), loaded["scheduler_state"]):
        raise RuntimeError("snapshot scheduler load_state_dict verification failed")

    before = feature_canary(check, canary_batch, device)
    repeat = feature_canary(check, canary_batch, device)
    reload_diff = float(np.max(np.abs(source_feature - before)))
    repeat_diff = float(np.max(np.abs(before - repeat)))
    if reload_diff != 0.0 or repeat_diff != 0.0:
        raise RuntimeError("snapshot feature canary is not bitwise deterministic")
    observed = sha256_file(destination)
    if observed != digest:
        raise RuntimeError("snapshot hash changed after closure")
    return {
        "immutable_path": str(destination),
        "immutable_sha256": digest,
        "size_bytes": destination.stat().st_size,
        "mode": oct(destination.stat().st_mode & 0o777),
        "strict_reload_pass": True,
        "model_parameter_exact_pass": True,
        "optimizer_state_exact_pass": True,
        "scheduler_state_exact_pass": True,
        "feature_sha256": array_sha(before),
        "feature_reload_max_abs_diff": reload_diff,
        "feature_repeat_max_abs_diff": repeat_diff,
    }


def run_validation(model, rows, contract_dir, batch_size, mask_ratio, device, index):
    model.eval()
    losses = []
    with torch.inference_mode():
        for batch_index, batch in enumerate(validation_batches(rows, contract_dir, batch_size)):
            seed = derived_seed("S2P-D1-validation-mask-v1", index, batch_index)
            loss = native_loss(model, batch, mask_ratio, device, seed)
            losses.append(float(loss.cpu()))
    value = float(np.mean(losses)) if losses else float("nan")
    if not np.isfinite(value):
        raise RuntimeError("non-finite validation loss")
    return value


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--cbramod-root", type=Path, required=True)
    parser.add_argument("--trajectory-id", required=True)
    parser.add_argument("--subset-seed", type=int, required=True)
    parser.add_argument("--init-seed", type=int, required=True)
    parser.add_argument("--unique-data-h", type=int, choices=(200, 1000), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol-dir", type=Path, default=Path("results/s2p_route_b_phase_d1_protocol"))
    parser.add_argument("--contract-dir", type=Path, default=Path("results/s2p_route_b_33ch_contract"))
    parser.add_argument("--data-root", type=Path, default=Path("/projects/EEG-foundation-model/datalake/processed/4704743c/TUEG"))
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--base-lr", type=float, default=5e-4)
    parser.add_argument("--eta-min", type=float, default=1e-5)
    parser.add_argument("--weight-decay", type=float, default=5e-2)
    parser.add_argument("--mask-ratio", type=float, default=0.5)
    parser.add_argument("--clip-value", type=float, default=1.0)
    parser.add_argument("--low-update", type=int, default=18_750)
    parser.add_argument("--high-update", type=int, default=93_750)
    parser.add_argument("--validation-cadence", type=int, default=1_875)
    parser.add_argument("--max-manifest-windows", type=int)
    parser.add_argument("--canary", action="store_true")
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    protocol_dir = (repo_root / args.protocol_dir).resolve()
    contract_dir = (repo_root / args.contract_dir).resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train_log.jsonl"
    if log_path.exists() and not args.canary:
        raise RuntimeError("refusing to overwrite an existing scientific trajectory")

    if args.canary:
        if (args.low_update, args.high_update, args.validation_cadence) != (2, 4, 2):
            raise RuntimeError("canary must use the frozen 2/4-update contract")
        if args.max_manifest_windows is None or args.max_manifest_windows % args.batch_size:
            raise RuntimeError("canary window cap must be a positive multiple of batch size")
    else:
        if (args.low_update, args.high_update, args.validation_cadence) != (18_750, 93_750, 1_875):
            raise RuntimeError("scientific trajectory update contract changed")
        if args.max_manifest_windows is not None:
            raise RuntimeError("scientific trajectory cannot cap the frozen manifest")

    schedule = load_schedule_row(
        protocol_dir / "phase_d1_update_schedule.csv", args.trajectory_id, args.unique_data_h
    )
    pairing = load_pairing_row(
        protocol_dir / "phase_d1_initial_state_pairing.csv", args.trajectory_id, args.unique_data_h
    )
    manifest = load_manifest(
        protocol_dir / "phase_d1_nested_subset_manifest.csv", args.subset_seed, args.unique_data_h
    )
    expected_subset_sha = pairing["subset_window_sha256"]
    observed_subset_sha = manifest_identity_sha(manifest)
    if observed_subset_sha != expected_subset_sha:
        raise RuntimeError(
            f"subset identity hash mismatch expected={expected_subset_sha} observed={observed_subset_sha}"
        )
    if not args.canary and (
        int(schedule["P_low"]["snapshot_update"]) != args.low_update
        or int(schedule["P_high"]["snapshot_update"]) != args.high_update
    ):
        raise RuntimeError("runtime snapshot updates differ from frozen schedule")
    if int(pairing["subset_seed"]) != args.subset_seed or int(pairing["init_seed"]) != args.init_seed:
        raise RuntimeError("trajectory identity differs from pairing manifest")

    sys.path.insert(0, str(args.cbramod_root.resolve()))
    sys.path.insert(0, str(repo_root / "s2p" / "scripts"))
    from models.cbramod import CBraMod

    configure_seed(args.init_seed)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if not args.canary and device.type != "cuda":
        raise RuntimeError("scientific D1 training requires CUDA")
    model = CBraMod(**MODEL_KWARGS).to(device)
    initial_hash = state_sha(model)
    if initial_hash != pairing["initial_state_sha256"]:
        raise RuntimeError(
            f"initial state mismatch expected={pairing['initial_state_sha256']} observed={initial_hash}"
        )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.high_update, eta_min=args.eta_min
    )
    stream = WindowStream(
        manifest,
        args.data_root / "metadata.parquet",
        contract_dir / "route_b_canonical_channel_order.json",
        args.data_root,
        args.batch_size,
        int(pairing["loader_seed_root"]),
        args.unique_data_h,
    )
    val_rows = validation_rows(repo_root, contract_dir)
    canary_batch = next(validation_batches(val_rows, contract_dir, batch_size=2))
    canary_input_sha = array_sha(canary_batch.numpy())

    try:
        code_commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=repo_root, text=True
        ).strip()
    except Exception:
        code_commit = "unknown"
    config = {
        **vars(args),
        "repo_root": str(repo_root),
        "protocol_dir": str(protocol_dir),
        "contract_dir": str(contract_dir),
        "output_dir": str(output_dir),
        "data_root": str(args.data_root),
    }
    config_sha = canonical_sha(config)
    run_contract = {
        "phase": "D1_unique_data_x_cumulative_exposure",
        "trajectory_id": args.trajectory_id,
        "subset_seed": args.subset_seed,
        "init_seed": args.init_seed,
        "stream_seed": int(pairing["stream_seed"]),
        "loader_seed_root": int(pairing["loader_seed_root"]),
        "mask_seed_root": int(pairing["mask_seed_root"]),
        "unique_data_h": args.unique_data_h,
        "subset_window_sha256": expected_subset_sha,
        "initial_state_sha256": initial_hash,
        "code_commit": code_commit,
        "config_sha256": config_sha,
        "low_update": args.low_update,
        "high_update": args.high_update,
        "validation_cadence": args.validation_cadence,
        "batch_size": args.batch_size,
        "best_val_used_for_primary": False,
        "target_labels_used": False,
        "canary": args.canary,
        "canary_input_sha256": canary_input_sha,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
    }
    write_json(output_dir / "run_contract.json", run_contract)
    append_jsonl(log_path, {"event": "start", **run_contract})

    global_update = 0
    epoch = 0
    validation_index = 0
    snapshot_rows = []
    started = time.time()
    snapshots = {args.low_update: "P_low", args.high_update: "P_high"}
    while global_update < args.high_update:
        model.train()
        epoch_batches = 0
        epoch_losses = []
        for batch in stream.batches(epoch, max_windows=args.max_manifest_windows):
            if global_update >= args.high_update:
                break
            update_number = global_update + 1
            seed = derived_seed(
                "S2P-D1-mask-update-v1", int(pairing["mask_seed_root"]), update_number
            )
            optimizer.zero_grad(set_to_none=True)
            loss = native_loss(model, batch, args.mask_ratio, device, seed)
            if not torch.isfinite(loss):
                raise RuntimeError(f"NaN/Inf at update {update_number}")
            loss.backward()
            if args.clip_value > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_value)
            optimizer.step()
            scheduler.step()
            global_update = update_number
            epoch_batches += 1
            epoch_losses.append(float(loss.detach().cpu()))

            if global_update in snapshots:
                role = snapshots[global_update]
                payload = {
                    "model_state": {k: v.detach().cpu() for k, v in model.state_dict().items()},
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "global_update": global_update,
                    "completed_epochs": epoch + (epoch_batches * args.batch_size) / (
                        args.max_manifest_windows or args.unique_data_h * 120
                    ),
                    "subset_seed": args.subset_seed,
                    "init_seed": args.init_seed,
                    "stream_seed": int(pairing["stream_seed"]),
                    "subset_window_sha256": expected_subset_sha,
                    "initial_state_sha256": initial_hash,
                    "config_sha256": config_sha,
                    "code_commit": code_commit,
                    "trajectory_id": args.trajectory_id,
                    "unique_data_h": args.unique_data_h,
                    "snapshot_role": role,
                    "lr_after_update": optimizer.param_groups[0]["lr"],
                    "target_labels_used": False,
                }
                closure = close_snapshot(
                    payload,
                    output_dir / "snapshots",
                    CBraMod,
                    model,
                    device,
                    canary_batch,
                )
                row = {**payload, **closure}
                for key in ("model_state", "optimizer_state", "scheduler_state"):
                    row.pop(key)
                snapshot_rows.append(row)
                write_json(output_dir / f"{role}_snapshot_closure.json", row)
                append_jsonl(log_path, {"event": "snapshot", **row})

            if global_update % args.validation_cadence == 0:
                validation_index += 1
                value = run_validation(
                    model,
                    val_rows,
                    contract_dir,
                    args.batch_size,
                    args.mask_ratio,
                    device,
                    validation_index,
                )
                append_jsonl(
                    log_path,
                    {
                        "event": "validation",
                        "global_update": global_update,
                        "validation_index": validation_index,
                        "val_loss": value,
                        "selection_use": "diagnostic_only",
                    },
                )
                model.train()
        if epoch_batches == 0:
            raise RuntimeError("data stream produced zero full batches")
        epoch += 1
        append_jsonl(
            log_path,
            {
                "event": "epoch",
                "epoch": epoch,
                "global_update": global_update,
                "batches": epoch_batches,
                "mean_train_loss": float(np.mean(epoch_losses)),
                "lr": optimizer.param_groups[0]["lr"],
            },
        )

    if len(snapshot_rows) != 2 or {row["snapshot_role"] for row in snapshot_rows} != {"P_low", "P_high"}:
        raise RuntimeError("trajectory did not close both fixed-update snapshots")
    final_state = {
        "status": "PASS_CANARY" if args.canary else "PASS_TRAINING",
        "trajectory_id": args.trajectory_id,
        "unique_data_h": args.unique_data_h,
        "global_update": global_update,
        "epochs_completed": epoch,
        "snapshots": [row["immutable_sha256"] for row in snapshot_rows],
        "initial_state_sha256": initial_hash,
        "common_lr_schedule": True,
        "fixed_update_snapshots": True,
        "best_val_used_for_primary": False,
        "target_labels_used": False,
        "wall_seconds": round(time.time() - started, 3),
    }
    write_json(output_dir / "run_complete.json", final_state)
    append_jsonl(log_path, {"event": "complete", **final_state})


if __name__ == "__main__":
    main()
