#!/usr/bin/env python
"""Extract immutable frozen CBraMod features for one Phase C object."""

import argparse
import json
import pickle
import re
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
import torch

from route_b_cross_task_common import (
    TAGS,
    build_model,
    canonical_sha,
    checkpoint_hash,
    extract_features,
    manifest_row,
    normalize_patches,
    sanitize_checkpoint_path,
    sha256_file,
    validate_manifest,
    write_json,
)


SEEDV_KEY = re.compile(
    r"^(?P<subject>[0-9]+)_(?P<session>[0-9]+)_(?P<date>[0-9]+)[.]cnt-"
    r"(?P<trial>[0-9]+)-(?P<window>[0-9]+)$"
)
ISRUC_FILE = re.compile(r"^ISRUC-group3-(?P<subject>[0-9]+)-(?P<sequence>[0-9]+)[.]npy$")


def seedv_groups(root):
    env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False)
    groups = defaultdict(list)
    with env.begin(write=False) as txn:
        keys = pickle.loads(txn.get(b"__keys__"))
    for split in ("train", "val", "test"):
        for key in keys[split]:
            text = key.decode() if isinstance(key, bytes) else key
            match = SEEDV_KEY.fullmatch(text)
            if match is None:
                raise RuntimeError(f"unparseable SEED-V key: {text}")
            record = f"{match.group('subject')}_{match.group('session')}_{match.group('date')}.cnt"
            groups[(split, record, int(match.group("trial")))].append(
                (int(match.group("window")), text)
            )
    for value in groups.values():
        value.sort()
    return env, sorted(groups.items())


def extract_seedv(model, root, device, batch_size):
    env, groups = seedv_groups(root)
    features = []
    labels = []
    subjects = []
    sessions = []
    splits = []
    trial_ids = []
    window_counts = []
    canary_first = None
    with env.begin(write=False) as txn:
        for index, ((split, record, trial), keys) in enumerate(groups):
            if index % 50 == 0:
                print(f"SEED-V trial={index}/{len(groups)}", flush=True)
            if [item[0] for item in keys] != list(range(len(keys))):
                raise RuntimeError(f"non-contiguous SEED-V trial: {record}-{trial}")
            samples = []
            trial_labels = set()
            for _, key in keys:
                item = pickle.loads(txn.get(key.encode()))
                sample = np.asarray(item["sample"], dtype=np.float32)
                if sample.shape != (62, 1, 200):
                    raise RuntimeError(f"SEED-V sample shape mismatch: {key}")
                samples.append(sample)
                trial_labels.add(int(item["label"]))
            if len(trial_labels) != 1:
                raise RuntimeError(f"SEED-V trial label is not constant: {record}-{trial}")
            data = normalize_patches(np.stack(samples))
            if canary_first is None:
                canary_data = data[: min(4, len(data))]
                first = extract_features(model, canary_data, device, batch_size)
                second = extract_features(model, canary_data, device, batch_size)
                repeat_diff = float(np.max(np.abs(first - second)))
                if repeat_diff != 0.0:
                    raise RuntimeError("SEED-V feature canary is not bitwise deterministic")
                canary_first = first
            encoded = extract_features(model, data, device, batch_size)
            features.append(encoded.mean(axis=0, dtype=np.float64).astype(np.float32))
            labels.append(next(iter(trial_labels)))
            subject, session = (int(value) for value in record.split("_")[:2])
            subjects.append(subject)
            sessions.append(session)
            splits.append(split)
            trial_ids.append(f"{record}-{trial}")
            window_counts.append(len(keys))
    env.close()
    payload = {
        "features": np.ascontiguousarray(np.stack(features)),
        "labels": np.asarray(labels, dtype=np.int64),
        "subjects": np.asarray(subjects, dtype=np.int64),
        "sessions": np.asarray(sessions, dtype=np.int64),
        "splits": np.asarray(splits),
        "trial_ids": np.asarray(trial_ids),
        "window_counts": np.asarray(window_counts, dtype=np.int64),
    }
    contract = {
        "dataset": "SEED-V",
        "items": len(features),
        "analysis_unit": "trial_mean",
        "native_channels": 62,
        "patches": 1,
        "feature_dimension": int(payload["features"].shape[1]),
        "canary_feature_sha256": canonical_sha(canary_first.tolist()),
        "canary_repeat_max_abs_diff": 0.0,
        "trial_ids_sha256": canonical_sha(trial_ids),
    }
    return payload, contract


def isruc_files(root):
    entries = []
    for subject in range(1, 11):
        seq_dir = Path(root) / "seq" / f"ISRUC-group3-{subject}"
        label_dir = Path(root) / "labels" / f"ISRUC-group3-{subject}"
        sequences = {}
        labels = {}
        for path in seq_dir.glob("*.npy"):
            match = ISRUC_FILE.fullmatch(path.name)
            if match is not None:
                sequences[int(match.group("sequence"))] = path
        for path in label_dir.glob("*.npy"):
            match = ISRUC_FILE.fullmatch(path.name)
            if match is not None:
                labels[int(match.group("sequence"))] = path
        if set(sequences) != set(labels) or set(sequences) != set(range(len(sequences))):
            raise RuntimeError(f"ISRUC chronology mismatch for subject {subject}")
        entries.extend((subject, seq, sequences[seq], labels[seq]) for seq in sorted(sequences))
    return entries


def extract_isruc(model, root, device, batch_size):
    entries = isruc_files(root)
    features = []
    labels = []
    subjects = []
    sequence_ids = []
    canary_first = None
    for index, (subject, sequence, data_path, label_path) in enumerate(entries):
        if index % 25 == 0:
            print(f"ISRUC sequence={index}/{len(entries)}", flush=True)
        data = np.asarray(np.load(data_path), dtype=np.float32)
        label = np.asarray(np.load(label_path), dtype=np.int64)
        if data.shape != (20, 6, 6000) or label.shape != (20,):
            raise RuntimeError(f"ISRUC sequence shape mismatch: subject={subject} seq={sequence}")
        data = normalize_patches(data.reshape(20, 6, 30, 200))
        if canary_first is None:
            canary_data = data[:4]
            first = extract_features(model, canary_data, device, batch_size)
            second = extract_features(model, canary_data, device, batch_size)
            repeat_diff = float(np.max(np.abs(first - second)))
            if repeat_diff != 0.0:
                raise RuntimeError("ISRUC feature canary is not bitwise deterministic")
            canary_first = first
        features.append(extract_features(model, data, device, batch_size))
        labels.append(label)
        subjects.append(subject)
        sequence_ids.append(sequence)
    payload = {
        "features": np.ascontiguousarray(np.stack(features)),
        "labels": np.ascontiguousarray(np.stack(labels)),
        "subjects": np.asarray(subjects, dtype=np.int64),
        "sequence_ids": np.asarray(sequence_ids, dtype=np.int64),
    }
    contract = {
        "dataset": "ISRUC_S3_Group_III",
        "items": len(features),
        "analysis_unit": "20_epoch_sequence",
        "native_channels": 6,
        "patches_per_epoch": 30,
        "epochs": int(payload["features"].shape[0] * payload["features"].shape[1]),
        "feature_dimension": int(payload["features"].shape[2]),
        "canary_feature_sha256": canonical_sha(canary_first.tolist()),
        "canary_repeat_max_abs_diff": 0.0,
        "subject_sequence_sha256": canonical_sha(
            list(zip(subjects, sequence_ids))
        ),
    }
    return payload, contract


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("seedv", "isruc_s3"), required=True)
    parser.add_argument("--tag", choices=TAGS, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--cbramod-root", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    rows = validate_manifest(args.checkpoint_manifest)
    row = manifest_row(rows, args.tag)
    before_hash = checkpoint_hash(row)
    if before_hash != row["immutable_sha256"]:
        raise RuntimeError(f"pre-extraction hash mismatch for {args.tag}")
    if not args.dataset_root.exists() or not args.cbramod_root.exists():
        raise RuntimeError("dataset or CBraMod root is unavailable")
    if args.dry_run:
        print(json.dumps({
            "status": "PASS_DRY_RUN",
            "dataset": args.dataset,
            "tag": args.tag,
            "checkpoint": sanitize_checkpoint_path(row),
            "trainer_called": False,
            "encoder_optimizer_created": False,
        }, indent=2, sort_keys=True))
        return

    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Phase C feature extraction requires CUDA")
    model = build_model(row, args.cbramod_root, device)
    if args.dataset == "seedv":
        payload, contract = extract_seedv(
            model, args.dataset_root, device, args.batch_size
        )
    else:
        payload, contract = extract_isruc(
            model, args.dataset_root, device, args.batch_size
        )
    after_hash = checkpoint_hash(row)
    if before_hash != after_hash:
        raise RuntimeError(f"post-extraction hash mismatch for {args.tag}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    feature_path = args.out_dir / f"{args.tag}_features.npz"
    if feature_path.exists():
        raise RuntimeError(f"refusing to overwrite feature cache: {feature_path}")
    np.savez_compressed(feature_path, **payload)
    report = {
        "status": "PASS",
        "dataset": args.dataset,
        "tag": args.tag,
        "checkpoint_path": sanitize_checkpoint_path(row),
        "checkpoint_sha256_before": before_hash,
        "checkpoint_sha256_after": after_hash,
        "strict_reload_pass": True,
        "encoder_frozen": True,
        "encoder_optimizer_created": False,
        "fine_tuning_used": False,
        "normalization": "per_channel_per_1s_patch_zscore",
        "payload": f"features/{feature_path.name}",
        "payload_sha256": sha256_file(feature_path),
        **contract,
    }
    write_json(args.out_dir / f"{args.tag}_feature_contract.json", report)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
