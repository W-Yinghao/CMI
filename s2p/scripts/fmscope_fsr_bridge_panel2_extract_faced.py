#!/usr/bin/env python
"""Extract FMScope-aligned pooled FACED features for one immutable object."""

import argparse
import hashlib
import json
import pickle
import re
from pathlib import Path

import lmdb
import numpy as np
import torch

from route_b_cross_task_common import (
    TAGS,
    build_model,
    canonical_sha,
    checkpoint_hash,
    manifest_row,
    normalize_patches,
    validate_manifest,
    write_json,
)


KEY_RE = re.compile(r"^sub(?P<subject>[0-9]+)[.]pkl-(?P<clip>[0-9]+)-(?P<segment>[0-9]+)$")
CLASS_TO_CLIPS = {
    0: [0, 1, 2],
    1: [3, 4, 5],
    2: [6, 7, 8],
    3: [9, 10, 11],
    4: [12, 13, 14, 15],
    5: [16, 17, 18],
    6: [19, 20, 21],
    7: [22, 23, 24],
    8: [25, 26, 27],
}
CLIP_TO_CLASS = {
    clip: label for label, clip_values in CLASS_TO_CLIPS.items() for clip in clip_values
}


@torch.inference_mode()
def pooled_features(model, data, device, batch_size):
    chunks = []
    for start in range(0, len(data), batch_size):
        batch = torch.from_numpy(data[start : start + batch_size]).to(device)
        patch = model.patch_embedding(batch, None)
        encoded = model.encoder(patch)
        chunks.append(encoded.mean(dim=(1, 2)).float().cpu().numpy())
    return np.ascontiguousarray(np.concatenate(chunks).astype(np.float32))


def ordered_faced_keys(path):
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    ordered = []
    with env.begin(write=False) as txn:
        keys_by_split = pickle.loads(txn.get(b"__keys__"))
    env.close()
    for lmdb_split, protocol_split in (
        ("train", "source_train"),
        ("val", "source_val"),
        ("test", "target_test"),
    ):
        for raw_key in keys_by_split[lmdb_split]:
            key = raw_key.decode() if isinstance(raw_key, bytes) else raw_key
            ordered.append((key, protocol_split))
    return ordered


def extract_faced(model, path, device, batch_size):
    ordered = ordered_faced_keys(path)
    env = lmdb.open(str(path), readonly=True, lock=False, readahead=False, meminit=False)
    feature_chunks = []
    labels = []
    subjects = []
    clips = []
    segments = []
    splits = []
    keys_out = []
    canary = None
    with env.begin(write=False) as txn:
        for start in range(0, len(ordered), batch_size):
            batch_samples = []
            for key, protocol_split in ordered[start : start + batch_size]:
                match = KEY_RE.fullmatch(key)
                if match is None:
                    raise RuntimeError(f"unparseable FACED key: {key}")
                item = pickle.loads(txn.get(key.encode()))
                sample = np.asarray(item["sample"], dtype=np.float32)
                if sample.shape != (32, 10, 200):
                    raise RuntimeError(f"FACED sample shape mismatch: {key} {sample.shape}")
                subject = int(match.group("subject")) + 1
                expected = (
                    "source_train" if subject <= 80 else "source_val" if subject <= 100 else "target_test"
                )
                if protocol_split != expected:
                    raise RuntimeError(f"FACED subject split mismatch: {key}")
                label = int(item["label"])
                clip = int(match.group("clip"))
                if CLIP_TO_CLASS[clip] != label:
                    raise RuntimeError(f"FACED clip/label mismatch: {key}")
                batch_samples.append(sample)
                labels.append(label)
                subjects.append(subject)
                clips.append(clip)
                segments.append(int(match.group("segment")))
                splits.append(protocol_split)
                keys_out.append(key)
            batch = normalize_patches(np.stack(batch_samples))
            encoded = pooled_features(model, batch, device, batch_size)
            feature_chunks.append(encoded)
            if canary is None:
                first = pooled_features(model, batch[:4], device, batch_size)
                second = pooled_features(model, batch[:4], device, batch_size)
                canary = (first, float(np.max(np.abs(first - second))))
    env.close()
    identity = list(zip(subjects, clips, segments, splits, labels))
    return np.ascontiguousarray(np.concatenate(feature_chunks)), {
        "labels": np.asarray(labels, dtype=np.int64),
        "subjects": np.asarray(subjects, dtype=np.int64),
        "clips": np.asarray(clips, dtype=np.int64),
        "segments": np.asarray(segments, dtype=np.int64),
        "splits": np.asarray(splits),
        "keys": np.asarray(keys_out),
        "identity_sha256": canonical_sha(identity),
    }, canary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", choices=TAGS, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--cbramod-root", type=Path, required=True)
    parser.add_argument("--faced-lmdb", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    rows = validate_manifest(args.checkpoint_manifest)
    row = manifest_row(rows, args.tag)
    before_hash = checkpoint_hash(row)
    if before_hash != row["immutable_sha256"]:
        raise RuntimeError(f"checkpoint hash mismatch before extraction: {args.tag}")
    device = torch.device(args.device)
    model = build_model(row, args.cbramod_root, device)
    features, metadata, canary = extract_faced(
        model, args.faced_lmdb, device, args.batch_size
    )
    first, repeat_diff = canary
    if repeat_diff != 0.0:
        raise RuntimeError("FACED pooled feature canary is not bitwise deterministic")
    if features.shape != (len(metadata["labels"]), 200) or not np.isfinite(features).all():
        raise RuntimeError(f"FACED pooled feature contract failed: {features.shape}")
    after_hash = checkpoint_hash(row)
    if after_hash != before_hash:
        raise RuntimeError("checkpoint hash changed during extraction")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    payload_path = args.out_dir / f"{args.tag}_faced_pooled_features.npz"
    np.savez_compressed(
        payload_path,
        features=features,
        labels=metadata["labels"],
        subjects=metadata["subjects"],
        clips=metadata["clips"],
        segments=metadata["segments"],
        splits=metadata["splits"],
        keys=metadata["keys"],
    )
    contract = {
        "phase": "FMScope_FSR_Bridge_Panel2",
        "dataset": "FACED",
        "tag": args.tag,
        "representation": "final_channel_and_patch_mean_200d",
        "items": len(features),
        "feature_shape": list(features.shape),
        "feature_sha256": hashlib.sha256(features.tobytes()).hexdigest(),
        "payload_sha256": None,
        "sample_identity_sha256": metadata["identity_sha256"],
        "checkpoint_sha256": before_hash,
        "checkpoint_hash_stable": True,
        "canary_feature_sha256": hashlib.sha256(first.tobytes()).hexdigest(),
        "canary_repeat_max_abs_diff": repeat_diff,
        "target_labels_used_for_selection": False,
    }
    digest = hashlib.sha256()
    with payload_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    contract["payload_sha256"] = digest.hexdigest()
    write_json(args.out_dir / f"{args.tag}_faced_pooled_contract.json", contract)
    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
