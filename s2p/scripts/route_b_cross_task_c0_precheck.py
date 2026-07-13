#!/usr/bin/env python
"""Metadata-only Phase C0 contract and provenance precheck."""

import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import stat
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np


TAGS = [
    "random",
    "released",
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
    "H2000_s0",
    "H2000_s1",
]
SEEDV_KEY = re.compile(
    r"^(?P<subject>[0-9]+)_(?P<session>[0-9]+)_(?P<date>[0-9]+)[.]cnt-"
    r"(?P<trial>[0-9]+)-(?P<window>[0-9]+)$"
)
ISRUC_FILE = re.compile(r"^ISRUC-group3-(?P<subject>[0-9]+)-(?P<sequence>[0-9]+)[.]npy$")

# The processed LMDB stores no channel-name metadata. This is the single order
# used by the upstream SEED-V 62-channel CNT conversion and is pinned here as a
# contract; C0 records that it cannot independently recover names from LMDB.
SEEDV_CHANNEL_ORDER = [
    "FP1", "FPZ", "FP2", "AF3", "AF4", "F7", "F5", "F3", "F1", "FZ",
    "F2", "F4", "F6", "F8", "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2",
    "FC4", "FC6", "FT8", "T7", "C5", "C3", "C1", "CZ", "C2", "C4",
    "C6", "T8", "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6",
    "TP8", "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8", "CB1", "O1", "OZ",
    "O2", "CB2",
]


def sha256_file(path, chunk_size=8 * 1024 * 1024):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fobj:
        while True:
            block = fobj.read(chunk_size)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def write_csv(path, rows):
    rows = list(rows)
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fobj:
        writer = csv.DictWriter(fobj, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def truth(value):
    return str(value).lower() == "true"


def public_path(path, roots):
    if isinstance(path, str) and path.startswith("random_init_contract://"):
        return path
    resolved = str(Path(path).resolve())
    for root, token in roots:
        prefix = str(Path(root).resolve())
        if resolved == prefix:
            return token
        if resolved.startswith(prefix + os.sep):
            return token + "/" + resolved[len(prefix) + 1:]
    return "${EXTERNAL_CONTENT_ADDRESSED_ARTIFACT}/" + Path(resolved).name


def checkpoint_precheck(manifest_path, artifact_root):
    rows = read_csv(manifest_path)
    if len(rows) != 10 or [row["tag"] for row in rows] != TAGS:
        raise RuntimeError("Phase C requires the ordered ten-object immutable manifest")
    report = []
    public = []
    for row in rows:
        tag = row["tag"]
        common_pass = all(
            truth(row[field])
            for field in ("strict_reload_pass", "parameter_exact_pass", "feature_equivalence_pass")
        ) and float(row["feature_max_abs_diff"]) == 0.0
        if not common_pass:
            raise RuntimeError(f"immutable closure not passed for {tag}")
        if tag == "random":
            observed = row["immutable_sha256"]
            path_pass = row["immutable_path"].startswith("random_init_contract://sha256_")
            mode = "logical_contract"
            size = "NA"
        else:
            payload = Path(row["immutable_path"])
            path_pass = (
                payload.is_file()
                and not payload.is_symlink()
                and not (stat.S_IMODE(payload.stat().st_mode) & 0o222)
                and row["immutable_sha256"] in payload.name
            )
            if not path_pass:
                raise RuntimeError(f"invalid immutable payload contract for {tag}")
            observed = sha256_file(payload)
            mode = oct(stat.S_IMODE(payload.stat().st_mode))
            size = payload.stat().st_size
        hash_pass = observed == row["immutable_sha256"]
        if not path_pass or not hash_pass:
            raise RuntimeError(f"checkpoint precheck failed for {tag}")
        report.append({
            "tag": tag,
            "expected_sha256": row["immutable_sha256"],
            "observed_sha256": observed,
            "size_bytes": size,
            "mode": mode,
            "path_contract_pass": path_pass,
            "hash_pass": hash_pass,
            "strict_reload_closed": truth(row["strict_reload_pass"]),
            "feature_equivalence_closed": truth(row["feature_equivalence_pass"]),
        })
        public.append({
            "tag": tag,
            "role": row["role"],
            "budget_h": row["budget_h"],
            "seed": row["seed"],
            "immutable_path": public_path(
                row["immutable_path"], [(artifact_root, "${PHASE_B_ARTIFACT_ROOT}")]
            ),
            "immutable_sha256": row["immutable_sha256"],
            "feature_hash": row["feature_hash"],
            "selection_metric": row["selection_metric"],
            "provenance_status": row["provenance_status"],
        })
    return report, public


def seedv_precheck(root):
    root = Path(root)
    env = lmdb.open(str(root), readonly=True, lock=False, readahead=False, meminit=False)
    expected_trials = {"train": set(range(5)), "val": set(range(5, 10)), "test": set(range(10, 15))}
    groups = defaultdict(lambda: {"labels": set(), "windows": [], "keys": []})
    split_rows = []
    checksum = hashlib.sha256()
    with env.begin(write=False) as txn:
        keys_by_split = pickle.loads(txn.get(b"__keys__"))
        if set(keys_by_split) != set(expected_trials):
            raise RuntimeError("SEED-V split manifest differs from train/val/test contract")
        for split in ("train", "val", "test"):
            for key in keys_by_split[split]:
                text = key.decode() if isinstance(key, bytes) else key
                match = SEEDV_KEY.fullmatch(text)
                if match is None:
                    raise RuntimeError(f"unparseable SEED-V key: {text}")
                trial = int(match.group("trial"))
                if trial not in expected_trials[split]:
                    raise RuntimeError(f"SEED-V trial in wrong split: {text}")
                subject = int(match.group("subject"))
                session = int(match.group("session"))
                window = int(match.group("window"))
                record = f"{subject}_{session}_{match.group('date')}.cnt"
                raw = txn.get(text.encode())
                item = pickle.loads(raw)
                sample = np.asarray(item["sample"])
                label = int(item["label"])
                if sample.shape != (62, 1, 200) or label not in range(5):
                    raise RuntimeError(f"SEED-V sample contract failed: {text}")
                group_id = (record, trial)
                groups[group_id]["labels"].add(label)
                groups[group_id]["windows"].append(window)
                groups[group_id]["keys"].append(text)
                groups[group_id]["split"] = split
                groups[group_id]["subject"] = subject
                groups[group_id]["session"] = session
                checksum.update(text.encode() + b"\0")
                checksum.update(str(label).encode() + b"\0")
    env.close()

    trial_rows = []
    split_groups = defaultdict(set)
    for (record, trial), group in sorted(groups.items()):
        if len(group["labels"]) != 1:
            raise RuntimeError(f"SEED-V trial has nonconstant label: {record}-{trial}")
        windows = sorted(group["windows"])
        if windows != list(range(len(windows))):
            raise RuntimeError(f"SEED-V window chronology is not contiguous: {record}-{trial}")
        trial_id = f"{record}-{trial}"
        split_groups[group["split"]].add(trial_id)
        trial_rows.append({
            "trial_id": trial_id,
            "subject": group["subject"],
            "session": group["session"],
            "record": record,
            "trial": trial,
            "split": group["split"],
            "label": next(iter(group["labels"])),
            "windows": len(windows),
            "first_window": windows[0],
            "last_window": windows[-1],
            "trial_label_constant": True,
            "window_chronology_contiguous": True,
        })
    overlap = {
        "train_val": len(split_groups["train"] & split_groups["val"]),
        "train_test": len(split_groups["train"] & split_groups["test"]),
        "val_test": len(split_groups["val"] & split_groups["test"]),
    }
    if any(overlap.values()):
        raise RuntimeError("SEED-V trial leakage across splits")
    for split in ("train", "val", "test"):
        current = [row for row in trial_rows if row["split"] == split]
        split_rows.append({
            "split": split,
            "windows": sum(row["windows"] for row in current),
            "trials": len(current),
            "subjects": len({row["subject"] for row in current}),
            "subject_sessions": len({(row["subject"], row["session"]) for row in current}),
            "trial_ids_sha256": canonical_sha([row["trial_id"] for row in current]),
            "trial_groups_disjoint": True,
        })
    all_pass = (
        len(trial_rows) == 705
        and all(row["trials"] == 235 for row in split_rows)
        and all(row["subjects"] == 16 for row in split_rows)
        and all(row["subject_sessions"] == 47 for row in split_rows)
    )
    if not all_pass:
        raise RuntimeError("SEED-V aggregate contract mismatch")
    channel = {
        "dataset": "SEED-V",
        "channel_count": 62,
        "channel_order": SEEDV_CHANNEL_ORDER,
        "channel_order_sha256": canonical_sha(SEEDV_CHANNEL_ORDER),
        "order_source": "pinned_upstream_SEED-V_CNT_conversion_contract",
        "lmdb_contains_channel_name_metadata": False,
        "multiple_channel_orders_detected": False,
        "native_input_only": True,
    }
    summary = {
        "dataset": "SEED-V",
        "root": "${SEEDV_ROOT}",
        "subjects": 16,
        "subject_sessions": 47,
        "trials": 705,
        "classes": 5,
        "sample_shape": [62, 1, 200],
        "split_trial_sets": {"train": "0-4", "val": "5-9", "test": "10-14"},
        "trial_group_overlap": overlap,
        "keys_and_labels_sha256": checksum.hexdigest(),
        "trial_mean_is_analysis_unit": True,
        "asset_contract_pass": all_pass,
    }
    return trial_rows, split_rows, channel, summary


def isruc_precheck(root, accepted_contract, accepted_split):
    root = Path(root)
    processed = json.loads((root / "processed_manifest.json").read_text())
    accepted = json.loads(Path(accepted_contract).read_text())
    if processed["tree_sha256"] != accepted["tree_sha256"]:
        raise RuntimeError("ISRUC processed-tree authority mismatch")
    split_rows = read_csv(accepted_split)
    if len(split_rows) != 10 or not all(truth(row["subject_disjoint"]) for row in split_rows):
        raise RuntimeError("ISRUC rotating split contract mismatch")
    sequence_rows = []
    epoch_total = 0
    for subject in range(1, 11):
        seq_dir = root / "seq" / f"ISRUC-group3-{subject}"
        label_dir = root / "labels" / f"ISRUC-group3-{subject}"
        seq_files = {}
        label_files = {}
        for path in seq_dir.glob("*.npy"):
            match = ISRUC_FILE.fullmatch(path.name)
            if match is None or int(match.group("subject")) != subject:
                raise RuntimeError(f"invalid ISRUC sequence filename: {path.name}")
            seq_files[int(match.group("sequence"))] = path
        for path in label_dir.glob("*.npy"):
            match = ISRUC_FILE.fullmatch(path.name)
            if match is None or int(match.group("subject")) != subject:
                raise RuntimeError(f"invalid ISRUC label filename: {path.name}")
            label_files[int(match.group("sequence"))] = path
        if set(seq_files) != set(label_files) or set(seq_files) != set(range(len(seq_files))):
            raise RuntimeError(f"ISRUC sequence chronology mismatch for subject {subject}")
        for sequence in sorted(seq_files):
            x = np.load(seq_files[sequence], mmap_mode="r")
            y = np.load(label_files[sequence], mmap_mode="r")
            if x.shape != (20, 6, 6000) or y.shape != (20,) or not set(np.unique(y)).issubset(range(5)):
                raise RuntimeError(f"ISRUC sequence payload mismatch: subject={subject} seq={sequence}")
            sequence_rows.append({
                "subject": subject,
                "sequence": sequence,
                "chronological_epoch_start": sequence * 20,
                "chronological_epoch_end": sequence * 20 + 19,
                "shape": "20;6;6000",
                "label_shape": "20",
                "classes_present": ";".join(str(int(v)) for v in sorted(np.unique(y))),
                "sequence_path": f"${{ISRUC_PROCESSED_ROOT}}/seq/ISRUC-group3-{subject}/{seq_files[sequence].name}",
                "label_path": f"${{ISRUC_PROCESSED_ROOT}}/labels/ISRUC-group3-{subject}/{label_files[sequence].name}",
                "sequence_contract_pass": True,
            })
            epoch_total += 20
    channel_order = processed["channel_order"]
    summary = {
        "dataset": "ISRUC_S3_Group_III",
        "root": "${ISRUC_PROCESSED_ROOT}",
        "subjects": 10,
        "sequences": len(sequence_rows),
        "epochs": epoch_total,
        "sequence_shape": [20, 6, 6000],
        "classes": 5,
        "rotations": len(split_rows),
        "split": "rotating_8_1_1_subject_wise",
        "tree_sha256": accepted["tree_sha256"],
        "channel_order": channel_order,
        "channel_order_sha256": canonical_sha(channel_order),
        "multiple_channel_orders_detected": False,
        "processed_filter_equivalence_max_abs_diff": accepted["vectorized_filter_equivalence"]["max_abs_diff"],
        "sequence_contract_pass": len(sequence_rows) == 425 and epoch_total == 8500,
    }
    if not summary["sequence_contract_pass"]:
        raise RuntimeError("ISRUC aggregate sequence contract mismatch")
    return sequence_rows, split_rows, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--phase-b-artifact-root", type=Path, required=True)
    parser.add_argument("--seedv-root", type=Path, required=True)
    parser.add_argument("--isruc-root", type=Path, required=True)
    parser.add_argument("--isruc-contract", type=Path, required=True)
    parser.add_argument("--isruc-split", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    checkpoint_rows, public_checkpoints = checkpoint_precheck(
        args.checkpoint_manifest, args.phase_b_artifact_root
    )
    seedv_trials, seedv_splits, seedv_channels, seedv_summary = seedv_precheck(args.seedv_root)
    isruc_sequences, isruc_splits, isruc_summary = isruc_precheck(
        args.isruc_root, args.isruc_contract, args.isruc_split
    )

    out = args.out_dir
    write_csv(out / "phase_c0_checkpoint_hash_recheck.csv", checkpoint_rows)
    write_csv(out / "phase_c0_representation_contracts.csv", public_checkpoints)
    write_csv(out / "seedv_trial_manifest.csv", seedv_trials)
    write_csv(out / "seedv_split_manifest.csv", seedv_splits)
    write_json(out / "seedv_channel_order.json", seedv_channels)
    write_json(out / "seedv_dataset_contract.json", seedv_summary)
    write_csv(out / "isruc_s3_sequence_manifest.csv", isruc_sequences)
    write_csv(out / "isruc_s3_rotation_manifest.csv", isruc_splits)
    write_json(out / "isruc_s3_dataset_contract.json", isruc_summary)

    protocol = args.repo_root / "docs/S2P_27_CBRAMOD_CROSS_TASK_PROTOCOL.md"
    redteam = args.repo_root / "docs/S2P_28_CBRAMOD_CROSS_TASK_REDTEAM.md"
    go = {
        "phase": "C0_cross_task_protocol_and_asset_precheck",
        "status": "PASS",
        "checkpoint_objects_expected": 10,
        "checkpoint_objects_hash_verified": len(checkpoint_rows),
        "mutable_checkpoint_path_used": False,
        "seedv_asset_contract_pass": seedv_summary["asset_contract_pass"],
        "seedv_trial_group_leakage": False,
        "seedv_primary_analysis_unit": "trial_mean",
        "isruc_s3_asset_contract_pass": isruc_summary["sequence_contract_pass"],
        "isruc_s3_sequence_overlap": False,
        "isruc_s3_split": "rotating_8_1_1_subject_wise",
        "protocol_sha256": sha256_file(protocol),
        "redteam_sha256": sha256_file(redteam),
        "target_labels_used_for_selection": False,
        "scientific_endpoint_computed": False,
        "encoder_training_launched": False,
        "encoder_fine_tuning_launched": False,
        "seedv_gate_ready": True,
        "isruc_s3_gate_ready": True,
        "auto_launch_gate": False,
        "auto_launch_fleet": False,
    }
    write_json(out / "phase_c0_go_nogo.json", go)
    print(json.dumps(go, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
