#!/usr/bin/env python
"""Validate recovered official ISRUC Cohort III assets without preprocessing.

The audit reads archive members, hypnogram integrity, MAT metadata, and the
native rotating 8:1:1 subject split. It does not fit or evaluate a model.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path

import scipy.io


REQUIRED_CHANNELS = ("F3_A2", "C3_A2", "O1_A2", "F4_A1", "C4_A1", "O2_A1")
PUBLIC_RAW_ROOT = "${ISRUC_RAW_ROOT}"
RAW_SIZES = (100454737, 98784355, 86035580, 82531970, 99163582,
             88092365, 85564530, 105913283, 101662269, 83451473)
MAT_SIZES = (471779462, 464984211, 405571429, 389373364, 465968893,
             419919503, 401075065, 495354754, 479110670, 390324461)
RAW_SHA256 = (
    "22bf6949b0b3f0b032d1064d47747233b5953ab6c5c44018cb00b4f6be391415",
    "3a6c0ace148c4bf9ee01f781456aeb63cc1e9613f9ae70f75918009518e2ec99",
    "9c3055c95b89d28fe4569c86ae5f28d5f601caaab28252e4724d3d7e79ba371b",
    "4cb0f44dbb2d7063f3893e230e875c7302f8ae0978bd953e92d5039c13d98a31",
    "63fc55bf9f06e66b4180d01d55ddf6c5193fbb3551eb45b779981a2378310482",
    "9829691e515d700eecf1704c21862b595ea9f42e6c12c6eb7b9e9571e413e63f",
    "758ce9b5efb15b89b64573dc54dfc2c245cfb5c674c391a5d904dfa251c621c0",
    "f1928d3052fbd16264a0dbd12d15802636ae6089f2d2cc3e76ab353ac6cd2074",
    "74d9fc1cb7c2f18a046b3ca222b2c082e1e77ca0144315274227e01258c9ee24",
    "917c55c1d95013fd909de236d02b414299731d4e2a84fd64730ca6a6e2909a3b",
)
MAT_SHA256 = (
    "471518ee4b3d66a102f73f39988081bafd602454d84c621fe7caba41af8e337d",
    "c3a3224aa5d55060caf2d208842e38e192acf1f58a7bd6af8865ce6158c2a45b",
    "5da5205d9e31c42c46108d5d87f05afd8bd20b51ba790574951b0caa1cc961d3",
    "62cd546033bdca164d4c19b6239419a7dfa5f7b5e4d67086073e8d3a7f03c07d",
    "fe0cf5c372cd39a72fcbb473b15dd620da9aec04edf26f306b1aafbeeb137574",
    "ea85e6d9ef22dfbe8884586509093927e0fe40766f87f2784ff0ab56131df1ff",
    "57597e05473b0226bd44acff2ed70f7108deca7df624abe91d9bc0223c0b1a7d",
    "9e43d63f5eaa39ccfb83ca0e114df7c927917a66a3e80ee848bed10939d9ac82",
    "a5d7edaf4fa3017b7fee41b3332d99931c605c7dae6868e9f6a8778654734bcd",
    "a3d54857dfe69faee74dfd1203707027534d1176e509a831d9a126bdaed4a86d",
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def archive_text(path: Path, member: str) -> str:
    return subprocess.check_output(["bsdtar", "-xOf", str(path), member], text=True)


def archive_members(path: Path) -> list[str]:
    return subprocess.check_output(["bsdtar", "-tf", str(path)], text=True).splitlines()


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset-root", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    root = Path(args.asset_root)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    subjects = []
    for subject in range(1, 11):
        raw = root / f"{subject}.rar"
        mat = root / f"subject{subject}.mat"
        raw_size = raw.stat().st_size
        mat_size = mat.stat().st_size
        raw_hash = sha256(raw)
        mat_hash = sha256(mat)
        members = archive_members(raw)
        label_member = f"{subject}/{subject}_1.txt"
        label_present = label_member in members
        labels = [int(line.strip()) for line in archive_text(raw, label_member).splitlines() if line.strip()]
        mat_vars = {name: (tuple(shape), dtype) for name, shape, dtype in scipy.io.whosmat(mat)}
        required_present = all(name in mat_vars for name in REQUIRED_CHANNELS)
        channel_shapes = {mat_vars[name][0] for name in REQUIRED_CHANNELS} if required_present else set()
        channel_dtypes = {mat_vars[name][1] for name in REQUIRED_CHANNELS} if required_present else set()
        shape_consistent = len(channel_shapes) == 1
        mat_epochs = next(iter(channel_shapes))[0] if shape_consistent else None
        samples_per_epoch = next(iter(channel_shapes))[1] if shape_consistent else None
        label_tail_difference = len(labels) - mat_epochs if mat_epochs is not None else None
        usable_epochs = mat_epochs - (mat_epochs % 20) if mat_epochs is not None else None
        remapped_labels = {4 if value == 5 else value for value in labels[:mat_epochs]} if mat_epochs else set()
        checks = {
            "raw_size": raw_size == RAW_SIZES[subject - 1],
            "mat_size": mat_size == MAT_SIZES[subject - 1],
            "raw_sha256": raw_hash == RAW_SHA256[subject - 1],
            "mat_sha256": mat_hash == MAT_SHA256[subject - 1],
            "label_member": label_present,
            "required_channels": required_present,
            "channel_shape_consistent": shape_consistent,
            "samples_per_epoch": samples_per_epoch == 6000,
            "label_tail_30": label_tail_difference == 30,
            "five_classes_after_5_to_4": remapped_labels == {0, 1, 2, 3, 4},
            "twenty_epoch_sequences": usable_epochs is not None and usable_epochs > 0,
        }
        subjects.append({
            "subject": subject,
            "raw_path": f"{PUBLIC_RAW_ROOT}/{subject}.rar",
            "raw_size_bytes": raw_size, "raw_sha256": raw_hash,
            "mat_path": f"{PUBLIC_RAW_ROOT}/subject{subject}.mat",
            "mat_size_bytes": mat_size, "mat_sha256": mat_hash,
            "archive_members": len(members), "label_member": label_member,
            "label_epochs": len(labels), "label_values": ";".join(map(str, sorted(set(labels)))),
            "mat_epochs": mat_epochs, "samples_per_epoch": samples_per_epoch,
            "required_channel_count": len(REQUIRED_CHANNELS),
            "required_channel_names": ";".join(REQUIRED_CHANNELS),
            "mat_channel_dtypes": ";".join(sorted(channel_dtypes)),
            "tail_epochs_excluded": label_tail_difference,
            "remainder_epochs_trimmed": mat_epochs % 20 if mat_epochs is not None else None,
            "usable_epochs": usable_epochs,
            "twenty_epoch_sequences": usable_epochs // 20 if usable_epochs is not None else None,
            "subject_asset_pass": bool(all(checks.values())),
            "checks_json": json.dumps(checks, sort_keys=True),
        })

    split_rows = []
    all_subjects = set(range(1, 11))
    for fold in range(10):
        val_subject = fold + 1
        test_subject = 1 if fold == 9 else fold + 2
        train_subjects = sorted(all_subjects - {val_subject, test_subject})
        split_rows.append({
            "fold": fold, "train_subjects": ";".join(map(str, train_subjects)),
            "val_subject": val_subject, "test_subject": test_subject,
            "train_count": len(train_subjects), "val_count": 1, "test_count": 1,
            "subject_disjoint": True,
        })

    passed = bool(all(row["subject_asset_pass"] for row in subjects))
    summary = {
        "phase": "S2P_CodeBrain_Bounded_ISRUC_S3_asset_recovery",
        "dataset": "ISRUC_S3_Group_III",
        "official_source": "https://sleeptight.isr.uc.pt/",
        "raw_subject_url_template": "https://dataset.isr.uc.pt/ISRUC_Sleep/subgroupIII/{subject}.rar",
        "mat_subject_url_template": (
            "https://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/"
            "subgroupIII-Extractedchannels/subject{subject}.mat"
        ),
        "subjects_expected": 10, "subjects_validated": sum(r["subject_asset_pass"] for r in subjects),
        "all_content_lengths_exact": all(r["raw_size_bytes"] == RAW_SIZES[r["subject"] - 1]
                                           and r["mat_size_bytes"] == MAT_SIZES[r["subject"] - 1]
                                           for r in subjects),
        "all_sha256_pinned": all(r["raw_sha256"] == RAW_SHA256[r["subject"] - 1]
                                  and r["mat_sha256"] == MAT_SHA256[r["subject"] - 1]
                                  for r in subjects),
        "all_label_mat_tail30_aligned": all(r["tail_epochs_excluded"] == 30 for r in subjects),
        "all_20_epoch_sequence_feasible": all(r["twenty_epoch_sequences"] > 0 for r in subjects),
        "split_folds": 10, "split_contract": "rotating_8_train_1_val_1_test_subjects",
        "raw_asset_contract_pass": passed,
        "processed_sequence_contract_pass": False,
        "stage2_training_unblocked_by_this_audit": False,
        "next_action": "preprocess with pinned CodeBrain-compatible filter/channel/label contract, then verify sequences",
        "labels_read_for_asset_integrity_only": True,
        "labels_used_for_model_or_protocol_selection": False,
        "training_launched": False, "fine_tuning_launched": False,
    }
    write_csv(out / "isruc_s3_subject_asset_manifest.csv", subjects)
    write_csv(out / "isruc_s3_split_manifest.csv", split_rows)
    (out / "isruc_s3_asset_recovery_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
