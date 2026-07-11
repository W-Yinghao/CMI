#!/usr/bin/env python
"""Build the native CodeBrain ISRUC_S3 20-epoch sequence tree.

The implementation preserves the upstream 0.3-35 Hz FIR, 50 Hz notch,
six-channel order, last-30-label exclusion, stage 5->4 remap, and tail trim
to a multiple of 20. Filtering is vectorized only after an exact equivalence
canary against the upstream per-epoch loop.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
from pathlib import Path

os.environ.setdefault("MNE_DONTWRITE_HOME", "true")
os.environ.setdefault("MNE_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("MPLCONFIGDIR", "/tmp/s2p_mplconfig")

import mne
import numpy as np
import scipy.io
from scipy import signal


CHANNELS = ("F3_A2", "C3_A2", "O1_A2", "F4_A1", "C4_A1", "O2_A1")
SFREQ = 200
EPOCH_SAMPLES = 6000
SEQUENCE_EPOCHS = 20
PUBLIC_ASSET_MANIFEST = (
    "results/s2p_codebrain_bounded_preflight/isruc_s3_recovery/"
    "isruc_s3_subject_asset_manifest.csv"
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def filter_native(x: np.ndarray) -> np.ndarray:
    x = mne.filter.filter_data(
        x, sfreq=SFREQ, l_freq=0.3, h_freq=35, fir_design="firwin", verbose=False
    )
    return mne.filter.notch_filter(x, Fs=SFREQ, freqs=50, verbose=False)


def archive_labels(archive: Path, subject: int) -> np.ndarray:
    member = f"{subject}/{subject}_1.txt"
    text = subprocess.check_output(["bsdtar", "-xOf", str(archive), member], text=True)
    return np.asarray([int(line.strip()) for line in text.splitlines() if line.strip()], dtype=np.int64)


def load_channels(mat_path: Path) -> np.ndarray:
    data = scipy.io.loadmat(mat_path, variable_names=CHANNELS)
    arrays = [signal.resample(np.asarray(data[channel]), EPOCH_SAMPLES, axis=-1) for channel in CHANNELS]
    return np.stack(arrays, axis=1)


def aggregate_tree_sha(root: Path) -> str:
    h = hashlib.sha256()
    for path in sorted(root.rglob("*.npy")):
        h.update(str(path.relative_to(root)).encode("ascii"))
        h.update(b"\0")
        with path.open("rb") as f:
            for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
                h.update(block)
    return h.hexdigest()


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
    ap.add_argument("--asset-manifest", required=True)
    ap.add_argument("--output-root", required=True)
    ap.add_argument("--report-dir", required=True)
    args = ap.parse_args()
    asset_root = Path(args.asset_root)
    output_root = Path(args.output_root)
    report_dir = Path(args.report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    if output_root.exists():
        raise RuntimeError(f"fail-closed: output root already exists: {output_root}")

    with Path(args.asset_manifest).open(newline="") as f:
        authority = {int(row["subject"]): row for row in csv.DictReader(f)}
    if set(authority) != set(range(1, 11)) or not all(row["subject_asset_pass"] == "True" for row in authority.values()):
        raise RuntimeError("raw asset authority is not 10/10 PASS")

    tmp_root = output_root.with_name(f"{output_root.name}.tmp_{os.getpid()}")
    if tmp_root.exists():
        raise RuntimeError(f"temporary root already exists: {tmp_root}")
    (tmp_root / "seq").mkdir(parents=True)
    (tmp_root / "labels").mkdir(parents=True)

    subject_rows = []
    equivalence = None
    for subject in range(1, 11):
        source = authority[subject]
        archive = asset_root / f"{subject}.rar"
        mat_path = asset_root / f"subject{subject}.mat"
        if sha256(archive) != source["raw_sha256"] or sha256(mat_path) != source["mat_sha256"]:
            raise RuntimeError(f"subject {subject}: source SHA drift")
        raw = load_channels(mat_path)
        if raw.ndim != 3 or raw.shape[1:] != (6, EPOCH_SAMPLES):
            raise RuntimeError(f"subject {subject}: malformed MAT tensor {raw.shape}")

        if equivalence is None:
            probe = raw[:2]
            upstream = np.stack([filter_native(epoch) for epoch in probe])
            vectorized = filter_native(probe)
            equivalence = {
                "probe_shape": list(probe.shape),
                "max_abs_diff": float(np.max(np.abs(upstream - vectorized))),
                "exact_equal": bool(np.array_equal(upstream, vectorized)),
                "mne_version": mne.__version__,
            }
            if not equivalence["exact_equal"]:
                raise RuntimeError(f"vectorized filter differs from native loop: {equivalence}")

        filtered = filter_native(raw)
        labels = archive_labels(archive, subject)
        if len(labels) - 30 != len(filtered):
            raise RuntimeError(f"subject {subject}: label/MAT mismatch {len(labels)} vs {len(filtered)}+30")
        labels = labels[:-30]
        labels[labels == 5] = 4
        remainder = len(filtered) % SEQUENCE_EPOCHS
        if remainder:
            filtered = filtered[:-remainder]
            labels = labels[:-remainder]
        sequences = filtered.reshape(-1, SEQUENCE_EPOCHS, 6, EPOCH_SAMPLES)
        label_sequences = labels.reshape(-1, SEQUENCE_EPOCHS)
        if len(sequences) != len(label_sequences):
            raise RuntimeError(f"subject {subject}: sequence/label count mismatch")

        seq_dir = tmp_root / "seq" / f"ISRUC-group3-{subject}"
        label_dir = tmp_root / "labels" / f"ISRUC-group3-{subject}"
        seq_dir.mkdir()
        label_dir.mkdir()
        for index, (sequence, label_sequence) in enumerate(zip(sequences, label_sequences)):
            stem = f"ISRUC-group3-{subject}-{index}"
            np.save(seq_dir / f"{stem}.npy", sequence)
            np.save(label_dir / f"{stem}.npy", label_sequence)

        first_x = np.load(sorted(seq_dir.glob("*.npy"))[0], mmap_mode="r")
        first_y = np.load(sorted(label_dir.glob("*.npy"))[0], mmap_mode="r")
        checks = {
            "source_sha": True,
            "sequence_count_match": len(sequences) == int(source["twenty_epoch_sequences"]),
            "sequence_shape": list(first_x.shape) == [20, 6, 6000],
            "label_shape": list(first_y.shape) == [20],
            "sequence_dtype_native": str(first_x.dtype) == "float64",
            "label_dtype": str(first_y.dtype) == "int64",
            "class_support": set(np.unique(label_sequences).tolist()) == {0, 1, 2, 3, 4},
        }
        subject_rows.append({
            "subject": subject, "input_epochs": int(raw.shape[0]),
            "tail_labels_excluded": 30, "remainder_epochs_trimmed": int(remainder),
            "usable_epochs": int(len(filtered)), "sequence_count": int(len(sequences)),
            "sequence_shape": "20;6;6000", "sequence_dtype": str(first_x.dtype),
            "label_shape": "20", "label_dtype": str(first_y.dtype),
            "class_support": ";".join(map(str, sorted(np.unique(label_sequences).tolist()))),
            "subject_sequence_pass": bool(all(checks.values())),
            "checks_json": json.dumps(checks, sort_keys=True),
        })
        del raw, filtered, sequences, label_sequences

    if not all(row["subject_sequence_pass"] for row in subject_rows):
        raise RuntimeError("one or more subject sequence contracts failed")
    tree_sha = aggregate_tree_sha(tmp_root)
    external_manifest = {
        "dataset": "ISRUC_S3_Group_III",
        "source_asset_manifest": PUBLIC_ASSET_MANIFEST,
        "channel_order": list(CHANNELS), "sfreq": SFREQ, "epoch_samples": EPOCH_SAMPLES,
        "filter": "MNE FIR 0.3-35Hz then 50Hz notch",
        "tail_labels_excluded": 30, "stage_remap": "5_to_4",
        "sequence_epochs": SEQUENCE_EPOCHS, "subjects": 10,
        "sequences": int(sum(row["sequence_count"] for row in subject_rows)),
        "usable_epochs": int(sum(row["usable_epochs"] for row in subject_rows)),
        "vectorized_filter_equivalence": equivalence,
        "tree_sha256": tree_sha,
        "processed_sequence_contract_pass": True,
    }
    (tmp_root / "processed_manifest.json").write_text(json.dumps(external_manifest, indent=2, sort_keys=True) + "\n")
    os.replace(tmp_root, output_root)

    report = dict(external_manifest)
    report.update({
        "output_root": "${ISRUC_PROCESSED_ROOT}",
        "rotating_8_1_1_split_manifest": "isruc_s3_split_manifest.csv",
        "stage2_training_unblocked_by_isruc_contract": True,
        "labels_used_for_model_or_protocol_selection": False,
        "training_launched": False, "fine_tuning_launched": False,
    })
    write_csv(report_dir / "isruc_s3_processed_subject_manifest.csv", subject_rows)
    (report_dir / "isruc_s3_processed_sequence_contract.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
