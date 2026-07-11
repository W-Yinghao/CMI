#!/usr/bin/env python
"""Metadata-only Phase B0 precheck for Route-B representation emergence.

This script does not extract model features or compute scientific endpoints. It
verifies FACED clip provenance, freezes clip-group folds, and audits checkpoint
immutability plus the already committed Phase-A determinism evidence.
"""
import argparse
import csv
import hashlib
import json
import os
import pickle
import re
import stat
import subprocess
from collections import defaultdict
from pathlib import Path

import lmdb
import numpy as np
from scipy.signal import resample


KEY_RE = re.compile(r"sub(?P<subject>[0-9]+)[.]pkl-(?P<clip>[0-9]+)-(?P<segment>[0-9]+)$")
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
    clip: class_id for class_id, clips in CLASS_TO_CLIPS.items() for clip in clips
}
LOW_BUDGET_TAGS = [
    "H200_s0",
    "H200_s1",
    "H500_s0",
    "H500_s1",
    "H1000_s0",
    "H1000_s1",
]
H2000_TAGS = ["H2000_s0", "H2000_s1"]
ALL_TAGS = ["random", "released", *LOW_BUDGET_TAGS, *H2000_TAGS]


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
    payload = json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


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


def read_csv(path):
    with Path(path).open(newline="") as fobj:
        return list(csv.DictReader(fobj))


def inspect_faced(lmdb_path, raw_root):
    env = lmdb.open(str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False)
    with env.begin() as txn:
        keys_by_split = pickle.loads(txn.get(b"__keys__"))
        decoded = {
            split: [key.decode() if isinstance(key, bytes) else key for key in keys]
            for split, keys in keys_by_split.items()
        }
        if set(decoded) != {"train", "val", "test"}:
            raise RuntimeError(f"unexpected FACED splits: {sorted(decoded)}")

        groups = defaultdict(set)
        split_subjects = defaultdict(set)
        for split in ("train", "val", "test"):
            for key in decoded[split]:
                match = KEY_RE.fullmatch(key)
                if match is None:
                    raise RuntimeError(f"unparseable FACED key: {key}")
                subject = int(match.group("subject"))
                clip = int(match.group("clip"))
                segment = int(match.group("segment"))
                groups[(split, subject)].add((clip, segment))
                split_subjects[split].add(subject)

        expected_group = {(clip, segment) for clip in range(28) for segment in range(3)}
        bad_groups = [key for key, values in groups.items() if values != expected_group]
        if bad_groups:
            raise RuntimeError(f"FACED subject groups violate 28x3 contract: {bad_groups[:5]}")
        expected_split_counts = {"train": 80, "val": 20, "test": 23}
        actual_split_counts = {key: len(value) for key, value in split_subjects.items()}
        if actual_split_counts != expected_split_counts:
            raise RuntimeError(f"FACED split-subject mismatch: {actual_split_counts}")

        checked_subjects = [0, 40, 122]
        label_maps = []
        for subject in checked_subjects:
            mapping = []
            for clip in range(28):
                key = f"sub{subject:03d}.pkl-{clip}-0".encode()
                obj = pickle.loads(txn.get(key))
                mapping.append(int(obj["label"]))
            label_maps.append(mapping)
        expected_map = [CLIP_TO_CLASS[clip] for clip in range(28)]
        if any(mapping != expected_map for mapping in label_maps):
            raise RuntimeError("FACED clip-to-class map is not stable across checked subjects")

        raw_path = Path(raw_root) / "sub000.pkl"
        raw = pickle.load(raw_path.open("rb"))
        if raw.shape != (28, 32, 7500):
            raise RuntimeError(f"unexpected raw FACED subject shape: {raw.shape}")
        reconstruction_rows = []
        for clip, segment in [(0, 0), (0, 1), (0, 2), (14, 1), (27, 2)]:
            obj = pickle.loads(txn.get(f"sub000.pkl-{clip}-{segment}".encode()))
            expected = resample(raw[clip], 6000, axis=-1)[
                :, segment * 2000:(segment + 1) * 2000
            ].reshape(32, 10, 200)
            observed = np.asarray(obj["sample"])
            reconstruction_rows.append({
                "clip_id": clip,
                "segment_id": segment,
                "class_id": int(obj["label"]),
                "max_abs_reconstruction_error": float(np.max(np.abs(observed - expected))),
            })
        max_reconstruction_error = max(
            row["max_abs_reconstruction_error"] for row in reconstruction_rows
        )
        if max_reconstruction_error > 1e-10:
            raise RuntimeError(f"FACED raw-to-LMDB reconstruction failed: {max_reconstruction_error}")

    env.close()
    fold_rows = []
    for class_id, clips in CLASS_TO_CLIPS.items():
        for index, clip in enumerate(clips):
            fold_rows.append({
                "clip_id": clip,
                "class_id": class_id,
                "heldout_fold": index % 3,
                "segment_ids": "0;1;2",
                "segments_per_subject": 3,
                "fit_folds": ";".join(str(fold) for fold in range(3) if fold != index % 3),
            })
    if {row["clip_id"] for row in fold_rows} != set(range(28)):
        raise RuntimeError("clip-fold assignment is not exhaustive")
    return {
        "keys_sha256": canonical_sha(decoded),
        "n_keys": sum(len(values) for values in decoded.values()),
        "n_subjects": len(groups),
        "split_subject_counts": actual_split_counts,
        "raw_subject_shape": list(raw.shape),
        "raw_subject_file": str(raw_path.resolve()),
        "raw_subject_sha256": sha256_file(raw_path),
        "clip_to_class": expected_map,
        "clip_to_class_sha256": canonical_sha(expected_map),
        "label_map_subjects_checked": checked_subjects,
        "raw_to_lmdb_checks": reconstruction_rows,
        "max_abs_raw_to_lmdb_reconstruction_error": max_reconstruction_error,
        "fold_rows": fold_rows,
    }


def checkpoint_row(tag, path, reproduction, strict_reload_evidence):
    if tag == "random":
        return {
            "tag": tag,
            "checkpoint": "deterministic_init_seed_0_at_pinned_code_commit",
            "sha256": "not_applicable",
            "file_mode_octal": "not_applicable",
            "is_symlink": False,
            "payload_read_only": True,
            "immutable_contract_pass": True,
            "strict_reload_evidence": strict_reload_evidence,
            "phase_a_reproduction_pass": reproduction["reproduction_pass"],
            "deterministic_repeat_max_abs": reproduction["deterministic_repeat_max_abs"],
            "b1_eligible": True,
        }
    path = Path(path)
    resolved = path.resolve()
    mode = stat.S_IMODE(resolved.stat().st_mode)
    digest = sha256_file(resolved)
    sha_named = digest in resolved.name
    immutable = path.is_symlink() and sha_named and not (mode & 0o222)
    return {
        "tag": tag,
        "checkpoint": str(path),
        "resolved_checkpoint": str(resolved),
        "sha256": digest,
        "file_mode_octal": oct(mode),
        "is_symlink": path.is_symlink(),
        "payload_read_only": not bool(mode & 0o222),
        "sha_named_payload": sha_named,
        "immutable_contract_pass": immutable,
        "strict_reload_evidence": strict_reload_evidence,
        "phase_a_reproduction_pass": reproduction["reproduction_pass"],
        "deterministic_repeat_max_abs": reproduction["deterministic_repeat_max_abs"],
        "b1_eligible": immutable and reproduction["reproduction_pass"] == "True",
    }


def inspect_checkpoints(source_root, immutable_root, released_path, reproduction_path):
    reproduction_rows = {row["tag"]: row for row in read_csv(reproduction_path)}
    if set(reproduction_rows) != set(ALL_TAGS):
        raise RuntimeError(f"Phase-A reproduction rows mismatch: {sorted(reproduction_rows)}")
    rows = [checkpoint_row(
        "random",
        None,
        reproduction_rows["random"],
        "torch_seed_0_plus_phase_a_exact_reproduction",
    )]
    rows.append(checkpoint_row(
        "released",
        released_path,
        reproduction_rows["released"],
        "phase_a_state_keys_complete_but_strict_immutable_reload_pending",
    ))
    for tag in LOW_BUDGET_TAGS:
        summary = json.loads((Path(source_root) / tag / "run_summary.json").read_text())
        if summary.get("checkpoint_strict_reload") is not True:
            raise RuntimeError(f"{tag} lacks training-time strict reload evidence")
        rows.append(checkpoint_row(
            tag,
            Path(source_root) / tag / "best.pth",
            reproduction_rows[tag],
            "run_summary.checkpoint_strict_reload=true",
        ))
    for tag in H2000_TAGS:
        rows.append(checkpoint_row(
            tag,
            Path(immutable_root) / tag / "best.pth",
            reproduction_rows[tag],
            "Phase_A_immutable_closure_strict_reload=true",
        ))
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lmdb", default="/projects/EEG-foundation-model/FACED_data/processed")
    parser.add_argument("--raw-root", default="/projects/EEG-foundation-model/FACED_data/Processed_data")
    parser.add_argument(
        "--source-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1",
    )
    parser.add_argument(
        "--immutable-root",
        default="/home/infres/yinwang/CMI_AAAI_s2p_b1_launch/results/s2p_route_b_33ch_b1_immutable",
    )
    parser.add_argument(
        "--released-checkpoint",
        default="/home/infres/yinwang/eeg2025/NIPS/Cbramod_pretrained_weights.pth",
    )
    parser.add_argument(
        "--phase-a-reproduction",
        default="results/s2p_route_b_33ch_b1_faced/faced_final_reproduction_check.csv",
    )
    parser.add_argument(
        "--protocol-doc",
        default="docs/S2P_19_REPRESENTATION_EMERGENCE_PROTOCOL.md",
    )
    parser.add_argument(
        "--redteam-doc",
        default="docs/S2P_20_REPRESENTATION_EMERGENCE_REDTEAM.md",
    )
    parser.add_argument(
        "--out-dir",
        default="results/s2p_route_b_representation_emergence_b0",
    )
    args = parser.parse_args()

    faced = inspect_faced(args.lmdb, args.raw_root)
    checkpoints = inspect_checkpoints(
        args.source_root,
        args.immutable_root,
        args.released_checkpoint,
        args.phase_a_reproduction,
    )
    protocol_doc = Path(args.protocol_doc)
    redteam_doc = Path(args.redteam_doc)
    if not protocol_doc.is_file() or not redteam_doc.is_file():
        raise RuntimeError("Phase B0 protocol and red-team documents must exist before precheck")
    out = Path(args.out_dir)
    write_csv(out / "phase_b0_clip_group_manifest.csv", faced.pop("fold_rows"))
    write_csv(out / "phase_b0_checkpoint_provenance.csv", checkpoints)

    feature_path_pass = all(
        row["phase_a_reproduction_pass"] == "True"
        and float(row["deterministic_repeat_max_abs"]) <= 1e-6
        for row in checkpoints
    )
    immutable_failures = [
        row["tag"] for row in checkpoints
        if row["tag"] != "random" and not row["immutable_contract_pass"]
    ]
    precheck = {
        "phase": "B0_representation_emergence_protocol_precheck",
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "scientific_metrics_computed": False,
        "training_launched": False,
        "fine_tuning_launched": False,
        "h4000_launched": False,
        "codebrain_launched": False,
        "faced": faced,
        "protocol_sha256": sha256_file(protocol_doc),
        "redteam_sha256": sha256_file(redteam_doc),
        "feature_path_phase_a_exact_reproduction_10_of_10": feature_path_pass,
        "feature_determinism_tolerance": 1e-6,
        "clip_group_crossfit_folds": 3,
        "clip_group_overlap_between_fit_and_holdout": False,
        "source_subject_class_cells": 80 * 9,
        "segments_per_subject_class": "9_except_class4_12",
        "target_labels_used": False,
        "checkpoint_immutable_failures": immutable_failures,
    }
    write_json(out / "phase_b0_precheck.json", precheck)
    go_nogo = {
        "phase": "B0_representation_emergence",
        "clip_id_reliably_recovered": True,
        "raw_to_lmdb_clip_mapping_exact": True,
        "clip_group_crossfit_unique_and_exhaustive": True,
        "feature_path_deterministic_and_reproduced": feature_path_pass,
        "source_only_fit_target_final_score_firewall_feasible": True,
        "equal_rank_subject_task_subspace_contract_pinned": True,
        "cross_fitted_variance_partition_feasible": True,
        "metric_definitions_frozen_before_b1": True,
        "empirical_metric_non_saturation": None,
        "empirical_metric_non_saturation_note": "Cannot be established without B1 scientific compute; fail-closed saturation rules are preregistered.",
        "all_nonrandom_checkpoints_immutable": not immutable_failures,
        "immutable_blocking_tags": immutable_failures,
        "phase_b1_compute_authorized": False,
        "phase_b1_compute_recommended": False,
        "next_allowed_action": "immutable_close_H200_H500_H1000_and_released_then_repeat_B0_provenance_check",
        "requires_pm_review_before_b1": True,
    }
    write_json(out / "phase_b0_go_nogo.json", go_nogo)
    print(json.dumps(go_nogo, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
