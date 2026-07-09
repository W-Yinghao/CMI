"""Utilities for the repaired W1 MI split.

The repaired split is a frozen benchmark construction step. Target labels are
used only here to make both adaptation and evaluation contain both MI classes;
runtime adaptation receives unlabeled target adaptation trials only.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


DATASETS = ["BNCI2014_001", "Cho2017", "Lee2019_MI"]
SOURCE_SEEDS = [0, 1, 2]
SPLIT_FAMILY = "class_stratified_half"
P0_BRANCHES = [
    "identity_uniform",
    "identity_joint_prior",
    "joint_geometry_uniform",
    "joint_geometry_joint_prior",
    "fixed_iterative_geometry_uniform",
    "fixed_reference_oneshot_uniform",
    "pooled_uniform",
    "latent_im_diag_uniform",
    "source_recolored_ea",
]
OUTPUT_BRANCHES = P0_BRANCHES + ["__decomposition__"]
EXPECTED_TARGET_UNITS = 115
EXPECTED_MANIFEST_ROWS = EXPECTED_TARGET_UNITS * len(SOURCE_SEEDS)
EXPECTED_H2CMI_ROWS = EXPECTED_MANIFEST_ROWS * len(OUTPUT_BRANCHES)


def json_compact(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), default=int)


def class_counts(y: np.ndarray) -> list[int]:
    return np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(int).tolist()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def trial_id(dataset: str, ep, idx: int) -> str:
    return (
        f"{dataset}:subject={int(ep.subject[idx])}:session={int(ep.session[idx])}:"
        f"run={int(ep.run[idx])}:epoch={int(idx)}"
    )


def trial_ids(dataset: str, ep, indices: np.ndarray) -> list[str]:
    return [trial_id(dataset, ep, int(i)) for i in np.asarray(indices, dtype=np.int64).tolist()]


def trial_id_to_epoch_index(value: str) -> int:
    marker = ":epoch="
    if marker not in value:
        raise ValueError(f"trial id lacks epoch marker: {value}")
    return int(value.rsplit(marker, 1)[1])


def indices_from_trial_ids(values: list[str]) -> np.ndarray:
    return np.asarray([trial_id_to_epoch_index(v) for v in values], dtype=np.int64)


def class_stratified_half_indices(ep, target_subject: int) -> tuple[int, np.ndarray, np.ndarray]:
    target_session = int(ep.session[ep.subject == target_subject].min())
    m = np.where((ep.subject == target_subject) & (ep.session == target_session))[0]
    adapt: list[int] = []
    evalm: list[int] = []
    for cls in (0, 1):
        cls_idx = m[ep.y[m] == cls]
        if len(cls_idx) < 2:
            return target_session, m[:0], m[:0]
        half = len(cls_idx) // 2
        adapt.extend(int(i) for i in cls_idx[:half])
        evalm.extend(int(i) for i in cls_idx[half:])
    return (
        target_session,
        np.asarray(sorted(adapt), dtype=np.int64),
        np.asarray(sorted(evalm), dtype=np.int64),
    )


def row_split_hash(row: dict[str, Any]) -> str:
    payload = {
        "dataset": row["dataset"],
        "target_subject": int(row["target_subject"]),
        "source_seed": int(row["source_seed"]),
        "source_subject_ids": row["source_subject_ids"],
        "adapt_trial_ids": row["adapt_trial_ids"],
        "eval_trial_ids": row["eval_trial_ids"],
        "class_counts_adapt": row["class_counts_adapt"],
        "class_counts_eval": row["class_counts_eval"],
        "split_family": row["split_family"],
    }
    return sha256_text(json_compact(payload))


def manifest_hash(rows: list[dict[str, Any]]) -> str:
    stable_rows = sorted(
        rows,
        key=lambda r: (r["dataset"], int(r["target_subject"]), int(r["source_seed"])),
    )
    payload = {
        "schema": "w1_repaired_split_manifest_v1",
        "split_family": SPLIT_FAMILY,
        "rows": stable_rows,
    }
    return sha256_text(json_compact(payload))


def manifest_rows_for_dataset(dataset: str, ep, seeds: list[int] | None = None) -> list[dict[str, Any]]:
    seeds = SOURCE_SEEDS if seeds is None else [int(s) for s in seeds]
    subjects = sorted(int(s) for s in np.unique(ep.subject))
    rows: list[dict[str, Any]] = []
    for target in subjects:
        target_session, adapt_idx, eval_idx = class_stratified_half_indices(ep, target)
        source_subject_ids = [int(s) for s in subjects if int(s) != int(target)]
        adapt_counts = class_counts(ep.y[adapt_idx]) if len(adapt_idx) else [0, 0]
        eval_counts = class_counts(ep.y[eval_idx]) if len(eval_idx) else [0, 0]
        adapt_ids = trial_ids(dataset, ep, adapt_idx)
        eval_ids = trial_ids(dataset, ep, eval_idx)
        disjoint = not (set(adapt_ids) & set(eval_ids))
        base = {
            "dataset": dataset,
            "target_subject": int(target),
            "target_session": int(target_session),
            "source_subject_ids": source_subject_ids,
            "adapt_trial_ids": adapt_ids,
            "eval_trial_ids": eval_ids,
            "n_adapt": int(len(adapt_idx)),
            "n_eval": int(len(eval_idx)),
            "class_counts_adapt": adapt_counts,
            "class_counts_eval": eval_counts,
            "adapt_eval_disjoint": bool(disjoint),
            "both_classes_adapt": bool(min(adapt_counts) > 0),
            "both_classes_eval": bool(min(eval_counts) > 0),
            "split_family": SPLIT_FAMILY,
            "labels_used_only_for_split_construction": True,
            "target_labels_hidden_from_adaptation": True,
        }
        for seed in seeds:
            row = dict(base, source_seed=int(seed))
            row["split_hash"] = row_split_hash(row)
            rows.append(row)
    return rows


CSV_FIELDNAMES = [
    "dataset",
    "target_subject",
    "source_subject_ids",
    "source_seed",
    "adapt_trial_ids",
    "eval_trial_ids",
    "n_adapt",
    "n_eval",
    "class_counts_adapt",
    "class_counts_eval",
    "adapt_eval_disjoint",
    "both_classes_adapt",
    "both_classes_eval",
    "split_family",
    "split_hash",
    "labels_used_only_for_split_construction",
    "target_labels_hidden_from_adaptation",
    "target_session",
]


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple)):
        return json_compact(value)
    return value


def write_manifest_csv(path: str | Path, rows: list[dict[str, Any]]) -> None:
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row[k]) for k in CSV_FIELDNAMES})


def _parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    if value in ("True", "true", "1"):
        return True
    if value in ("False", "false", "0"):
        return False
    raise ValueError(f"cannot parse bool: {value}")


def load_manifest_csv(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open(newline="") as f:
        for row in csv.DictReader(f):
            parsed = dict(row)
            for key in ("source_subject_ids", "adapt_trial_ids", "eval_trial_ids",
                        "class_counts_adapt", "class_counts_eval"):
                parsed[key] = json.loads(parsed[key])
            for key in ("target_subject", "source_seed", "n_adapt", "n_eval", "target_session"):
                parsed[key] = int(parsed[key])
            for key in ("adapt_eval_disjoint", "both_classes_adapt", "both_classes_eval",
                        "labels_used_only_for_split_construction",
                        "target_labels_hidden_from_adaptation"):
                parsed[key] = _parse_bool(parsed[key])
            rows.append(parsed)
    return rows
