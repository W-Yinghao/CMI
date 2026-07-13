"""C84R metadata-only registry for the exact 20-channel interface.

The C84P v1 registry remains the historical 21-channel object.  This module
reuses its dataset metadata and subject-partition functions while replacing
only the harmonized interface.  It imports no EEG, ML, or array library.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from typing import Sequence

from . import c84_dataset_registry as v1
from .c84r_montage_repair import (
    CLASS_MAPPING_VERSION,
    COMMON_CHANNELS,
    EPOCH_RULE,
    INTERFACE_ID,
    MONTAGE_SHA256,
    SAMPLE_RATE_HZ,
    validate_montage,
)


SCHEMA_VERSION = "c84_dataset_metadata_snapshot_v2"
PRIMARY_CHANNELS = COMMON_CHANNELS
EXPECTED_N_TIMES = 480
EXPECTED_INPUT_SHAPE = (20, 480)
DATASETS = v1.DATASETS
SUBJECT_PARTITION_SALT = v1.SUBJECT_PARTITION_SALT
SOURCE_AUDIT_SALT = v1.SOURCE_AUDIT_SALT
TARGET_SPLIT_SALT = v1.TARGET_SPLIT_SALT


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def partition_subjects(spec: v1.DatasetSpec) -> dict[str, tuple[int, ...]]:
    return v1.partition_subjects(spec)


def source_train_audit_split(dataset: str, panel: str, subjects: Sequence[int]) -> dict[str, tuple[int, ...]]:
    return v1.source_train_audit_split(dataset, panel, subjects)


def target_trial_split(dataset: str, subject: int, class_name: str, trial_ids):
    return v1.target_trial_split(dataset, subject, class_name, trial_ids)


def canonicalize_channel_name(name: str) -> str:
    """Apply case canonicalization only; anatomical aliases are forbidden."""
    key = str(name).strip().lower()
    matches = [channel for channel in PRIMARY_CHANNELS if channel.lower() == key]
    if len(matches) != 1:
        raise ValueError(f"C84 V2 channel is absent or ambiguous: {name!r}")
    return matches[0]


def ordered_dataset_channels(dataset: str) -> tuple[str, ...]:
    spec = DATASETS[dataset]
    native_by_lower = {channel.lower(): channel for channel in spec.native_eeg_channels}
    missing = [channel for channel in PRIMARY_CHANNELS if channel.lower() not in native_by_lower]
    if missing:
        raise ValueError(f"{dataset} lacks C84 V2 channels: {missing}")
    ordered = tuple(canonicalize_channel_name(native_by_lower[channel.lower()]) for channel in PRIMARY_CHANNELS)
    validate_montage(ordered)
    return ordered


def dataset_registry_payload() -> dict[str, object]:
    return {
        "schema_version": SCHEMA_VERSION,
        "historical_registry_schema": "c84_dataset_metadata_snapshot_v1",
        "interface_id": INTERFACE_ID,
        "primary_channel_allowlist": list(PRIMARY_CHANNELS),
        "primary_channel_sha256": MONTAGE_SHA256,
        "epoch_rule": EPOCH_RULE,
        "resample_sfreq_hz": SAMPLE_RATE_HZ,
        "expected_n_times": EXPECTED_N_TIMES,
        "expected_input_shape": list(EXPECTED_INPUT_SHAPE),
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "Fz_substitution": False,
        "FCz_interpolation": False,
        "dataset_specific_masks": False,
        "zero_filling": False,
        "metadata_environment": {
            "moabb": "1.5.0",
            "mne": "1.11.0",
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
            "dataset_downloads": 0,
        },
        "datasets": {
            code: {
                **asdict(spec),
                "ordered_primary_channels": list(ordered_dataset_channels(code)),
                "primary_channels_available": len(ordered_dataset_channels(code)),
            }
            for code, spec in DATASETS.items()
        },
    }


def validate_registry() -> dict[str, object]:
    partitions = {code: partition_subjects(spec) for code, spec in DATASETS.items()}
    expected_targets = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
    v1_partition_payload = {
        code: {key: list(values) for key, values in partitions[code].items()}
        for code in DATASETS
    }
    checks = {
        "schema_v2": SCHEMA_VERSION == "c84_dataset_metadata_snapshot_v2",
        "interface_v2": INTERFACE_ID == "C84_LEFT_RIGHT_20CH_160HZ_0_3S_V2",
        "montage_count_20": len(PRIMARY_CHANNELS) == 20,
        "montage_digest": sha256_json(list(PRIMARY_CHANNELS)) == MONTAGE_SHA256,
        "FCz_absent": "FCz" not in PRIMARY_CHANNELS,
        "Fz_absent": "Fz" not in PRIMARY_CHANNELS,
        "all_datasets_exact_order": all(ordered_dataset_channels(code) == PRIMARY_CHANNELS for code in DATASETS),
        "target_counts_unchanged": {code: len(partitions[code]["targets"]) for code in DATASETS} == expected_targets,
        "source_panels_unchanged": all(
            len(partitions[code][key]) == 16
            for code in DATASETS for key in ("source_panel_A", "source_panel_B")
        ),
        "canary_targets": {code: partitions[code]["targets"][0] for code in DATASETS}
        == {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106},
        "physionet_88_excluded": 88 not in DATASETS["PhysionetMI"].eligible_subjects,
    }
    return {
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "ready": all(checks.values()),
        "partition_sha256": sha256_json(v1_partition_payload),
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
    }


if __name__ == "__main__":
    print(json.dumps(validate_registry(), sort_keys=True))
