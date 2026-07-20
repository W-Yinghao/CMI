"""Frozen metadata registry for the C84 fixed-zoo external-validity program.

This module deliberately contains no EEG-loader entrypoint.  The records below
are snapshots of MOABB 1.5.0 metadata and loader source inspected during C84P;
they make the protocol and synthetic tests reproducible without touching a raw
dataset path, trial array, or label view.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from typing import Iterable, Sequence


SUBJECT_PARTITION_SALT = "C84_FIXED_ZOO_LEFT_RIGHT_V1"
SOURCE_AUDIT_SALT = "C84_SOURCE_AUDIT_SPLIT_V1"
TARGET_SPLIT_SALT = "C84_TARGET_SPLIT_V1"

PRIMARY_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6",
    "C5", "C3", "C1", "Cz", "C2", "C4", "C6",
    "CP5", "CP3", "CP1", "CPz", "CP2", "CP4", "CP6",
)


@dataclass(frozen=True)
class DatasetSpec:
    code: str
    moabb_class: str
    eligible_subjects: tuple[int, ...]
    excluded_subjects: tuple[int, ...]
    sessions: int
    native_sfreq_hz: int
    native_eeg_channels: tuple[str, ...]
    class_names: tuple[str, str]
    event_mapping: tuple[tuple[str, int], ...]
    task_runs: str
    trials_per_class: str
    dataset_doi: str
    paper_doi: str
    repository: str
    license: str
    official_metadata_url: str
    loader_source_sha256: str
    metadata_notes: str

    @property
    def subject_count(self) -> int:
        return len(self.eligible_subjects)

    @property
    def missing_primary_channels(self) -> tuple[str, ...]:
        available = set(self.native_eeg_channels)
        return tuple(channel for channel in PRIMARY_CHANNELS if channel not in available)

    @property
    def available_primary_channels(self) -> tuple[str, ...]:
        available = set(self.native_eeg_channels)
        return tuple(channel for channel in PRIMARY_CHANNELS if channel in available)


LEE_CHANNELS = (
    "AF3", "AF4", "AF7", "AF8", "C1", "C2", "C3", "C4", "C5", "C6",
    "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz", "F10", "F3",
    "F4", "F7", "F8", "F9", "FC1", "FC2", "FC3", "FC4", "FC5", "FC6",
    "FT10", "FT9", "FTT10h", "FTT9h", "Fp1", "Fp2", "Fz", "O1", "O2", "Oz",
    "P1", "P2", "P3", "P4", "P7", "P8", "PO10", "PO3", "PO4", "PO9",
    "POz", "Pz", "T7", "T8", "TP10", "TP7", "TP8", "TP9", "TPP10h",
    "TPP8h", "TPP9h", "TTP7h",
)

CHO_CHANNELS = (
    "AF3", "AF4", "AF7", "AF8", "AFz", "C1", "C2", "C3", "C4", "C5",
    "C6", "CP1", "CP2", "CP3", "CP4", "CP5", "CP6", "CPz", "Cz", "F1",
    "F2", "F3", "F4", "F5", "F6", "F7", "F8", "FC1", "FC2", "FC3",
    "FC4", "FC5", "FC6", "FCz", "FT7", "FT8", "Fp1", "Fp2", "Fpz", "Fz",
    "Iz", "O1", "O2", "Oz", "P1", "P10", "P2", "P3", "P4", "P5", "P6",
    "P7", "P8", "P9", "PO3", "PO4", "PO7", "PO8", "POz", "Pz", "T7",
    "T8", "TP7", "TP8",
)

PHYSIONET_CHANNELS = (
    "FC5", "FC3", "FC1", "FCz", "FC2", "FC4", "FC6", "C5", "C3", "C1",
    "Cz", "C2", "C4", "C6", "CP5", "CP3", "CP1", "CPz", "CP2", "CP4",
    "CP6", "Fp1", "Fpz", "Fp2", "AF7", "AF3", "AFz", "AF4", "AF8", "F7",
    "F5", "F3", "F1", "Fz", "F2", "F4", "F6", "F8", "FT7", "FT8", "T7",
    "T8", "T9", "T10", "TP7", "TP8", "P7", "P5", "P3", "P1", "Pz", "P2",
    "P4", "P6", "P8", "PO7", "PO3", "POz", "PO4", "PO8", "O1", "Oz", "O2",
    "Iz",
)


DATASETS: dict[str, DatasetSpec] = {
    "Lee2019_MI": DatasetSpec(
        code="Lee2019_MI",
        moabb_class="moabb.datasets.Lee2019_MI(train_run=True,test_run=False)",
        eligible_subjects=tuple(range(1, 55)),
        excluded_subjects=(),
        sessions=2,
        native_sfreq_hz=1000,
        native_eeg_channels=LEE_CHANNELS,
        class_names=("left_hand", "right_hand"),
        event_mapping=(("left_hand", 2), ("right_hand", 1)),
        task_runs="offline_training_run_only_in_each_of_two_sessions;online_test_excluded",
        trials_per_class="100 total across two offline sessions (50/session/class)",
        dataset_doi="10.5524/100542",
        paper_doi="10.1093/gigascience/giz002",
        repository="GigaDB",
        license="MOABB_1.5.0_metadata_reports_GPL-3.0;access_terms_recheck_at_canary",
        official_metadata_url="https://moabb.neurotechx.com/docs/generated/moabb.datasets.Lee2019_MI.html",
        loader_source_sha256="a0234b81923fed15e4a221e011399f76a83873cd43d598ad5c8c71ba54678a6f",
        metadata_notes="62 EEG; FCz absent; Fz present; online MI run has no usable labels",
    ),
    "Cho2017": DatasetSpec(
        code="Cho2017",
        moabb_class="moabb.datasets.Cho2017",
        eligible_subjects=tuple(range(1, 53)),
        excluded_subjects=(),
        sessions=1,
        native_sfreq_hz=512,
        native_eeg_channels=CHO_CHANNELS,
        class_names=("left_hand", "right_hand"),
        event_mapping=(("left_hand", 1), ("right_hand", 2)),
        task_runs="single MOABB run assembled from imagery trials",
        trials_per_class="100_or_120_before_loader/availability validation",
        dataset_doi="10.5524/100295",
        paper_doi="10.1093/gigascience/gix034",
        repository="GigaDB",
        license="CC-BY-4.0",
        official_metadata_url="https://moabb.neurotechx.com/docs/generated/moabb.datasets.Cho2017.html",
        loader_source_sha256="42e2ef372762cb86aab11a886e1707675477ac776e0468448233de7a4ba71e32",
        metadata_notes="64 EEG plus 4 EMG; source MAT contains bad_trial_indices not applied by MOABB loader",
    ),
    "PhysionetMI": DatasetSpec(
        code="PhysionetMI",
        moabb_class="moabb.datasets.PhysionetMI(imagined=True,executed=False)",
        eligible_subjects=tuple(subject for subject in range(1, 110) if subject != 88),
        excluded_subjects=(88,),
        sessions=1,
        native_sfreq_hz=160,
        native_eeg_channels=PHYSIONET_CHANNELS,
        class_names=("left_hand", "right_hand"),
        event_mapping=(("left_hand", 2), ("right_hand", 3)),
        task_runs="imagined unilateral hand runs 4,8,12 only;executed and bilateral runs excluded",
        trials_per_class="approximately_23",
        dataset_doi="10.13026/C28G6P",
        paper_doi="10.1109/TBME.2004.827072",
        repository="PhysioNet",
        license="ODC-By-1.0",
        official_metadata_url="https://physionet.org/content/eegmmidb/1.0.0/",
        loader_source_sha256="a8abe8097870d804a2d78f500f3c6820962c1c3402f53368e92e7a91068b84ba",
        metadata_notes="subject 88 excluded prospectively because native sfreq is 128 Hz rather than 160 Hz",
    ),
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_json(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def subject_hash(dataset: str, subject: int) -> str:
    return hashlib.sha256(f"{SUBJECT_PARTITION_SALT}|{dataset}|{int(subject)}".encode("ascii")).hexdigest()


def partition_subjects(spec: DatasetSpec) -> dict[str, tuple[int, ...]]:
    ordered = tuple(sorted(spec.eligible_subjects, key=lambda subject: subject_hash(spec.code, subject)))
    if len(ordered) < 33:
        raise ValueError(f"{spec.code} cannot support two 16-subject panels plus targets")
    partition = {
        "source_panel_A": ordered[:16],
        "source_panel_B": ordered[16:32],
        "targets": ordered[32:],
    }
    sets = [set(values) for values in partition.values()]
    if any(left & right for index, left in enumerate(sets) for right in sets[index + 1 :]):
        raise RuntimeError(f"{spec.code} subject partition overlaps")
    if set().union(*sets) != set(spec.eligible_subjects):
        raise RuntimeError(f"{spec.code} subject partition is incomplete")
    return partition


def source_train_audit_split(dataset: str, panel: str, subjects: Sequence[int]) -> dict[str, tuple[int, ...]]:
    if panel not in {"A", "B"} or len(subjects) != 16:
        raise ValueError("C84 source train/audit split requires a named 16-subject panel")
    ordered = tuple(sorted(
        (int(subject) for subject in subjects),
        key=lambda subject: hashlib.sha256(
            f"{SOURCE_AUDIT_SALT}|{dataset}|panel={panel}|{subject}".encode("ascii")
        ).hexdigest(),
    ))
    return {"source_training": ordered[:12], "source_audit": ordered[12:]}


def target_trial_split(dataset: str, subject: int, class_name: str, trial_ids: Iterable[str]) -> dict[str, tuple[str, ...]]:
    trial_ids = tuple(str(trial_id) for trial_id in trial_ids)
    if len(set(trial_ids)) != len(trial_ids):
        raise ValueError("target trial IDs must be unique within dataset/subject/class")
    ordered = tuple(sorted(
        trial_ids,
        key=lambda trial_id: hashlib.sha256(
            f"{TARGET_SPLIT_SALT}|{dataset}|{int(subject)}|{trial_id}".encode("ascii")
        ).hexdigest(),
    ))
    split = len(ordered) // 2
    construction, evaluation = ordered[:split], ordered[split:]
    if set(construction) & set(evaluation):
        raise RuntimeError("construction/evaluation split overlaps")
    return {"construction": construction, "evaluation": evaluation}


def dataset_registry_payload() -> dict[str, object]:
    return {
        "schema_version": "c84_dataset_metadata_snapshot_v1",
        "metadata_environment": {
            "moabb": "1.5.0",
            "mne": "1.11.0",
            "moabb_paradigm_base_sha256": "06dbf9bf8a8a4c9bd3bc06a969a388c5d8c6c26ec004997e9333c3fc27b51c24",
            "moabb_preprocessing_sha256": "30b39efa7e0c7edbd5b6f7482795b895ba9b6297e48817e488d86b9eb6bd0c72",
            "real_EEG_arrays_loaded": 0,
            "real_labels_read": 0,
        },
        "primary_channel_allowlist": list(PRIMARY_CHANNELS),
        "datasets": {code: asdict(spec) for code, spec in DATASETS.items()},
    }


def validate_registry() -> dict[str, object]:
    partitions = {code: partition_subjects(spec) for code, spec in DATASETS.items()}
    expected_targets = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
    checks = {
        "dataset_count": len(DATASETS) == 3,
        "eligible_subject_counts": {code: spec.subject_count for code, spec in DATASETS.items()} == {
            "Lee2019_MI": 54, "Cho2017": 52, "PhysionetMI": 108,
        },
        "target_counts": {code: len(partitions[code]["targets"]) for code in DATASETS} == expected_targets,
        "source_panels": all(
            len(partitions[code][panel]) == 16
            for code in DATASETS for panel in ("source_panel_A", "source_panel_B")
        ),
        "physionet_88_excluded": 88 not in DATASETS["PhysionetMI"].eligible_subjects,
        "requested_allowlist_size": len(PRIMARY_CHANNELS) == 21,
        "cho_allowlist_complete": not DATASETS["Cho2017"].missing_primary_channels,
        "physionet_allowlist_complete": not DATASETS["PhysionetMI"].missing_primary_channels,
        "lee_FCz_missing": DATASETS["Lee2019_MI"].missing_primary_channels == ("FCz",),
    }
    return {
        "checks": checks,
        "passed": sum(bool(value) for value in checks.values()),
        "total": len(checks),
        "channel_contract_blocked": bool(DATASETS["Lee2019_MI"].missing_primary_channels),
    }


if __name__ == "__main__":
    print(json.dumps(validate_registry(), sort_keys=True))
