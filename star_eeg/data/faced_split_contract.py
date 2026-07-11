"""Exact FACED firewall contract inherited from the frozen S2P audit."""

import hashlib
import json
from typing import Dict, Iterable, Mapping, Tuple


SOURCE_TRAIN_SUBJECTS: Tuple[int, ...] = tuple(range(1, 81))
SOURCE_VAL_SUBJECTS: Tuple[int, ...] = tuple(range(81, 101))
TARGET_TEST_SUBJECTS: Tuple[int, ...] = tuple(range(101, 124))

FACED_SPLITS = {
    "source_train": SOURCE_TRAIN_SUBJECTS,
    "source_val": SOURCE_VAL_SUBJECTS,
    "target_test": TARGET_TEST_SUBJECTS,
}

EXPECTED_SEGMENTS = {
    "source_train": 6720,
    "source_val": 1680,
    "target_test": 1932,
}


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def canonical_hash(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def contract_payload() -> Dict[str, object]:
    core = {
        "dataset": "FACED",
        "splits": {name: list(subjects) for name, subjects in FACED_SPLITS.items()},
        "expected_segments": EXPECTED_SEGMENTS,
        "input_shape": [32, 10, 200],
        "n_classes": 9,
        "source_train_labels_allowed_for_anchor": True,
        "source_val_labels_allowed_for_gradient": False,
        "target_test_labels_allowed_before_final_scoring": False,
        "target_class_distribution_allowed_during_training": False,
    }
    return {**core, "faced_split_hash": canonical_hash(core)}


def role_for_subject(subject: int) -> str:
    subject = int(subject)
    for role, subjects in FACED_SPLITS.items():
        if subject in subjects:
            return role
    raise ValueError(f"FACED subject outside frozen 1..123 contract: {subject}")


def assert_exact_split(mapping: Mapping[str, Iterable[int]]) -> None:
    if set(mapping) != set(FACED_SPLITS):
        raise ValueError(f"split names differ from frozen contract: {sorted(mapping)}")
    for role, expected in FACED_SPLITS.items():
        observed = tuple(int(value) for value in mapping[role])
        if observed != expected:
            raise ValueError(f"{role} differs from frozen FACED subject list")
    all_subjects = [subject for values in mapping.values() for subject in values]
    if len(all_subjects) != len(set(all_subjects)):
        raise ValueError("FACED split subjects overlap")


def assert_source_train_subject(subject: int) -> None:
    if role_for_subject(subject) != "source_train":
        raise PermissionError(f"subject {subject} is not allowed in the source anchor stream")
