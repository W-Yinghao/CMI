"""Frozen within-subject shuffled-label control for the source anchor stream."""

import hashlib
import random
from collections import Counter, defaultdict
from typing import Dict, Iterable, List, Mapping

from star_eeg.data.faced_split_contract import assert_source_train_subject, canonical_hash


def _subject_seed(permutation_seed: int, subject: int) -> int:
    token = f"STAR-FACED-SHUFFLE-v1|{int(permutation_seed)}|{int(subject)}".encode("utf-8")
    return int(hashlib.sha256(token).hexdigest()[:16], 16)


def _histogram(labels: Iterable[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts.get(label, 0)) for label in range(9)}


def build_shuffled_manifest(anchor_manifest: Mapping[str, object], permutation_seed: int) -> Dict[str, object]:
    if anchor_manifest.get("split") != "source_train":
        raise PermissionError("shuffle input must be the FACED source_train anchor manifest")
    records = list(anchor_manifest.get("records", []))
    if not records:
        raise ValueError("shuffle input has no records")
    by_subject: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        if row.get("split") != "source_train":
            raise PermissionError("shuffle input contains a non-source_train record")
        subject = int(row["subject"])
        assert_source_train_subject(subject)
        by_subject[subject].append(row)

    output = []
    histograms = {}
    for subject in sorted(by_subject):
        rows = sorted(by_subject[subject], key=lambda row: str(row["sample_id"]))
        labels = [int(row["label"]) for row in rows]
        shuffled = list(labels)
        random.Random(_subject_seed(permutation_seed, subject)).shuffle(shuffled)
        if len(set(labels)) > 1 and shuffled == labels:
            shuffled = shuffled[1:] + shuffled[:1]
        before = _histogram(labels)
        after = _histogram(shuffled)
        if before != after:
            raise AssertionError(f"within-subject class histogram changed for subject {subject}")
        histograms[str(subject)] = {"before": before, "after": after}
        for row, shuffled_label in zip(rows, shuffled):
            output.append({
                "sample_id": str(row["sample_id"]),
                "subject": subject,
                "split": "source_train",
                "shuffled_label": int(shuffled_label),
            })

    core = {
        "schema_version": 1,
        "dataset": "FACED",
        "split": "source_train",
        "permutation_scope": "fixed_within_subject",
        "permutation_seed": int(permutation_seed),
        "permutation_rng_separate_from_training_rng": True,
        "reshuffle_each_epoch": False,
        "same_semantic_manifest_for_model_seeds": [0, 1],
        "anchor_manifest_hash": anchor_manifest.get("anchor_manifest_hash"),
        "n_records": len(output),
        "records": output,
        "within_subject_histograms": histograms,
        "source_val_participated": False,
        "target_test_participated": False,
        "preview_only": bool(anchor_manifest.get("preview_only", False)),
    }
    return {**core, "shuffled_manifest_hash": canonical_hash(core)}


def validate_frozen_shuffled_manifest(manifest: Mapping[str, object]) -> None:
    if manifest.get("split") != "source_train":
        raise ValueError("shuffled manifest split is not source_train")
    if manifest.get("reshuffle_each_epoch") is not False:
        raise ValueError("shuffled manifest is not fixed across epochs")
    if manifest.get("source_val_participated") or manifest.get("target_test_participated"):
        raise PermissionError("non-training FACED split participated in the permutation")
    core = {key: value for key, value in manifest.items() if key != "shuffled_manifest_hash"}
    if canonical_hash(core) != manifest.get("shuffled_manifest_hash"):
        raise ValueError("shuffled manifest hash mismatch")
    for subject, pair in manifest.get("within_subject_histograms", {}).items():
        assert_source_train_subject(int(subject))
        if pair.get("before") != pair.get("after"):
            raise ValueError(f"class histogram changed for subject {subject}")
