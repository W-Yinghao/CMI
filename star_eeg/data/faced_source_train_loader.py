"""Dedicated FACED source_train-only loader for STAR anchoring.

The loader selects only ``keys_by_split["train"]``. It never iterates,
deserializes, counts, or tensors source_val/test samples. Evaluation remains a
separate future process.
"""

import hashlib
import json
import pickle
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from star_eeg.data.faced_split_contract import (
    SOURCE_TRAIN_SUBJECTS,
    canonical_hash,
    contract_payload,
)


FACED_LMDB = Path("/projects/EEG-foundation-model/FACED_data/processed")
EXPECTED_SOURCE_RECORDS = 6720
EXPECTED_RECORDS_PER_SUBJECT = 84
EXPECTED_CLASS_HISTOGRAM = {
    "0": 9,
    "1": 9,
    "2": 9,
    "3": 9,
    "4": 12,
    "5": 9,
    "6": 9,
    "7": 9,
    "8": 9,
}
_KEY_PATTERN = re.compile(r"sub(?P<subject>\d+)\.pkl-(?P<condition>\d+)-(?P<segment>\d+)$")


def parse_source_key(key: object) -> Dict[str, object]:
    text = key.decode() if isinstance(key, bytes) else str(key)
    match = _KEY_PATTERN.fullmatch(text)
    if match is None:
        raise ValueError(f"unexpected FACED source key: {text}")
    subject = int(match.group("subject")) + 1
    if subject not in SOURCE_TRAIN_SUBJECTS:
        raise PermissionError(f"FACED key is outside source_train subjects: {text}")
    return {
        "sample_id": text,
        "subject": subject,
        "condition_id": int(match.group("condition")),
        "segment_id": int(match.group("segment")),
    }


def _source_keys(transaction) -> List[object]:
    raw_index = transaction.get(b"__keys__")
    if raw_index is None:
        raise RuntimeError("FACED LMDB is missing __keys__")
    keys_by_split = pickle.loads(raw_index)
    if "train" not in keys_by_split:
        raise RuntimeError("FACED LMDB has no train key list")
    # Do not inspect or compute any property of other split lists.
    return list(keys_by_split["train"])


def _histogram(labels: Sequence[int]) -> Dict[str, int]:
    counts = Counter(int(label) for label in labels)
    return {str(label): int(counts.get(label, 0)) for label in range(9)}


def _validate_inventory_records(records: Sequence[Mapping[str, object]]) -> None:
    if len(records) != EXPECTED_SOURCE_RECORDS:
        raise RuntimeError(f"source_train record count {len(records)} != {EXPECTED_SOURCE_RECORDS}")
    by_subject: Dict[int, List[Mapping[str, object]]] = defaultdict(list)
    for row in records:
        if row.get("split") != "source_train":
            raise PermissionError("source inventory contains a non-source_train row")
        subject = int(row["subject"])
        if subject not in SOURCE_TRAIN_SUBJECTS:
            raise PermissionError(f"source inventory contains subject {subject}")
        if int(row["label"]) not in range(9):
            raise ValueError("source inventory label outside 0..8")
        by_subject[subject].append(row)
    if sorted(by_subject) != list(SOURCE_TRAIN_SUBJECTS):
        raise RuntimeError("source inventory does not contain exact subjects 1..80")
    for subject, rows in by_subject.items():
        if len(rows) != EXPECTED_RECORDS_PER_SUBJECT:
            raise RuntimeError(f"subject {subject} record count {len(rows)} != 84")
        observed = _histogram([int(row["label"]) for row in rows])
        if observed != EXPECTED_CLASS_HISTOGRAM:
            raise RuntimeError(f"subject {subject} class histogram differs: {observed}")


def scan_source_train_inventory(lmdb_path: Path = FACED_LMDB) -> Dict[str, object]:
    """Read and validate all 6,720 source records, and no other samples."""
    import lmdb
    import numpy as np

    environment = lmdb.open(
        str(lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
    )
    records = []
    per_subject_ordinal: Dict[int, int] = defaultdict(int)
    source_sample_reads = 0
    try:
        with environment.begin() as transaction:
            keys = _source_keys(transaction)
            for key in keys:
                parsed = parse_source_key(key)
                raw = transaction.get(key if isinstance(key, bytes) else str(key).encode())
                if raw is None:
                    raise RuntimeError(f"missing source_train sample: {parsed['sample_id']}")
                source_sample_reads += 1
                obj = pickle.loads(raw)
                sample = np.asarray(obj["sample"], dtype=np.float32)
                label = int(obj["label"])
                if sample.shape != (32, 10, 200):
                    raise RuntimeError(f"source sample {parsed['sample_id']} shape {sample.shape}")
                if label not in range(9):
                    raise RuntimeError(f"source sample {parsed['sample_id']} label {label}")
                subject = int(parsed["subject"])
                ordinal = per_subject_ordinal[subject]
                per_subject_ordinal[subject] += 1
                records.append({
                    **parsed,
                    "split": "source_train",
                    "label": label,
                    "subject_item_index": ordinal,
                    "shape": [32, 10, 200],
                    "dtype": "float32",
                    "sample_sha256": hashlib.sha256(sample.tobytes(order="C")).hexdigest(),
                    "serialized_value_sha256": hashlib.sha256(raw).hexdigest(),
                })
    finally:
        environment.close()
    _validate_inventory_records(records)
    records_hash = canonical_hash(records)
    core = {
        "schema_version": 1,
        "dataset": "FACED",
        "lmdb_path": str(lmdb_path),
        "split": "source_train",
        "n_records": len(records),
        "n_subjects": len(SOURCE_TRAIN_SUBJECTS),
        "subjects": list(SOURCE_TRAIN_SUBJECTS),
        "records_per_subject": EXPECTED_RECORDS_PER_SUBJECT,
        "per_subject_class_histogram": EXPECTED_CLASS_HISTOGRAM,
        "input_shape": [32, 10, 200],
        "normalization": "per_channel_per_1s_patch_zscore_eps_1e-6",
        "records_hash": records_hash,
        "records": records,
        "access_audit": {
            "index_key_reads": 1,
            "selected_index_member": "train",
            "source_sample_reads": source_sample_reads,
            "source_val_sample_reads": 0,
            "test_sample_reads": 0,
            "non_source_labels_read": 0,
            "non_source_tensors_created": 0,
            "non_source_class_counts_computed": False,
        },
    }
    return {**core, "faced_source_train_inventory_hash": canonical_hash(core)}


def build_real_anchor_manifest(inventory: Mapping[str, object]) -> Dict[str, object]:
    records = list(inventory.get("records", []))
    _validate_inventory_records(records)
    anchor_records = [
        {
            "sample_id": str(row["sample_id"]),
            "subject": int(row["subject"]),
            "label": int(row["label"]),
            "split": "source_train",
            "condition_id": int(row["condition_id"]),
            "segment_id": int(row["segment_id"]),
            "subject_item_index": int(row["subject_item_index"]),
            "sample_sha256": str(row["sample_sha256"]),
        }
        for row in records
    ]
    core = {
        "schema_version": 2,
        "dataset": "FACED",
        "split": "source_train",
        "n_records": len(anchor_records),
        "n_subjects": 80,
        "subjects": list(SOURCE_TRAIN_SUBJECTS),
        "faced_split_hash": contract_payload()["faced_split_hash"],
        "source_inventory_hash": inventory["faced_source_train_inventory_hash"],
        "records": anchor_records,
        "real_source_manifest": True,
        "source_val_records_present": False,
        "test_records_present": False,
    }
    return {**core, "anchor_manifest_hash": canonical_hash(core)}


def load_anchor_manifest(path: Path) -> Mapping[str, object]:
    payload = json.loads(path.read_text())
    if payload.get("split") != "source_train" or payload.get("n_records") != 6720:
        raise RuntimeError("anchor manifest is not the full source_train contract")
    core = {key: value for key, value in payload.items() if key != "anchor_manifest_hash"}
    if canonical_hash(core) != payload.get("anchor_manifest_hash"):
        raise RuntimeError("anchor manifest hash mismatch")
    _validate_inventory_records(payload["records"])
    return payload


class FACEDSourceTrainAnchorLoader:
    """Batch loader restricted by a prevalidated source_train manifest."""

    def __init__(self, lmdb_path: Path, anchor_manifest: Mapping[str, object]):
        records = list(anchor_manifest.get("records", []))
        _validate_inventory_records(records)
        self.lmdb_path = Path(lmdb_path)
        self.records = {str(row["sample_id"]): row for row in records}
        if len(self.records) != EXPECTED_SOURCE_RECORDS:
            raise RuntimeError("anchor manifest sample IDs are not unique")
        self.source_sample_reads = 0
        self.non_source_sample_reads = 0

    def load_batch(self, sample_ids: Sequence[str]) -> Tuple[object, object, List[Mapping[str, object]]]:
        import lmdb
        import numpy as np

        if not sample_ids:
            raise ValueError("anchor batch is empty")
        environment = lmdb.open(
            str(self.lmdb_path), readonly=True, lock=False, readahead=False, meminit=False
        )
        samples, labels, rows = [], [], []
        try:
            with environment.begin() as transaction:
                for sample_id in sample_ids:
                    row = self.records.get(str(sample_id))
                    if row is None:
                        self.non_source_sample_reads += 1
                        raise PermissionError(f"sample ID is not in frozen source_train manifest: {sample_id}")
                    parse_source_key(sample_id)
                    raw = transaction.get(str(sample_id).encode())
                    if raw is None:
                        raise RuntimeError(f"missing source_train sample: {sample_id}")
                    obj = pickle.loads(raw)
                    sample = np.asarray(obj["sample"], dtype=np.float32)
                    label = int(obj["label"])
                    if sample.shape != (32, 10, 200) or label != int(row["label"]):
                        raise RuntimeError(f"source_train manifest/data mismatch: {sample_id}")
                    if hashlib.sha256(sample.tobytes(order="C")).hexdigest() != row["sample_sha256"]:
                        raise RuntimeError(f"source_train sample SHA mismatch: {sample_id}")
                    samples.append(sample)
                    labels.append(label)
                    rows.append(row)
                    self.source_sample_reads += 1
        finally:
            environment.close()
        batch = np.stack(samples).astype(np.float32)
        batch = (batch - batch.mean(-1, keepdims=True)) / (batch.std(-1, keepdims=True) + 1e-6)
        return batch, np.asarray(labels, dtype=np.int64), rows

    def access_audit(self) -> Dict[str, object]:
        return {
            "source_sample_reads": self.source_sample_reads,
            "source_val_sample_reads": 0,
            "test_sample_reads": 0,
            "non_source_sample_reads": self.non_source_sample_reads,
            "target_tensors_created": 0,
            "status": "PASS" if self.non_source_sample_reads == 0 else "FAIL",
        }
