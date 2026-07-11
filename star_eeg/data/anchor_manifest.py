"""Deterministic source-only FACED anchor-manifest construction."""

from dataclasses import asdict, dataclass
from typing import Dict, Iterable, List

from star_eeg.data.faced_split_contract import (
    assert_source_train_subject,
    canonical_hash,
    contract_payload,
)


@dataclass(frozen=True)
class AnchorRecord:
    sample_id: str
    subject: int
    label: int
    split: str = "source_train"


def validate_anchor_records(records: Iterable[AnchorRecord]) -> List[AnchorRecord]:
    ordered = sorted(records, key=lambda row: (int(row.subject), str(row.sample_id)))
    if not ordered:
        raise ValueError("anchor manifest cannot be empty")
    sample_ids = set()
    for row in ordered:
        if row.split != "source_train":
            raise PermissionError(f"anchor record {row.sample_id} has forbidden split {row.split}")
        assert_source_train_subject(row.subject)
        if int(row.label) not in range(9):
            raise ValueError(f"anchor record {row.sample_id} label outside 0..8")
        if row.sample_id in sample_ids:
            raise ValueError(f"duplicate anchor sample_id: {row.sample_id}")
        sample_ids.add(row.sample_id)
    return ordered


def build_anchor_manifest(
    records: Iterable[AnchorRecord],
    dataset_manifest_hash: str,
    preview_only: bool,
) -> Dict[str, object]:
    ordered = validate_anchor_records(records)
    entries = [asdict(row) for row in ordered]
    core = {
        "schema_version": 1,
        "dataset": "FACED",
        "split": "source_train",
        "subjects": sorted({int(row.subject) for row in ordered}),
        "n_records": len(entries),
        "n_classes": 9,
        "records": entries,
        "dataset_manifest_hash": str(dataset_manifest_hash),
        "faced_split_hash": contract_payload()["faced_split_hash"],
        "preview_only": bool(preview_only),
        "real_eeg_loaded": False if preview_only else None,
        "target_records_present": False,
        "source_val_records_present": False,
    }
    return {**core, "anchor_manifest_hash": canonical_hash(core)}


def synthetic_preview_records() -> List[AnchorRecord]:
    """Small fake records for STAR_00A; these are not FACED observations."""
    labels = (0, 1, 2, 3, 4, 4, 5, 6, 7, 8)
    return [
        AnchorRecord(sample_id=f"synthetic-sub{subject:03d}-{index:02d}", subject=subject, label=label)
        for subject in (1, 2, 3)
        for index, label in enumerate(labels)
    ]
