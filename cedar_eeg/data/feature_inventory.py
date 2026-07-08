"""Inventory CEDAR frozen feature supply candidates."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .feature_schema import FeatureInventoryRecord, inspect_feature_file


FIELDS = [
    "dataset",
    "backbone",
    "fold",
    "seed",
    "path",
    "has_z",
    "has_y",
    "has_domain",
    "has_groups",
    "has_role",
    "has_subject",
    "has_session_or_recording",
    "n_samples",
    "z_dim",
    "status",
    "reject_reason",
    "provenance",
    "cedar_role",
    "deployable",
]


def iter_feature_candidates(roots: list[str | Path]) -> list[Path]:
    paths: list[Path] = []
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        if root_path.is_file():
            paths.append(root_path)
            continue
        paths.extend(sorted(root_path.rglob("*.npz")))
    return sorted(set(paths))


def inventory_paths(paths: list[Path], *, include_archive: bool = False) -> list[FeatureInventoryRecord]:
    return [inspect_feature_file(path, include_archive=include_archive) for path in paths]


def write_csv(records: list[FeatureInventoryRecord], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec.to_dict())


def write_json(records: list[FeatureInventoryRecord], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump([rec.to_dict() for rec in records], f, indent=2, sort_keys=True)


def summarize(records: list[FeatureInventoryRecord]) -> dict[str, int]:
    out = {"total": len(records), "COMPLIANT": 0, "ADAPTER_POSSIBLE": 0, "REJECT": 0}
    for rec in records:
        out[rec.status] = out.get(rec.status, 0) + 1
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", action="append", default=[], help="Root or file to inventory; repeatable.")
    ap.add_argument("--include-archive", action="store_true")
    ap.add_argument("--csv-out", default="")
    ap.add_argument("--json-out", default="")
    args = ap.parse_args()

    roots = args.root or ["results", "cedar_eeg"]
    paths = iter_feature_candidates(roots)
    if not args.include_archive:
        paths = [p for p in paths if "archive" not in p.parts]
    records = inventory_paths(paths, include_archive=args.include_archive)
    if args.csv_out:
        write_csv(records, args.csv_out)
    if args.json_out:
        write_json(records, args.json_out)
    print(json.dumps(summarize(records), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
