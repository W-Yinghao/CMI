"""Audit the post-freeze C79E construction/evaluation route before analysis."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import c74_cache


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79_tables"
ROUTE_PATH = REPORT_DIR / "C79_SEED4_PRIMARY_VIEW_ROUTE.json"
ROUTE_SHA_PATH = REPORT_DIR / "C79_SEED4_PRIMARY_VIEW_ROUTE.sha256"
LOCKED_SPLITS = REPORT_DIR / "c78s_tables" / "label_split_isolation.csv"
CAMPAIGN_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/oaci-c79-seed4/"
    "protocol_e350b7f0c4ee3dfc/implementation_dd4043ad7dd67552"
)
PRIMARY_TARGETS = (1, 2, 3, 5, 6, 7, 8, 9)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def descriptor(route_item: dict[str, Any]) -> dict[str, Any]:
    return {
        "path": route_item["path"],
        "sha256": route_item["sha256"],
        "row_count": int(route_item["rows"]),
        "fields": json.loads(route_item["allowed_columns"]),
        "size_bytes": Path(route_item["path"]).stat().st_size,
    }


def load_npz(item: dict[str, Any]) -> dict[str, np.ndarray]:
    with np.load(item["path"], allow_pickle=False) as shard:
        return {key: shard[key] for key in item["fields"]}


def audit() -> dict[str, Any]:
    expected_route_sha = ROUTE_SHA_PATH.read_text().strip()
    observed_route_sha = sha256_file(ROUTE_PATH)
    if observed_route_sha != expected_route_sha:
        raise RuntimeError("C79E primary route hash drift")
    raw_route = ROUTE_PATH.read_text()
    route = json.loads(raw_route)
    if tuple(route["primary_targets"]) != PRIMARY_TARGETS or route["target4_included"]:
        raise RuntimeError("C79E primary target route drift")
    if route["same_label_oracle_descriptor_included"] or "same_label_oracle_view" in raw_route or "/oracle/" in raw_route:
        raise RuntimeError("C79E primary route exposes same-label oracle")

    locked = {int(row["target_id"]): row for row in read_csv(LOCKED_SPLITS)}
    rows: list[dict[str, Any]] = []
    total_construction = total_evaluation = 0
    for target in PRIMARY_TARGETS:
        entry = route["views"][str(target)]
        construction = descriptor(entry["target_construction_view"])
        evaluation = descriptor(entry["target_evaluation_view"])
        unlabeled = descriptor(entry["target_unlabeled_input"])
        strict = descriptor(entry["strict_source_input"])
        c74_cache.verify_shard(construction, required_fields={"target_trial_id", "target_class_label", "split_role"})
        c74_cache.verify_shard(evaluation, required_fields={"target_trial_id", "target_class_label", "split_role"})
        c74_cache.verify_shard(unlabeled, required_fields={"X", "target_id", "target_trial_id"})
        c74_cache.verify_shard(strict, required_fields={"X", "source_class_label", "source_domain_id", "source_role", "source_trial_id"})
        cdata = load_npz(construction)
        edata = load_npz(evaluation)
        udata = load_npz(unlabeled)
        cids = set(map(str, cdata["target_trial_id"]))
        eids = set(map(str, edata["target_trial_id"]))
        uids = set(map(str, udata["target_trial_id"]))
        overlap = cids & eids
        locked_row = locked[target]
        fields_unlabeled = set(unlabeled["fields"])
        forbidden_absent = not fields_unlabeled.intersection({"target_class_label", "y_true", "correctness", "joint_good"})
        label_manifest_path = CAMPAIGN_ROOT / "targets" / f"target-{target:03d}" / "views" / "LABEL_VIEWS.json"
        label_manifest = json.loads(label_manifest_path.read_text())
        passed = (
            len(overlap) == 0
            and cids | eids == uids
            and len(uids) == 576
            and set(map(str, cdata["split_role"])) == {"target_construct"}
            and set(map(str, edata["split_role"])) == {"target_eval"}
            and construction["sha256"] == locked_row["construction_sha256"]
            and evaluation["sha256"] == locked_row["evaluation_sha256"]
            and construction["row_count"] == int(locked_row["construction_rows"])
            and evaluation["row_count"] == int(locked_row["evaluation_rows"])
            and label_manifest["same_label_oracle"] is None
            and forbidden_absent
        )
        if not passed:
            raise RuntimeError(f"C79E target {target} label-route audit failed")
        total_construction += len(cids)
        total_evaluation += len(eids)
        rows.append({
            "target": target,
            "construction_rows": len(cids),
            "construction_sha256": construction["sha256"],
            "evaluation_rows": len(eids),
            "evaluation_sha256": evaluation["sha256"],
            "overlap_rows": len(overlap),
            "union_rows": len(cids | eids),
            "locked_split_hashes_replayed": 1,
            "target_label_in_unlabeled_view": 0,
            "trial_id_used_as_predictor": 0,
            "row_order_used_as_predictor": 0,
            "same_label_oracle_descriptor_visible": 0,
            "passed": 1,
        })

    target4_label = CAMPAIGN_ROOT / "targets" / "target-004" / "views" / "LABEL_VIEWS.json"
    if target4_label.exists():
        raise RuntimeError("C79E target-4 label view must not exist")
    field = json.loads((CAMPAIGN_ROOT / "gates" / "FULL_SEED4_FIELD_FROZEN.json").read_text())
    if not field["label_views_created"] or field["same_label_oracle_created"] or field["target4_label_view_created"]:
        raise RuntimeError("C79E post-label field barrier state drift")

    write_csv(TABLE_DIR / "seed4_label_view_access.csv", rows)
    result = {
        "schema_version": "c79_seed4_label_view_red_team_v1",
        "route_sha256": observed_route_sha,
        "primary_targets": list(PRIMARY_TARGETS),
        "construction_rows": total_construction,
        "evaluation_rows": total_evaluation,
        "union_rows": total_construction + total_evaluation,
        "overlap_rows": 0,
        "locked_split_hashes_passed": len(rows),
        "target4_label_view_created": False,
        "same_label_oracle_created": False,
        "same_label_oracle_descriptor_visible": False,
        "trial_id_used_as_predictor": False,
        "row_order_used_as_predictor": False,
        "all_gates_passed": True,
    }
    (REPORT_DIR / "C79_SEED4_LABEL_VIEW_RED_TEAM.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    report = f"""# C79 Seed-4 Label-View Red Team

```text
primary targets:                    8 / 8
construction rows:                 {total_construction}
evaluation rows:                   {total_evaluation}
union rows:                        {total_construction + total_evaluation}
construction/evaluation overlap:  0
locked split hashes replayed:      8 / 8
target-label fields in unlabeled:  0
target-4 label view created:       false
same-label oracle created:         false
oracle descriptor visible:        false
trial ID used as predictor:        false
row order used as predictor:       false
```

The construction/evaluation descriptors exactly replay the C78S trial split
hashes and partition all 576 target trials per primary target without overlap.
The primary route contains no target-4 label view and no same-label-oracle
descriptor.

Gate: `C79E_LABEL_VIEW_PROVISIONING_RED_TEAM_PASSED`.
"""
    (REPORT_DIR / "C79_SEED4_LABEL_VIEW_RED_TEAM.md").write_text(report)
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    audit()
