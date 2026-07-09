"""Read-only completion checks for CEDAR_01F feature-supply artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from cedar_eeg.probes.crossfit_grouped import make_folds

from .feature_inventory import iter_feature_candidates
from .feature_schema import inspect_feature_file, sha256_file, stable_json_hash
from .load_frozen_features import FrozenFeatureSchemaError, load_frozen_feature_npz


FORBIDDEN_SELECTOR_KEYS = {
    "candidate",
    "candidates",
    "drop_dims",
    "dropped_units",
    "mask",
    "selected",
    "selected_candidate",
    "selected_mask",
    "target_bacc_after",
    "target_bacc_before",
}


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: object = None) -> object:
    if key not in data:
        return default
    arr = np.asarray(data[key])
    if arr.shape == ():
        return arr.item()
    if arr.size == 1:
        return arr.reshape(-1)[0].item()
    return default


def _role_counts(role: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(role.astype(str), return_counts=True)
    return {str(v): int(c) for v, c in zip(values, counts)}


def _manifest_expected(path: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    bundle = load_frozen_feature_npz(path)
    manifest = {
        **bundle.metadata,
        "n_samples": int(bundle.z.shape[0]),
        "z_dim": int(bundle.z.shape[1]),
        "deployable": False,
        "cedar_role": "feature_supply_candidate_only",
    }
    manifest_hash = stable_json_hash(manifest)
    manifest["manifest_hash"] = manifest_hash
    return manifest, bundle.metadata


def validate_feature_file(path: str | Path, *, n_splits: int = 3) -> dict[str, Any]:
    path = Path(path)
    record: dict[str, Any] = {
        "path": str(path),
        "status": "FAIL",
        "errors": [],
    }
    try:
        inventory = inspect_feature_file(path).to_dict()
        bundle = load_frozen_feature_npz(path)
        data = np.load(path, allow_pickle=False)
        keys = set(data.files)

        role = np.asarray(bundle.role).astype(str)
        source_keep = np.char.startswith(role, "source_")
        target_keep = np.char.startswith(role, "target_")
        source_view = bundle.source_selection_view()
        folds = make_folds(
            len(source_view["groups"]),
            groups=np.asarray(source_view["groups"]).astype(str),
            n_splits=n_splits,
            seed=0,
        )
        for tr, ev in folds:
            train_groups = set(np.asarray(source_view["groups"])[tr].astype(str))
            eval_groups = set(np.asarray(source_view["groups"])[ev].astype(str))
            if train_groups & eval_groups:
                raise FrozenFeatureSchemaError("grouped split produced overlapping groups")

        forbidden_keys = sorted(keys & FORBIDDEN_SELECTOR_KEYS)
        deployable = bool(_scalar(data, "deployable", False))
        cedar_role = str(_scalar(data, "cedar_role", ""))
        if deployable:
            record["errors"].append("artifact marked deployable")
        if cedar_role != "feature_supply_candidate_only":
            record["errors"].append(f"unexpected cedar_role:{cedar_role}")
        if forbidden_keys:
            record["errors"].append("selector_keys_present:" + ",".join(forbidden_keys))
        if "role" in source_view:
            record["errors"].append("source_selection_view_exposes_role")
        if int(source_keep.sum()) != int(source_view["z"].shape[0]):
            record["errors"].append("source_selection_view_row_count_mismatch")
        if int(target_keep.sum()) <= 0:
            record["errors"].append("missing_target_audit_rows")

        manifest_path = path.with_suffix(".manifest.json")
        manifest_valid = False
        manifest_hash_valid = False
        if not manifest_path.exists():
            record["errors"].append("missing_manifest_json")
            manifest = {}
            expected_manifest = {}
        else:
            manifest = json.loads(manifest_path.read_text())
            expected_manifest, _ = _manifest_expected(path)
            manifest_valid = manifest.get("file_sha256") == sha256_file(path)
            manifest_hash_valid = manifest.get("manifest_hash") == expected_manifest["manifest_hash"]
            if not manifest_valid:
                record["errors"].append("manifest_file_sha256_mismatch")
            if not manifest_hash_valid:
                record["errors"].append("manifest_hash_mismatch")

        record.update(
            {
                "status": "PASS" if not record["errors"] else "FAIL",
                "dataset": bundle.metadata.get("dataset"),
                "backbone": bundle.metadata.get("backbone"),
                "fold_id": bundle.metadata.get("fold_id"),
                "seed": bundle.metadata.get("seed"),
                "file_sha256": bundle.metadata.get("file_sha256"),
                "manifest": str(manifest_path),
                "manifest_file_sha256": manifest.get("file_sha256"),
                "manifest_hash": manifest.get("manifest_hash"),
                "expected_manifest_hash": expected_manifest.get("manifest_hash"),
                "manifest_valid": manifest_valid,
                "manifest_hash_valid": manifest_hash_valid,
                "inventory_status": inventory["status"],
                "n_samples": int(bundle.z.shape[0]),
                "z_dim": int(bundle.z.shape[1]),
                "n_groups": int(len(np.unique(bundle.groups.astype(str)))),
                "n_domains": int(len(np.unique(bundle.domain.astype(str)))),
                "role_counts": _role_counts(role),
                "source_selection_view": {
                    "source_rows": int(source_view["z"].shape[0]),
                    "target_rows_quarantined": int(target_keep.sum()),
                    "keys": sorted(source_view.keys()),
                    "target_role_exposed": False,
                },
                "grouped_split": {
                    "required": True,
                    "feasible": True,
                    "n_folds": int(len(folds)),
                    "n_source_groups": int(len(np.unique(np.asarray(source_view["groups"]).astype(str)))),
                    "split_policy": "group_disjoint_crossfit",
                },
                "no_selector_proof": {
                    "selection_run": False,
                    "scientific_readout_run": False,
                    "deployable": deployable,
                    "cedar_role": cedar_role,
                    "forbidden_selector_keys_present": forbidden_keys,
                },
            }
        )
    except Exception as exc:
        record["errors"].append(f"{type(exc).__name__}:{exc}")
    return record


def validate_feature_root(
    root: str | Path,
    *,
    expected_count: int | None = None,
    expected_backbones: list[str] | None = None,
    n_splits: int = 3,
) -> dict[str, Any]:
    root = Path(root)
    paths = iter_feature_candidates([root])
    records = [validate_feature_file(path, n_splits=n_splits) for path in paths]
    backbones = sorted({str(r.get("backbone")) for r in records if r.get("backbone")})
    errors: list[str] = []
    if expected_count is not None and len(records) != expected_count:
        errors.append(f"artifact_count:{len(records)}!=expected:{expected_count}")
    if expected_backbones is not None and backbones != sorted(expected_backbones):
        errors.append("backbone_set:" + ",".join(backbones))
    failed = [r["path"] for r in records if r["status"] != "PASS"]
    if failed:
        errors.append(f"artifact_failures:{len(failed)}")
    return {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01F_feature_supply_route_c",
        "root": str(root),
        "selection_run": False,
        "scientific_readout_run": False,
        "deployable": False,
        "expected_count": expected_count,
        "artifact_count": len(records),
        "backbones": backbones,
        "complete": not errors,
        "errors": errors,
        "records": records,
    }


def write_json(payload: dict[str, Any], path: str | Path) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--expected-count", type=int, default=None)
    ap.add_argument("--expected-backbones", nargs="*", default=None)
    ap.add_argument("--n-splits", type=int, default=3)
    ap.add_argument("--schema-out", default="")
    ap.add_argument("--manifest-out", default="")
    ap.add_argument("--source-view-out", default="")
    args = ap.parse_args()

    payload = validate_feature_root(
        args.root,
        expected_count=args.expected_count,
        expected_backbones=args.expected_backbones,
        n_splits=args.n_splits,
    )
    if args.schema_out:
        write_json(payload, args.schema_out)
    if args.manifest_out:
        manifest_payload = {
            k: payload[k]
            for k in (
                "project",
                "phase",
                "root",
                "selection_run",
                "scientific_readout_run",
                "deployable",
                "expected_count",
                "artifact_count",
                "backbones",
                "complete",
                "errors",
            )
        }
        manifest_payload["records"] = [
            {
                "path": r["path"],
                "manifest": r.get("manifest"),
                "file_sha256": r.get("file_sha256"),
                "manifest_file_sha256": r.get("manifest_file_sha256"),
                "manifest_hash": r.get("manifest_hash"),
                "expected_manifest_hash": r.get("expected_manifest_hash"),
                "manifest_valid": r.get("manifest_valid"),
                "manifest_hash_valid": r.get("manifest_hash_valid"),
                "status": r["status"],
                "errors": r["errors"],
            }
            for r in payload["records"]
        ]
        write_json(manifest_payload, args.manifest_out)
    if args.source_view_out:
        source_payload = {
            "project": payload["project"],
            "phase": payload["phase"],
            "root": payload["root"],
            "complete": payload["complete"],
            "records": [
                {
                    "path": r["path"],
                    "status": r["status"],
                    "source_selection_view": r.get("source_selection_view"),
                    "grouped_split": r.get("grouped_split"),
                    "no_selector_proof": r.get("no_selector_proof"),
                    "errors": r["errors"],
                }
                for r in payload["records"]
            ],
        }
        write_json(source_payload, args.source_view_out)
    print(json.dumps({k: payload[k] for k in ("artifact_count", "backbones", "complete", "errors")}, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
