"""Canonical handoff manifest for CEDAR_01F -> CEDAR_01."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from .feature_schema import sha256_file, stable_json_hash


PLAN_HASH_RE = re.compile(r'"plan_hash"\s*:\s*"([0-9a-f]{64})"')
RUN_MANIFEST_HASH_RE = re.compile(r'"run_manifest_hash"\s*:\s*"([0-9a-f]{64})"')


def _load_json(path: str | Path) -> Any:
    with Path(path).open() as f:
        return json.load(f)


def _first_hash(path: Path, regex: re.Pattern[str]) -> str:
    text = path.read_text(errors="replace")
    match = regex.search(text)
    if not match:
        raise ValueError(f"missing hash matching {regex.pattern} in {path}")
    return match.group(1)


def _artifact_records(schema_validation: dict[str, Any]) -> list[dict[str, Any]]:
    records = []
    for rec in schema_validation.get("records", []):
        if rec.get("status") != "PASS":
            raise ValueError(f"cannot hand off failing artifact {rec.get('path')}")
        path = str(rec["path"])
        manifest = str(rec["manifest"])
        records.append(
            {
                "path": path,
                "manifest": manifest,
                "dataset": rec.get("dataset"),
                "backbone": rec.get("backbone"),
                "fold_id": rec.get("fold_id"),
                "seed": rec.get("seed"),
                "file_sha256": rec.get("file_sha256"),
                "manifest_hash": rec.get("manifest_hash"),
                "z_dim": rec.get("z_dim"),
                "n_samples": rec.get("n_samples"),
                "deployable": False,
            }
        )
    return sorted(records, key=lambda r: (str(r["backbone"]), str(r["fold_id"]), str(r["path"])))


def build_handoff_manifest(
    *,
    root: str | Path,
    source_commit: str,
    array0_stdout: str | Path,
    array0_stderr: str | Path,
    array1_stdout: str | Path,
    array1_stderr: str | Path,
) -> dict[str, Any]:
    root = Path(root)
    inventory_path = root / "feature_inventory.json"
    schema_path = root / "schema_validation.json"
    source_view_path = root / "source_selection_view_quarantine.json"
    manifest_freeze_path = root / "manifest_freeze.json"
    completion_readout = Path("cedar_eeg/reports/CEDAR_01F_ROUTE_C_COMPLETION_READOUT.md")

    inventory = _load_json(inventory_path)
    schema_validation = _load_json(schema_path)
    source_view_validation = _load_json(source_view_path)
    manifest_freeze = _load_json(manifest_freeze_path)
    artifacts = _artifact_records(schema_validation)
    if len(artifacts) != 18:
        raise ValueError(f"expected 18 artifacts, found {len(artifacts)}")
    if not schema_validation.get("complete"):
        raise ValueError("schema validation is not complete")
    if not source_view_validation.get("complete"):
        raise ValueError("source-view validation is not complete")
    if not manifest_freeze.get("complete"):
        raise ValueError("manifest freeze validation is not complete")
    if len(inventory) != 18 or any(rec.get("status") != "COMPLIANT" for rec in inventory):
        raise ValueError("inventory is not 18/18 COMPLIANT")

    array0_stdout = Path(array0_stdout)
    array0_stderr = Path(array0_stderr)
    array1_stdout = Path(array1_stdout)
    array1_stderr = Path(array1_stderr)
    payload: dict[str, Any] = {
        "project": "CEDAR-EEG",
        "phase": "CEDAR_01F_to_CEDAR_01_handoff",
        "source_commit": source_commit,
        "feature_root": str(root),
        "planned_items": 18,
        "completed_items": len(artifacts),
        "inventory_hash": sha256_file(inventory_path),
        "schema_validation_hash": sha256_file(schema_path),
        "source_view_validation_hash": sha256_file(source_view_path),
        "manifest_freeze_hash": sha256_file(manifest_freeze_path),
        "completion_readout_hash": sha256_file(completion_readout) if completion_readout.exists() else "",
        "per_array_plan_hashes": {
            "890263_0": _first_hash(array0_stdout, PLAN_HASH_RE),
            "890263_1": _first_hash(array1_stdout, PLAN_HASH_RE),
        },
        "per_array_run_manifest_hashes": {
            "890263_0": _first_hash(array0_stdout, RUN_MANIFEST_HASH_RE),
            "890263_1": _first_hash(array1_stdout, RUN_MANIFEST_HASH_RE),
        },
        "array_stdout_hashes": {
            "890263_0": sha256_file(array0_stdout),
            "890263_1": sha256_file(array1_stdout),
        },
        "array_stderr_hashes": {
            "890263_0": sha256_file(array0_stderr),
            "890263_1": sha256_file(array1_stderr),
        },
        "cluster_provenance": {
            "sacct_available": False,
            "scontrol_record_complete": False,
            "compensating_evidence": [
                "array_stdout_hash",
                "array_stderr_hash",
                "per_artifact_manifest_hash",
                "feature_inventory_hash",
                "schema_validation_hash",
            ],
            "pm_disposition": "accepted_for_shadow_audit_not_for_deployment",
        },
        "per_artifact_hashes": artifacts,
        "shared_plan_overwritten": True,
        "handoff_manifest_is_canonical": True,
        "deployable": False,
    }
    payload["canonical_payload_hash"] = stable_json_hash(payload)
    return payload


def write_handoff_manifest(payload: dict[str, Any], out: str | Path) -> dict[str, Any]:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    return payload


def validate_handoff_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    payload = _load_json(path)
    if payload.get("handoff_manifest_is_canonical") is not True:
        raise ValueError("handoff manifest is not canonical")
    if payload.get("deployable") is not False:
        raise ValueError("handoff manifest unexpectedly marked deployable")
    if payload.get("planned_items") != 18 or payload.get("completed_items") != 18:
        raise ValueError("handoff manifest does not describe 18/18 artifacts")
    artifacts = payload.get("per_artifact_hashes", [])
    if len(artifacts) != 18:
        raise ValueError("handoff manifest missing per-artifact hash records")
    for rec in artifacts:
        path_text = rec.get("path")
        expected = rec.get("file_sha256")
        if not path_text or not expected:
            raise ValueError("artifact record missing path or file_sha256")
        observed = sha256_file(path_text)
        if observed != expected:
            raise ValueError(f"feature artifact hash mismatch for {path_text}")
        manifest_path = rec.get("manifest")
        if manifest_path:
            manifest = _load_json(manifest_path)
            if manifest.get("file_sha256") != expected:
                raise ValueError(f"manifest file_sha256 mismatch for {path_text}")
            if manifest.get("manifest_hash") != rec.get("manifest_hash"):
                raise ValueError(f"manifest_hash mismatch for {path_text}")
    return payload


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--source-commit", required=True)
    ap.add_argument("--array0-stdout", default="logs/cedar01f-srcerm-890263_0.out")
    ap.add_argument("--array0-stderr", default="logs/cedar01f-srcerm-890263_0.err")
    ap.add_argument("--array1-stdout", default="logs/cedar01f-srcerm-890263_1.out")
    ap.add_argument("--array1-stderr", default="logs/cedar01f-srcerm-890263_1.err")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    payload = build_handoff_manifest(
        root=args.root,
        source_commit=args.source_commit,
        array0_stdout=args.array0_stdout,
        array0_stderr=args.array0_stderr,
        array1_stdout=args.array1_stdout,
        array1_stderr=args.array1_stderr,
    )
    payload = write_handoff_manifest(payload, args.out)
    validate_handoff_manifest(args.out)
    print(
        json.dumps(
            {
                "out": args.out,
                "completed_items": payload["completed_items"],
                "handoff_manifest_file_sha256": sha256_file(args.out),
                "canonical_payload_hash": payload["canonical_payload_hash"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
