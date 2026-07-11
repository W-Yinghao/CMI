#!/usr/bin/env python
"""Fail-closed immutable closure for the two frozen Route-B H200 starts.

No training or inference is performed. Source payloads are checked against the
committed STAR_00A inventory, strictly reloaded, copied to SHA-named read-only
files, and rechecked while the source SHA is held constant.
"""

import argparse
import json
import os
import shutil
import stat
from pathlib import Path
from typing import Dict, List, Mapping

from star_eeg.config import DEPENDENCY_COMMIT
from star_eeg.data.checkpoint_registry import (
    CheckpointSpec,
    inspect_spec,
    sha256_file,
    strict_reload,
)
from star_eeg.data.faced_split_contract import canonical_hash


DEFAULT_IMMUTABLE_ROOT = Path(
    "/home/infres/yinwang/CMI_AAAI_star_runtime/results/star_h200_starts_immutable"
)
H200_TAGS = ("H200_s0", "H200_s1")


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")


def _load_expected_inventory(path: Path) -> Dict[str, Mapping[str, object]]:
    payload = json.loads(path.read_text())
    entries = {str(row["tag"]): row for row in payload.get("entries", [])}
    if not all(tag in entries for tag in H200_TAGS):
        raise RuntimeError("STAR_00A inventory does not contain both H200 starts")
    if payload.get("h200_start_checkpoints_ready") is not True:
        raise RuntimeError("STAR_00A did not mark both H200 starts ready")
    return entries


def _copy_once(source: Path, payload: Path) -> None:
    payload.parent.mkdir(parents=True, exist_ok=True)
    if payload.exists():
        return
    temporary = payload.with_name(f".{payload.name}.tmp-{os.getpid()}")
    if temporary.exists():
        raise RuntimeError(f"stale closure temporary exists: {temporary}")
    try:
        with source.open("rb") as reader, temporary.open("xb") as writer:
            shutil.copyfileobj(reader, writer, length=8 * 1024 * 1024)
            writer.flush()
            os.fsync(writer.fileno())
        if payload.exists():
            raise RuntimeError(f"immutable payload appeared during closure: {payload}")
        temporary.rename(payload)
    finally:
        if temporary.exists():
            temporary.unlink()


def _ensure_stable_link(link: Path, payload: Path) -> None:
    if os.path.lexists(link):
        if not link.is_symlink() or link.resolve() != payload.resolve():
            raise RuntimeError(f"immutable best.pth link conflict: {link}")
        return
    link.symlink_to(payload.name)


def freeze_one(
    tag: str,
    expected: Mapping[str, object],
    repo_root: Path,
    immutable_root: Path,
) -> Dict[str, object]:
    seed = int(tag.rsplit("_s", 1)[1])
    source_spec = CheckpointSpec(
        tag=tag,
        budget_h=200,
        seed=seed,
        path=str(expected["path"]),
        kind="route_b_checkpoint",
        usable_as_star_start=True,
        usable_as_reference_only=False,
    )
    source_audit = inspect_spec(source_spec, repo_root=repo_root)
    required = {
        "exists": True,
        "strict_reload_pass": True,
        "training_complete": True,
        "provenance_complete": True,
        "repeated_sha_identical": True,
        "resolved_path_stable": True,
    }
    for key, value in required.items():
        if source_audit.get(key) != value:
            raise RuntimeError(f"{tag} source audit failed {key}: {source_audit.get(key)!r}")
    if source_audit["sha256"] != expected.get("sha256"):
        raise RuntimeError(f"{tag} differs from committed STAR_00A SHA")

    source = Path(source_audit["resolved_path"])
    source_sha_before = sha256_file(source)
    destination_dir = immutable_root / tag
    payload = destination_dir / f"best.{source_sha_before}.pth"
    _copy_once(source, payload)
    destination_sha_before_chmod = sha256_file(payload)
    if destination_sha_before_chmod != source_sha_before:
        raise RuntimeError(f"{tag} copied SHA differs from source")
    payload.chmod(0o444)
    link = destination_dir / "best.pth"
    _ensure_stable_link(link, payload)

    destination_spec = CheckpointSpec(
        tag=tag,
        budget_h=200,
        seed=seed,
        path=str(payload),
        kind="route_b_immutable_checkpoint",
        usable_as_star_start=True,
        usable_as_reference_only=False,
    )
    reload_info = strict_reload(destination_spec)
    destination_sha_after_reload = sha256_file(payload)
    source_sha_after = sha256_file(source)
    mode = stat.S_IMODE(payload.stat().st_mode)
    checks = {
        "source_sha_before_equals_star00a": source_sha_before == expected.get("sha256"),
        "source_sha_stable_during_copy_reload": source_sha_before == source_sha_after,
        "destination_sha_equals_source": destination_sha_after_reload == source_sha_before,
        "destination_sha_stable_during_reload": destination_sha_before_chmod == destination_sha_after_reload,
        "destination_strict_reload_pass": reload_info.get("strict_reload_pass") is True,
        "destination_read_only": (mode & 0o222) == 0,
        "best_link_is_symlink": link.is_symlink(),
        "best_link_targets_sha_payload": link.resolve() == payload.resolve(),
        "sha_named_payload": payload.name == f"best.{source_sha_before}.pth",
    }
    if not all(checks.values()):
        raise RuntimeError(f"{tag} immutable closure checks failed: {checks}")
    return {
        "tag": tag,
        "budget_h": 200,
        "seed": seed,
        "source_checkpoint": str(source),
        "source_sha256_before": source_sha_before,
        "source_sha256_after": source_sha_after,
        "immutable_checkpoint": str(payload.resolve()),
        "immutable_best_link": str(link),
        "sha256": destination_sha_after_reload,
        "bytes": payload.stat().st_size,
        "immutable_mode_octal": oct(mode),
        "source_git_commit": source_audit.get("source_git_commit"),
        "route_manifest_hash": source_audit.get("route_manifest_hash"),
        "channel_contract_hash": source_audit.get("channel_contract_hash"),
        "training_complete": source_audit.get("training_complete"),
        "strict_reload_immutable": True,
        "launcher_accepted_path": str(payload.resolve()),
        "checks": checks,
    }


def freeze_h200_starts(
    repo_root: Path,
    inventory_path: Path,
    immutable_root: Path,
) -> Dict[str, object]:
    expected = _load_expected_inventory(inventory_path)
    rows: List[Mapping[str, object]] = [
        freeze_one(tag, expected[tag], repo_root, immutable_root)
        for tag in H200_TAGS
    ]
    core = {
        "schema_version": 1,
        "phase": "STAR_00B_H200_immutable_start_closure",
        "dependency_commit": DEPENDENCY_COMMIT,
        "immutable_root": str(immutable_root.resolve()),
        "launcher_path_policy": "manifest_sha_named_payload_only",
        "checkpoints": rows,
        "training_run": False,
        "target_data_read": False,
    }
    return {**core, "h200_immutable_manifest_hash": canonical_hash(core)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=str(Path(__file__).resolve().parents[2]))
    parser.add_argument(
        "--inventory",
        default="results/star/star00a_preflight/checkpoint_inventory.json",
    )
    parser.add_argument("--immutable-root", default=str(DEFAULT_IMMUTABLE_ROOT))
    parser.add_argument("--out-dir", default="results/star/star00b_preflight")
    args = parser.parse_args()
    repo_root = Path(args.repo_root).resolve()
    inventory = Path(args.inventory)
    if not inventory.is_absolute():
        inventory = repo_root / inventory
    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = repo_root / out_dir
    immutable_root = Path(args.immutable_root)
    try:
        manifest = freeze_h200_starts(repo_root, inventory, immutable_root)
        checks = [
            check
            for row in manifest["checkpoints"]
            for check in row["checks"].values()
        ]
        go_core = {
            "phase": "STAR_00B_H200_immutable_start_closure",
            "status": "PASS" if all(checks) else "FAIL",
            "immutable_checkpoint_count": len(manifest["checkpoints"]),
            "all_read_only": all(row["checks"]["destination_read_only"] for row in manifest["checkpoints"]),
            "all_strict_reload_pass": all(row["strict_reload_immutable"] for row in manifest["checkpoints"]),
            "all_source_sha_stable": all(row["checks"]["source_sha_stable_during_copy_reload"] for row in manifest["checkpoints"]),
            "launcher_restricted_to_sha_named_payload": True,
            "training_run": False,
            "target_metrics_computed": False,
        }
        go_nogo = {**go_core, "h200_immutable_go_nogo_hash": canonical_hash(go_core)}
        _write_json(out_dir / "h200_immutable_manifest.json", manifest)
        _write_json(out_dir / "h200_immutable_go_nogo.json", go_nogo)
        print(json.dumps(go_nogo, indent=2, sort_keys=True))
        if go_nogo["status"] != "PASS":
            raise SystemExit(1)
    except Exception as exc:
        fail_core = {
            "phase": "STAR_00B_H200_immutable_start_closure",
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "training_run": False,
            "target_metrics_computed": False,
        }
        _write_json(out_dir / "h200_immutable_go_nogo.json", {
            **fail_core,
            "h200_immutable_go_nogo_hash": canonical_hash(fail_core),
        })
        raise


if __name__ == "__main__":
    main()
