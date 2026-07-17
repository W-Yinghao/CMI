"""Attempt-bound U2 input registry for C85U V2.

No label, target-artifact, or logit location is defined in this module.  Real
U2 locations are supplied only by a semantically replayed V2 execution lock.
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from oaci.multidataset.c84s_common import require, sha256_file


@dataclass(frozen=True)
class U2RuntimeRegistry:
    selection_root: Path
    selection_manifest_path: Path
    selection_manifest_sha256: str
    candidate_ranks_path: Path
    candidate_ranks_sha256: str
    fixed_actions_path: Path
    fixed_actions_sha256: str
    q0_index_path: Path
    q0_index_sha256: str
    result_manifest_path: Path
    result_manifest_sha256: str
    method_context_path: Path
    method_context_sha256: str


def resolve_u2_runtime_registry(lock: Mapping[str, Any]) -> U2RuntimeRegistry:
    """Resolve paths from the validated lock without opening protected U2 files."""
    raw = lock.get("U2_runtime_input_registry")
    require(isinstance(raw, Mapping), "C85U V2 U2 lock registry absent")

    def identity(name: str) -> tuple[Path, str]:
        value = raw.get(name)
        require(isinstance(value, Mapping), f"C85U V2 U2 input absent: {name}")
        path = Path(str(value.get("path", "")))
        require(path.is_absolute(), f"C85U V2 U2 input path is not absolute: {name}")
        digest = str(value.get("sha256", ""))
        require(len(digest) == 64, f"C85U V2 U2 input SHA malformed: {name}")
        return path.resolve(), digest

    selection_manifest, selection_manifest_sha = identity("selection_manifest")
    ranks, ranks_sha = identity("candidate_ranks")
    fixed, fixed_sha = identity("fixed_actions")
    q0_index, q0_index_sha = identity("q0_shard_index")
    result_manifest, result_manifest_sha = identity("result_manifest")
    decisions, decisions_sha = identity("method_context_decisions")
    selection_root = selection_manifest.parent
    require(
        ranks.parent == fixed.parent == q0_index.parent == selection_root,
        "C85U V2 U2 selection input roots differ",
    )
    require(result_manifest.parent == decisions.parent,
            "C85U V2 U2 historical result roots differ")
    return U2RuntimeRegistry(
        selection_root=selection_root,
        selection_manifest_path=selection_manifest,
        selection_manifest_sha256=selection_manifest_sha,
        candidate_ranks_path=ranks,
        candidate_ranks_sha256=ranks_sha,
        fixed_actions_path=fixed,
        fixed_actions_sha256=fixed_sha,
        q0_index_path=q0_index,
        q0_index_sha256=q0_index_sha,
        result_manifest_path=result_manifest,
        result_manifest_sha256=result_manifest_sha,
        method_context_path=decisions,
        method_context_sha256=decisions_sha,
    )


def replay_u2_runtime_registry(registry: U2RuntimeRegistry) -> dict[str, Any]:
    """Open and hash U2 inputs only after the caller creates its stage receipt."""
    identities = {
        "selection_manifest": (
            registry.selection_manifest_path, registry.selection_manifest_sha256,
        ),
        "candidate_ranks": (registry.candidate_ranks_path, registry.candidate_ranks_sha256),
        "fixed_actions": (registry.fixed_actions_path, registry.fixed_actions_sha256),
        "q0_shard_index": (registry.q0_index_path, registry.q0_index_sha256),
        "result_manifest": (registry.result_manifest_path, registry.result_manifest_sha256),
        "method_context_decisions": (
            registry.method_context_path, registry.method_context_sha256,
        ),
    }
    rows: list[dict[str, Any]] = []
    for object_id, (path, expected) in identities.items():
        require(path.is_file() and sha256_file(path) == expected,
                f"C85U V2 U2 input identity drift: {object_id}")
        rows.append({
            "object_id": object_id,
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": expected,
        })
    selection = json.loads(registry.selection_manifest_path.read_text(encoding="utf-8"))
    result = json.loads(registry.result_manifest_path.read_text(encoding="utf-8"))
    require(selection.get("contexts") == 944 and selection.get("Q0_records") == 8_750_000,
            "C85U V2 U2 selection arithmetic drift")
    require(selection.get("evaluation_label_descriptor_received") is False,
            "C85U V2 U2 selection input contains evaluation descriptor")
    artifacts = result.get("artifacts")
    require(isinstance(artifacts, (dict, list)), "C85U V2 U2 result manifest malformed")
    return {"objects": rows, "objects_opened": len(rows), "status": "PASS"}


def u2_allowed_paths(registry: U2RuntimeRegistry) -> frozenset[Path]:
    return frozenset({
        registry.selection_manifest_path,
        registry.candidate_ranks_path,
        registry.fixed_actions_path,
        registry.q0_index_path,
        registry.result_manifest_path,
        registry.method_context_path,
    })


__all__ = [
    "U2RuntimeRegistry",
    "replay_u2_runtime_registry",
    "resolve_u2_runtime_registry",
    "u2_allowed_paths",
]
