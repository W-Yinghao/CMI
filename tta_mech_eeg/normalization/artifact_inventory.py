"""Artifact inventory for TTA_MECH_02B0 normalization / BN preflight."""

from __future__ import annotations

import csv
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from tta_mech_eeg.baselines.registry import stable_hash
from tta_mech_eeg.data.artifact_inventory import sha256_file
from tta_mech_eeg.data.handoff_schema import DEFAULT_CEDAR01F_HANDOFF


CRITICAL_READY_FIELDS: tuple[str, ...] = (
    "has_model_checkpoint",
    "has_classifier_head",
    "has_bn_buffers",
    "has_target_X",
    "has_source_X",
    "has_raw_or_preprocessed_input",
    "can_forward_model",
    "can_copy_model_without_mutation",
    "can_recompute_bn_buffers_on_copy",
    "can_disable_dropout",
    "can_eval_frozen_bn",
    "can_emit_logits_without_target_y",
)


@dataclass(frozen=True)
class BnArtifactInventoryRecord:
    dataset: str
    backbone: str
    fold_id: str
    seed: str
    feature_artifact_path: str
    feature_artifact_hash: str
    feature_artifact_hash_matches_handoff: bool
    has_model_checkpoint: bool
    has_classifier_head: bool
    has_bn_buffers: bool
    has_feature_normalizer: bool
    has_source_split: bool
    has_target_X: bool
    has_source_X: bool
    has_raw_or_preprocessed_input: bool
    can_forward_model: bool
    can_copy_model_without_mutation: bool
    can_recompute_bn_buffers_on_copy: bool
    can_disable_dropout: bool
    can_eval_frozen_bn: bool
    can_emit_logits_without_target_y: bool
    status: str
    reject_reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _npz_keys(path: str | Path) -> set[str]:
    try:
        with zipfile.ZipFile(path) as zf:
            return {name[:-4] for name in zf.namelist() if name.endswith(".npy")}
    except zipfile.BadZipFile:
        return set()


def _role_values(path: str | Path) -> set[str]:
    try:
        import numpy as np

        data = np.load(path, allow_pickle=False)
        if "role" not in data.files:
            return set()
        return {str(x) for x in data["role"].astype(str)}
    except Exception:
        return set()


def _nearby_exists(path: Path, suffixes: tuple[str, ...]) -> bool:
    stem = path.with_suffix("")
    return any(stem.with_suffix(suffix).exists() for suffix in suffixes)


def _record_from_handoff(rec: dict[str, Any]) -> BnArtifactInventoryRecord:
    path = Path(rec["path"])
    observed_hash = sha256_file(path) if path.exists() else ""
    hash_matches = observed_hash == rec.get("file_sha256")
    keys = _npz_keys(path) if path.exists() else set()
    roles = _role_values(path) if path.exists() else set()

    has_model_checkpoint = _nearby_exists(path, (".pt", ".pth", ".ckpt"))
    has_classifier_head = _nearby_exists(path, (".classifier.json", ".head.json"))
    has_bn_buffers = _nearby_exists(path, (".bn.json", ".bn_buffers.json"))
    has_feature_normalizer = _nearby_exists(path, (".normalizer.json", ".feature_normalizer.json"))
    has_source_split = {"source_train", "source_audit"} <= roles
    has_target_x = bool({"target_x", "X_target", "target_X"} & keys)
    has_source_x = bool({"source_x", "X_source", "source_X"} & keys)
    has_raw_or_preprocessed_input = bool({"X", "raw_X", "preprocessed_X"} & keys)
    can_forward_model = (
        has_model_checkpoint
        and has_classifier_head
        and has_raw_or_preprocessed_input
        and (has_target_x or "target_audit" in roles)
    )
    can_copy_model_without_mutation = has_model_checkpoint
    can_recompute_bn_buffers_on_copy = (
        can_copy_model_without_mutation
        and has_bn_buffers
        and has_raw_or_preprocessed_input
        and has_target_x
    )
    can_disable_dropout = can_forward_model
    can_eval_frozen_bn = can_forward_model and has_bn_buffers
    can_emit_logits_without_target_y = can_forward_model and has_classifier_head

    fields = {
        "has_model_checkpoint": has_model_checkpoint,
        "has_classifier_head": has_classifier_head,
        "has_bn_buffers": has_bn_buffers,
        "has_target_X": has_target_x,
        "has_source_X": has_source_x,
        "has_raw_or_preprocessed_input": has_raw_or_preprocessed_input,
        "can_forward_model": can_forward_model,
        "can_copy_model_without_mutation": can_copy_model_without_mutation,
        "can_recompute_bn_buffers_on_copy": can_recompute_bn_buffers_on_copy,
        "can_disable_dropout": can_disable_dropout,
        "can_eval_frozen_bn": can_eval_frozen_bn,
        "can_emit_logits_without_target_y": can_emit_logits_without_target_y,
    }
    missing = [name for name in CRITICAL_READY_FIELDS if not fields[name]]
    if not hash_matches:
        status = "REJECT"
        reject_reason = "feature artifact hash does not match CEDAR_01F handoff"
    elif missing:
        status = "REJECT"
        reject_reason = "missing " + ", ".join(missing)
    else:
        status = "READY"
        reject_reason = ""

    return BnArtifactInventoryRecord(
        dataset=str(rec.get("dataset", "")),
        backbone=str(rec.get("backbone", "")),
        fold_id=str(rec.get("fold_id", "")),
        seed=str(rec.get("seed", "")),
        feature_artifact_path=str(path),
        feature_artifact_hash=observed_hash,
        feature_artifact_hash_matches_handoff=hash_matches,
        has_model_checkpoint=has_model_checkpoint,
        has_classifier_head=has_classifier_head,
        has_bn_buffers=has_bn_buffers,
        has_feature_normalizer=has_feature_normalizer,
        has_source_split=has_source_split,
        has_target_X=has_target_x,
        has_source_X=has_source_x,
        has_raw_or_preprocessed_input=has_raw_or_preprocessed_input,
        can_forward_model=can_forward_model,
        can_copy_model_without_mutation=can_copy_model_without_mutation,
        can_recompute_bn_buffers_on_copy=can_recompute_bn_buffers_on_copy,
        can_disable_dropout=can_disable_dropout,
        can_eval_frozen_bn=can_eval_frozen_bn,
        can_emit_logits_without_target_y=can_emit_logits_without_target_y,
        status=status,
        reject_reason=reject_reason,
    )


def _backbone_readiness(records: list[BnArtifactInventoryRecord]) -> dict[str, Any]:
    by_backbone: dict[str, list[BnArtifactInventoryRecord]] = {}
    for record in records:
        by_backbone.setdefault(record.backbone, []).append(record)
    out = {}
    for backbone, rows in sorted(by_backbone.items()):
        ready = [row for row in rows if row.status == "READY"]
        out[backbone] = {
            "folds_total": len(rows),
            "folds_ready": len(ready),
            "full_loso_ready": len(rows) == 9 and len(ready) == 9,
            "status": "READY_FOR_02B" if len(rows) == 9 and len(ready) == 9 else "NOT_READY",
        }
    return out


def build_bn_artifact_inventory(
    handoff_manifest: str | Path = DEFAULT_CEDAR01F_HANDOFF,
) -> dict[str, Any]:
    with Path(handoff_manifest).open() as f:
        handoff = json.load(f)
    records = [
        _record_from_handoff(rec)
        for rec in sorted(
            handoff.get("per_artifact_hashes", []),
            key=lambda r: (str(r.get("backbone", "")), int(r.get("fold_id", 0)), str(r.get("path", ""))),
        )
    ]
    if any(not row.feature_artifact_hash_matches_handoff for row in records):
        feasibility = "ARTIFACT_HASH_MISMATCH_HARD_FAIL"
    elif any(row.status == "READY" for row in records):
        readiness = _backbone_readiness(records)
        feasibility = (
            "READY_FOR_02B"
            if any(item["full_loso_ready"] for item in readiness.values())
            else "PARTIAL_NOT_READY_FOR_02B"
        )
    else:
        feasibility = "TTA_MECH_02B_NOT_FEASIBLE_FROM_CURRENT_ARTIFACTS"
    payload = {
        "project": "TTA-MECH-EEG",
        "phase": "TTA_MECH_02B0_normalization_bn_audit_preflight",
        "handoff_manifest": str(handoff_manifest),
        "handoff_hash": sha256_file(handoff_manifest),
        "real_forward_run": False,
        "bn_refresh_run": False,
        "target_metrics_computed": False,
        "records": [row.to_dict() for row in records],
        "summary": {
            "total_records": len(records),
            "ready_records": sum(row.status == "READY" for row in records),
            "partial_records": sum(row.status == "PARTIAL" for row in records),
            "rejected_records": sum(row.status == "REJECT" for row in records),
            "feature_artifact_hashes_match_handoff": all(row.feature_artifact_hash_matches_handoff for row in records),
            "has_any_model_checkpoint": any(row.has_model_checkpoint for row in records),
            "has_any_bn_buffers": any(row.has_bn_buffers for row in records),
            "has_any_raw_or_preprocessed_input": any(row.has_raw_or_preprocessed_input for row in records),
            "has_any_forward_ready_artifact": any(row.can_forward_model for row in records),
            "backbone_readiness": _backbone_readiness(records),
            "feasibility": feasibility,
        },
    }
    payload["bn_artifact_inventory_hash"] = stable_hash(payload)
    return payload


def write_inventory_csv(payload: dict[str, Any], out_path: str | Path) -> None:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows = payload.get("records", [])
    if not rows:
        out.write_text("")
        return
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
