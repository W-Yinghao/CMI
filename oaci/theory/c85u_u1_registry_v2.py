"""U1-only runtime input registry for C85U V2.

This module intentionally contains no Stage-B or scientific-result path.  It
reconstructs the 944 context descriptors solely from protected-input metadata
that U1 is allowed to receive.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file
from oaci.multidataset.c84sr1_context_enumerator import (
    CandidateDescriptor,
    ContextDescriptor,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
READINESS_TABLE_DIR = REPORT_DIR / "c85urp_tables"

C84F_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84-full-field-target-replay-v2/lock_f0c369ee273352b47e36"
)
COMPLETE_FIELD_MANIFEST = C84F_ROOT / "C84F_COMPLETE_FIELD_MANIFEST.json"
TARGET_TRIAL_REGISTRY = C84F_ROOT / "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json"
STAGE_A_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84s-analysis-v3/stage_a_labels"
)
EVALUATION_SEAL = STAGE_A_ROOT / "C84S_STAGE_A_EVALUATION_SEAL.json"
EVALUATION_VIEW_MANIFEST = STAGE_A_ROOT / "target_evaluation_label_view/manifest.json"
EVALUATION_LABEL_TABLE = STAGE_A_ROOT / "target_evaluation_label_view/labels.csv"
OPERATIVE_CANDIDATE_REGISTRY = (
    REPORT_DIR / "c84fl2_tables/operative_complete_unit_registry_replay.csv"
)
TARGET_ARTIFACT_REGISTRY = READINESS_TABLE_DIR / "target_artifact_registry.csv"
CANDIDATE_ORDER_REGISTRY = READINESS_TABLE_DIR / "candidate_order_registry.csv"
CONTEXT_DESCRIPTOR_REGISTRY = READINESS_TABLE_DIR / "context_descriptor_registry.csv"

EXPECTED_FILE_SHA256 = {
    COMPLETE_FIELD_MANIFEST: "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
    TARGET_TRIAL_REGISTRY: "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
    EVALUATION_SEAL: "54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502",
    EVALUATION_VIEW_MANIFEST: "6fad247629eb48340a4badf9ab1a0669652757a58216e46826e4dfd8bfd608bd",
    OPERATIVE_CANDIDATE_REGISTRY: "b0117d94f221eaab1b49b7181f3a026a77ee19ee4e1f7e8e9b9de541c7d45591",
}
EVALUATION_LABEL_TABLE_SHA256 = (
    "ea76c34663edac1e6e7e844fee6af3f06058aaaf3846febda1dff94df343a371"
)
EXPECTED_EVALUATION_ROWS = 4_848
EXPECTED_TARGET_ARTIFACTS = 1_944
EXPECTED_TARGET_ARTIFACT_BYTES = 48_018_748_054
EXPECTED_CONTEXTS = 944
EXPECTED_CANDIDATES = 81


@dataclass(frozen=True)
class U1RuntimeRegistry:
    contexts: tuple[ContextDescriptor, ...]
    target_artifact_rows: tuple[dict[str, Any], ...]
    context_rows: tuple[dict[str, Any], ...]
    candidate_rows: tuple[dict[str, Any], ...]
    evaluation_label_table_path: Path
    evaluation_label_table_sha256: str
    evaluation_label_table_rows: int
    evaluation_view_manifest_sha256: str
    target_artifact_registry_sha256: str
    target_sidecar_registry_sha256: str
    target_artifact_total_bytes: int


def _read_csv(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"C85U V2 U1 metadata table absent: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    require(rows, f"C85U V2 U1 metadata table empty: {path}")
    return rows


def _verify_exact_metadata(path: Path) -> None:
    require(path.is_file(), f"C85U V2 U1 metadata absent: {path}")
    require(
        sha256_file(path) == EXPECTED_FILE_SHA256[path],
        f"C85U V2 U1 metadata SHA drift: {path}",
    )


def _artifact_registry_digest(
    rows: list[dict[str, str]], *, sidecar: bool,
) -> str:
    prefix = "target_sidecar" if sidecar else "target_artifact"
    return canonical_sha256(
        [
            {
                "unit_id": row["unit_id"],
                "path": row[f"{prefix}_path"],
                "bytes": int(row[f"{prefix}_bytes"]),
                "sha256": row[f"{prefix}_sha256"],
            }
            for row in rows
        ]
    )


def _candidate_descriptor(
    candidate: Mapping[str, str], artifact: Mapping[str, str],
) -> CandidateDescriptor:
    require(candidate["unit_id"] == artifact["unit_id"],
            "C85U V2 U1 candidate/artifact unit drift")
    require(candidate["target_artifact_sha256"] == artifact["target_artifact_sha256"],
            "C85U V2 U1 candidate/artifact SHA drift")
    # Source and training paths are structurally required by the historical
    # descriptor type but are never inputs to U1 utility construction.
    return CandidateDescriptor(
        dataset=candidate["dataset"],
        panel=candidate["panel"],
        training_seed=int(candidate["training_seed"]),
        level=int(candidate["level"]),
        regime=candidate["regime"],
        epoch=int(candidate["epoch"]),
        trajectory_order=int(candidate["trajectory_order"]),
        unit_id=candidate["unit_id"],
        level_intervention_id=candidate["level_intervention_id"],
        source_audit_path="U1_NOT_AVAILABLE",
        source_audit_sha256="0" * 64,
        target_artifact_path=artifact["target_artifact_path"],
        target_artifact_sha256=artifact["target_artifact_sha256"],
        training_sidecar_path="U1_NOT_AVAILABLE",
        training_sidecar_sha256="0" * 64,
        target_sidecar_path=artifact["target_sidecar_path"],
        target_sidecar_sha256=artifact["target_sidecar_sha256"],
    )


def build_u1_runtime_registry() -> U1RuntimeRegistry:
    """Build the exact U1 registry without defining or opening U2 objects."""
    for path in EXPECTED_FILE_SHA256:
        _verify_exact_metadata(path)
    require(EVALUATION_LABEL_TABLE.is_file(), "C85U V2 evaluation label table absent")
    require(EVALUATION_LABEL_TABLE.stat().st_size == 394_109,
            "C85U V2 evaluation label table size drift")

    seal = json.loads(EVALUATION_SEAL.read_text(encoding="utf-8"))
    view = json.loads(EVALUATION_VIEW_MANIFEST.read_text(encoding="utf-8"))
    descriptor = seal.get("evaluation_descriptor", {})
    require(
        descriptor.get("kind") == "evaluation"
        and int(descriptor.get("row_count", -1)) == EXPECTED_EVALUATION_ROWS
        and descriptor.get("manifest_sha256")
        == EXPECTED_FILE_SHA256[EVALUATION_VIEW_MANIFEST]
        and seal.get("released_to_Stage_B") is False,
        "C85U V2 U1 evaluation seal drift",
    )
    require(
        view.get("kind") == "evaluation"
        and int(view.get("row_count", -1)) == EXPECTED_EVALUATION_ROWS
        and view.get("candidate_artifacts") == 0
        and view.get("EEG_arrays") == 0
        and view.get("table")
        == {"path": "labels.csv", "sha256": EVALUATION_LABEL_TABLE_SHA256},
        "C85U V2 U1 evaluation view drift",
    )

    artifacts = _read_csv(TARGET_ARTIFACT_REGISTRY)
    candidates = _read_csv(CANDIDATE_ORDER_REGISTRY)
    context_rows = _read_csv(CONTEXT_DESCRIPTOR_REGISTRY)
    require(len(artifacts) == len(candidates) == EXPECTED_TARGET_ARTIFACTS,
            "C85U V2 U1 candidate registry coverage drift")
    require(len(context_rows) == EXPECTED_CONTEXTS,
            "C85U V2 U1 context registry coverage drift")
    artifact_by_unit = {row["unit_id"]: row for row in artifacts}
    require(len(artifact_by_unit) == EXPECTED_TARGET_ARTIFACTS,
            "C85U V2 U1 duplicate target artifact unit")

    zoos_indexed: dict[
        tuple[str, str, int, int], list[tuple[int, CandidateDescriptor]]
    ] = {}
    for row in candidates:
        descriptor_row = _candidate_descriptor(row, artifact_by_unit[row["unit_id"]])
        key = (
            descriptor_row.dataset,
            descriptor_row.panel,
            descriptor_row.training_seed,
            descriptor_row.level,
        )
        zoos_indexed.setdefault(key, []).append((int(row["candidate_index"]), descriptor_row))
    zoos: dict[tuple[str, str, int, int], tuple[CandidateDescriptor, ...]] = {}
    for key, indexed in zoos_indexed.items():
        indexed.sort(key=lambda item: item[0])
        require(
            len(indexed) == EXPECTED_CANDIDATES
            and [index for index, _ in indexed] == list(range(EXPECTED_CANDIDATES)),
            f"C85U V2 U1 canonical candidate order drift: {key}",
        )
        zoos[key] = tuple(descriptor for _, descriptor in indexed)

    contexts: list[ContextDescriptor] = []
    observed_contexts: set[str] = set()
    for row in context_rows:
        key = (
            row["dataset"], row["panel"], int(row["training_seed"]), int(row["level"]),
        )
        ordered = zoos[key]
        require(int(row["candidate_count"]) == EXPECTED_CANDIDATES,
                "C85U V2 U1 context candidate count drift")
        require(
            canonical_sha256([candidate.unit_id for candidate in ordered])
            != "",  # Canonical serialization sanity; the historical digest is checked below.
            "C85U V2 U1 candidate identity serialization failed",
        )
        historical_order_digest = __import__("hashlib").sha256(
            "\n".join(candidate.unit_id for candidate in ordered).encode("ascii")
        ).hexdigest()
        require(historical_order_digest == row["candidate_id_order_sha256"],
                "C85U V2 U1 context candidate-order digest drift")
        descriptor_row = ContextDescriptor(
            context_id=row["context_id"],
            dataset=row["dataset"],
            target_subject_id=row["target_subject_id"],
            panel=row["panel"],
            training_seed=int(row["training_seed"]),
            level=int(row["level"]),
            candidates=ordered,
        )
        require(descriptor_row.context_id not in observed_contexts,
                "C85U V2 U1 duplicate context ID")
        observed_contexts.add(descriptor_row.context_id)
        contexts.append(descriptor_row)

    target_bytes = sum(int(row["target_artifact_bytes"]) for row in artifacts)
    require(target_bytes == EXPECTED_TARGET_ARTIFACT_BYTES,
            "C85U V2 U1 target-artifact byte total drift")
    return U1RuntimeRegistry(
        contexts=tuple(contexts),
        target_artifact_rows=tuple(dict(row) for row in artifacts),
        context_rows=tuple(dict(row) for row in context_rows),
        candidate_rows=tuple(dict(row) for row in candidates),
        evaluation_label_table_path=EVALUATION_LABEL_TABLE,
        evaluation_label_table_sha256=EVALUATION_LABEL_TABLE_SHA256,
        evaluation_label_table_rows=EXPECTED_EVALUATION_ROWS,
        evaluation_view_manifest_sha256=EXPECTED_FILE_SHA256[EVALUATION_VIEW_MANIFEST],
        target_artifact_registry_sha256=_artifact_registry_digest(artifacts, sidecar=False),
        target_sidecar_registry_sha256=_artifact_registry_digest(artifacts, sidecar=True),
        target_artifact_total_bytes=target_bytes,
    )


def u1_allowed_paths(registry: U1RuntimeRegistry) -> frozenset[Path]:
    metadata = set(EXPECTED_FILE_SHA256) | {
        EVALUATION_LABEL_TABLE,
        TARGET_ARTIFACT_REGISTRY,
        CANDIDATE_ORDER_REGISTRY,
        CONTEXT_DESCRIPTOR_REGISTRY,
    }
    payloads = {
        Path(row["target_artifact_path"]).resolve()
        for row in registry.target_artifact_rows
    }
    sidecars = {
        Path(row["target_sidecar_path"]).resolve()
        for row in registry.target_artifact_rows
    }
    return frozenset(path.resolve() for path in metadata) | frozenset(payloads) | frozenset(sidecars)


__all__ = [
    "CANDIDATE_ORDER_REGISTRY",
    "COMPLETE_FIELD_MANIFEST",
    "CONTEXT_DESCRIPTOR_REGISTRY",
    "EVALUATION_LABEL_TABLE",
    "EVALUATION_LABEL_TABLE_SHA256",
    "EVALUATION_SEAL",
    "EVALUATION_VIEW_MANIFEST",
    "EXPECTED_CONTEXTS",
    "EXPECTED_EVALUATION_ROWS",
    "EXPECTED_TARGET_ARTIFACT_BYTES",
    "EXPECTED_TARGET_ARTIFACTS",
    "OPERATIVE_CANDIDATE_REGISTRY",
    "TARGET_ARTIFACT_REGISTRY",
    "TARGET_TRIAL_REGISTRY",
    "U1RuntimeRegistry",
    "build_u1_runtime_registry",
    "u1_allowed_paths",
]
