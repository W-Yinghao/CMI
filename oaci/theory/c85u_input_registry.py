"""Metadata-only frozen input registry for C85U readiness and execution."""
from __future__ import annotations

import csv
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from oaci.multidataset.c84s_common import canonical_sha256, require, sha256_file, write_csv
from oaci.multidataset.c84sr1_common import context_id, context_identity
from oaci.multidataset.c84sr1_context_enumerator import (
    CandidateDescriptor,
    ContextDescriptor,
    canonical_candidates,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c85urp_tables"
PROTOCOL_PATH = REPORT_DIR / "C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.json"
PROTOCOL_SHA_PATH = REPORT_DIR / "C85U_CANDIDATE_UTILITY_RECONSTRUCTION_PROTOCOL.sha256"
PROTOCOL_SHA256 = "c9ed7081cf8cb1a6c8a05181d1660da2015b4e1716a05c8916f7fe5b09efc160"

C84F_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84-full-field-target-replay-v2/lock_f0c369ee273352b47e36"
)
COMPLETE_FIELD_MANIFEST = C84F_ROOT / "C84F_COMPLETE_FIELD_MANIFEST.json"
TARGET_TRIAL_REGISTRY = C84F_ROOT / "C84F_TARGET_UNLABELED_TRIAL_REGISTRY.json"
C84S_V5_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5")
SELECTION_MANIFEST = (
    C84S_V5_ROOT / "stage_b_selection_freeze/C84S_SELECTION_FREEZE_MANIFEST_V3.json"
)
SCIENTIFIC_RESULT = C84S_V5_ROOT / "stage_c_scientific_result/C84S_RESULT.json"
RESULT_MANIFEST = (
    C84S_V5_ROOT / "stage_c_scientific_result/C84S_RESULT_ARTIFACT_MANIFEST.json"
)
STAGE_A_ROOT = Path(
    "/projects/EEG-foundation-model/yinghao/"
    "oaci-c84s-analysis-v3/stage_a_labels"
)
EVALUATION_SEAL = STAGE_A_ROOT / "C84S_STAGE_A_EVALUATION_SEAL.json"
EVALUATION_VIEW_MANIFEST = STAGE_A_ROOT / "target_evaluation_label_view/manifest.json"
EVALUATION_LABEL_TABLE = STAGE_A_ROOT / "target_evaluation_label_view/labels.csv"
V5_LOCK = REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V5.json"
OPERATIVE_REGISTRY = REPORT_DIR / "c84fl2_tables/operative_complete_unit_registry_replay.csv"

EXPECTED_SHA256 = {
    COMPLETE_FIELD_MANIFEST: "cfffcac1a55148941b809b69bed2c9a8957a94729ed7f2c2c29ed8d48c0134d8",
    TARGET_TRIAL_REGISTRY: "52526aaf7d9bd941bac693a0947971dc35b9083c1c783619f97055926aceabb8",
    SELECTION_MANIFEST: "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
    SCIENTIFIC_RESULT: "5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7",
    RESULT_MANIFEST: "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
    EVALUATION_SEAL: "54e06dff60d80255631dc4faa20c8c7db651f2af8fc5415671dd9ab6681b5502",
    EVALUATION_VIEW_MANIFEST: "6fad247629eb48340a4badf9ab1a0669652757a58216e46826e4dfd8bfd608bd",
    V5_LOCK: "030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846",
    OPERATIVE_REGISTRY: "b0117d94f221eaab1b49b7181f3a026a77ee19ee4e1f7e8e9b9de541c7d45591",
}
EVALUATION_LABEL_SHA256 = "ea76c34663edac1e6e7e844fee6af3f06058aaaf3846febda1dff94df343a371"
DATASET_TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
PANELS = ("A", "B")
SEEDS = (5, 6)
LEVELS = (0, 1)
REGIME_RANK = {"ERM": 0, "OACI": 1, "SRC": 2}


@dataclass(frozen=True)
class FrozenInputRegistry:
    contexts: tuple[ContextDescriptor, ...]
    input_rows: tuple[dict[str, Any], ...]
    target_artifact_rows: tuple[dict[str, Any], ...]
    candidate_rows: tuple[dict[str, Any], ...]
    context_rows: tuple[dict[str, Any], ...]
    access_counters: Mapping[str, int]


def _json_exact(path: Path) -> dict[str, Any]:
    require(path.is_file(), f"C85U metadata object absent: {path}")
    require(sha256_file(path) == EXPECTED_SHA256[path], f"C85U metadata SHA drift: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(value, dict), f"C85U metadata object malformed: {path}")
    return value


def _csv_exact(path: Path) -> list[dict[str, str]]:
    require(path.is_file(), f"C85U metadata table absent: {path}")
    require(sha256_file(path) == EXPECTED_SHA256[path], f"C85U metadata table SHA drift: {path}")
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_protocol() -> dict[str, Any]:
    require(PROTOCOL_PATH.is_file() and PROTOCOL_SHA_PATH.is_file(), "C85U protocol absent")
    expected = PROTOCOL_SHA_PATH.read_text(encoding="ascii").split()[0]
    require(expected == PROTOCOL_SHA256, "C85U protocol sidecar drift")
    require(sha256_file(PROTOCOL_PATH) == expected, "C85U protocol SHA drift")
    value = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    require(value["schema_version"] == "c85u_candidate_utility_reconstruction_protocol_v1",
            "C85U protocol schema drift")
    return value


def _candidate_descriptor(
    registry_row: Mapping[str, str], field_row: Mapping[str, Any],
) -> CandidateDescriptor:
    unit_id = str(registry_row["unit_id"])
    require(str(field_row["unit_id"]) == unit_id, "C85U field/unit identity drift")
    target = field_row["complete_target_unlabeled"]
    target_sidecar = field_row["target_context_digest_index"]
    source = field_row["source_audit"]
    training = field_row["training_sidecar"]
    return CandidateDescriptor(
        dataset=str(registry_row["dataset"]),
        panel=str(registry_row["panel"]),
        training_seed=int(registry_row["training_seed"]),
        level=int(registry_row["level"]),
        regime=str(registry_row["regime"]),
        epoch=int(registry_row["epoch"]),
        trajectory_order=int(registry_row["trajectory_order"]),
        unit_id=unit_id,
        level_intervention_id=str(field_row["level_intervention_id"]),
        source_audit_path=str(source["path"]),
        source_audit_sha256=str(source["sha256"]),
        target_artifact_path=str(target["path"]),
        target_artifact_sha256=str(target["sha256"]),
        training_sidecar_path=str(training["path"]),
        training_sidecar_sha256=str(training["sha256"]),
        target_sidecar_path=str(target_sidecar["path"]),
        target_sidecar_sha256=str(target_sidecar["sha256"]),
    )


def _target_subjects(registry: Mapping[str, Any]) -> dict[str, tuple[str, ...]]:
    require(registry["schema_version"] == "c84f_target_unlabeled_trial_registry_v1",
            "C85U target registry schema drift")
    require(registry.get("target_label_access") == 0 and
            registry.get("target_y_operations") in (0, []),
            "C85U target registry protected counters drift")
    trials = registry["trials"]
    require(len(trials) == 9621, "C85U target trial count drift")
    forbidden = {key for row in trials for key in row if "label" in key.lower() or key.lower() == "y"}
    require(not forbidden, "C85U label-like field in target registry")
    grouped = {dataset: set() for dataset in DATASET_TARGET_COUNTS}
    for row in trials:
        grouped[str(row["dataset"])].add(str(row["target_subject_id"]))
    require({key: len(value) for key, value in grouped.items()} == DATASET_TARGET_COUNTS,
            "C85U target subject coverage drift")
    return {
        dataset: tuple(sorted(values, key=lambda value: int(value)))
        for dataset, values in grouped.items()
    }


def _file_identity_row(
    object_id: str, path: Path, *, access_class: str,
    expected_sha256: str, expected_bytes: int | None = None,
) -> dict[str, Any]:
    require(path.is_file(), f"C85U input absent: {path}")
    observed_bytes = path.stat().st_size
    if expected_bytes is not None:
        require(observed_bytes == expected_bytes, f"C85U input size drift: {path}")
    opened = access_class == "HASH_REPLAYED_METADATA"
    observed = sha256_file(path) if opened else "NOT_OPENED_C85URP"
    require(not opened or observed == expected_sha256, f"C85U input SHA drift: {path}")
    return {
        "object_id": object_id,
        "path": str(path),
        "bytes": observed_bytes,
        "expected_sha256": expected_sha256,
        "observed_sha256": observed,
        "access_class": access_class,
        "status": "PASS" if opened else "BOUND_BY_PARENT_MANIFEST_NOT_OPENED",
    }


def build_frozen_input_registry() -> FrozenInputRegistry:
    verify_protocol()
    field_manifest = _json_exact(COMPLETE_FIELD_MANIFEST)
    target_registry = _json_exact(TARGET_TRIAL_REGISTRY)
    selection_manifest = _json_exact(SELECTION_MANIFEST)
    _json_exact(SCIENTIFIC_RESULT)
    result_manifest = _json_exact(RESULT_MANIFEST)
    evaluation_seal = _json_exact(EVALUATION_SEAL)
    evaluation_manifest = _json_exact(EVALUATION_VIEW_MANIFEST)
    _json_exact(V5_LOCK)
    operative = _csv_exact(OPERATIVE_REGISTRY)

    require(field_manifest["gate"] ==
            "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
            "C85U complete-field gate drift")
    require(field_manifest["target_evaluation_labels"] == 0 and
            field_manifest["selector_scores"] == 0 and
            field_manifest["scientific_statistics"] == 0,
            "C85U complete-field protected counters drift")
    raw_descriptors = field_manifest["field_descriptors"]
    require(len(raw_descriptors) == 1944, "C85U field descriptor count drift")
    by_unit = {str(row["unit_id"]): row for row in raw_descriptors}
    require(len(by_unit) == 1944, "C85U field descriptor unit IDs are not unique")
    require(len(operative) == 1944 and len({row["unit_id"] for row in operative}) == 1944,
            "C85U operative registry coverage drift")
    require(set(by_unit) == {row["unit_id"] for row in operative},
            "C85U operative/field unit set drift")

    candidates = [_candidate_descriptor(row, by_unit[row["unit_id"]]) for row in operative]
    zoos: dict[tuple[str, str, int, int], list[CandidateDescriptor]] = {}
    for candidate in candidates:
        key = (candidate.dataset, candidate.panel, candidate.training_seed, candidate.level)
        zoos.setdefault(key, []).append(candidate)
    expected_zoos = {
        (dataset, panel, seed, level)
        for dataset in DATASET_TARGET_COUNTS for panel in PANELS
        for seed in SEEDS for level in LEVELS
    }
    require(set(zoos) == expected_zoos and len(zoos) == 24, "C85U zoo coverage drift")
    canonical = {key: canonical_candidates(value) for key, value in zoos.items()}

    target_artifact_rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    for zoo_key in sorted(canonical):
        for index, candidate in enumerate(canonical[zoo_key]):
            target_path = Path(candidate.target_artifact_path)
            sidecar_path = Path(candidate.target_sidecar_path)
            require(target_path.is_file() and sidecar_path.is_file(),
                    "C85U target artifact or digest sidecar absent")
            target_artifact_rows.append({
                "unit_id": candidate.unit_id,
                "dataset": candidate.dataset,
                "panel": candidate.panel,
                "training_seed": candidate.training_seed,
                "level": candidate.level,
                "candidate_index": index,
                "target_artifact_path": str(target_path),
                "target_artifact_bytes": target_path.stat().st_size,
                "target_artifact_sha256": candidate.target_artifact_sha256,
                "target_artifact_opened": 0,
                "target_sidecar_path": str(sidecar_path),
                "target_sidecar_bytes": sidecar_path.stat().st_size,
                "target_sidecar_sha256": candidate.target_sidecar_sha256,
                "target_sidecar_opened": 0,
            })
            candidate_rows.append({
                "dataset": candidate.dataset,
                "panel": candidate.panel,
                "training_seed": candidate.training_seed,
                "level": candidate.level,
                "candidate_index": index,
                "unit_id": candidate.unit_id,
                "regime": candidate.regime,
                "epoch": candidate.epoch,
                "trajectory_order": candidate.trajectory_order,
                "level_intervention_id": candidate.level_intervention_id,
                "target_artifact_sha256": candidate.target_artifact_sha256,
            })
    require(len(target_artifact_rows) == len(candidate_rows) == 1944,
            "C85U candidate registry row-count drift")
    require(len({row["target_artifact_path"] for row in target_artifact_rows}) == 1944,
            "C85U target artifact paths are not unique")

    targets = _target_subjects(target_registry)
    contexts: list[ContextDescriptor] = []
    context_rows: list[dict[str, Any]] = []
    for dataset in DATASET_TARGET_COUNTS:
        for target in targets[dataset]:
            for panel in PANELS:
                for seed in SEEDS:
                    for level in LEVELS:
                        identity = context_identity(dataset, target, panel, seed, level)
                        ordered = canonical[(dataset, panel, seed, level)]
                        descriptor = ContextDescriptor(
                            context_id=context_id(identity), dataset=dataset,
                            target_subject_id=target, panel=panel,
                            training_seed=seed, level=level, candidates=ordered,
                        )
                        contexts.append(descriptor)
                        context_rows.append({
                            **identity,
                            "context_id": descriptor.context_id,
                            "candidate_count": 81,
                            "candidate_id_order_sha256": hashlib.sha256(
                                "\n".join(row.unit_id for row in ordered).encode("ascii")
                            ).hexdigest(),
                            "target_artifact_input_sha256": canonical_sha256([
                                {"unit_id": row.unit_id, "sha256": row.target_artifact_sha256}
                                for row in ordered
                            ]),
                        })
    require(len(contexts) == len(context_rows) == 944 and
            len({row.context_id for row in contexts}) == 944,
            "C85U context coverage drift")

    descriptor = evaluation_seal["evaluation_descriptor"]
    require(descriptor["kind"] == "evaluation" and descriptor["row_count"] == 4848,
            "C85U evaluation descriptor drift")
    require(descriptor["manifest_sha256"] == EXPECTED_SHA256[EVALUATION_VIEW_MANIFEST],
            "C85U evaluation descriptor manifest linkage drift")
    require(evaluation_seal["released_to_Stage_B"] is False,
            "C85U evaluation descriptor historical release drift")
    require(evaluation_manifest["kind"] == "evaluation" and
            evaluation_manifest["row_count"] == 4848 and
            evaluation_manifest["candidate_artifacts"] == 0 and
            evaluation_manifest["EEG_arrays"] == 0,
            "C85U evaluation view manifest drift")
    require(evaluation_manifest["table"] == {
        "path": "labels.csv", "sha256": EVALUATION_LABEL_SHA256,
    }, "C85U evaluation table linkage drift")
    require(selection_manifest["contexts"] == 944 and len(selection_manifest["artifacts"]) == 10,
            "C85U selection manifest metadata drift")
    require(len(result_manifest["artifacts"]) == 18,
            "C85U result manifest metadata drift")

    input_rows = [
        _file_identity_row("C84F_complete_field_manifest", COMPLETE_FIELD_MANIFEST,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[COMPLETE_FIELD_MANIFEST], expected_bytes=6334399),
        _file_identity_row("C84F_target_trial_registry", TARGET_TRIAL_REGISTRY,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[TARGET_TRIAL_REGISTRY]),
        _file_identity_row("C84S_V5_lock", V5_LOCK, access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[V5_LOCK]),
        _file_identity_row("C84S_selection_manifest", SELECTION_MANIFEST,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[SELECTION_MANIFEST]),
        _file_identity_row("C84S_scientific_result", SCIENTIFIC_RESULT,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[SCIENTIFIC_RESULT]),
        _file_identity_row("C84S_result_manifest", RESULT_MANIFEST,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[RESULT_MANIFEST]),
        _file_identity_row("Stage_A_evaluation_seal", EVALUATION_SEAL,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[EVALUATION_SEAL], expected_bytes=587),
        _file_identity_row("evaluation_view_manifest", EVALUATION_VIEW_MANIFEST,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[EVALUATION_VIEW_MANIFEST], expected_bytes=347),
        _file_identity_row("evaluation_label_table", EVALUATION_LABEL_TABLE,
                           access_class="BOUND_BY_VIEW_MANIFEST_NOT_OPENED",
                           expected_sha256=EVALUATION_LABEL_SHA256, expected_bytes=394109),
        _file_identity_row("operative_unit_registry", OPERATIVE_REGISTRY,
                           access_class="HASH_REPLAYED_METADATA",
                           expected_sha256=EXPECTED_SHA256[OPERATIVE_REGISTRY]),
    ]
    access_counters = {
        "evaluation_label_rows_opened": 0,
        "target_artifact_payloads_opened": 0,
        "target_sidecar_payloads_opened": 0,
        "Q0_shards_opened": 0,
        "direct_C84S_tables_opened": 0,
        "candidate_utilities_computed": 0,
    }
    return FrozenInputRegistry(
        contexts=tuple(contexts), input_rows=tuple(input_rows),
        target_artifact_rows=tuple(target_artifact_rows),
        candidate_rows=tuple(candidate_rows), context_rows=tuple(context_rows),
        access_counters=access_counters,
    )


def write_readiness_registries(output_dir: str | Path = TABLE_DIR) -> dict[str, Any]:
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    registry = build_frozen_input_registry()
    outputs = {
        "frozen_input_identity_registry.csv": write_csv(
            root / "frozen_input_identity_registry.csv", registry.input_rows,
        ),
        "target_artifact_registry.csv": write_csv(
            root / "target_artifact_registry.csv", registry.target_artifact_rows,
        ),
        "candidate_order_registry.csv": write_csv(
            root / "candidate_order_registry.csv", registry.candidate_rows,
        ),
        "context_descriptor_registry.csv": write_csv(
            root / "context_descriptor_registry.csv", registry.context_rows,
        ),
    }
    return {
        "schema_version": "c85urp_metadata_only_input_registry_v1",
        "status": "PASS",
        "contexts": len(registry.contexts),
        "candidate_units": len(registry.candidate_rows),
        "target_artifacts": len(registry.target_artifact_rows),
        "target_artifact_bytes": sum(
            int(row["target_artifact_bytes"]) for row in registry.target_artifact_rows
        ),
        "outputs": outputs,
        "protected_access_counters": dict(registry.access_counters),
    }


__all__ = [
    "EVALUATION_LABEL_SHA256", "EVALUATION_LABEL_TABLE", "EVALUATION_SEAL",
    "EVALUATION_VIEW_MANIFEST", "FrozenInputRegistry", "PROTOCOL_SHA256",
    "SELECTION_MANIFEST", "SCIENTIFIC_RESULT", "RESULT_MANIFEST",
    "build_frozen_input_registry", "write_readiness_registries",
]
