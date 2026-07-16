"""Deterministic C84SR1 context enumeration from frozen field descriptors."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from .c84s_common import read_json, require, sha256_file
from .c84sr1_common import (
    CANDIDATES, COMPLETE_FIELD_MANIFEST_PATH, DATASET_TARGET_COUNTS, LEVELS,
    PANELS, SEEDS, TARGET_TRIAL_REGISTRY_PATH, context_id, context_identity,
)


REGIME_RANK = {"ERM": 0, "OACI": 1, "SRC": 2}
LEVEL_INTERVENTION_IDS = {
    0: "C84_LEVEL0_FULL_SOURCE_PANEL_V1",
    1: "C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1",
}


@dataclass(frozen=True)
class CandidateDescriptor:
    dataset: str
    panel: str
    training_seed: int
    level: int
    regime: str
    epoch: int
    trajectory_order: int
    unit_id: str
    level_intervention_id: str
    source_audit_path: str
    source_audit_sha256: str
    target_artifact_path: str
    target_artifact_sha256: str
    training_sidecar_path: str
    training_sidecar_sha256: str
    target_sidecar_path: str
    target_sidecar_sha256: str


@dataclass(frozen=True)
class ContextDescriptor:
    context_id: str
    dataset: str
    target_subject_id: str
    panel: str
    training_seed: int
    level: int
    candidates: tuple[CandidateDescriptor, ...]

    def identity(self) -> dict[str, Any]:
        return context_identity(
            self.dataset, self.target_subject_id, self.panel,
            self.training_seed, self.level,
        )


def resolve_level_intervention_id(
    raw: Mapping[str, Any], sidecar: Mapping[str, Any],
) -> str:
    """Resolve the frozen intervention identity across historical sidecar schemas."""
    require("level_intervention_id" in raw,
            "complete-field descriptor lacks level intervention identity")
    level = int(sidecar["level"])
    require(level in LEVEL_INTERVENTION_IDS, "candidate level intervention is undefined")
    expected = LEVEL_INTERVENTION_IDS[level]
    authoritative = str(raw["level_intervention_id"])
    require(authoritative == expected,
            "complete-field level intervention differs from locked level definition")
    if "level_intervention_id" in sidecar:
        require(str(sidecar["level_intervention_id"]) == authoritative,
                "sidecar/complete-field level intervention drift")
        return authoritative

    require(
        raw.get("model_reuse_provenance") == "C84C"
        and str(sidecar["panel"]) == "A"
        and int(sidecar["seed"]) == 5
        and level == 0,
        "level intervention absent outside exact historical C84C compatibility scope",
    )
    return authoritative


def _candidate_from_field_descriptor(raw: Mapping[str, Any]) -> CandidateDescriptor:
    sidecar_identity = raw["training_sidecar"]
    sidecar_path = Path(sidecar_identity["path"])
    require(sidecar_path.is_file(), "training sidecar absent")
    require(sha256_file(sidecar_path) == sidecar_identity["sha256"], "training sidecar SHA drift")
    sidecar = read_json(sidecar_path)
    require(str(sidecar["unit_id"]) == str(raw["unit_id"]), "unit/sidecar identity drift")
    dataset = str(sidecar["dataset"])
    panel = str(sidecar["panel"])
    seed = int(sidecar["seed"])
    level = int(sidecar["level"])
    regime = str(sidecar["regime"])
    trajectory_order = int(sidecar["trajectory_order"])
    require(dataset in DATASET_TARGET_COUNTS, "candidate dataset drift")
    require(panel in PANELS and seed in SEEDS and level in LEVELS, "candidate repeated-factor drift")
    require(regime in REGIME_RANK, "candidate regime drift")
    require((regime == "ERM" and trajectory_order == 0) or
            (regime != "ERM" and 1 <= trajectory_order <= 40),
            "candidate trajectory order drift")
    level_intervention_id = resolve_level_intervention_id(raw, sidecar)
    return CandidateDescriptor(
        dataset=dataset, panel=panel, training_seed=seed, level=level,
        regime=regime, epoch=int(sidecar["epoch"]), trajectory_order=trajectory_order,
        unit_id=str(raw["unit_id"]),
        level_intervention_id=level_intervention_id,
        source_audit_path=str(raw["source_audit"]["path"]),
        source_audit_sha256=str(raw["source_audit"]["sha256"]),
        target_artifact_path=str(raw["complete_target_unlabeled"]["path"]),
        target_artifact_sha256=str(raw["complete_target_unlabeled"]["sha256"]),
        training_sidecar_path=str(sidecar_path),
        training_sidecar_sha256=str(sidecar_identity["sha256"]),
        target_sidecar_path=str(raw["target_context_digest_index"]["path"]),
        target_sidecar_sha256=str(raw["target_context_digest_index"]["sha256"]),
    )


def canonical_candidates(candidates: Sequence[CandidateDescriptor]) -> tuple[CandidateDescriptor, ...]:
    require(len(candidates) == CANDIDATES, "candidate zoo does not contain 81 units")
    ordered = tuple(sorted(
        candidates,
        key=lambda row: (REGIME_RANK[row.regime], row.trajectory_order, row.unit_id),
    ))
    require(len({row.unit_id for row in ordered}) == CANDIDATES, "candidate IDs are not unique")
    require([row.regime for row in ordered] == ["ERM"] + ["OACI"] * 40 + ["SRC"] * 40,
            "canonical regime order drift")
    require([row.trajectory_order for row in ordered] == [0] + list(range(1, 41)) + list(range(1, 41)),
            "canonical trajectory order drift")
    return ordered


def target_subjects_from_registry(path: str | Path = TARGET_TRIAL_REGISTRY_PATH) -> dict[str, tuple[str, ...]]:
    payload = read_json(path)
    require(payload["schema_version"] == "c84f_target_unlabeled_trial_registry_v1",
            "target registry schema drift")
    targets: dict[str, set[str]] = {dataset: set() for dataset in DATASET_TARGET_COUNTS}
    for row in payload["trials"]:
        targets[str(row["dataset"])].add(str(row["target_subject_id"]))
    require({dataset: len(values) for dataset, values in targets.items()} == DATASET_TARGET_COUNTS,
            "target subject counts drift")
    return {
        dataset: tuple(sorted(values, key=lambda value: (int(value) if value.isdigit() else value)))
        for dataset, values in targets.items()
    }


def enumerate_contexts(
    manifest_path: str | Path = COMPLETE_FIELD_MANIFEST_PATH,
    target_registry_path: str | Path = TARGET_TRIAL_REGISTRY_PATH,
) -> list[ContextDescriptor]:
    manifest = read_json(manifest_path)
    require(manifest["gate"] == "C84_MULTI_DATASET_DUAL_LEVEL_FIXED_ZOO_FIELD_EXECUTED_AND_MANIFESTED_ANALYSIS_NOT_STARTED",
            "complete-field gate drift")
    require(len(manifest["field_descriptors"]) == 1944, "field descriptor count drift")
    zoos: dict[tuple[str, str, int, int], list[CandidateDescriptor]] = {}
    for raw in manifest["field_descriptors"]:
        candidate = _candidate_from_field_descriptor(raw)
        key = (candidate.dataset, candidate.panel, candidate.training_seed, candidate.level)
        zoos.setdefault(key, []).append(candidate)
    expected_zoos = {
        (dataset, panel, seed, level)
        for dataset in DATASET_TARGET_COUNTS for panel in PANELS for seed in SEEDS for level in LEVELS
    }
    require(set(zoos) == expected_zoos and len(zoos) == 24, "candidate zoo coverage drift")
    canonical = {key: canonical_candidates(values) for key, values in zoos.items()}
    targets = target_subjects_from_registry(target_registry_path)
    contexts: list[ContextDescriptor] = []
    for dataset in DATASET_TARGET_COUNTS:
        for target in targets[dataset]:
            for panel in PANELS:
                for seed in SEEDS:
                    for level in LEVELS:
                        identity = context_identity(dataset, target, panel, seed, level)
                        contexts.append(ContextDescriptor(
                            context_id=context_id(identity), dataset=dataset,
                            target_subject_id=target, panel=panel,
                            training_seed=seed, level=level,
                            candidates=canonical[(dataset, panel, seed, level)],
                        ))
    require(len(contexts) == 944 and len({row.context_id for row in contexts}) == 944,
            "target context coverage/identity drift")
    observed = {dataset: sum(row.dataset == dataset for row in contexts) for dataset in DATASET_TARGET_COUNTS}
    require(observed == {"Lee2019_MI": 176, "Cho2017": 160, "PhysionetMI": 608},
            "dataset context arithmetic drift")
    return contexts


def context_registry_rows(contexts: Sequence[ContextDescriptor]) -> list[dict[str, Any]]:
    return [{
        **row.identity(), "context_id": row.context_id,
        "candidate_count": len(row.candidates),
        "candidate_id_sha256": __import__("hashlib").sha256(
            "\n".join(candidate.unit_id for candidate in row.candidates).encode("utf-8")
        ).hexdigest(),
    } for row in contexts]
