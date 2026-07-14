"""Generate the additive C84 fixed-panel level-1 protocol family.

This module is metadata-only. It reads committed protocol tables and never
imports a dataset loader, array framework, training package, or real artifact.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import c84r_v2_protocols as historical
from .c84r_montage_repair import (
    CLASS_MAPPING_VERSION,
    EPOCH_RULE,
    INTERFACE_ID,
    MONTAGE_SHA256,
    SAMPLE_RATE_HZ,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84l1p_tables"
CREATED_AT_UTC = "2026-07-14T15:30:00Z"

C84FL_HEAD = "6d6030f17dc2cdf8c8b180a9376632e238d42e75"
C84FL_MD_SHA256 = "f3ff4dd24dac12c4886e868d1b9d786bd20df2b479fd37c9f523090400265b17"
C84FL_JSON_SHA256 = "b5f61a79aeccba0d054ee66e5e80236eb30dae2025935c61a6db0764db1cf4a5"
C84C_HEAD = "f7bbd27579308e01ed5c0388cb728cc7417978ac"
C84C_MANIFEST_SHA256 = "530471ef370d5fa13a88e7e53cf1add558b8444b66675496187aa192b0606f2b"
EXTERNAL_V2_SHA256 = "522e6fe8372f8c73741ed146a27068076db8c3d7087f4c4a36760fe0328b7c2f"
FIELD_V4_SHA256 = "eff7ebbc2e4f91830a3df1d679adfcae6eae2ab8a1e91c64ed28df7fce96aa12"
SCIENCE_V2_SHA256 = "dc33b22527352bd42989c26f6771b4a49dc1443d458962587ca3d70ad76dd631"

LEVEL0_ID = "C84_LEVEL0_FULL_SOURCE_PANEL_V1"
LEVEL1_ID = "C84_LEVEL1_FIXED_PANEL_LEFT_HAND_CELL_DELETION_V1"
LEVEL1_UNIT_SALT = "C84_FIXED_ZOO_LEFT_RIGHT_20CH_LEVEL1_SUPPORT_DELETION_V1"
DELETED_CLASS = "left_hand"
DELETED_CLASS_ID = 0
MIN_CELL_SUPPORT = 8
SUCCESS_GATE = "C84_LEVEL1_FIXED_PANEL_SUPPORT_DELETION_LOCKED_READY_FOR_ENGINEERING_CANARY_AUTHORIZATION"
FAILURE_GATE = "C84_LEVEL1_SUPPORT_IDENTITY_INPUT_IMPLEMENTATION_OR_PROTOCOL_RECONCILIATION_REQUIRED"
CANARY_TARGETS = {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106}
DELETED_SUBJECTS = {
    ("Lee2019_MI", "A"): 31,
    ("Lee2019_MI", "B"): 16,
    ("Cho2017", "A"): 17,
    ("Cho2017", "B"): 37,
    ("PhysionetMI", "A"): 103,
    ("PhysionetMI", "B"): 109,
}


def canonical_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii")


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_json(path: str | Path, value: Any) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(canonical_bytes(value) + b"\n")
    digest = sha256_file(target)
    target.with_suffix(".sha256").write_text(f"{digest}  {target.name}\n", encoding="ascii")
    return digest


def write_csv(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    fields: Sequence[str] | None = None,
) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise RuntimeError(f"refusing empty C84L1P table: {path}")
    fieldnames = list(fields or materialized[0])
    if any(set(row) != set(fieldnames) for row in materialized):
        raise RuntimeError(f"C84L1P CSV schema drift: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _checked_json(name: str, expected: str) -> dict[str, Any]:
    path = REPORT_DIR / name
    observed = sha256_file(path)
    if observed != expected:
        raise RuntimeError(f"C84L1P predecessor hash drift: {name}: {observed}")
    return read_json(path)


def source_contract_rows() -> list[dict[str, str]]:
    rows = read_csv(REPORT_DIR / "c84fl_tables/source_view_contract.csv")
    if len(rows) != 6:
        raise RuntimeError("C84L1P requires six dataset/panel source contracts")
    by_key = {(row["dataset"], row["panel"]): row for row in rows}
    if set(by_key) != set(DELETED_SUBJECTS):
        raise RuntimeError("C84L1P deletion registry differs from source-view universe")
    for key, subject in DELETED_SUBJECTS.items():
        order = tuple(int(value) for value in by_key[key]["source_training_subjects"].split("|"))
        if len(order) != 12 or order[0] != subject:
            raise RuntimeError(f"C84L1P fixed deletion subject is not first in locked order: {key}")
    return rows


def level_intervention_rows() -> list[dict[str, Any]]:
    source = {(row["dataset"], row["panel"]): row for row in source_contract_rows()}
    rows = []
    for dataset in historical.DATASET_ORDER:
        for panel in historical.PANELS:
            contract = source[(dataset, panel)]
            order = tuple(int(value) for value in contract["source_training_subjects"].split("|"))
            rows.append({
                "dataset": dataset,
                "panel": panel,
                "level": 1,
                "level_intervention_id": LEVEL1_ID,
                "source_training_order": "|".join(map(str, order)),
                "deleted_source_subject": order[0],
                "deleted_subject_is_first_locked": 1,
                "deleted_class": DELETED_CLASS,
                "deleted_class_id": DELETED_CLASS_ID,
                "minimum_cell_support": MIN_CELL_SUPPORT,
                "target_independent": 1,
                "alternative_cell_allowed": 0,
                "outcome_dependent_choice": 0,
            })
    return rows


def level1_unit_identity(old: Mapping[str, Any], registry_sha256: str) -> dict[str, Any]:
    if int(old["level"]) != 1:
        raise ValueError("new C84L1 identity is defined only for level 1")
    deleted_subject = DELETED_SUBJECTS[(str(old["dataset"]), str(old["source_panel"]))]
    return {
        "identity_salt": LEVEL1_UNIT_SALT,
        "milestone": "C84",
        "dataset": old["dataset"],
        "source_panel": old["source_panel"],
        "training_seed": int(old["training_seed"]),
        "level": 1,
        "regime": old["regime"],
        "epoch": int(old["epoch"]),
        "trajectory_order": int(old["trajectory_order"]),
        "interface_id": INTERFACE_ID,
        "montage_sha256": MONTAGE_SHA256,
        "epoch_rule": EPOCH_RULE,
        "sample_rate_hz": SAMPLE_RATE_HZ,
        "class_mapping_version": CLASS_MAPPING_VERSION,
        "level_intervention_id": LEVEL1_ID,
        "deleted_source_subject": int(deleted_subject),
        "deleted_class": DELETED_CLASS,
        "level_intervention_registry_sha256": registry_sha256,
    }


def level1_unit_id(old: Mapping[str, Any], registry_sha256: str) -> str:
    return "c84l1_" + sha256_bytes(canonical_bytes(level1_unit_identity(old, registry_sha256)))[:32]


def identity_rows(registry_sha256: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    historical_units = historical.candidate_units()
    if len(historical_units) != 1944:
        raise RuntimeError("historical C84 unit registry is not 1,944 rows")
    level0: list[dict[str, Any]] = []
    level1: list[dict[str, Any]] = []
    supersession: list[dict[str, Any]] = []
    operative: list[dict[str, Any]] = []
    for old in historical_units:
        common = {
            "dataset": old["dataset"],
            "panel": old["source_panel"],
            "training_seed": old["training_seed"],
            "level": old["level"],
            "regime": old["regime"],
            "epoch": old["epoch"],
            "trajectory_order": old["trajectory_order"],
        }
        if int(old["level"]) == 0:
            row = {
                **common,
                "historical_unit_id": old["unit_id"],
                "operative_unit_id": old["unit_id"],
                "identity_unchanged": 1,
                "level_intervention_id": LEVEL0_ID,
                "C84C_reusable": int(bool(old["canary_subset"])),
            }
            level0.append(row)
            operative.append({
                **common,
                "unit_id": old["unit_id"],
                "historical_planned_unit_id": old["unit_id"],
                "identity_status": "UNCHANGED_LEVEL0",
                "level_intervention_id": LEVEL0_ID,
                "deleted_source_subject": "NONE",
                "deleted_class": "NONE",
                "level_intervention_registry_sha256": registry_sha256,
                "C84C_reusable": int(bool(old["canary_subset"])),
                "C84L1C_canary": 0,
            })
            continue
        identity = level1_unit_identity(old, registry_sha256)
        new_id = level1_unit_id(old, registry_sha256)
        level1.append({
            **common,
            "unit_id": new_id,
            "historical_planned_unit_id": old["unit_id"],
            "level_intervention_id": LEVEL1_ID,
            "deleted_source_subject": identity["deleted_source_subject"],
            "deleted_class": DELETED_CLASS,
            "level_intervention_registry_sha256": registry_sha256,
            "identity_json_sha256": sha256_bytes(canonical_bytes(identity)),
            "C84L1C_canary": int(old["source_panel"] == "A" and old["training_seed"] == 5),
        })
        supersession.append({
            **common,
            "historical_planned_level1_unit_id": old["unit_id"],
            "operative_level1_unit_id": new_id,
            "historical_identity_preserved": 1,
            "historical_identity_operative": 0,
            "new_identity_operative": 1,
            "reason": "historical_ID_predated_scientific_level1_intervention",
        })
        operative.append({
            **common,
            "unit_id": new_id,
            "historical_planned_unit_id": old["unit_id"],
            "identity_status": "SUPERSEDED_LEVEL1",
            "level_intervention_id": LEVEL1_ID,
            "deleted_source_subject": identity["deleted_source_subject"],
            "deleted_class": DELETED_CLASS,
            "level_intervention_registry_sha256": registry_sha256,
            "C84C_reusable": 0,
            "C84L1C_canary": int(old["source_panel"] == "A" and old["training_seed"] == 5),
        })
    if len(level0) != 972 or len(level1) != 972 or len(supersession) != 972 or len(operative) != 1944:
        raise RuntimeError("C84L1P level identity arithmetic drift")
    ids = [row["unit_id"] for row in operative]
    historical_level0 = {row["unit_id"] for row in historical_units if row["level"] == 0}
    historical_level1 = {row["unit_id"] for row in historical_units if row["level"] == 1}
    new_level1 = {row["unit_id"] for row in level1}
    if len(set(ids)) != 1944 or historical_level1 & new_level1:
        raise RuntimeError("C84L1P operative unit IDs are not unique or reused blocked level-1 IDs")
    if {row["operative_unit_id"] for row in level0} != historical_level0:
        raise RuntimeError("C84L1P migrated a level-0 unit ID")
    return level0, level1, supersession, operative


def support_contract_rows() -> list[dict[str, Any]]:
    conditions = (
        ("exact_source_subject_set", "12 locked subjects", "BLOCK"),
        ("registered_deleted_subject_present", "true", "BLOCK"),
        ("registered_deleted_cell_pre_count", ">=8", "BLOCK"),
        ("registered_deleted_cell_post_count", "0", "BLOCK"),
        ("deleted_subject_right_hand_post_count", ">=8", "BLOCK"),
        ("all_other_observed_cell_counts", ">=8", "BLOCK"),
        ("pre_deletion_observed_cells", "24", "BLOCK"),
        ("post_deletion_observed_cells", "23", "BLOCK"),
        ("absent_post_deletion_cells", "exactly_deleted_cell", "BLOCK"),
        ("source_training_trial_ids", "unique_and_otherwise_unchanged", "BLOCK"),
        ("source_audit_trial_ids", "unchanged", "BLOCK"),
        ("target_trial_ids", "unchanged", "BLOCK"),
        ("alternative_cell_selection", "forbidden", "BLOCK"),
    )
    return [{
        "level_intervention_id": LEVEL1_ID,
        "check_order": order,
        "condition": condition,
        "required_value": value,
        "failure_action": action,
        "minimum_cell_support": MIN_CELL_SUPPORT,
        "real_data_access_in_C84L1P": 0,
    } for order, (condition, value, action) in enumerate(conditions, start=1)]


def fail_closed_case_rows() -> list[dict[str, Any]]:
    cases = (
        ("registered_first_subject_deleted", "PASS"),
        ("numeric_min_subject_substituted", "FAIL"),
        ("right_hand_substituted", "FAIL"),
        ("target_dependent_deleted_subject", "FAIL"),
        ("outcome_selected_cell", "FAIL"),
        ("deleted_cell_absent_before_deletion", "FAIL"),
        ("deleted_cell_below_support_minimum", "FAIL"),
        ("second_cell_absent", "FAIL"),
        ("remaining_observed_cell_below_8", "FAIL"),
        ("source_audit_row_deleted", "FAIL"),
        ("target_row_deleted", "FAIL"),
        ("level0_plan_or_hash_drift", "FAIL"),
        ("different_model_initialization_across_levels", "FAIL"),
        ("historical_planned_level1_ID_used", "FAIL"),
        ("new_level1_ID_missing_intervention_digest", "FAIL"),
        ("panel_A_three_dataset_canary_243_units_9_phases", "PASS"),
        ("target_y_access", "FAIL"),
        ("scientific_metric_emission", "FAIL"),
    )
    return [{
        "case_id": f"L1F{index:02d}",
        "fixture": fixture,
        "expected": expected,
        "alternative_cell_selected": 0,
        "real_data_access": 0,
    } for index, (fixture, expected) in enumerate(cases, start=1)]


def paired_rng_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in historical.DATASET_ORDER:
        for panel in historical.PANELS:
            for seed in historical.SEEDS:
                rows.append({
                    "dataset": dataset,
                    "panel": panel,
                    "training_seed": seed,
                    "levels": "0|1",
                    "model_init_seed_rule": "derive_seed(training_seed,dataset,model_init)",
                    "same_model_init_across_levels": 1,
                    "plan_seed": seed,
                    "plans_materialized_per_level_population_signature": 1,
                    "architecture_optimizer_epochs_cadence_identical": 1,
                    "outcome_dependent_seed_choice": 0,
                })
    return rows


def canary_scope_rows() -> list[dict[str, Any]]:
    return [{
        "dataset": dataset,
        "panel": "A",
        "training_seed": 5,
        "level": 1,
        "deleted_source_subject": DELETED_SUBJECTS[(dataset, "A")],
        "deleted_class": DELETED_CLASS,
        "target_subject": CANARY_TARGETS[dataset],
        "candidate_units": 81,
        "training_phases": 3,
        "engineering_only": 1,
        "target_scientific_metrics": 0,
    } for dataset in historical.DATASET_ORDER]


def view_contract_rows() -> list[dict[str, Any]]:
    values = (
        ("source_training_pre_deletion", 1, 1, 0, "support_validation_only"),
        ("source_training_post_deletion", 1, 1, 0, "training"),
        ("source_audit", 1, 1, 0, "instrumentation_only_no_metric"),
        ("target_unlabeled", 1, 0, 1, "instrumentation_without_y"),
        ("target_construction", 0, 0, 0, "not_provisioned"),
        ("target_evaluation", 0, 0, 0, "not_provisioned"),
        ("same_label_oracle", 0, 0, 0, "unreachable"),
    )
    return [{
        "view": view,
        "provisioned": provisioned,
        "source_y_allowed": source_y,
        "target_X_allowed": target_x,
        "target_y_allowed": 0,
        "purpose": purpose,
        "deletion_allowed": int(view == "source_training_post_deletion"),
        "scientific_metric_allowed": 0,
    } for view, provisioned, source_y, target_x, purpose in values]


def artifact_schema_rows() -> list[dict[str, Any]]:
    fields = (
        ("sidecar", "unit_id", "scalar"),
        ("sidecar", "level_intervention_id", "scalar"),
        ("sidecar", "level_intervention_registry_sha256", "sha256"),
        ("sidecar", "deleted_source_subject", "scalar"),
        ("sidecar", "deleted_class", "scalar"),
        ("sidecar", "pre_deletion_trial_count", "integer"),
        ("sidecar", "post_deletion_trial_count", "integer"),
        ("sidecar", "deleted_trial_id_sha256", "sha256"),
        ("sidecar", "population_signature", "sha256"),
        ("sidecar", "support_hash", "sha256"),
        ("sidecar", "model_init_hash", "sha256"),
        ("sidecar", "plan_hashes", "mapping"),
        ("source_audit", "source_class_label", "array"),
        ("source_audit", "source_domain_id", "array"),
        ("source_audit", "source_trial_id", "array"),
        ("target_unlabeled", "target_subject_id", "scalar"),
        ("target_unlabeled", "target_trial_id", "array"),
        ("target_unlabeled", "logits_probabilities_z", "arrays"),
    )
    return [{
        "artifact": artifact,
        "field_order": order,
        "field": field,
        "type_contract": kind,
        "required": 1,
        "target_label_or_derived": 0,
    } for order, (artifact, field, kind) in enumerate(fields)]


def risk_rows() -> list[dict[str, Any]]:
    risks = (
        "historical_C84FL_blocker_rewritten",
        "level0_ID_or_C84C_artifact_changed",
        "historical_level1_ID_reused",
        "deletion_subject_not_first_locked_subject",
        "deleted_class_not_left_hand",
        "alternative_cell_selected_at_runtime",
        "support_minimum_bypassed",
        "more_than_one_support_cell_absent",
        "source_audit_or_target_row_deleted",
        "target_specific_retraining_reintroduced",
        "outcome_dependent_deletion_choice",
        "level0_level1_model_init_unpaired",
        "plan_population_signature_unbound",
        "target_y_access",
        "target_scientific_metric_emission",
        "C84F_or_C84S_lock_created",
        "real_EEG_access_in_C84L1P",
        "raw_EEG_weights_or_cache_in_Git",
        "payload_over_50MiB_in_Git",
    )
    return [{
        "risk": risk,
        "status": "CLOSED_BY_PROTOCOL_AND_SYNTHETIC_CONTROL",
        "blocking": 0,
        "control": "fail_closed_and_lock_bound",
        "real_data_access_in_C84L1P": 0,
    } for risk in risks]


def build_repair_protocol(
    registry_sha256: str,
    level1_digest: str,
    canary_digest: str,
    operative_registry_sha256: str,
) -> dict[str, Any]:
    return {
        "schema_version": "c84l1_fixed_panel_support_deletion_protocol_v1",
        "milestone": "C84L1P",
        "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_PROTOCOL_IMPLEMENTATION_AND_CANARY_LOCK_PENDING_NOT_AUTHORIZED",
        "supersession": {
            "C84FL_HEAD": C84FL_HEAD,
            "C84FL_markdown_sha256": C84FL_MD_SHA256,
            "C84FL_json_sha256": C84FL_JSON_SHA256,
            "C84FL_gate": "C84F_CANARY_REUSE_DATA_VIEW_IMPLEMENTATION_RESOURCE_OR_MANIFEST_RECONCILIATION_REQUIRED",
            "historical_objects_rewritten": False,
            "historical_level1_IDs_preserved_but_nonoperative": 972,
        },
        "timing": {
            "designed_before_level1_real_data_access": True,
            "level1_real_EEG_access_before_protocol": 0,
            "level1_label_reads_before_protocol": 0,
            "level1_training_forward_GPU_before_protocol": 0,
            "target_scientific_outcomes_before_protocol": 0,
        },
        "levels": {
            "0": {
                "id": LEVEL0_ID,
                "definition": "exact_full_locked_12_subject_source_training_panel",
                "deletion": None,
                "unit_IDs_changed": False,
                "C84C_reusable_units": 243,
            },
            "1": {
                "id": LEVEL1_ID,
                "definition": "delete_registered_source_subject_x_left_hand_before_support_and_plan_materialization",
                "target_independent": True,
                "source_only": True,
                "deleted_class": DELETED_CLASS,
                "deleted_class_id": DELETED_CLASS_ID,
                "minimum_cell_support": MIN_CELL_SUPPORT,
                "post_deletion_observed_cells": 23,
                "alternative_cell_selection": False,
                "exact_C78_replication": False,
            },
        },
        "deletion_registry": {
            "path": "oaci/reports/c84l1p_tables/level_intervention_registry.csv",
            "sha256": registry_sha256,
            "cells": {f"{dataset}/{panel}": {"subject": subject, "class": DELETED_CLASS}
                      for (dataset, panel), subject in DELETED_SUBJECTS.items()},
        },
        "support_contract": {
            "minimum_rows": MIN_CELL_SUPPORT,
            "pre_observed_cells": 24,
            "post_observed_cells": 23,
            "absent_cell": "registered_deleted_cell_only",
            "alternative_cell_on_failure": False,
            "source_audit_or_target_deletion": False,
        },
        "paired_training": {
            "same_model_initialization_across_levels": True,
            "same_architecture_optimizer_hyperparameters_epochs_cadence": True,
            "same_base_training_seed": True,
            "level_specific_population_signature": True,
            "level_specific_plan_materialization": True,
            "seeds": [5, 6],
        },
        "candidate_identity": {
            "level0_unchanged": 972,
            "historical_level1_superseded": 972,
            "new_level1": 972,
            "operative_total": 1944,
            "new_level1_salt": LEVEL1_UNIT_SALT,
            "level1_candidate_ID_digest": level1_digest,
            "level1_canary_243_ID_digest": canary_digest,
            "operative_registry_sha256": operative_registry_sha256,
        },
        "canary": {
            "datasets": list(historical.DATASET_ORDER),
            "panel": "A",
            "training_seed": 5,
            "level": 1,
            "target_subjects": CANARY_TARGETS,
            "candidate_units": 243,
            "training_phases": 9,
            "engineering_only": True,
            "scientific_metrics": False,
        },
        "authorization": {
            "C84L1C_authorized": False,
            "C84F_authorized": False,
            "C84S_authorized": False,
            "fresh_direct_authorization_required": True,
            "shortest_future_statement": "授权 C84L1C",
        },
    }


def build_external_v3(prior: Mapping[str, Any], repair_sha: str, registry_sha: str, operative_sha: str) -> dict[str, Any]:
    return {
        **prior,
        "schema_version": "c84_multidataset_external_validity_protocol_v3",
        "milestone": "C84L1P",
        "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_LEVEL_DEFINITIONS_FUTURE_STAGES_SEPARATELY_AUTHORIZED",
        "supersession": {
            **prior["supersession"],
            "C84L1_repair_protocol_sha256": repair_sha,
            "C84FL_HEAD": C84FL_HEAD,
            "historical_external_V2_sha256": EXTERNAL_V2_SHA256,
            "historical_objects_rewritten": False,
        },
        "candidate_field": {
            **prior["candidate_field"],
            "level_definitions": {
                "0": LEVEL0_ID,
                "1": LEVEL1_ID,
            },
            "level1_intervention_registry_sha256": registry_sha,
            "operative_complete_unit_registry_sha256": operative_sha,
            "level0_unit_IDs_unchanged": 972,
            "level1_unit_IDs_superseded": 972,
        },
        "authorization": {
            **prior["authorization"],
            "C84L1C_authorized": False,
            "prior_C84C_authorization_reusable": False,
        },
    }


def build_canary_v1(repair_sha: str, external_sha: str, registry_sha: str, canary_digest: str) -> dict[str, Any]:
    return {
        "schema_version": "c84_level1_canary_protocol_v1",
        "milestone": "C84L1C",
        "created_at_utc": CREATED_AT_UTC,
        "status": "LOCKED_PROTOCOL_IMPLEMENTATION_AND_EXECUTION_LOCK_REQUIRED_NOT_AUTHORIZED",
        "repair_protocol_sha256": repair_sha,
        "parent_external_protocol_v3_sha256": external_sha,
        "scope": {
            "datasets": list(historical.DATASET_ORDER),
            "source_panel": "A",
            "training_seed": 5,
            "level": 1,
            "target_subjects": CANARY_TARGETS,
            "units_per_dataset": 81,
            "total_units": 243,
            "training_phases": 9,
            "engineering_only": True,
        },
        "intervention": {
            "id": LEVEL1_ID,
            "registry_sha256": registry_sha,
            "deleted_cells": {dataset: {"subject": DELETED_SUBJECTS[(dataset, "A")], "class": DELETED_CLASS}
                              for dataset in historical.DATASET_ORDER},
            "minimum_support": MIN_CELL_SUPPORT,
            "alternative_cell_allowed": False,
        },
        "candidate_identity": {
            "salt": LEVEL1_UNIT_SALT,
            "canary_count": 243,
            "canary_unit_ID_digest": canary_digest,
        },
        "engineering_checks": [
            "runtime_lock_to_bytes_replay",
            "environment_and_loader_identity",
            "exact_pre_post_trial_IDs_and_counts",
            "exactly_one_absent_support_cell",
            "population_and_support_hashes",
            "paired_level0_level1_model_init_hash",
            "level_specific_plan_hashes",
            "checkpoint_optimizer_sidecar_243",
            "source_audit_and_target_unlabeled_243",
            "persisted_artifact_replay",
            "target_y_access_zero",
            "target_scientific_metrics_zero",
        ],
        "forbidden_outputs": [
            "target_accuracy", "target_calibration", "target_regret", "selector_scores",
            "Q1", "Q2", "label_budget_frontier", "level0_level1_target_performance_comparison",
        ],
        "authorization": {
            "fresh_direct_PI_authorization_required": True,
            "shortest_statement": "授权 C84L1C",
            "record_path": "oaci/reports/C84L1C_PI_AUTHORIZATION_RECORD.json",
            "C84F": False,
            "C84S": False,
        },
    }


def build_field_v5(prior: Mapping[str, Any], repair_sha: str, external_sha: str, operative_sha: str) -> dict[str, Any]:
    return {
        **prior,
        "schema_version": "c84_field_generation_protocol_v5",
        "status": "LOCKED_PROTOCOL_ONLY_C84L1C_REVIEW_REQUIRED_NO_C84F_LOCK_NOT_AUTHORIZED",
        "parent_external_protocol_v3_sha256": external_sha,
        "historical_field_protocol_v4_sha256": FIELD_V4_SHA256,
        "C84L1_repair_protocol_sha256": repair_sha,
        "candidate_identity": {
            "level0_salt": historical.UNIT_ID_SALT,
            "level1_salt": LEVEL1_UNIT_SALT,
            "interface_id": INTERFACE_ID,
            "montage_sha256": MONTAGE_SHA256,
            "operative_complete_unit_registry_sha256": operative_sha,
        },
        "levels": {
            "0": {"id": LEVEL0_ID, "units": 972, "intervention": "none"},
            "1": {"id": LEVEL1_ID, "units": 972, "intervention": "fixed_panel_left_hand_cell_deletion"},
        },
        "canary_reuse": {
            **prior["canary_reuse"],
            "accepted_level0_units": 243,
            "future_level1_units_after_successful_C84L1C": 243,
            "future_total_reusable_units": 486,
            "future_remaining_units": 1458,
            "future_remaining_training_phases": 54,
            "canary_target_slices_are_subset_witnesses_only": True,
        },
        "scope_specific_execution_lock_created_in_C84L1P": False,
        "fresh_direct_PI_authorization_after_C84L1C_review": True,
    }


def build_science_v3(prior: Mapping[str, Any], repair_sha: str, external_sha: str) -> dict[str, Any]:
    return {
        **prior,
        "schema_version": "c84_scientific_analysis_protocol_v3",
        "status": "LOCKED_PROTOCOL_ONLY_NO_SCIENTIFIC_EXECUTION_LOCK_NOT_AUTHORIZED",
        "parent_external_protocol_v3_sha256": external_sha,
        "historical_scientific_protocol_v2_sha256": SCIENCE_V2_SHA256,
        "C84L1_repair_protocol_sha256": repair_sha,
        "level_reporting": {
            "report_each_level_before_averaging": True,
            "primary_registered_average_retained": True,
            "qualitative_level_disagreement_tag": "LEVEL_HETEROGENEITY",
            "level_disagreement_may_not_be_hidden_by_averaging": True,
            "level0_id": LEVEL0_ID,
            "level1_id": LEVEL1_ID,
        },
        "within_dataset_aggregation": "report_levels_separately_then_apply_registered_average_across_levels_and_panel_x_seed_cells_within_target",
        "scope_specific_execution_lock_created_in_C84L1P": False,
    }


def timing_audit_markdown(repair_sha: str, protocol_hashes: Mapping[str, str]) -> str:
    return f"""# C84L1 Protocol Timing Audit

## Accepted base

- C84FL HEAD: `{C84FL_HEAD}`
- C84FL Markdown SHA-256: `{C84FL_MD_SHA256}`
- C84FL JSON SHA-256: `{C84FL_JSON_SHA256}`
- C84C accepted HEAD: `{C84C_HEAD}`
- C84C complete manifest SHA-256: `{C84C_MANIFEST_SHA256}`

## Additive protocol identity

- C84L1 repair protocol SHA-256: `{repair_sha}`
- External protocol V3 SHA-256: `{protocol_hashes['external']}`
- Level-1 canary protocol V1 SHA-256: `{protocol_hashes['canary']}`
- Field protocol V5 SHA-256: `{protocol_hashes['field']}`
- Scientific protocol V3 SHA-256: `{protocol_hashes['science']}`

## Prospective boundary

The fixed deletion cells were committed from the already locked source-training
order before any C84 level-1 real-data access or implementation execution.

| Protected event before protocol | Count |
|---|---:|
| New real EEG array access | 0 |
| New source or target label read | 0 |
| Level-1 training phases | 0 |
| Level-1 candidate units | 0 |
| Level-1 forward/instrumentation | 0 |
| Target scientific metrics | 0 |
| GPU jobs | 0 |

The accepted C84C level-0 artifacts are historical engineering evidence. They
were not modified, reopened, or recomputed by C84L1P protocol generation.

## Stop boundary

C84L1P may implement and lock C84L1C only. It does not execute C84L1C and does
not create C84F or C84S execution locks.
"""


def generate() -> dict[str, Any]:
    _checked_json("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json", EXTERNAL_V2_SHA256)
    _checked_json("C84_FIELD_GENERATION_PROTOCOL_V4.json", FIELD_V4_SHA256)
    _checked_json("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json", SCIENCE_V2_SHA256)
    if sha256_file(REPORT_DIR / "C84FL_OVERALL_REPORT.md") != C84FL_MD_SHA256:
        raise RuntimeError("C84FL Markdown identity drift")
    if sha256_file(REPORT_DIR / "C84FL_OVERALL_REPORT.json") != C84FL_JSON_SHA256:
        raise RuntimeError("C84FL JSON identity drift")

    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    intervention = level_intervention_rows()
    intervention_path = TABLE_DIR / "level_intervention_registry.csv"
    write_csv(intervention_path, intervention)
    registry_sha = sha256_file(intervention_path)

    level0, level1, supersession, operative = identity_rows(registry_sha)
    write_csv(TABLE_DIR / "level0_identity_replay.csv", level0)
    write_csv(TABLE_DIR / "level1_candidate_id_registry.csv", level1)
    write_csv(TABLE_DIR / "historical_level1_unit_id_supersession.csv", supersession)
    operative_path = TABLE_DIR / "operative_complete_unit_registry_v2.csv"
    write_csv(operative_path, operative)
    operative_sha = sha256_file(operative_path)
    level1_digest = sha256_bytes(canonical_bytes(sorted(row["unit_id"] for row in level1)))
    canary_digest = sha256_bytes(canonical_bytes(sorted(
        row["unit_id"] for row in level1 if int(row["C84L1C_canary"]) == 1
    )))
    (TABLE_DIR / "level1_candidate_id_digest.txt").write_text(
        f"{level1_digest}  canonical_sorted_level1_candidate_IDs\n", encoding="ascii",
    )

    write_csv(TABLE_DIR / "level_support_contract.csv", support_contract_rows())
    write_csv(TABLE_DIR / "level1_fail_closed_support_cases.csv", fail_closed_case_rows())
    write_csv(TABLE_DIR / "paired_rng_plan_contract.csv", paired_rng_rows())
    write_csv(TABLE_DIR / "level1_canary_scope.csv", canary_scope_rows())
    write_csv(TABLE_DIR / "level1_canary_view_contract.csv", view_contract_rows())
    write_csv(TABLE_DIR / "level1_artifact_schema.csv", artifact_schema_rows())
    write_csv(TABLE_DIR / "risk_register.csv", risk_rows())
    write_csv(TABLE_DIR / "failure_reason_ledger.csv", [{
        "failure_id": "NONE",
        "stage": "C84L1P_protocol_generation",
        "blocking": 0,
        "reason": "protocol_and_identity_generation_passed",
        "real_data_access": 0,
        "scientific_outcome_access": 0,
        "repair_required": 0,
    }])

    repair = build_repair_protocol(registry_sha, level1_digest, canary_digest, operative_sha)
    repair_sha = write_json(REPORT_DIR / "C84L1_FIXED_PANEL_SUPPORT_DELETION_PROTOCOL.json", repair)
    prior_external = _checked_json("C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json", EXTERNAL_V2_SHA256)
    external = build_external_v3(prior_external, repair_sha, registry_sha, operative_sha)
    external_sha = write_json(REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V3.json", external)
    canary_sha = write_json(
        REPORT_DIR / "C84_LEVEL1_CANARY_PROTOCOL_V1.json",
        build_canary_v1(repair_sha, external_sha, registry_sha, canary_digest),
    )
    prior_field = _checked_json("C84_FIELD_GENERATION_PROTOCOL_V4.json", FIELD_V4_SHA256)
    field_sha = write_json(
        REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V5.json",
        build_field_v5(prior_field, repair_sha, external_sha, operative_sha),
    )
    prior_science = _checked_json("C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json", SCIENCE_V2_SHA256)
    science_sha = write_json(
        REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V3.json",
        build_science_v3(prior_science, repair_sha, external_sha),
    )
    hashes = {"external": external_sha, "canary": canary_sha, "field": field_sha, "science": science_sha}
    (REPORT_DIR / "C84L1_PROTOCOL_TIMING_AUDIT.md").write_text(
        timing_audit_markdown(repair_sha, hashes), encoding="utf-8",
    )
    return {
        "repair_protocol_sha256": repair_sha,
        "external_protocol_v3_sha256": external_sha,
        "canary_protocol_v1_sha256": canary_sha,
        "field_protocol_v5_sha256": field_sha,
        "science_protocol_v3_sha256": science_sha,
        "level_intervention_registry_sha256": registry_sha,
        "operative_registry_sha256": operative_sha,
        "level1_candidate_ID_digest": level1_digest,
        "level1_canary_243_ID_digest": canary_digest,
        "level0_IDs_unchanged": len(level0),
        "level1_IDs_superseded": len(level1),
        "operative_units": len(operative),
        "real_data_access": 0,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
