"""C86R2 public-metadata-only adult cohort resolution.

The module consumes a frozen, safe catalog projection containing dataset names,
task names, subject counts, licenses, and public catalog identities. It has no
EEG, label, prediction, training, active-acquisition, or execution-lock path.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c86r2_tables"
PROTOCOL = REPORT_DIR / "C86R2_ADULT_PARTICIPANT_AND_COHORT_RESOLUTION_PROTOCOL.json"
EFFECTIVE_MANIFEST_V2 = REPORT_DIR / "C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json"
EFFECTIVE_MANIFEST_V3 = REPORT_DIR / "C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V3.json"

PROTOCOL_COMMIT = "0da4ec39f26ac4bc0d89035e9ad951f452217f05"
PROTOCOL_SHA256 = "2e88e2fef7500b12ca8b3c5b19e6aab06df5a7f388781855b73793a1fe75df92"
IMPLEMENTATION_COMMIT = "IMPLEMENTATION_COMMIT_PENDING"
CREATED_AT_UTC = "2026-07-18T03:24:00Z"

ADULT_THRESHOLD_YEARS = 18
MINIMUM_ADULT_SUBJECTS = 12
MINIMUM_PRIMARY_COHORTS = 2
FINAL_GATE = "C86_ADULT_UNTOUCHED_MULTI_COHORT_ELIGIBILITY_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW"
V3_STATUS = "EFFECTIVE_PROGRAM_RESOLVED_READY_FOR_C86LP_PROTOCOL_REVIEW"

COMMON_SOURCE_TARGET_11 = (
    "FC5", "FC1", "FC2", "FC6", "C3", "Cz", "C4", "CP5", "CP1", "CP2", "CP6",
)
BRANDL_SUBJECT_IDS = tuple(str(index) for index in range(1, 17))
DS007221_HYBRID_SUBJECT_IDS = tuple(f"sub-{index:02d}" for index in range(37, 74))

CATALOG_SNAPSHOTS: tuple[dict[str, Any], ...] = (
    {
        "catalog_id": "MOABB_INSTALLED_IMAGERY_CATALOG_C86R2",
        "source": "installed_MOABB_summary_imagery",
        "identity": "/home/infres/yinwang/anaconda3/envs/c84c-eeg2025-v3-exact/lib/python3.13/site-packages/moabb/datasets/summary_imagery.csv",
        "sha256": "5d7b3abca6f56c83ce90a90d3e0c252783a24f04e237d8d09b11f3862f0ce7e4",
        "bytes": 2331,
        "records": 53,
        "safe_candidate_records": 53,
    },
    {
        "catalog_id": "EEGDASH_PUBLIC_CATALOG_C86R2",
        "source": "EEGDash_chart_data_API",
        "identity": "https://data.eegdash.org/api/eegdash/datasets/chart-data",
        "sha256": "ff12eabbe4832e7303a2289acce0384a69dc478231e5949fad68d4772784760f",
        "bytes": 721760,
        "records": 824,
        "safe_candidate_records": 71,
    },
    {
        "catalog_id": "NEMAR_PUBLIC_CATALOG_C86R2",
        "source": "NEMAR_dataset_API_complete_four_page_snapshot",
        "identity": "https://nemar.org/api/datasets?limit=200&offset=0,200,400,600",
        "sha256": "5b6f4c3cca54a6a466a0549f9e21a48220e6e515881abd0306376c53d0afbfdf",
        "bytes": 1169143,
        "records": 760,
        "safe_candidate_records": 78,
    },
)

DS007221_PUBLIC_IDENTITIES: tuple[tuple[str, str], ...] = (
    ("participants_tsv", "2555ab380eeb7dacde655fc7d4b350cdd5921c28921678506469b6ca51e74f14"),
    ("participants_json", "95d03b1c94e7bfcb4a0459090452ce18ed4d7b71bc16ee0f8944d19955374264"),
    ("dataset_description", "e3760628ff3dc5ee9dd97f6d5846c447428954be748827998b3411e2f42df36f"),
    ("README_recording_summary", "25db7fc2d599433ebe0b4363a99a80e42e1bd1b9e6e7f784e9d8580c592507bc"),
    ("representative_hybrid_channels_tsv", "23927e3967b0c2d357509cc3653b7ddcb07fdd5d7cb98f5052a76e0d63dcbc17"),
    ("repository_tree_API", "e2cdf3b01590e85d1d460d347e1d55fff0ae655ce0d19bb79dab6fe3aec1a0d1"),
)


class C86R2ContractError(RuntimeError):
    """Raised when a public-metadata resolution contract does not replay."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise C86R2ContractError(message)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def subject_registry_sha256(subject_ids: Sequence[str]) -> str:
    payload = "".join(f"{subject_id}\n" for subject_id in subject_ids).encode("ascii")
    return hashlib.sha256(payload).hexdigest()


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("ascii") + b"\n"


def _write_json(path: Path, value: Any) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_canonical_json_bytes(value))
    digest = sha256_file(path)
    path.with_suffix(".sha256").write_text(f"{digest}  {path.name}\n", encoding="ascii")
    return digest


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> str:
    values = [dict(row) for row in rows]
    _require(values, f"refusing empty C86R2 table: {path.name}")
    fields = list(values[0])
    _require(all(list(row) == fields for row in values), f"schema drift in {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(values)
    return sha256_file(path)


def deterministic_adult_subset(
    canonical_subject_ids: Sequence[str],
    age_at_recording: Mapping[str, float | int | None],
) -> tuple[str, ...]:
    """Return all and only canonically ordered subjects with verified adult age."""
    _require(len(canonical_subject_ids) == len(set(canonical_subject_ids)), "duplicate canonical subject ID")
    _require(set(age_at_recording) == set(canonical_subject_ids), "age map does not match canonical subject IDs")
    return tuple(
        subject_id
        for subject_id in canonical_subject_ids
        if age_at_recording[subject_id] is not None
        and float(age_at_recording[subject_id]) >= ADULT_THRESHOLD_YEARS
    )


def adult_interface_eligible(subject_ids: Sequence[str]) -> bool:
    return len(subject_ids) >= MINIMUM_ADULT_SUBJECTS


def _blocker_supersession_rows() -> list[dict[str, Any]]:
    return [
        {
            "milestone": "C86R",
            "historical_gate": "C86_UNTOUCHED_COHORT_AGE_ACCESS_OR_INTERFACE_ELIGIBILITY_RECONCILIATION_REQUIRED",
            "historical_proven_adult_cohorts": 1,
            "historical_decision_correct": 1,
            "retroactively_rewritten": 0,
            "C86R2_role": "PROSPECTIVE_PUBLIC_METADATA_ONLY_RESOLUTION",
            "new_protected_access": 0,
        }
    ]


def _adult_rule_rows() -> list[dict[str, Any]]:
    return [
        {
            "rule_id": "C86_ADULT_PARTICIPANT_INTERFACE_V1",
            "age_basis": "age_at_recording",
            "threshold_operator": ">=",
            "threshold_years": ADULT_THRESHOLD_YEARS,
            "include_rule": "all_canonical_subjects_with_valid_age_at_recording_at_least_18",
            "exclude_rule": "all_minors_and_unknown_or_invalid_age_subjects",
            "minimum_retained_subjects": MINIMUM_ADULT_SUBJECTS,
            "loader_ID_mapping_required": 1,
            "aggregate_mean_or_SD_sufficient": 0,
            "present_day_age_allowed": 0,
            "outcome_or_signal_selection_allowed": 0,
            "subject_list_frozen_before_EEG_access": 1,
        }
    ]


def _yang_audit_rows() -> list[dict[str, Any]]:
    return [
        {
            "native_cohort": "Yang2025_2C",
            "candidate_interface": "Yang2025_2C_ADULT_V1",
            "canonical_subject_count": 51,
            "canonical_loader_subject_IDs": "sub-1_through_sub-51",
            "primary_paper_age_range": "17_to_30",
            "public_participant_age_field": "constant_29",
            "field_interpretation": "INVALID_ANONYMIZED_VALUE_NOT_SUBJECT_AGE_AT_RECORDING",
            "paper_supplement_disclosure": "age_removed_during_anonymization",
            "exact_demographic_to_loader_mapping": 0,
            "verified_adult_count": 0,
            "known_minor_ID_count": 0,
            "unknown_age_ID_count": 51,
            "minimum_subject_rule_pass": 0,
            "decision": "NO_DETERMINISTIC_ADULT_INTERFACE",
            "retained_role": "AGE_MIXED_STRESS_TRACK_ONLY",
            "participants_tsv_sha256": "1b4a41957d6bb6ebfec66df7d740b5c3f05a80384cf0fbd47c8e0a969d1b62f9",
            "primary_article_sha256": "0fa7fca9ea8648fce7a74941f8bc411739aaaab1e0abc74e76d5198fc0325c9b",
            "supplement_sha256": "f78c1ddfc9c3ae952887fea374dbf581cf5ee76e72adff64e6baab5af8aa5af1",
        }
    ]


def _yang_subject_rows() -> list[dict[str, Any]]:
    return [
        {
            "native_cohort": "Yang2025_2C",
            "loader_subject_id": f"sub-{index}",
            "valid_age_at_recording_available": 0,
            "adult_eligible": 0,
            "exclusion_reason": "UNKNOWN_AGE_INVALID_ANONYMIZED_PUBLIC_FIELD",
        }
        for index in range(1, 52)
    ]


def _kumar_audit_rows() -> list[dict[str, Any]]:
    return [
        {
            "native_cohort": "Kumar2024",
            "candidate_interface": "Kumar2024_ADULT_V1",
            "canonical_subject_count": 18,
            "primary_public_evidence": "reported_mean_and_SD_without_all_adult_inclusion_criterion",
            "public_participant_age_field": "constant_2020",
            "field_interpretation": "INVALID_YEAR_LIKE_VALUE_NOT_AGE_AT_RECORDING",
            "exact_verified_adult_count": 0,
            "unknown_age_count": 18,
            "minimum_subject_rule_pass": 0,
            "decision": "KUMAR_AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED",
            "retained_role": "AGE_UNCERTAIN_STRESS_TRACK_ONLY",
            "participants_tsv_sha256": "1596034e5ba33af6df6870ec82bc17cf4cfd5198d91ffb35daf22e7674410bcd",
            "README_sha256": "42b1d38790300854b0746eaad1a9f385e6887211b2ec3c6e1c478e54c52a986f",
        }
    ]


def _kumar_subject_rows() -> list[dict[str, Any]]:
    return [
        {
            "native_cohort": "Kumar2024",
            "loader_subject_id": f"sub-{index}",
            "valid_age_at_recording_available": 0,
            "adult_eligible": 0,
            "exclusion_reason": "UNKNOWN_AGE_INVALID_YEAR_LIKE_PUBLIC_FIELD",
        }
        for index in range(1, 19)
    ]


def _snapshot_rows() -> list[dict[str, Any]]:
    return [
        {
            **row,
            "captured_after_protocol_commit": 1,
            "safe_projection_only": 1,
            "EEG_or_label_payload": 0,
            "catalog_scope_frozen_before_screen": 1,
        }
        for row in CATALOG_SNAPSHOTS
    ]


def _read_safe_catalog(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle, delimiter="\t")
        for raw in reader:
            _require(len(raw) == 9, "safe catalog projection schema drift")
            source_id, catalog_id, name, subjects, tasks, license_name, _, _, source = raw
            rows.append({
                "source_id": source_id,
                "catalog_id": catalog_id,
                "name": name.replace("\n", " "),
                "subjects": int(subjects),
                "tasks": tasks,
                "license": license_name,
                "source": source,
            })
    _require(len(rows) == 79, "deduplicated safe candidate catalog must contain 79 rows")
    _require(len({row["source_id"] for row in rows}) == 79, "duplicate safe candidate source ID")
    return rows


def _early_catalog_decision(row: Mapping[str, Any]) -> tuple[str, str]:
    source_id = str(row["source_id"])
    name = str(row["name"]).lower()
    if source_id in {"nm000329"}:
        return "PASS_REGISTERED_INTERFACE", "Brandl2020"
    if source_id in {"nm000177"}:
        return "FAIL_ADULT_EVIDENCE", "Kumar2024"
    if source_id in {"nm000246", "nm000348"}:
        return "FAIL_ADULT_EVIDENCE", "Yang2025_2C"
    if source_id in {"nm000250"}:
        return "FAIL_HISTORICAL_PROJECT_ACCESS", "Dreyer2023"
    if source_id in {"nm000139", "nm000245", "nm000338", "ds004362"}:
        return "FAIL_PRIOR_PROJECT_TARGET_OR_OUTCOME_ACCESS", source_id
    if int(row["subjects"]) < MINIMUM_ADULT_SUBJECTS:
        return "FAIL_MINIMUM_TARGET_SUBJECT_COUNT", source_id
    if not str(row["license"]).strip():
        return "FAIL_EXPLICIT_LICENSE_NOT_PRESENT_IN_FROZEN_CATALOG", source_id
    nonhealthy = ("patient", "stroke", "impairment", "intracerebral hemorrhage", "knee pain")
    if any(token in name for token in nonhealthy):
        return "FAIL_HEALTHY_PARTICIPANT_RULE", source_id
    task_mismatch = (
        "speech", "visual imagery", "mental imagery", "animal", "alcohol", "finger", "reaching",
        "kinematic", "motor execution", "motor movement", "upper-limb", "upper limb", "lower-limb",
        "lower limb", "character", "mental tasks", "multisensory", "metaphor", "standing and sitting",
    )
    if any(token in name for token in task_mismatch):
        return "FAIL_NATIVE_EXACT_BINARY_LEFT_RIGHT_MI_TASK", source_id
    return "FAIL_CLOSED_CATALOG_DOES_NOT_PROVE_ALL_INTERFACE_CRITERIA", source_id


def _external_eligibility_rows(catalog_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in catalog_rows:
        if item["source_id"] == "ds007221":
            continue
        decision, native = _early_catalog_decision(item)
        rows.append({
            "catalog_source_id": item["source_id"],
            "native_cohort": native,
            "interface_variant": "canonical_default",
            "catalog_subjects": item["subjects"],
            "task_metadata": item["tasks"],
            "license_metadata": item["license"] or "NOT_PRESENT",
            "native_exact_binary_left_right_MI": int(decision == "PASS_REGISTERED_INTERFACE"),
            "healthy_participant_evidence": int(decision == "PASS_REGISTERED_INTERFACE"),
            "adult_subject_evidence": int(decision == "PASS_REGISTERED_INTERFACE"),
            "verified_adult_targets": 16 if native == "Brandl2020" else 0,
            "minimum_trials_per_subject": 200 if native == "Brandl2020" else 0,
            "compatible_channel_count": 11 if native == "Brandl2020" else 0,
            "usable_event_seconds": 3 if native == "Brandl2020" else 0,
            "public_repository_and_license": int(bool(item["license"])),
            "untouched_target_status": int(native == "Brandl2020"),
            "common_interface_status": "PASS" if native == "Brandl2020" else "NOT_REACHED_AFTER_FAIL_CLOSED_GATE",
            "decision": decision,
            "selected_primary": int(decision == "PASS_REGISTERED_INTERFACE"),
            "outcome_fields_present": 0,
        })
    for interface, subjects, events, trial_range, decision in (
        ("OpenNeuro_ds007221_GRAZ_V1", 14, "left_hand|right_hand|feet|rest", "159_to_520", "FAIL_MULTICLASS_NO_CLASS_FILTERING"),
        ("OpenNeuro_ds007221_SSMVEPMI_V1", 22, "left_MI|right_MI|left_AO|right_AO", "352_to_384", "FAIL_MULTICLASS_NON_NATIVE_EXACT_TASK"),
        ("OpenNeuro_ds007221_HYBRID_ADULT_V1", 37, "left_hand|right_hand", "600_to_640", "PASS_ALL_LOCKED_CRITERIA"),
        ("OpenNeuro_ds007221_HYBRIDONLINE_ADULT_V1", 11, "left_hand|right_hand", "959_to_960", "FAIL_MINIMUM_TARGET_SUBJECT_COUNT"),
    ):
        passed = decision == "PASS_ALL_LOCKED_CRITERIA"
        rows.append({
            "catalog_source_id": "ds007221",
            "native_cohort": "OpenNeuro_ds007221",
            "interface_variant": interface,
            "catalog_subjects": subjects,
            "task_metadata": events,
            "license_metadata": "CC0",
            "native_exact_binary_left_right_MI": int(events == "left_hand|right_hand"),
            "healthy_participant_evidence": 1,
            "adult_subject_evidence": 1,
            "verified_adult_targets": subjects,
            "minimum_trials_per_subject": int(trial_range.split("_to_")[0]),
            "compatible_channel_count": 11,
            "usable_event_seconds": 3,
            "public_repository_and_license": 1,
            "untouched_target_status": 1,
            "common_interface_status": "PASS",
            "decision": decision,
            "selected_primary": int(passed),
            "outcome_fields_present": 0,
        })
    _require(len(rows) == 82, "external interface audit must contain 82 rows")
    return rows


def _deduplication_rows() -> list[dict[str, Any]]:
    return [
        {"public_identity": "nm000329", "native_identity": "Brandl2020", "duplicate_role": "NEMAR_MIRROR_OF_REGISTERED_MOABB_COHORT", "independent_cohort_count": 1},
        {"public_identity": "nm000246|nm000348", "native_identity": "Yang2025_2C", "duplicate_role": "TWO_PUBLIC_REPRESENTATIONS_ONE_NATIVE_COHORT", "independent_cohort_count": 1},
        {"public_identity": "nm000177", "native_identity": "Kumar2024", "duplicate_role": "NEMAR_REPRESENTATION_OF_REGISTERED_COHORT", "independent_cohort_count": 1},
        {"public_identity": "nm000250", "native_identity": "Dreyer2023", "duplicate_role": "NEMAR_REPRESENTATION_OF_HISTORICALLY_ACCESSED_COHORT", "independent_cohort_count": 1},
        {"public_identity": "ds007221|on007221", "native_identity": "OpenNeuro_ds007221", "duplicate_role": "OPENNEURO_AND_NEMAR_MIRROR_ONE_NATIVE_COHORT", "independent_cohort_count": 1},
        {"public_identity": "EEGDash_71|NEMAR_78", "native_identity": "79_deduplicated_safe_candidate_datasets", "duplicate_role": "CATALOG_LEVEL_DEDUPLICATION", "independent_cohort_count": 79},
    ]


def _final_cohort_rows() -> list[dict[str, Any]]:
    return [
        {
            "native_cohort": "Brandl2020",
            "interface_variant": "Brandl2020_CANONICAL_ADULT_V1",
            "canonical_adult_subject_ids": "|".join(BRANDL_SUBJECT_IDS),
            "subject_registry_sha256": subject_registry_sha256(BRANDL_SUBJECT_IDS),
            "adult_count": 16,
            "task_events": "left_hand|right_hand",
            "minimum_trials_per_subject": 200,
            "channels_interface": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3",
            "license": "CC-BY-NC-ND-4.0",
            "loader_or_BIDS_identity": "MOABB_BI2015a_via_Brandl2020_loader_identity",
            "historical_access_status": "CERTIFIABLY_UNTOUCHED_TARGET_OUTCOMES",
            "role": "PRIMARY_UNTOUCHED_CONFIRMATION",
            "decision_reason": "PASSES_ALL_LOCKED_ADULT_AND_INTERFACE_RULES",
        },
        {
            "native_cohort": "OpenNeuro_ds007221",
            "interface_variant": "OpenNeuro_ds007221_HYBRID_ADULT_V1",
            "canonical_adult_subject_ids": "|".join(DS007221_HYBRID_SUBJECT_IDS),
            "subject_registry_sha256": subject_registry_sha256(DS007221_HYBRID_SUBJECT_IDS),
            "adult_count": 37,
            "task_events": "left_hand|right_hand",
            "minimum_trials_per_subject": 600,
            "channels_interface": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3",
            "license": "CC0",
            "loader_or_BIDS_identity": "OpenNeuro_ds007221_v1.0.1_NEMAR_on007221_v1.0.0_task-hybrid",
            "historical_access_status": "NO_COMMITTED_PROJECT_REFERENCE_OR_TARGET_ACCESS_FOUND",
            "role": "PRIMARY_UNTOUCHED_CONFIRMATION",
            "decision_reason": "PASSES_ALL_LOCKED_ADULT_AND_INTERFACE_RULES",
        },
        {
            "native_cohort": "Kumar2024",
            "interface_variant": "Kumar2024_whole_cohort",
            "canonical_adult_subject_ids": "",
            "subject_registry_sha256": "NOT_APPLICABLE",
            "adult_count": 0,
            "task_events": "left_hand|right_hand",
            "minimum_trials_per_subject": 0,
            "channels_interface": "NOT_PRIMARY",
            "license": "CC-BY-4.0",
            "loader_or_BIDS_identity": "NEMAR_nm000177",
            "historical_access_status": "CERTIFIABLY_UNTOUCHED_TARGET_OUTCOMES",
            "role": "AGE_UNCERTAIN_STRESS_TRACK_ONLY",
            "decision_reason": "KUMAR_AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED",
        },
        {
            "native_cohort": "Yang2025_2C",
            "interface_variant": "Yang2025_2C_whole_cohort",
            "canonical_adult_subject_ids": "",
            "subject_registry_sha256": "NOT_APPLICABLE",
            "adult_count": 0,
            "task_events": "left_hand|right_hand",
            "minimum_trials_per_subject": 0,
            "channels_interface": "NOT_PRIMARY",
            "license": "CC-BY-4.0",
            "loader_or_BIDS_identity": "NEMAR_nm000348",
            "historical_access_status": "CERTIFIABLY_UNTOUCHED_TARGET_OUTCOMES",
            "role": "AGE_MIXED_STRESS_TRACK_ONLY",
            "decision_reason": "NO_DETERMINISTIC_ADULT_INTERFACE",
        },
        {
            "native_cohort": "Dreyer2023",
            "interface_variant": "Dreyer2023_whole_cohort",
            "canonical_adult_subject_ids": "",
            "subject_registry_sha256": "NOT_APPLICABLE",
            "adult_count": 0,
            "task_events": "left_hand|right_hand",
            "minimum_trials_per_subject": 0,
            "channels_interface": "NOT_PRIMARY",
            "license": "CC-BY-4.0",
            "loader_or_BIDS_identity": "NEMAR_nm000250",
            "historical_access_status": "HISTORICAL_PROJECT_ACCESS_NOT_CERTIFIABLY_ABSENT",
            "role": "DEVELOPMENT_ONLY_HISTORICAL_ACCESS",
            "decision_reason": "PRESERVED_C86R_ROLE",
        },
    ]


def _all_passing_rows(external_rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    expected = {"Brandl2020_CANONICAL_ADULT_V1", "OpenNeuro_ds007221_HYBRID_ADULT_V1"}
    observed = {
        "Brandl2020_CANONICAL_ADULT_V1" if row["native_cohort"] == "Brandl2020" else row["interface_variant"]
        for row in external_rows
        if row["selected_primary"] == 1
    }
    _require(observed == expected, "all passing interface inclusion drift")
    return [
        {
            "interface_variant": interface,
            "passes_all_locked_criteria": 1,
            "included_primary_confirmation": 1,
            "post_screen_cap_applied": 0,
            "excluded_after_pass": 0,
            "at_least_two_rule_count": len(expected),
            "at_least_two_rule": "PASS",
        }
        for interface in sorted(expected)
    ]


def _common_interface_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset, role, native_channels in (
        ("Lee2019_MI", "LEGACY_SOURCE", 62),
        ("Cho2017", "LEGACY_SOURCE", 64),
        ("PhysionetMI", "LEGACY_SOURCE", 64),
        ("Brandl2020_CANONICAL_ADULT_V1", "UNTOUCHED_TARGET", 63),
        ("OpenNeuro_ds007221_HYBRID_ADULT_V1", "UNTOUCHED_TARGET", 69),
    ):
        rows.append({
            "interface_id": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3",
            "dataset_or_interface": dataset,
            "role": role,
            "native_EEG_channels": native_channels,
            "common_channel_count": len(COMMON_SOURCE_TARGET_11),
            "common_channels": "|".join(COMMON_SOURCE_TARGET_11),
            "channel_identity_public_metadata_verified": 1,
            "resample_Hz": 160,
            "epoch_half_open_seconds": "0|3",
            "bandpass_Hz": "4|38",
            "events": "left_hand|right_hand",
            "target_in_own_training": 0,
            "same_candidate_zoo_across_primary_cohorts": 1,
            "metadata_interface_status": "PASS",
        })
    return rows


def _cluster_rows() -> list[dict[str, Any]]:
    return [
        {
            "interface_variant": interface,
            "target_subject_clusters": targets,
            "minimum_clusters_required": MINIMUM_ADULT_SUBJECTS,
            "cluster_count_pass": 1,
            "exact_sign_configurations": str(2**targets),
            "registered_maxT_draws": 65536,
            "plus_one_min_p": "1/65537",
            "favorable_75pct_targets": math.ceil(0.75 * targets),
            "LOTO_fits": targets,
            "scientific_unit": "target_subject",
        }
        for interface, targets in (
            ("Brandl2020_CANONICAL_ADULT_V1", 16),
            ("OpenNeuro_ds007221_HYBRID_ADULT_V1", 37),
        )
    ]


def _resource_rows() -> list[dict[str, Any]]:
    return [
        {"resource": "primary_adult_target_subjects", "estimate": 53, "unit": "subjects", "envelope": 53, "basis": "16+37", "locked_for_real_execution": 0},
        {"resource": "unique_candidate_units", "estimate": 648, "unit": "models", "envelope": 648, "basis": "2_panels*2_seeds*2_levels*81", "locked_for_real_execution": 0},
        {"resource": "training_phases", "estimate": 24, "unit": "phases", "envelope": 24, "basis": "2*2*2*3_regimes", "locked_for_real_execution": 0},
        {"resource": "target_contexts", "estimate": 424, "unit": "contexts", "envelope": 424, "basis": "53*8", "locked_for_real_execution": 0},
        {"resource": "candidate_context_slices", "estimate": 34344, "unit": "slices", "envelope": 34344, "basis": "424*81", "locked_for_real_execution": 0},
        {"resource": "unit_cohort_artifacts", "estimate": 1296, "unit": "artifacts", "envelope": 1296, "basis": "648*2", "locked_for_real_execution": 0},
        {"resource": "raw_download_storage", "estimate": 192, "unit": "GiB", "envelope": 256, "basis": "public_metadata_sizes_plus_staging_margin", "locked_for_real_execution": 0},
        {"resource": "target_prediction_field_storage", "estimate": 48, "unit": "GiB", "envelope": 64, "basis": "34344_slices_scaled_from_C84", "locked_for_real_execution": 0},
        {"resource": "total_scratch_storage", "estimate": 640, "unit": "GiB", "envelope": 768, "basis": "raw_models_outputs_staging_and_replay", "locked_for_real_execution": 0},
        {"resource": "GPU_time", "estimate": 24, "unit": "GPU_hours_scaled", "envelope": 96, "basis": "future_canary_required", "locked_for_real_execution": 0},
        {"resource": "RAM", "estimate": 128, "unit": "GiB", "envelope": 128, "basis": "future_field_generation", "locked_for_real_execution": 0},
    ]


def _license_rows() -> list[dict[str, Any]]:
    return [
        {
            "interface_variant": "Brandl2020_CANONICAL_ADULT_V1",
            "license": "CC-BY-NC-ND-4.0",
            "public_identity": "MOABB_loader_metadata_and_primary_dataset_paper",
            "internal_analysis": "TERMS_REPLAY_REQUIRED_BEFORE_FUTURE_ACCESS",
            "derived_artifact_policy": "RESTRICTED_INSTITUTIONAL_REVIEW_REQUIRED",
            "attribution_required": 1,
            "legal_advice_claimed": 0,
        },
        {
            "interface_variant": "OpenNeuro_ds007221_HYBRID_ADULT_V1",
            "license": "CC0",
            "public_identity": "OpenNeuro_ds007221_v1.0.1_NEMAR_on007221_v1.0.0",
            "internal_analysis": "TERMS_REPLAY_REQUIRED_BEFORE_FUTURE_ACCESS",
            "derived_artifact_policy": "CC0_WITH_PROVENANCE_AND_REPOSITORY_TERMS_REPLAY",
            "attribution_required": 0,
            "legal_advice_claimed": 0,
        },
    ]


def _build_manifest(table_hashes: Mapping[str, str]) -> dict[str, Any]:
    primary = [row for row in _final_cohort_rows() if row["role"] == "PRIMARY_UNTOUCHED_CONFIRMATION"]
    _require(len(primary) >= MINIMUM_PRIMARY_COHORTS, "adult multi-cohort rule not satisfied")
    return {
        "schema_version": "c86_active_testing_effective_program_manifest_v3",
        "created_at_utc": CREATED_AT_UTC,
        "status": V3_STATUS,
        "protocol_precedence_low_to_high": [
            "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json",
            "C86P_ACTIVE_ESTIMATOR_OPERATIONALIZATION_PROTOCOL.json",
            "C86P_UNTOUCHED_COHORT_VARIANT_ELIGIBILITY_CORRECTION_PROTOCOL.json",
            "C86P_HISTORICAL_ACCESS_ELIGIBILITY_CORRECTION_PROTOCOL.json",
            "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL.json",
            "C86R_ELIGIBILITY_BASELINE_AND_DEVELOPMENT_VIEW_REPAIR_PROTOCOL.json",
            "C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json",
            PROTOCOL.name,
        ],
        "identities": {
            "C86R2_protocol_commit": PROTOCOL_COMMIT,
            "C86R2_protocol_sha256": PROTOCOL_SHA256,
            "C86R2_implementation_commit": IMPLEMENTATION_COMMIT,
            "C86_effective_program_manifest_V2_sha256": "c19ca4090b64dec3cc98971e44cbf09f7a1367e4a754535529d972b691c7ca65",
            "C86_synthetic_operationalization_V2_sha256": "3413f8f13d7e11657823120b386acd7659f291d927617aa97d141b3981147862",
            "ds007221_public_metadata": dict(DS007221_PUBLIC_IDENTITIES),
        },
        "authoritative_program": {
            "target_population": "adult_age_at_recording_at_least_18",
            "primary_untouched_confirmation_interfaces": [row["interface_variant"] for row in primary],
            "primary_untouched_confirmation_count": len(primary),
            "minimum_primary_cohorts": MINIMUM_PRIMARY_COHORTS,
            "at_least_two_rule": "PASS",
            "all_passing_interfaces_included": True,
            "common_field_interface": "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V3",
            "same_candidate_zoo_across_primary_cohorts": True,
            "target_in_own_training": False,
            "primary_adult_target_subjects": 53,
            "target_contexts": 424,
            "candidate_context_slices": 34344,
            "active_method_registry": "c86r_tables/active_method_registry_v2.csv",
            "C86L_development_view_contract": "c86r_tables/C86L_development_view_contract_v2.csv",
            "synthetic_contract": "C86P_SYNTHETIC_CALIBRATION_OPERATIONALIZATION_PROTOCOL_V2.json",
            "taxonomy": "UNCHANGED_C86_A_TO_E_AND_C86_L1_TO_L4",
            "stage_sequence": ["C86LP", "C86L", "C86DP", "C86D", "C86C_F", "C86H"],
        },
        "table_hashes": dict(sorted(table_hashes.items())),
        "final_gate": FINAL_GATE,
        "downstream_contract": {
            "stale_manifest_V2_may_drive_C86LP": False,
            "manifest_V3_required_for_C86LP_review": True,
            "C86LP_authorized": False,
            "C86L_authorized": False,
            "C86D_authorized": False,
            "C86C_F_authorized": False,
            "C86H_authorized": False,
            "real_data_execution_lock_created": False,
            "authorization_record_created": False,
        },
        "protected_counters": {
            "new_EEG_downloaded_or_opened": 0,
            "new_target_labels_opened": 0,
            "candidate_outputs_opened": 0,
            "active_acquisition_executed": 0,
            "new_candidate_training_or_forward": 0,
            "registered_C86_synthetic_results": 0,
            "GPU": 0,
        },
    }


def table_rows(catalog_rows: Sequence[Mapping[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    external = _external_eligibility_rows(catalog_rows)
    return {
        "c86r_blocker_supersession_ledger.csv": _blocker_supersession_rows(),
        "adult_participant_interface_rule.csv": _adult_rule_rows(),
        "yang_2c_adult_interface_audit.csv": _yang_audit_rows(),
        "yang_2c_adult_subject_registry.csv": _yang_subject_rows(),
        "kumar_adult_evidence_audit.csv": _kumar_audit_rows(),
        "kumar_adult_subject_registry.csv": _kumar_subject_rows(),
        "external_registry_snapshot.csv": _snapshot_rows(),
        "external_adult_cohort_eligibility_registry.csv": external,
        "native_cohort_deduplication_ledger.csv": _deduplication_rows(),
        "all_passing_cohort_inclusion_truth_table.csv": _all_passing_rows(external),
        "final_adult_untouched_cohort_registry_v3.csv": _final_cohort_rows(),
        "common_field_interface_v3.csv": _common_interface_rows(),
        "adult_target_cluster_resolution.csv": _cluster_rows(),
        "field_resource_envelope_v3.csv": _resource_rows(),
        "license_and_artifact_policy_v3.csv": _license_rows(),
    }


def load_effective_manifest_v3(path: Path = EFFECTIVE_MANIFEST_V3) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    expected = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    _require(sha256_file(path) == expected, "effective program V3 sidecar drift")
    _require(value["schema_version"] == "c86_active_testing_effective_program_manifest_v3", "V3 schema drift")
    _require(value["status"] == V3_STATUS, "V3 status drift")
    _require(value["authoritative_program"]["at_least_two_rule"] == "PASS", "adult cohort count drift")
    _require(value["downstream_contract"]["manifest_V3_required_for_C86LP_review"] is True, "V3 guard drift")
    return value


def require_effective_manifest_v3_for_c86lp(bound_paths: Sequence[str | Path]) -> None:
    normalized = {Path(path).resolve() for path in bound_paths}
    _require(EFFECTIVE_MANIFEST_V3.resolve() in normalized, "stale C86R V2 cohort list cannot drive C86LP")
    load_effective_manifest_v3()


def write_readiness_artifacts(catalog_candidate_tsv: Path, table_dir: Path = TABLE_DIR) -> dict[str, Any]:
    _require(sha256_file(PROTOCOL) == PROTOCOL_SHA256, "C86R2 protocol identity drift")
    _require(sha256_file(EFFECTIVE_MANIFEST_V2) == "c19ca4090b64dec3cc98971e44cbf09f7a1367e4a754535529d972b691c7ca65", "V2 identity drift")
    catalog_rows = _read_safe_catalog(catalog_candidate_tsv)
    rows = table_rows(catalog_rows)
    hashes = {name: _write_csv(table_dir / name, values) for name, values in rows.items()}
    manifest_sha = _write_json(EFFECTIVE_MANIFEST_V3, _build_manifest(hashes))
    return {
        "tables": len(hashes),
        "table_hashes": hashes,
        "effective_manifest_V3_sha256": manifest_sha,
        "primary_adult_interfaces": [
            "Brandl2020_CANONICAL_ADULT_V1",
            "OpenNeuro_ds007221_HYBRID_ADULT_V1",
        ],
        "primary_adult_target_subjects": 53,
        "final_gate": FINAL_GATE,
        "protected_access": 0,
    }


__all__ = [
    "ADULT_THRESHOLD_YEARS",
    "BRANDL_SUBJECT_IDS",
    "C86R2ContractError",
    "DS007221_HYBRID_SUBJECT_IDS",
    "EFFECTIVE_MANIFEST_V3",
    "FINAL_GATE",
    "adult_interface_eligible",
    "deterministic_adult_subset",
    "load_effective_manifest_v3",
    "require_effective_manifest_v3_for_c86lp",
    "subject_registry_sha256",
    "table_rows",
    "write_readiness_artifacts",
]


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Materialize public-metadata-only C86R2 readiness artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)
    write = subparsers.add_parser("write-readiness")
    write.add_argument("--catalog-candidate-tsv", type=Path, required=True)
    subparsers.add_parser("validate-effective-manifest")
    args = parser.parse_args(argv)
    if args.command == "write-readiness":
        result = write_readiness_artifacts(args.catalog_candidate_tsv)
    else:
        result = load_effective_manifest_v3()
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
