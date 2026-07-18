"""C86R2 exhaustive safe-catalog and common-field replay tests."""
from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
MODULE = ROOT / "oaci/theory/c86r2_adult_cohort_resolution.py"
TABLES = ROOT / "oaci/reports/c86r2_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_catalog_snapshots_are_complete_and_frozen_after_protocol() -> None:
    rows = {row["catalog_id"]: row for row in _rows("external_registry_snapshot.csv")}
    assert rows["MOABB_INSTALLED_IMAGERY_CATALOG_C86R2"]["records"] == "53"
    assert rows["EEGDASH_PUBLIC_CATALOG_C86R2"]["records"] == "824"
    assert rows["NEMAR_PUBLIC_CATALOG_C86R2"]["records"] == "760"
    assert rows["EEGDASH_PUBLIC_CATALOG_C86R2"]["safe_candidate_records"] == "71"
    assert rows["NEMAR_PUBLIC_CATALOG_C86R2"]["safe_candidate_records"] == "78"
    assert all(row["captured_after_protocol_commit"] == "1" for row in rows.values())
    assert all(row["EEG_or_label_payload"] == "0" for row in rows.values())


def test_external_registry_dispositions_every_safe_candidate_and_ds007221_interfaces() -> None:
    rows = _rows("external_adult_cohort_eligibility_registry.csv")
    assert len(rows) == 92
    assert len({row["catalog_source_id"] for row in rows}) == 89
    assert all(row["decision"] for row in rows)
    assert all(row["outcome_fields_present"] == "0" for row in rows)
    assert not any("performance" in field.lower() for field in rows[0])
    selected = [row for row in rows if row["selected_primary"] == "1"]
    assert {(row["native_cohort"], row["interface_variant"]) for row in selected} == {
        ("Brandl2020", "canonical_default"),
        ("OpenNeuro_ds007221", "OpenNeuro_ds007221_HYBRID_ADULT_V1"),
    }
    hybrid = next(row for row in rows if row["interface_variant"] == "OpenNeuro_ds007221_HYBRID_ADULT_V1")
    assert hybrid["catalog_subjects"] == "37"
    assert hybrid["verified_adult_targets"] == "37"
    assert hybrid["minimum_trials_per_subject"] == "600"
    assert hybrid["native_exact_binary_left_right_MI"] == "1"
    assert hybrid["common_interface_status"] == "PASS"


def test_ds007221_nonpassing_native_interfaces_are_not_class_filtered() -> None:
    rows = [row for row in _rows("external_adult_cohort_eligibility_registry.csv") if row["catalog_source_id"] == "ds007221"]
    decisions = {row["interface_variant"]: row["decision"] for row in rows}
    assert decisions["OpenNeuro_ds007221_GRAZ_V1"] == "FAIL_MULTICLASS_NO_CLASS_FILTERING"
    assert decisions["OpenNeuro_ds007221_SSMVEPMI_V1"] == "FAIL_MULTICLASS_NON_NATIVE_EXACT_TASK"
    assert decisions["OpenNeuro_ds007221_HYBRIDONLINE_ADULT_V1"] == "FAIL_MINIMUM_TARGET_SUBJECT_COUNT"


def test_native_catalog_deduplication_does_not_double_count_mirrors() -> None:
    rows = _rows("native_cohort_deduplication_ledger.csv")
    openneuro = next(row for row in rows if row["native_identity"] == "OpenNeuro_ds007221")
    yang = next(row for row in rows if row["native_identity"] == "Yang2025_2C")
    assert openneuro["public_identity"] == "ds007221|on007221"
    assert openneuro["independent_cohort_count"] == "1"
    assert yang["public_identity"] == "nm000246|nm000348"
    assert yang["independent_cohort_count"] == "1"


def test_common_interface_and_cluster_arithmetic_derive_from_final_registry() -> None:
    interface = _rows("common_field_interface_v3.csv")
    assert len(interface) == 5
    assert {row["common_channel_count"] for row in interface} == {"11"}
    assert {row["metadata_interface_status"] for row in interface} == {"PASS"}
    assert {row["same_candidate_zoo_across_primary_cohorts"] for row in interface} == {"1"}
    clusters = {row["interface_variant"]: row for row in _rows("adult_target_cluster_resolution.csv")}
    assert clusters["Brandl2020_CANONICAL_ADULT_V1"]["target_subject_clusters"] == "16"
    assert clusters["Brandl2020_CANONICAL_ADULT_V1"]["exact_sign_configurations"] == str(2**16)
    assert clusters["OpenNeuro_ds007221_HYBRID_ADULT_V1"]["target_subject_clusters"] == "37"
    assert clusters["OpenNeuro_ds007221_HYBRID_ADULT_V1"]["exact_sign_configurations"] == str(2**37)
    resources = {row["resource"]: row for row in _rows("field_resource_envelope_v3.csv")}
    assert resources["primary_adult_target_subjects"]["estimate"] == "53"
    assert resources["target_contexts"]["estimate"] == str(53 * 8)
    assert resources["candidate_context_slices"]["estimate"] == str(53 * 8 * 81)
    assert resources["unit_cohort_artifacts"]["estimate"] == str(648 * 2)


def test_module_has_no_protected_runtime_import_or_real_execution_entrypoint() -> None:
    source = MODULE.read_text(encoding="utf-8")
    tree = ast.parse(source)
    imports = {
        alias.name.split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    }
    imports |= {
        (node.module or "").split(".")[0]
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom)
    }
    assert imports.isdisjoint({"mne", "moabb", "torch", "sklearn"})
    assert "run-locked" not in source
    assert "C86L_EXECUTION_LOCK" not in source
    assert "C86H_EXECUTION_LOCK" not in source
    assert "AuthorizationRecord" not in source
    assert "performance" not in source.lower()

