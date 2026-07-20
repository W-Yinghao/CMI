"""C86R development-view and untouched-field metadata feasibility tests."""
from __future__ import annotations

import ast
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c86r_tables"
MODULE = ROOT / "oaci/theory/c86r_program_repair.py"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_c86l_acquisition_and_held_development_views_are_disjoint() -> None:
    rows = {row["view"]: row for row in _rows("C86L_development_view_contract_v2.csv")}
    assert rows["acquisition_pool"]["identity"] == "C84_immutable_construction_trial_IDs_and_matching_frozen_probabilities"
    assert rows["held_development_outcome"]["identity"] == "accepted_C85U_held_evaluation_candidate_utility_field"
    assert all(row["direct_C84_evaluation_labels"] == "0" for row in rows.values())
    truth = _rows("acquisition_evaluation_nonoverlap_truth_table.csv")
    assert len(truth) == 3
    assert all(row["label_rows_opened_in_C86R"] == "0" for row in truth)
    assert truth[0]["identity_overlap"] == truth[1]["identity_overlap"] == "0"
    assert truth[2]["status"] == "FORBIDDEN_AS_C86L_ACQUISITION_POOL"


def test_development_budget_availability_is_complete_and_fail_closed() -> None:
    rows = _rows("C84_development_budget_availability.csv")
    assert len(rows) == 118 * 5
    assert {row["budget"] for row in rows} == {"4", "8", "16", "32", "FULL"}
    assert all(row["replacement_sampling"] == "0" for row in rows)
    assert all(row["budget_substitution"] == "0" for row in rows)
    assert all(row["target_deleted"] == "0" for row in rows)
    by_dataset = {}
    for row in rows:
        by_dataset.setdefault(row["dataset"], []).append(row)
    assert len({row["target_subject_id"] for row in by_dataset["Lee2019_MI"]}) == 22
    assert len({row["target_subject_id"] for row in by_dataset["Cho2017"]}) == 20
    assert len({row["target_subject_id"] for row in by_dataset["PhysionetMI"]}) == 76
    phys_b32 = [row for row in by_dataset["PhysionetMI"] if row["budget"] == "32"]
    assert len(phys_b32) == 76
    assert all(row["available"] == "0" and row["disposition"] == "INPUT_UNAVAILABLE" for row in phys_b32)
    assert all(row["available"] == "1" for dataset in ("Lee2019_MI", "Cho2017") for row in by_dataset[dataset])


def test_linear_moments_and_nonlinear_plugin_claims_are_separated() -> None:
    queried = {row["object"]: row for row in _rows("queried_information_object_contract.csv")}
    assert queried["clipped_binary_NLL"]["linear_additive_moment"] == "1"
    for name in ("balanced_accuracy", "ECE", "candidate_midranks", "historical_composite_utility", "selected_action"):
        assert queried[name]["plugin_nonlinear"] == "1"
        assert queried[name]["unbiased_LURE_claim"].startswith("NONE")
    boundary = {row["estimator_object"]: row for row in _rows("estimator_claim_boundary_v2.csv")}
    assert boundary["LURE_linear_moment"]["claim"] == "UNBIASED_UNDER_LOCKED_SAMPLING_AND_POSITIVITY"
    assert boundary["midrank_composite_plugin"]["claim"] == "NO_UNBIASEDNESS_CLAIM"


def test_common_interface_is_metadata_feasible_but_requires_new_training() -> None:
    rows = [row for row in _rows("common_montage_metadata_audit.csv")
            if row["interface_id"] == "C86_C84SOURCE_TARGET_11CH_160HZ_0_3S_V1"]
    assert {row["dataset"] for row in rows} == {
        "Lee2019_MI", "Cho2017", "PhysionetMI", "Brandl2020", "Kumar2024", "Yang2025_2C",
    }
    assert all(row["common_channel_count"] == "11" for row in rows)
    assert all(row["common_channels"] == "FC5|FC1|FC2|FC6|C3|Cz|C4|CP5|CP1|CP2|CP6" for row in rows)
    assert all(row["resample_Hz"] == "160" for row in rows)
    assert all(row["epoch_half_open_seconds"] == "0|3" for row in rows)
    assert all(row["reuses_C84_20ch_model_bytes"] == "0" for row in rows)
    assert all(row["requires_new_candidate_training"] == "1" for row in rows)
    assert all(row["metadata_feasibility"] == "PASS" for row in rows)


def test_subject_allocation_and_inference_resolution_are_metadata_explicit() -> None:
    allocation = _rows("confirmation_subject_allocation_options.csv")
    preferred = [row for row in allocation if row["option_id"] == "O1_LEGACY_SOURCE_ALL_UNTOUCHED_TARGETS"]
    assert len(preferred) == 1
    assert preferred[0]["target_in_own_training"] == "0"
    assert preferred[0]["same_zoo_across_cohorts"] == "1"
    assert preferred[0]["target_counts"] == "Brandl=16|Kumar=18|Yang2C=51"
    inference = {row["cohort"]: row for row in _rows("inference_resolution_audit.csv")}
    assert {key: int(value["target_subject_clusters"]) for key, value in inference.items()} == {
        "Brandl2020": 16, "Kumar2024": 18, "Yang2025_2C": 51,
    }
    assert all(row["registered_maxT_draws"] == "65536" for row in inference.values())
    assert all(row["plus_one_Monte_Carlo_min_p_exact"] == "1/65537" for row in inference.values())
    assert inference["Brandl2020"]["adult_primary_eligible"] == "1"
    assert inference["Kumar2024"]["adult_primary_eligible"] == "0"
    assert inference["Yang2025_2C"]["adult_primary_eligible"] == "0"


def test_candidate_field_resource_arithmetic_and_licenses_are_locked_metadata() -> None:
    rows = {row["resource"]: row for row in _rows("candidate_field_resource_envelope.csv")}
    assert rows["unique_candidate_units"]["estimate"] == "648"
    assert rows["training_phases"]["estimate"] == "24"
    assert rows["three_cohort_target_contexts"]["estimate"] == "680"
    assert rows["three_cohort_candidate_context_slices"]["estimate"] == "55080"
    assert rows["GPU_time"]["envelope"] == "96"
    assert all(row["locked_for_execution"] == "0" for row in rows.values())
    licenses = {row["cohort"]: row for row in _rows("license_and_derived_artifact_policy.csv")}
    assert licenses["Brandl2020"]["license"] == "CC-BY-NC-ND-4.0"
    assert licenses["Brandl2020"]["derived_artifact_redistribution"] == "RESTRICTED_INSTITUTIONAL_REVIEW_REQUIRED"
    assert licenses["Kumar2024"]["license"] == "CC-BY-4.0"
    assert licenses["Yang2025_2C"]["license"] == "CC-BY-4.0"


def test_repair_module_has_no_real_data_or_execution_dependencies() -> None:
    tree = ast.parse(MODULE.read_text(encoding="utf-8"))
    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    forbidden_prefixes = (
        "moabb", "mne", "torch", "oaci.multidataset.c84s_stage_a",
        "oaci.multidataset.c84s_evaluation", "oaci.theory.c85u",
    )
    assert not any(name.startswith(forbidden_prefixes) for name in imports)
    assert not list(REPORTS.glob("C86*EXECUTION_LOCK*.json"))
