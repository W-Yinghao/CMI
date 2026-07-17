"""Prospective untouched-cohort and fair-budget eligibility tests."""
from __future__ import annotations

import csv
from pathlib import Path

import pytest

from oaci.theory.c86_active_program import (
    C86PContractError,
    ELIGIBLE_DATASETS,
    TOTAL_QUERY_GRID,
    _SOURCE_INFO,
    _read_catalog,
    canonical_trial_split,
    validate_total_query_grid,
)


ROOT = Path(__file__).resolve().parents[2]
TABLES = ROOT / "oaci/reports/c86p_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_all_and_only_locked_eligible_cohorts_are_selected() -> None:
    rows = _rows("untouched_dataset_eligibility_registry.csv")
    assert len(rows) == 53
    selected = tuple(sorted(row["confirmation_interface_id"] for row in rows
                            if row["selected_for_C86H_engineering"] == "1"))
    assert selected == ELIGIBLE_DATASETS
    for row in rows:
        assert row["published_performance_used"] == "0"
        if row["selected_for_C86H_engineering"] == "1":
            assert row["native_exact_left_right_MI"] == "1"
            assert row["healthy_adults_only"] == "1"
            assert int(row["eligible_interface_subjects"]) >= 12
            assert int(row["documented_min_total_trials_per_subject"]) >= 80
            assert row["prior_project_target_science"] == "0"
            assert row["prior_project_target_or_label_access_certifiably_absent"] == "1"
            assert row["canonical_nondeduplicated_cohort"] == "1"
            assert row["loader_determinism_audit"] == "SOURCE_IDENTITY_BOUND_NO_DATA_OPEN"

    yang = next(row for row in rows if row["confirmation_interface_id"] == "Yang2025_2C")
    assert yang["dataset"] == "Yang2025"
    assert yang["interface_variant"] == "paradigm_type=2C;subjects=1..51"
    assert yang["eligible_interface_subjects"] == "51"
    assert yang["documented_min_total_trials_per_subject"] == "600"
    assert yang["event_labels"] == "left_hand|right_hand"
    assert yang["catalog_classes"] == "3"
    expected_loader_hashes = {
        "Brandl2020": "313c912ac42ea89d0a176a7baa972330067b6ca559cdef81acde074757416edd",
        "Kumar2024": "b06d387564032456eb44c2bc24787d09fab4d5be3ae5d755b0178e190c07989d",
        "Yang2025_2C": "b6a1ea3ee0e415a43d6a4dc04c0d798160b9e740efedd08cf4fd82b35c35220f",
    }
    selected_rows = {
        row["confirmation_interface_id"]: row for row in rows
        if row["selected_for_C86H_engineering"] == "1"
    }
    assert {key: row["loader_source_sha256"] for key, row in selected_rows.items()} == expected_loader_hashes
    assert all(row["metadata_url"].startswith("https://") for row in selected_rows.values())


def test_every_catalog_binary_row_has_loader_source_task_audit() -> None:
    binary_rows = {row["Dataset"] for row in _read_catalog() if row["#Classes"] == "2"}
    assert binary_rows <= set(_SOURCE_INFO)


def test_near_eligible_cohorts_fail_the_predeclared_rule() -> None:
    by_name = {row["dataset"]: row for row in _rows("untouched_dataset_eligibility_registry.csv")}
    assert by_name["BNCI2014_004"]["decision_reason"] == "SUBJECT_COUNT_BELOW_12"
    assert by_name["Shin2017A"]["decision_reason"] == "GUARANTEED_TRIAL_SUPPORT_BELOW_80"
    assert by_name["Forenzo2023"]["decision_reason"] == "GUARANTEED_TRIAL_SUPPORT_BELOW_80"
    assert by_name["GuttmannFlury2025_MI"]["decision_reason"] == "GUARANTEED_TRIAL_SUPPORT_BELOW_80"
    assert by_name["HefmiIch2025"]["decision_reason"] == "POPULATION_NOT_HEALTHY_ONLY"
    assert by_name["Dreyer2023A"]["decision_reason"] == "HISTORICAL_TARGET_ACCESS_NOT_CERTIFIABLY_ABSENT"
    assert by_name["Dreyer2023"]["decision_reason"] == "HISTORICAL_TARGET_ACCESS_NOT_CERTIFIABLY_ABSENT"


def test_dataset_selection_rule_includes_every_passing_cohort() -> None:
    rows = _rows("dataset_selection_rule_truth_table.csv")
    primary = {row["confirmation_interface_id"] for row in rows
               if row["confirmation_role"] == "PRIMARY_UNTOUCHED_COHORT"}
    assert primary == set(ELIGIBLE_DATASETS)
    assert all(row["all_eligible_included"] == "1" for row in rows)
    assert all(row["performance_used"] == row["hand_selected"] == "0" for row in rows)


def test_historical_access_ledger_fails_closed_for_dreyer() -> None:
    rows = {row["dataset"]: row for row in _rows("historical_access_ledger.csv")}
    assert rows["Dreyer2023"]["untouched_access_certified"] == "0"
    assert rows["Dreyer2023"]["committed_access_evidence"] == "HISTORICAL_LOCAL_PREPROCESSED_STORE_VERIFIED"
    assert rows["Dreyer2023"]["evidence_path"] == "archive/lpc-cmi-failed/notes/preprocessing_decision.md"
    for dataset in ("Brandl2020", "Kumar2024", "Yang2025"):
        assert rows[dataset]["untouched_access_certified"] == "1"


def test_label_blind_split_is_deterministic_disjoint_and_balanced_by_count() -> None:
    trial_ids = [f"trial-{index:03d}" for index in range(81)]
    pool_a, eval_a = canonical_trial_split("Shadow", "S1", trial_ids)
    pool_b, eval_b = canonical_trial_split("Shadow", "S1", list(reversed(trial_ids)))
    assert (pool_a, eval_a) == (pool_b, eval_b)
    assert len(pool_a) == 40 and len(eval_a) == 41
    assert set(pool_a).isdisjoint(eval_a)
    assert set(pool_a) | set(eval_a) == set(trial_ids)
    with pytest.raises(C86PContractError, match="fewer than 80"):
        canonical_trial_split("Shadow", "S1", trial_ids[:79])


def test_total_query_grid_is_feasible_and_not_labels_per_class() -> None:
    validate_total_query_grid(40, TOTAL_QUERY_GRID)
    with pytest.raises(C86PContractError, match="grid drift"):
        validate_total_query_grid(40, (4, 8, 16, "FULL"))
    with pytest.raises(C86PContractError, match="cannot support"):
        validate_total_query_grid(39, TOTAL_QUERY_GRID)
    rows = _rows("total_query_budget_contract.csv")
    assert [row["budget"] for row in rows] == ["4", "8", "16", "32", "FULL"]
    assert all(row["unit"] == "TOTAL_QUERIES_PER_TARGET" for row in rows)
    assert all(row["labels_per_class"] == "0" for row in rows)
    assert rows[-1]["primary_comparative"] == "0"


def test_primary_passive_comparator_does_not_observe_class_allocation() -> None:
    rows = {row["comparator"]: row for row in _rows("passive_comparator_fairness_audit.csv")}
    assert rows["P0"]["primary"] == "1"
    assert rows["P0"]["class_known_before_query"] == "0"
    assert rows["P1"]["primary"] == "0"
    assert rows["P1"]["class_known_before_query"] == "1"
