"""C86R2 participant-rule and adult-interface contract tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from oaci.theory.c86r2_adult_cohort_resolution import (
    ADULT_THRESHOLD_YEARS,
    BRANDL_SUBJECT_IDS,
    C86R2ContractError,
    DS007221_HYBRID_SUBJECT_IDS,
    EFFECTIVE_MANIFEST_V3,
    FINAL_GATE,
    adult_interface_eligible,
    deterministic_adult_subset,
    load_effective_manifest_v3,
    require_effective_manifest_v3_for_c86lp,
    subject_registry_sha256,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c86r2_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_adult_rule_is_age_at_recording_at_least_18() -> None:
    assert ADULT_THRESHOLD_YEARS == 18
    canonical = ("s1", "s2", "s3", "s4")
    ages = {"s1": 17, "s2": 18, "s3": 29, "s4": None}
    assert deterministic_adult_subset(canonical, ages) == ("s2", "s3")
    assert not adult_interface_eligible(("s2", "s3"))
    assert adult_interface_eligible(tuple(f"s{i}" for i in range(12)))
    with pytest.raises(C86R2ContractError, match="does not match"):
        deterministic_adult_subset(canonical, {"s1": 18})


def test_subject_interfaces_include_all_verified_adults_and_have_frozen_digests() -> None:
    assert BRANDL_SUBJECT_IDS == tuple(str(index) for index in range(1, 17))
    assert DS007221_HYBRID_SUBJECT_IDS == tuple(f"sub-{index:02d}" for index in range(37, 74))
    assert subject_registry_sha256(BRANDL_SUBJECT_IDS) == "cd5cb9fb5ac3c4f4007e8b41d117da21622439cd05c1728f3e82f90e4f869dad"
    assert subject_registry_sha256(DS007221_HYBRID_SUBJECT_IDS) == "26520d5b53cfdb76915b552f4e3ac8d918deb584a430a6d03c4ac804fefa203c"


def test_yang_and_kumar_remain_fail_closed() -> None:
    yang = _rows("yang_2c_adult_interface_audit.csv")[0]
    kumar = _rows("kumar_adult_evidence_audit.csv")[0]
    assert yang["primary_paper_age_range"] == "17_to_30"
    assert yang["field_interpretation"].startswith("INVALID_ANONYMIZED")
    assert yang["verified_adult_count"] == "0"
    assert yang["decision"] == "NO_DETERMINISTIC_ADULT_INTERFACE"
    assert kumar["primary_public_evidence"].startswith("reported_mean_and_SD")
    assert kumar["field_interpretation"].startswith("INVALID_YEAR_LIKE")
    assert kumar["decision"] == "KUMAR_AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"
    assert all(row["adult_eligible"] == "0" for row in _rows("yang_2c_adult_subject_registry.csv"))
    assert all(row["adult_eligible"] == "0" for row in _rows("kumar_adult_subject_registry.csv"))


def test_final_registry_contains_all_and_only_two_passing_primary_interfaces() -> None:
    rows = _rows("final_adult_untouched_cohort_registry_v3.csv")
    primary = [row for row in rows if row["role"] == "PRIMARY_UNTOUCHED_CONFIRMATION"]
    assert {row["interface_variant"] for row in primary} == {
        "Brandl2020_CANONICAL_ADULT_V1",
        "OpenNeuro_ds007221_HYBRID_ADULT_V1",
    }
    assert sum(int(row["adult_count"]) for row in primary) == 53
    hybrid = next(row for row in primary if row["native_cohort"] == "OpenNeuro_ds007221")
    assert hybrid["subject_registry_sha256"] == subject_registry_sha256(DS007221_HYBRID_SUBJECT_IDS)
    assert hybrid["task_events"] == "left_hand|right_hand"
    assert hybrid["minimum_trials_per_subject"] == "600"
    assert hybrid["license"] == "CC0"


def test_all_passing_interfaces_are_included_without_post_screen_cap() -> None:
    rows = _rows("all_passing_cohort_inclusion_truth_table.csv")
    assert len(rows) == 2
    assert all(row["passes_all_locked_criteria"] == "1" for row in rows)
    assert all(row["included_primary_confirmation"] == "1" for row in rows)
    assert all(row["post_screen_cap_applied"] == "0" for row in rows)
    assert {row["at_least_two_rule"] for row in rows} == {"PASS"}


def test_effective_manifest_v3_is_required_and_authorizes_nothing() -> None:
    manifest = load_effective_manifest_v3()
    assert manifest["final_gate"] == FINAL_GATE
    assert manifest["authoritative_program"]["primary_untouched_confirmation_count"] == 2
    assert manifest["authoritative_program"]["primary_adult_target_subjects"] == 53
    assert manifest["authoritative_program"]["all_passing_interfaces_included"] is True
    assert set(manifest["protected_counters"].values()) == {0}
    assert all(value is False for key, value in manifest["downstream_contract"].items() if key.endswith("authorized"))
    assert manifest["downstream_contract"]["real_data_execution_lock_created"] is False
    assert manifest["downstream_contract"]["authorization_record_created"] is False
    require_effective_manifest_v3_for_c86lp([EFFECTIVE_MANIFEST_V3])
    with pytest.raises(C86R2ContractError, match="stale C86R V2"):
        require_effective_manifest_v3_for_c86lp([REPORTS / "C86_ACTIVE_TESTING_EFFECTIVE_PROGRAM_MANIFEST_V2.json"])


def test_manifest_and_all_bound_table_hashes_replay() -> None:
    manifest = json.loads(EFFECTIVE_MANIFEST_V3.read_text(encoding="utf-8"))
    sidecar = EFFECTIVE_MANIFEST_V3.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
    assert _sha(EFFECTIVE_MANIFEST_V3) == sidecar
    assert len(manifest["table_hashes"]) == 15
    for name, expected in manifest["table_hashes"].items():
        assert _sha(TABLES / name) == expected

