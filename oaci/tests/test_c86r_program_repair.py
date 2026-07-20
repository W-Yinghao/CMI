"""C86R effective-program, age-eligibility, and baseline-fidelity tests."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from oaci.theory.c86r_program_repair import (
    C86RContractError,
    EFFECTIVE_MANIFEST,
    FINAL_GATE,
    SYNTHETIC_V2,
    adult_eligibility,
    hara_general_k_score,
    load_effective_manifest,
    project_max_pair_score,
    require_effective_manifest_for_downstream,
)


ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / "oaci/reports"
TABLES = REPORTS / "c86r_tables"


def _rows(name: str) -> list[dict[str, str]]:
    with (TABLES / name).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_effective_manifest_is_the_only_authoritative_downstream_stack() -> None:
    manifest = load_effective_manifest()
    assert manifest["schema_version"] == "c86_active_testing_effective_program_manifest_v2"
    assert manifest["status"] == "EFFECTIVE_PROGRAM_RESOLVED_WITH_ELIGIBILITY_BLOCKER_NO_DOWNSTREAM_EXECUTION_AUTHORITY"
    assert manifest["final_gate"] == FINAL_GATE
    assert manifest["authoritative_program"]["primary_untouched_confirmation_interfaces"] == ["Brandl2020"]
    assert manifest["authoritative_program"]["at_least_two_cohort_rule"] == "FAIL"
    assert manifest["downstream_contract"]["stale_parent_protocol_alone_accepted"] is False
    assert manifest["downstream_contract"]["execution_lock_created"] is False
    require_effective_manifest_for_downstream([EFFECTIVE_MANIFEST])
    with pytest.raises(C86RContractError, match="stale parent"):
        require_effective_manifest_for_downstream([REPORTS / "C86_ACTIVE_TESTING_PROGRAM_PROTOCOL.json"])


def test_effective_manifest_and_synthetic_v2_sidecars_replay() -> None:
    for path in (EFFECTIVE_MANIFEST, SYNTHETIC_V2):
        expected = path.with_suffix(".sha256").read_text(encoding="ascii").split()[0]
        assert _sha(path) == expected
    manifest = _json(EFFECTIVE_MANIFEST)
    assert len(manifest["table_hashes"]) == 20
    for name, expected in manifest["table_hashes"].items():
        assert _sha(TABLES / name) == expected


def test_adult_rule_fails_closed_without_minimum_age_evidence() -> None:
    assert adult_eligibility(minimum_age=22, explicit_all_at_least_18=False) == "ADULT_ELIGIBILITY_PROVEN_PASS"
    assert adult_eligibility(minimum_age=17, explicit_all_at_least_18=False) == "AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"
    assert adult_eligibility(minimum_age=None, explicit_all_at_least_18=False) == "AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"
    rows = {row["cohort"]: row for row in _rows("participant_age_eligibility_audit.csv")}
    assert rows["Brandl2020"]["decision"] == "ADULT_ELIGIBILITY_PROVEN_PASS"
    assert rows["Kumar2024"]["decision"] == "AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"
    assert rows["Yang2025_2C"]["decision"] == "AGE_ELIGIBILITY_NOT_PROVEN_FAIL_CLOSED"
    assert rows["Yang2025_2C"]["minimum_age"] == "17"


def test_final_cohort_registry_preserves_stress_tracks_and_blocks_primary_rule() -> None:
    rows = {row["cohort"]: row for row in _rows("final_untouched_cohort_registry_v2.csv")}
    assert rows["Brandl2020"]["selected_primary_confirmation"] == "1"
    assert rows["Kumar2024"]["selected_primary_confirmation"] == "0"
    assert rows["Yang2025_2C"]["selected_primary_confirmation"] == "0"
    assert rows["Kumar2024"]["authoritative_C86R_role"] == "AGE_UNCERTAIN_STRESS_TRACK_ONLY"
    assert rows["Yang2025_2C"]["authoritative_C86R_role"] == "AGE_MIXED_STRESS_TRACK_ONLY"
    assert {row["at_least_two_primary_rule"] for row in rows.values()} == {"FAIL"}
    assert {row["registered_gate"] for row in rows.values()} == {FINAL_GATE}


def test_hara_general_k_sum_is_not_the_project_max_pair_heuristic() -> None:
    pairwise = np.array(
        [
            [[0.0, 1.0, 2.0], [1.0, 0.0, 4.0], [2.0, 4.0, 0.0]],
            [[0.0, 0.5, 0.25], [0.5, 0.0, 0.75], [0.25, 0.75, 0.0]],
        ],
        dtype=np.float64,
    )
    np.testing.assert_array_equal(hara_general_k_score(pairwise), np.array([7.0, 1.5]))
    np.testing.assert_array_equal(project_max_pair_score(pairwise), np.array([4.0, 0.75]))
    assert not np.array_equal(hara_general_k_score(pairwise), project_max_pair_score(pairwise))


def test_method_registry_corrects_hara_and_declares_major_baselines() -> None:
    rows = {row["method_id"]: row for row in _rows("active_method_registry_v2.csv")}
    assert rows["A2H"]["disposition"] == "PRIMARY_BASELINE"
    assert rows["A2H"]["query_score"].startswith("sum_over_k_lt_kprime")
    assert rows["A2H"]["interface_81_candidates"] == "EXACT_GENERAL_K_SCORE"
    assert rows["A2M"]["disposition"] == "DEVELOPMENT_ONLY"
    assert rows["A2M"]["name"] == "A2M_project_max_pair_heuristic"
    assert rows["A2M"]["confirmation_eligible"] == "0"
    expected = {"ASE_XWED", "ONLINE_AMS", "MODEL_SELECTOR", "CODA", "PPAT"}
    assert expected <= rows.keys()
    assert all(rows[name]["disposition"] for name in expected)
    assert rows["MODEL_SELECTOR"]["disposition"] == "SECONDARY_BASELINE"
    assert rows["CODA"]["disposition"] == "SECONDARY_BASELINE"
    assert rows["PPAT"]["disposition"] == "UNAVAILABLE_WITH_EXACT_REASON"


def test_reference_fidelity_does_not_mislabel_a2m_as_hara() -> None:
    rows = {row["method_id"]: row for row in _rows("reference_fidelity_registry_v2.csv")}
    assert rows["A2H"]["claimed_exact_reference"] == "1"
    assert rows["A2H"]["faithful_object"] == "general_K_sum_over_pairs_query_score"
    assert rows["A2M"]["claimed_exact_reference"] == "0"
    assert rows["A2M"]["fidelity_status"] == "NOT_HARA_PROJECT_HEURISTIC"


def test_synthetic_v2_updates_dispatchers_without_running_registered_draws() -> None:
    value = _json(SYNTHETIC_V2)
    assert value["schema_version"] == "c86p_synthetic_calibration_operationalization_v2"
    assert len(value["scenarios"]) == 11
    assert value["publication_rule"]["C86R_registered_draws"] == 0
    dispatchers = {row["method_id"]: row["dispatcher_id"] for row in value["primary_dispatcher_contract"]}
    assert set(dispatchers) == {"P0", "A1", "A2H"}
    assert "sum_pairwise" in dispatchers["A2H"]
    assert value["publication_rule"]["C86D_or_C86H_authorization_created"] is False


def test_protected_counters_and_authorizations_remain_zero_or_false() -> None:
    manifest = _json(EFFECTIVE_MANIFEST)
    assert set(manifest["protected_counters"].values()) == {0}
    downstream = manifest["downstream_contract"]
    assert all(downstream[key] is False for key in (
        "C86L_authorized", "C86D_authorized", "C86C_F_authorized", "C86H_authorized",
        "new_EEG_or_label_access_authorized", "execution_lock_created",
    ))
