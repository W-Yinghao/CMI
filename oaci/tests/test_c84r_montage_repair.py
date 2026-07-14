from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset import c84_dataset_registry as v1
from oaci.multidataset import c84_dataset_registry_v2 as v2
from oaci.multidataset import c84r_montage_repair as repair
from oaci.multidataset import c84r_regression_suite as suites
from oaci.multidataset import c84r_v2_protocols as protocols


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _rows(path: Path):
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def test_historical_21_channel_objects_remain_content_valid():
    assert len(v1.PRIMARY_CHANNELS) == 21
    assert "FCz" in v1.PRIMARY_CHANNELS
    for _, (name, digest) in repair.HISTORICAL_PROTOCOLS.items():
        assert _digest(repair.REPORT_DIR / name) == digest


def test_repair_protocol_and_v2_stage_hashes_replay():
    assert _digest(repair.REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json") == (
        repair.REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.sha256"
    ).read_text().split()[0]
    for stem in (
        "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2",
        "C84_CANARY_PROTOCOL_V2",
        "C84_FIELD_GENERATION_PROTOCOL_V2",
        "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2",
    ):
        path = protocols.REPORT_DIR / f"{stem}.json"
        assert _digest(path) == (protocols.REPORT_DIR / f"{stem}.sha256").read_text().split()[0]


def test_exact_20_channel_intersection_and_digest():
    assert v2.PRIMARY_CHANNELS == repair.COMMON_CHANNELS
    assert len(v2.PRIMARY_CHANNELS) == 20
    assert "FCz" not in v2.PRIMARY_CHANNELS
    assert "Fz" not in v2.PRIMARY_CHANNELS
    assert v2.sha256_json(list(v2.PRIMARY_CHANNELS)) == repair.MONTAGE_SHA256
    assert all(v2.ordered_dataset_channels(dataset) == v2.PRIMARY_CHANNELS for dataset in v2.DATASETS)


@pytest.mark.parametrize("channels,kwargs", [
    (("FCz",) + repair.COMMON_CHANNELS, {}),
    (("Fz",) + repair.COMMON_CHANNELS[1:], {"substituted_channels": {"FCz": "Fz"}}),
    (repair.COMMON_CHANNELS[:-1], {}),
    (tuple(reversed(repair.COMMON_CHANNELS)), {}),
    (repair.COMMON_CHANNELS, {"interpolation": True}),
    (repair.COMMON_CHANNELS, {"zero_fill": True}),
    (repair.COMMON_CHANNELS, {"dataset_specific_mask": True}),
])
def test_forbidden_montage_repairs_fail_closed(channels, kwargs):
    with pytest.raises(repair.C84RMontageError):
        repair.validate_montage(channels, **kwargs)


def test_subject_partitions_and_canary_targets_replay_exactly():
    assert v2.validate_registry()["ready"] is True
    expected = {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106}
    for dataset, spec in v2.DATASETS.items():
        assert v2.partition_subjects(spec) == v1.partition_subjects(spec)
        assert v2.partition_subjects(spec)["targets"][0] == expected[dataset]


def test_v2_candidate_ids_bind_interface_and_all_migrate():
    old = v1_ids = {row["unit_id"] for row in __import__(
        "oaci.multidataset.c84_fixed_zoo_protocol", fromlist=["candidate_units"]
    ).candidate_units()}
    new_rows = protocols.candidate_units()
    new = {row["unit_id"] for row in new_rows}
    assert len(old) == len(new) == 1944
    assert not old & new
    assert sum(row["canary_subset"] for row in new_rows) == 243
    assert all(row["interface_id"] == repair.INTERFACE_ID for row in new_rows)
    assert all(row["montage_sha256"] == repair.MONTAGE_SHA256 for row in new_rows)


def test_v2_stage_authorization_is_fresh_and_only_canary_lock_is_future_scope():
    external = json.loads((protocols.REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json").read_text())
    assert external["authorization"]["prior_conversational_C84_authorization_active"] is False
    assert external["authorization"]["C84C_authorized"] is False
    assert external["authorization"]["C84F_authorized"] is False
    assert external["authorization"]["C84S_authorized"] is False
    field = json.loads((protocols.REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V2.json").read_text())
    science = json.loads((protocols.REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json").read_text())
    assert field["scope_specific_execution_lock_created_in_C84R"] is False
    assert science["scope_specific_execution_lock_created_in_C84R"] is False


def test_leading_numeric_suite_parser_restores_c34s():
    files = {path.name for path in suites.suite_files("c23")}
    assert "test_c34s_artifact_hygiene.py" in files
    assert "test_c84r2_canary_runtime_repair.py" in files
    assert suites.milestone_number("test_c34s_artifact_hygiene.py") == 34
    assert suites.milestone_number("test_c84r2_canary_runtime_repair.py") == 84
    assert {"test_c84_multidataset_external_validity.py", "test_c84r_montage_repair.py",
            "test_c84c_canary_contract.py", "test_c84l1_intervention.py",
            "test_c84l1_protocol_lock.py", "test_c84l1_canary_contract.py"} <= {
                path.name for path in suites.suite_files("focused")
            }


def test_migration_table_is_complete_and_changed():
    rows = _rows(protocols.TABLE_DIR / "candidate_unit_id_migration.csv")
    assert len(rows) == 1944
    assert all(row["identity_changed"] == "1" for row in rows)
    assert len({row["v2_unit_id"] for row in rows}) == 1944
