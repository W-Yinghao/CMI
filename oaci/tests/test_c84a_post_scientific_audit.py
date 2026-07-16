from __future__ import annotations

import ast
import csv
import hashlib
import json
from pathlib import Path

import pytest

from oaci.multidataset import c84a_post_scientific_audit as audit


@pytest.fixture(scope="module")
def tables() -> dict[str, list[dict]]:
    result: dict[str, list[dict]] = {}
    for builder in (
        audit.build_identity_replays,
        audit.build_gate_matrices,
        audit.build_cott_audit,
        audit.build_mano_audit,
        audit.build_label_frontier_audit,
        audit.build_transport_audit,
        audit.build_theory_and_next_experiment_tables,
    ):
        result.update(builder())
    result["C84A_CLAIM_CONTRACT.csv"] = audit.build_claim_contract()
    return result


def test_c84s_identity_and_all_18_tables_replay(tables: dict[str, list[dict]]) -> None:
    assert len(tables["c84s_identity_replay.csv"]) == 10
    assert all(row["status"] == "PASS" for row in tables["c84s_identity_replay.csv"])
    assert len(tables["result_table_manifest_replay.csv"]) == 18
    assert all(row["status"] == "PASS" for row in tables["result_table_manifest_replay.csv"])


def test_lifecycle_and_protected_counters_replay(tables: dict[str, list[dict]]) -> None:
    assert len(tables["lifecycle_stage_replay.csv"]) == 3
    assert all(row["status"] == "PASS" for row in tables["lifecycle_stage_replay.csv"])
    assert len(tables["protected_counter_replay.csv"]) == 11
    assert all(row["status"] == "PASS" for row in tables["protected_counter_replay.csv"])


def test_full_and_level_gate_component_coverage(tables: dict[str, list[dict]]) -> None:
    assert len(tables["full_panel_gate_component_matrix.csv"]) == 18
    assert len(tables["level_specific_gate_component_matrix.csv"]) == 36
    assert all(row["Q2_upper_component_status"] == "NOT_RECOMPUTED" for row in tables["level_specific_gate_component_matrix.csv"])
    assert all("nearest_available_failing" in "|".join(row) for row in tables["level_specific_gate_component_matrix.csv"])


def test_lee_cott_is_single_target_near_boundary_failure(tables: dict[str, list[dict]]) -> None:
    row = next(row for row in tables["cott_average_tail_separation.csv"] if row["dataset"] == "Lee2019_MI")
    assert row["adverse_targets"] == 2
    assert row["registered_floor_breach_targets"] == 1
    assert row["registered_floor_breach_target_ids"] == "8"
    assert row["tail_interpretation"] == "single_target_near_boundary_floor_failure"
    assert float(row["minimum"]) == pytest.approx(-0.10787324378695262)


def test_cho_and_physionet_cott_tail_depth(tables: dict[str, list[dict]]) -> None:
    rows = {row["dataset"]: row for row in tables["cott_average_tail_separation.csv"]}
    assert rows["Cho2017"]["adverse_targets"] == 2
    assert rows["Cho2017"]["registered_floor_breach_target_ids"] == "3"
    assert rows["PhysionetMI"]["adverse_targets"] == 19
    assert rows["PhysionetMI"]["registered_floor_breach_targets"] == 9


def test_only_lee_target_8_changes_loto_category(tables: dict[str, list[dict]]) -> None:
    changed = [row for row in tables["cott_target_influence.csv"] if row["category_changed_frozen"]]
    assert [(row["dataset"], row["left_out_target"], row["LOTO_category_frozen"]) for row in changed] == [
        ("Lee2019_MI", "8", "A")
    ]


def test_mano_cho_decision_without_global_rank(tables: dict[str, list[dict]]) -> None:
    row = next(row for row in tables["mano_cross_dataset_decision_profile.csv"] if row["dataset"] == "Cho2017")
    assert row["Q1_pass"] == row["Q2_pass"] == 1
    assert float(row["mean_Spearman"]) == pytest.approx(0.0009604159822539928)
    assert float(row["top1"]) == pytest.approx(0.05)
    assert row["dominant_selected_regime"] == "ERM"
    assert float(row["dominant_regime_fraction"]) == 1.0
    assert float(row["exact_B1_utility_and_regret_context_fraction_descriptive"]) == 1.0
    assert row["near_optimal_action_density_identified"] == 0


def test_label_frontier_failure_decomposition(tables: dict[str, list[dict]]) -> None:
    rows = {(row["dataset"], row["budget"]): row for row in tables["label_frontier_component_matrix.csv"]}
    assert rows[("Lee2019_MI", "FULL")]["mean_component_pass"] == 1
    assert rows[("Lee2019_MI", "FULL")]["maxT_component_pass"] == 1
    assert rows[("Lee2019_MI", "FULL")]["worst_component_pass"] == 0
    assert rows[("Cho2017", "8")]["direct_qualification_frozen"] == 1
    assert rows[("PhysionetMI", "FULL")]["maxT_component_pass"] == 0
    assert rows[("PhysionetMI", "FULL")]["favorable_component_pass"] == 0
    assert rows[("PhysionetMI", "FULL")]["worst_component_pass"] == 0


def test_transport_preserves_method_identity_without_pooling(tables: dict[str, list[dict]]) -> None:
    rows = tables["c82_c84_method_transport_matrix.csv"]
    assert len(rows) == 45
    assert {row["method_id"] for row in rows} == {"S1", *audit.PRIMARY_METHODS, "Q0_B1", "Q0_FULL"}
    assert {row["cohort_or_seed"] for row in rows} == {
        "BNCI2014_001_seed3", "BNCI2014_001_seed4", *audit.DATASETS,
    }


def test_every_new_row_is_exploratory_source_keyed_and_gate_immutable(tables: dict[str, list[dict]]) -> None:
    for rows in tables.values():
        for row in rows:
            assert row["analysis_status"] == audit.STATUS
            assert int(row["confirmatory_gate_changed"]) == 0
            assert row["source_artifacts"]
            assert row["source_row_keys"]
    frozen = json.loads((audit.C84S_RESULT_DIR / "C84S_RESULT.json").read_text())
    assert frozen["primary_gate"] == audit.GATE
    assert frozen["label_frontier_tag"] == audit.FRONTIER_TAG


def test_module_has_no_scientific_runtime_import_or_array_loader() -> None:
    source = Path(audit.__file__).read_text()
    tree = ast.parse(source)
    imports = []
    calls = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imports.append(node.module or "")
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Name):
                calls.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                calls.append(node.func.attr)
    forbidden = ("c84s_select", "c84s_q0", "c84s_inference", "c84sr3_stage", "numpy", "torch", "mne", "moabb")
    assert not any(any(token in name for token in forbidden) for name in imports)
    assert not {"load", "forward", "train"}.intersection(calls)


def test_validation_is_complete(tables: dict[str, list[dict]]) -> None:
    checks = audit.validate_tables(tables)
    assert len(checks) >= 70
    assert all(row["passed"] for row in checks)


def test_build_writes_complete_machine_and_human_reports(tmp_path: Path) -> None:
    result = audit.build(tmp_path)
    report = json.loads((tmp_path / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.json").read_text())
    assert report["final_gate"] == audit.SUCCESS_GATE
    assert report["immutable_C84S_primary_gate"] == audit.GATE
    assert report["immutable_C84S_label_frontier_tag"] == audit.FRONTIER_TAG
    assert report["new_pvalues"] == 0
    assert report["authorization"]["C85"] is False
    assert len(result["tables"]) == 23
    assert len(list((tmp_path / "c84a_tables").glob("*.csv"))) == 22
    checksum_lines = (tmp_path / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.sha256").read_text().splitlines()
    assert len(checksum_lines) == 2
    for line in checksum_lines:
        expected, name = line.split("  ")
        assert hashlib.sha256((tmp_path / name).read_bytes()).hexdigest() == expected


def test_claim_contract_forbids_expansion_and_authorization(tables: dict[str, list[dict]]) -> None:
    rows = tables["C84A_CLAIM_CONTRACT.csv"]
    assert sum(row["status"] == "SUPPORTED" for row in rows) == 8
    assert sum(row["status"] == "FORBIDDEN" for row in rows) == 6
    assert all(row["manuscript_authorized"] == 0 and row["C85_authorized"] == 0 for row in rows)
