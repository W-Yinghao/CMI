from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np

from oaci.conditioned_ceiling_coverage import c77_independent_multiregime_replication_protocol as c77_analysis
from oaci.conditioned_ceiling_coverage import c77_protocol
from oaci.conditioned_ceiling_coverage import synthetic_multiregime_generator as synthetic


def _sha(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _rows(name: str) -> list[dict]:
    return list(csv.DictReader((c77_protocol.TABLE_DIR / name).open()))


def test_c77_parent_and_no_execution_boundary_are_locked():
    assert c77_protocol.PARENT_COMMIT.startswith("ce23753")
    protocol = json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    assert protocol["parent_C76_result_commit"] == c77_protocol.PARENT_COMMIT
    assert protocol["execution_boundary"] == {
        "BNCI2014_004_access": False,
        "GPU": False,
        "checkpoints_created": 0,
        "manuscript": False,
        "raw_cache_or_weights_in_git": False,
        "re_inference": False,
        "real_forward": False,
        "seed3_access": False,
        "seed4_access": False,
        "selector_or_recommendation_artifact": False,
        "training": False,
    }


def test_c77_and_c78_protocol_hashes_replay():
    assert _sha(c77_protocol.PROTOCOL_PATH) == c77_protocol.PROTOCOL_SHA_PATH.read_text().strip()
    assert _sha(c77_protocol.C78_PROTOCOL_PATH) == c77_protocol.C78_PROTOCOL_SHA_PATH.read_text().strip()


def test_c77_all_static_registry_hashes_replay():
    protocol = json.loads(c77_protocol.PROTOCOL_PATH.read_text())
    for item in protocol["locked_tables"].values():
        path = Path(item["path"])
        assert _sha(path) == item["sha256"]
        assert path.stat().st_size == item["size_bytes"]


def test_c77_replays_c76_gate_metrics_and_orbit_identity():
    metrics = {row["metric"]: row for row in _rows("c76_metric_identity_replay.csv")}
    assert metrics["final_gate"]["reported"] == "LOCAL_NONLINEAR_MEASUREMENT_NONTRANSPORTABLE"
    assert all(row["match"] == "1" for row in metrics.values())
    orbit = _rows("c76_orbit_identity_replay.csv")[0]
    assert orbit["orbit_variants"] == "29"
    assert float(orbit["max_projection_error"]) < 1e-8
    assert float(orbit["max_probability_error"]) < 1e-8
    assert orbit["prediction_disagreements"] == "0"


def test_c77_closes_current_representation_branch_without_reopening_T3():
    rows = {row["branch"]: row for row in _rows("c76_branch_closure_ledger.csv")}
    assert rows["additional_unregistered_T2_representation_features_or_kernels"]["C77_action"] == "forbidden"
    assert rows["T3_HO_representation_generation"]["C77_action"] == "forbidden"
    assert rows["new_seed_multiregime_replication"]["C77_action"] == "prepare_not_execute"


def test_c77_recovers_three_primary_regimes_and_two_equal_length_trajectories():
    rows = {row["regime_id"]: row for row in _rows("regime_reconstruction_status.csv")}
    assert {name for name, row in rows.items() if row["qualifies_primary_R1"] == "1"} == {"ERM", "OACI", "SRC"}
    assert {name for name, row in rows.items() if row["comparable_40_checkpoint_trajectory_per_level"] == "1"} == {"OACI", "SRC"}
    assert rows["ERM"]["comparable_40_checkpoint_trajectory_per_level"] == "0"
    assert all(rows[name]["exact_config_recoverable"] == "1" for name in ("ERM", "OACI", "SRC"))


def test_c77_src_is_disclosed_as_historical_falsification_control():
    rows = {row["regime_id"]: row for row in _rows("historical_regime_inventory.csv")}
    assert "after C10" in rows["SRC"]["historical_context"]
    assert "C12 falsified transfer" in rows["SRC"]["historical_context"]
    assert rows["SRC"]["C14_C76_target_outcome_used_to_select_for_C77"] == "0"
    assert rows["SRC"]["role_in_R1"] == "preexisting_negative_control_trajectory_C11_C12"


def test_c77_primary_regime_config_hashes_are_distinct_and_bound():
    rows = _rows("regime_config_hash_manifest.csv")
    assert {row["regime_id"] for row in rows} == {"ERM", "OACI", "SRC"}
    assert len({row["regime_config_sha256"] for row in rows}) == 3
    assert all(len(row["regime_config_sha256"]) == 64 for row in rows)
    assert all(json.loads(row["payload_json"])["base"] for row in rows)


def test_c77_target_isolation_audit_is_regime_complete():
    rows = _rows("regime_target_isolation_audit.csv")
    assert {row["regime_id"] for row in rows} == {"ERM", "OACI", "SRC"}
    assert all(row["target_label_training_access"] == "0" and row["passed"] == "1" for row in rows)


def test_c78_is_bnci001_seed3_only_and_exact_token_gated():
    protocol = json.loads(c77_protocol.C78_PROTOCOL_PATH.read_text())
    assert protocol["status"] == "LOCKED_READY_BUT_NOT_AUTHORIZED"
    assert protocol["authorization"]["exact_token"] == c77_protocol.C78_AUTHORIZATION_TOKEN
    assert protocol["authorization"]["accepted_channel"] == "exact_CLI_argument_only"
    assert not protocol["authorization"]["prompt_text_is_authorization"]
    assert protocol["execution_boundary"]["dataset_allowlist"] == ["BNCI2014_001"]
    assert "BNCI2014_004" in protocol["execution_boundary"]["dataset_denylist"]
    assert protocol["execution_boundary"]["seed_allowlist"] == [3]
    assert protocol["execution_boundary"]["seed_denylist"] == [4]


def test_c78_matrix_covers_two_levels_and_discloses_erm_asymmetry():
    protocol = json.loads(c77_protocol.C78_PROTOCOL_PATH.read_text())
    matrix = protocol["execution_matrix"]
    assert len(matrix) == 9 * 2 * 3
    assert {row["level"] for row in matrix} == {0, 1}
    assert sum(row["retained_checkpoints"] for row in matrix) == 1458
    assert protocol["matrix_summary"]["retained_checkpoint_target_level_units"] == 1458
    assert protocol["regimes"]["ERM_role"] == "shared_stage1_final_anchor_only"


def test_c78_pilot_selection_is_deterministic_and_outcome_blind():
    protocol = json.loads(c77_protocol.C78_PROTOCOL_PATH.read_text())
    assert protocol["pilot"]["target"] == 4
    assert protocol["pilot"]["regime"] == "OACI"
    assert protocol["pilot"]["outcome_blind"]
    assert protocol["pilot"]["planned_retained_units"] == 82


def test_c77_seed_roles_keep_seed4_physically_future_only():
    roles = {row["seed"]: row for row in _rows("seed_role_contract.csv")}
    assert roles["3"]["confirmation_claim_allowed"] == "0"
    assert roles["4"]["access_in_C77"] == "0"
    skeleton = json.loads(c77_protocol.C79_SKELETON_PATH.read_text())
    assert skeleton["status"] == "SKELETON_ONLY_NOT_FINAL_NOT_AUTHORIZED"
    assert skeleton["C77_seed4_access"] is False


def test_c77_physical_views_separate_information_classes():
    rows = {row["view"]: row for row in _rows("physical_view_schema.csv")}
    assert rows["strict_source_trial_view"]["uses_target_rows"] == "0"
    assert rows["target_unlabeled_trial_view"]["uses_target_labels"] == "0"
    assert rows["target_construction_view"]["uses_evaluation_labels"] == "0"
    assert rows["target_evaluation_view"]["uses_evaluation_labels"] == "1"
    assert all(row["physically_separate"] == "1" for row in rows.values())


def test_c77_hypotheses_and_actionability_are_small_and_conjunctive():
    hypotheses = _rows("primary_hypothesis_registry.csv")
    assert [row["hypothesis"] for row in hypotheses] == [f"H{i}" for i in range(1, 8)]
    gates = _rows("actionability_gate_registry.csv")
    assert len(gates) == 16
    assert all(row["all_required"] == "1" and row["association_p_alone_sufficient"] == "0" for row in gates)
    assert {row["gate"] for row in gates} >= {"incremental_R2", "leave_regime_median", "material_topk_or_regret"}


def test_c77_hierarchical_plan_forbids_row_iid_inference():
    rows = _rows("hierarchical_inference_plan.csv")
    assert all(row["mandatory"] == "1" and row["row_iid_allowed"] == "0" for row in rows)
    assert {row["unit"] for row in rows} >= {"target", "regime", "trajectory", "checkpoint", "trial_id", "checkpoint_x_trial_id"}


def test_c77_dummy_hook_ABI_is_functionally_exact_without_real_data():
    row = c77_analysis._dummy_abi()[0]
    assert row["passed"] == 1
    assert row["Wz_plus_b_max_abs"] == 0.0
    assert row["repeat_logit_max_abs"] == 0.0
    assert row["real_EEG_rows_loaded"] == 0
    assert json.loads(row["z_shape"]) == [2, 800]


def test_c77_sidecar_schema_locks_dataset_seed_and_regime():
    schema = json.loads((c77_protocol.TABLE_DIR / "checkpoint_sidecar_schema.json").read_text())
    assert schema["properties"]["dataset"]["const"] == "BNCI2014_001"
    assert schema["properties"]["seed"]["const"] == 3
    assert schema["properties"]["regime"]["enum"] == ["ERM", "OACI", "SRC"]


def test_c77_synthetic_grid_is_complete_and_hash_identified():
    cells = synthetic.grid()
    assert len(cells) == 486
    assert len({row["cell_id"] for row in cells}) == len(cells)
    assert {row["candidate_count"] for row in cells} == {20, 40, 80}
    assert {row["effective_multiplicity"] for row in cells} == {2, 8, 20}


def test_c77_synthetic_cell_is_deterministic_and_reports_transport_separately():
    cell = {
        "candidate_count": 20, "effective_multiplicity": 8, "top_gap": 0.03,
        "association_strength": 0.5, "transport_heterogeneity": 0.75,
        "label_budget": 32, "cell_id": "syn_0000000000000077",
    }
    left = synthetic.simulate_cell(cell, replicates=3, base_seed=77)
    right = synthetic.simulate_cell(cell, replicates=3, base_seed=77)
    assert left == right
    assert 0 <= left["association_detection_rate"] <= 1
    assert 0 <= left["transport_qualification_rate"] <= 1
    assert 0 <= left["actionability_qualification_rate"] <= 1
    assert np.isfinite(left["median_trajectory_incremental_R2"])


def test_c77_risk_register_has_no_silently_open_blocker():
    rows = _rows("risk_register.csv")
    assert len(rows) >= 20
    assert not [row for row in rows if row["blocking_open"] == "1"]
    assert {row["risk"] for row in rows} >= {"seed3_seed4_contamination", "ERM_trajectory_asymmetry", "SRC_historical_negative_control_overread"}


def test_c77_external_dataset_is_readiness_only():
    rows = _rows("external_dataset_readiness.csv")
    access = next(row for row in rows if row["requirement"] == "dataset_access")
    assert access["status"] == "not_accessed_in_C77"
    graph = _rows("R2_dependency_graph.csv")
    assert all(row["executes_in_C77"] == "0" for row in graph)


def test_c77_final_artifacts_if_present_require_red_team_pass():
    main = c77_protocol.REPORT_DIR / "C77_INDEPENDENT_MULTIREGIME_REPLICATION_PROTOCOL.md"
    if main.exists():
        red_team = (c77_protocol.REPORT_DIR / "C77_RED_TEAM_VERIFICATION.md").read_text()
        assert "Final status: `PASS`" in red_team
        state = json.loads(c77_analysis.STATE_PATH.read_text())
        assert state["execution_boundary"]["training"] == 0
        assert state["execution_boundary"]["seed3_access"] == 0
        assert state["execution_boundary"]["seed4_access"] == 0
