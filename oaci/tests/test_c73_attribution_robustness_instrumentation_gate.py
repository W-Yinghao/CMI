"""C73 attribution-robustness, instrumentation, and artifact contracts."""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np

from oaci.conditioned_ceiling_coverage import c72_measurement_control_gap as c72
from oaci.conditioned_ceiling_coverage import c73_attribution_robustness_instrumentation_gate as c73
from oaci.conditioned_ceiling_coverage import c73_robustness as rb
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C73_ATTRIBUTION_ROBUSTNESS_INSTRUMENTATION_GATE.json"
TABLE_DIR = "oaci/reports/c73_tables"


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _rows(name: str) -> list[dict]:
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _summary() -> dict:
    with open(REPORT_JSON) as f:
        return json.load(f)


def test_c73_protocol_taxonomy_and_static_execution_boundary():
    assert c73._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    protocol = json.load(open(c73.PROTOCOL_JSON))
    assert protocol["schema_version"] == "c73_attribution_robustness_protocol_v1"
    assert protocol["protocol_lock_source_commit"] == "ea844d5ad7021a48da96bdf6c20beb80ebf849b1"
    assert open(c73.PROTOCOL_SHA).read().strip() == _sha256(c73.PROTOCOL_JSON)
    assert protocol["known_before_lock"]["full_frozen_physical_universe_consumed"] is True
    assert protocol["known_before_lock"]["C73_status"].startswith("prospectively specified retrospective")
    assert protocol["execution_boundary"]["real_EEG_forward_passes"] is False
    assert protocol["execution_boundary"]["training_or_parameter_updates"] is False
    assert protocol["instrumentation_readiness"]["dummy_CPU_forward_allowed"] is True
    assert protocol["instrumentation_readiness"]["real_data_forward_allowed"] is False
    assert protocol["decomposition"]["order_count"] == 120
    assert set(protocol["decomposition"]["components"]) == set(rb.COMPONENTS)
    assert set(c73.PRIMARY_DECISIONS) == set(protocol["decision_taxonomy"]["primary"])
    assert set(c73.SECONDARY_DECISIONS) == set(protocol["decision_taxonomy"]["secondary"])
    assert set(c73.FINAL_GATES) == set(protocol["final_gates"])


def test_c73_effective_multiplicity_and_small_synthetic_identity():
    assert np.isclose(rb.effective_multiplicity(np.zeros(8), 0.1), 8.0)
    assert rb.effective_multiplicity(np.asarray([1.0, 0.0, -1.0]), 0.01) < 1.01
    protocol = json.load(open(c73.PROTOCOL_JSON))
    protocol["synthetic_grid"] = {
        "candidate_counts": [8], "effective_near_tie_counts": [2, 4],
        "gauge_tail": ["gaussian"], "dependence": [0.0],
        "label_budgets": [8], "replicates_per_cell": 16, "seed": 73673,
    }
    result = rb.build_synthetic_robustness(protocol)
    assert len(result["synthetic_attribution_robustness_rows"]) == 2
    assert result["synthetic_validation"]["common_offset_identity_passed"] is True
    assert result["synthetic_validation"]["candidate_specific_crossings_present"] is True
    assert result["synthetic_validation"]["raw_draws_persisted"] == 0


def test_c73_required_artifacts_provenance_and_no_execution():
    summary = _summary()
    assert summary["milestone"] == "C73"
    assert summary["protocol_commit"] == "26d3d34"
    assert summary["protocol_sha256"] == _sha256(c73.PROTOCOL_JSON)
    assert summary["confirmation_status"] == "retrospective_robustness_not_independent_confirmation"
    assert summary["key_numbers"]["T2_units"] == 216
    assert summary["key_numbers"]["T3_HO_units"] == 1052
    assert summary["key_numbers"]["full_frozen_units"] == 1268
    assert summary["forward_passes"] == summary["reinference_runs"] == 0
    assert summary["training_attempted"] == summary["gpu_used"] == 0
    assert summary["real_EEG_trials_loaded"] == 0
    assert summary["bnci004_used"] == summary["reserved_seeds_used"] == 0
    assert summary["selector_artifact_emitted"] == 0
    assert summary["checkpoint_recommendation_artifact_emitted"] == 0
    assert summary["selected_checkpoint_ids_emitted"] == 0
    assert summary["raw_cache_rows_copied_to_git"] == 0
    for name in c73.TABLE_SPECS:
        assert os.path.exists(os.path.join(TABLE_DIR, name)), name
    for path in (c73.MAIN_REPORT, c73.RED_REPORT, c73.THEORY_NOTE, c73.TIMING_REPORT):
        assert os.path.exists(path), path
    assert {row["passed"] for row in _rows("c72_protocol_replay.csv")} == {"1"}
    assert {row["passed"] for row in _rows("c72_cache_identity_replay.csv")} == {"1"}
    assert {row["passed"] for row in _rows("c72_metric_identity_replay.csv")} == {"1"}
    assert {row["passed"] for row in _rows("c72_red_team_repair_ledger.csv")} == {"1"}


def test_c73_attribution_order_shapley_and_context_contracts():
    order_rows = _rows("attribution_order_sensitivity.csv")
    groups = {(row["budget"], row["endpoint"]) for row in order_rows}
    assert len(groups) == 6
    for group in groups:
        rows = [row for row in order_rows if (row["budget"], row["endpoint"]) == group]
        assert len({row["order"] for row in rows}) == 120
        for order in {row["order"] for row in rows}:
            assert len([row for row in rows if row["order"] == order]) == 5
    shapley = _rows("attribution_shapley_summary.csv")
    full = [row for row in shapley if row["budget"] == c72.FULL_BUDGET and row["endpoint"] == "bAcc"]
    assert {row["component_code"] for row in full} == set(rb.COMPONENTS)
    for row in shapley:
        peers = [other for other in shapley if other["budget"] == row["budget"] and other["endpoint"] == row["endpoint"] and other is not row]
        expected = int(
            float(row["largest_order_fraction"]) >= 0.90
            and all(float(row["ci_lower"]) > float(other["ci_upper"]) for other in peers)
        )
        assert int(row["dominant_by_registered_rule"]) == expected
    random_rows = _rows("cell_specific_random_baselines.csv")
    assert all(np.isclose(float(row["random_top1"]), 1.0 / int(row["candidate_count"])) for row in random_rows)
    assert {row["reliable_control_inferred"] for row in random_rows} == {"0"}
    assert _rows("effective_candidate_multiplicity.csv")
    assert _rows("topk_and_regret_context.csv")
    assert _rows("near_tie_set_size.csv")


def test_c73_residual_h3_h4_h5_h6_claim_contracts():
    summary = _summary()
    validity = {row["criterion"]: row for row in _rows("residual_construct_validity.csv")}
    overall = int(validity["overall_construct_validity"]["passed"])
    assert bool(summary["key_numbers"]["residual_construct_validated"]) == bool(overall)
    report = open(c73.MAIN_REPORT).read().lower()
    if not overall:
        assert "unexplained candidate-specific residual" in report
    h3 = _rows("shared_calibration_equivalence.csv")
    assert {row["intervention_family"] for row in h3} == {"shared_class_vector", "shared_temperature_T1_identity"}
    assert all(float(row["beneficial_SESOI"]) > 0 for row in h3)
    power = _rows("shared_calibration_power_audit.csv")
    assert {row["p_gt_0p05_used_as_insufficiency"] for row in power if row.get("p_gt_0p05_used_as_insufficiency", "") != ""} == {"0"}
    h4 = _rows("h4_identity_vs_empirical_effect.csv")
    assert all(float(row["mean_identity_accuracy"]) == 1.0 for row in h4)
    assert {row["validates_residual_origin"] for row in h4} == {"0"}
    h5 = _rows("intervention_family_sensitivity.csv")
    assert {row["T3_tuned"] for row in h5 if row.get("T3_tuned", "") != ""} == {"0"}
    assert _rows("alpha_zero_local_geometry.csv")
    assert _rows("cross_fitted_intervention_stability.csv")
    assert _rows("candidate_count_confounding.csv")
    assert _rows("raw_M_vs_effective_M.csv")
    assert _rows("top_gap_adjusted_effects.csv")
    hierarchical = _rows("hierarchical_inference_summary.csv")
    assert hierarchical
    assert {row["row_iid_used"] for row in hierarchical} == {"0"}
    assert {row["inference_family"] for row in hierarchical} >= {
        "target_cluster", "checkpoint_cluster", "trial_id_cluster", "crossed_pigeonhole"
    }


def test_c73_theory_synthetic_and_instrumentation_scopes():
    assert {row["distributional_theorem_claimed"] for row in _rows("effective_candidate_bound.csv")} == {"0"}
    assert {row["simulation_or_empirical_proxy_only"] for row in _rows("empirical_tail_bound.csv")} == {"1"}
    assert _rows("finite_population_best_arm_bound.csv")
    synthetic = _rows("synthetic_attribution_robustness.csv")
    assert len(synthetic) > 0
    assert {row["common_offset_rank_flip_rate"] for row in synthetic} == {"0.0"}
    assert any(row["high_reliability_poor_top1"] == "1" for row in synthetic)
    assert {row["raw_draws_persisted"] for row in synthetic} == {"0"}
    hooks = _rows("hook_ABI_validation.csv")
    assert len(hooks) == 6
    assert {row["passed"] for row in hooks} == {"1"}
    assert {row["dummy_tensor_only"] for row in hooks} == {"1"}
    assert {row["real_EEG_forward"] for row in hooks} == {"0"}
    assert max(float(row["Wz_plus_b_logit_max_abs"]) for row in hooks) <= 1e-6
    feasibility = {row["criterion"]: row for row in _rows("frozen_instrumentation_feasibility.csv")}
    assert {row["passed"] for row in feasibility.values()} == {"1"}
    stages = {row["stage"]: row for row in _rows("replication_stage_decision.csv")}
    assert stages["R1"]["authorized"] == "0"
    assert stages["R1"]["new_training"] == "0"
    assert stages["R2"]["authorized"] == stages["R3"]["authorized"] == "0"


def test_c73_red_team_schema_and_artifact_hygiene():
    summary = _summary()
    assert summary["key_numbers"]["red_team_failure_count"] == 0
    assert {row["failed"] for row in _rows("red_team_failure_ledger.csv")} == {"0"}
    assert {row["blocking"] for row in _rows("risk_register.csv")} == {"0"}
    assert {row["passed"] for row in _rows("forbidden_claim_scan.csv")} == {"1"}
    assert {row["passed"] for row in _rows("large_artifact_scan.csv")} == {"1"}
    assert {row["passed"] for row in _rows("schema_validation_summary.csv")} == {"1"}
    assert {row["status"] for row in _rows("test_command_manifest.csv")} == {"green"}
    manifest = _rows("artifact_manifest.csv")
    assert manifest
    for row in manifest:
        assert os.path.exists(row["path"])
        assert _sha256(row["path"]) == row["sha256"]
        assert int(row["size_bytes"]) <= c73.MAX_REPORT_BYTES
