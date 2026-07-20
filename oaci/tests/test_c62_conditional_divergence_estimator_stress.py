"""C62 Conditional-Divergence Estimator Stress tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c62_conditional_divergence_estimator_stress as c62
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.json"
TABLE_DIR = "oaci/reports/c62_tables"


def _summary():
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name):
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_c62_decision_scope_and_training_gate_are_frozen():
    assert c62._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c62.DECISIONS) == {
        "C62-A_C61_ladder_reproduced",
        "C62-B_partition_cod_ladder_stable_under_smoothing_and_support_stress",
        "C62-C_full_conditional_cs_estimator_supported_and_matches_ladder",
        "C62-D_summary_kernel_proxy_only_not_full_conditional_cs",
        "C62-E_kernel_or_cs_proxy_unstable_but_partition_metrics_stable",
        "C62-F_endpoint_scalar_dominates_incremental_observability_across_estimators",
        "C62-G_template_partial_observability_but_no_screen_off",
        "C62-H_source_observable_estimator_escape_hatch_found",
        "C62-I_no_source_observable_estimator_escape_hatch_found",
        "C62-J_synthetic_rank_gauge_estimator_validation_successful",
        "C62-K_trial_level_or_atom_instrumentation_needed_for_full_cs_or_split_label",
        "C62-L_no_new_training_authorized",
        "C62-M_claim_or_availability_inconsistency_found",
    }
    d = _summary()
    assert d["milestone"] == "C62"
    assert d["config_hash"] == "664007686afb520f"
    assert d["c61_commit"] == "0c2f5b8"
    assert d["c61_decision"] == "C61-A_conditional_observability_divergence_ladder_established"
    assert d["decision"]["primary"] == "C62-A_C61_ladder_reproduced"
    for active in (
        "C62-B_partition_cod_ladder_stable_under_smoothing_and_support_stress",
        "C62-D_summary_kernel_proxy_only_not_full_conditional_cs",
        "C62-E_kernel_or_cs_proxy_unstable_but_partition_metrics_stable",
        "C62-F_endpoint_scalar_dominates_incremental_observability_across_estimators",
        "C62-I_no_source_observable_estimator_escape_hatch_found",
        "C62-L_no_new_training_authorized",
    ):
        assert active in d["decision"]["active"]
    assert "C62-C_full_conditional_cs_estimator_supported_and_matches_ladder" in d["decision"]["inactive"]
    assert "C62-H_source_observable_estimator_escape_hatch_found" in d["decision"]["inactive"]
    assert "C62-M_claim_or_availability_inconsistency_found" in d["decision"]["inactive"]
    assert d["decision"]["training_gate"] == c62.TRAINING_GATE
    assert d["decision"]["instrumentation_gate"] == c62.INSTRUMENTATION_GATE
    assert d["decision"]["red_team_failure_count"] == 0


def test_c62_table_shapes_and_reports_are_complete():
    d = _summary()
    assert d["table_row_counts"] == {
        "artifact_feasibility_audit": 10,
        "artifact_manifest": 20,
        "c61_identity_replay": 6,
        "estimator_agreement_ladder": 6,
        "estimator_inventory": 8,
        "forbidden_claim_scan": 21,
        "instrumentation_gate": 7,
        "kernel_proxy_feasibility": 7,
        "large_artifact_scan": 20,
        "null_calibration_summary": 6,
        "partition_cod_sensitivity": 45,
        "red_team_failure_ledger": 13,
        "schema_validation_summary": 15,
        "screening_off_summary": 5,
        "source_observable_adversary_summary": 5,
        "synthetic_rank_gauge_estimator_grid": 11,
        "test_command_manifest": 4,
    }
    required_reports = {
        "C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.json",
        "C62_CONDITIONAL_DIVERGENCE_ESTIMATOR_STRESS.md",
        "C62_ESTIMATOR_STRESS_NOTES.md",
        "C62_INSTRUMENTATION_GATE.md",
        "C62_RED_TEAM_VERIFICATION.md",
    }
    assert required_reports <= {p for p in os.listdir("oaci/reports") if p.startswith("C62_")}
    expected_tables = {
        "artifact_feasibility_audit.csv",
        "artifact_manifest.csv",
        "c61_identity_replay.csv",
        "estimator_agreement_ladder.csv",
        "estimator_inventory.csv",
        "forbidden_claim_scan.csv",
        "instrumentation_gate.csv",
        "kernel_proxy_feasibility.csv",
        "large_artifact_scan.csv",
        "null_calibration_summary.csv",
        "partition_cod_sensitivity.csv",
        "red_team_failure_ledger.csv",
        "schema_validation_summary.csv",
        "screening_off_summary.csv",
        "source_observable_adversary_summary.csv",
        "synthetic_rank_gauge_estimator_grid.csv",
        "test_command_manifest.csv",
    }
    assert expected_tables == {p for p in os.listdir(TABLE_DIR) if p.endswith(".csv")}


def test_c62_c61_identity_replay_is_exact():
    replay = {r["comparison_id"]: r for r in _rows("c61_identity_replay.csv")}
    assert len(replay) == 6
    assert {r["identity_pass"] for r in replay.values()} == {"1"}
    assert float(replay["COD_key_given_source"]["delta_hit"]) == 0.4876543209876543 - 0.5061728395061729
    assert float(replay["COD_template_given_source"]["delta_hit"]) == 0.7037037037037037 - 0.5061728395061729
    assert float(replay["COD_endpoint_given_source_template"]["delta_hit"]) == 0.9444444444444444 - 0.7037037037037037


def test_c62_estimator_inventory_and_feasibility_keep_full_cs_boundary():
    inventory = {r["estimator_id"]: r for r in _rows("estimator_inventory.csv")}
    assert inventory["E1"]["estimator_family"] == "finite_partition_plugin_cod"
    assert inventory["E1"]["supported_by_existing_artifacts"] == "1"
    assert inventory["E4"]["estimator_family"] == "summary_level_kernel_cs_proxy"
    assert inventory["E4"]["supported_by_existing_artifacts"] == "1"
    assert inventory["E4"]["full_conditional_cs"] == "0"
    assert inventory["E6"]["estimator_family"] == "sample_level_kde_gram_conditional_cs"
    assert inventory["E6"]["supported_by_existing_artifacts"] == "0"
    assert inventory["E6"]["requires_sample_level_pairs"] == "1"
    feasibility = {r["artifact_requirement"]: r for r in _rows("artifact_feasibility_audit.csv")}
    assert feasibility["overall_feasibility_decision"]["decision"] == "SUMMARY_LEVEL_KERNEL_PROXY_ONLY_AND_TRIAL_LEVEL_CACHE_REQUIRED"
    assert feasibility["per_trial_predictions_probabilities_logits"]["present"] == "0"
    assert feasibility["per_trial_labels"]["present"] == "0"
    assert feasibility["representation_tensors"]["present"] == "0"


def test_c62_partition_smoothing_keeps_ladder_order_and_template_boundary():
    rows = _rows("partition_cod_sensitivity.csv")
    assert len(rows) == 45
    assert {r["stable_order"] for r in rows} == {"1"}
    template_rows = [r for r in rows if r["comparison"] == "source_to_template"]
    assert {r["template_below_max_null_p95"] for r in template_rows} == {"1"}
    endpoint_rows = [r for r in rows if r["comparison"] == "source_to_endpoint"]
    assert {r["endpoint_dominates_setting"] for r in endpoint_rows} == {"1"}
    raw = {(r["setting"], r["comparison"]): r for r in rows}
    assert float(raw["raw_partition", "source_to_endpoint"]["smoothed_gain"]) == (
        0.9444444444444444 - 0.5061728395061729
    )


def test_c62_kernel_proxy_is_summary_only_and_bandwidth_sensitive():
    kernel = {r["row_id"]: r for r in _rows("kernel_proxy_feasibility.csv")}
    assert kernel["K0_full_sample_conditional_cs"]["supported"] == "0"
    assert kernel["K0_full_sample_conditional_cs"]["sample_level_pairs_required"] == "1"
    proxy_rows = [r for r in kernel.values() if r["row_id"].startswith("K1_summary")]
    assert len(proxy_rows) == 6
    assert {r["proxy_only"] for r in proxy_rows} == {"1"}
    assert "1" in {r["unstable"] for r in proxy_rows}
    assert {r["rank_order_endpoint_gt_template"] for r in proxy_rows} == {"1"}
    assert float(kernel["K1_summary_rbf_bw_0.1"]["endpoint_proxy"]) > float(
        kernel["K1_summary_rbf_bw_0.1"]["template_proxy"]
    )


def test_c62_estimator_agreement_and_nulls_preserve_endpoint_boundary():
    agreement = {r["comparison_id"]: r for r in _rows("estimator_agreement_ladder.csv")}
    assert float(agreement["COD_key_given_source"]["raw_hit_gain"]) < 0.0
    assert agreement["COD_template_given_source"]["template_partial"] == "1"
    assert agreement["COD_endpoint_given_source"]["endpoint_dominates"] == "1"
    assert float(agreement["COD_endpoint_given_source"]["cs_binary_proxy"]) > float(
        agreement["COD_template_given_source"]["cs_binary_proxy"]
    )
    assert float(agreement["COD_endpoint_given_source"]["kernel_proxy_bw_0p10"]) > float(
        agreement["COD_template_given_source"]["kernel_proxy_bw_0p10"]
    )
    nulls = {r["null_id"]: r for r in _rows("null_calibration_summary.csv")}
    assert nulls["N3_trajectory_block_shuffle"]["passes"] == "1"
    assert float(nulls["N3_trajectory_block_shuffle"]["observed_value"]) == 0.9444444444444444
    assert float(nulls["N3_trajectory_block_shuffle"]["null_p95"]) == 0.7712962962962961
    assert nulls["N4_template_only_vs_max_null"]["passes"] == "0"
    assert float(nulls["N4_template_only_vs_max_null"]["observed_value"]) == 0.7037037037037037


def test_c62_screening_and_source_adversary_close_escape_hatches():
    screening = {r["condition_set"]: r for r in _rows("screening_off_summary.csv")}
    assert screening["source_plus_template"]["screens_off_endpoint"] == "0"
    assert float(screening["source_plus_template"]["endpoint_remaining_gain"]) == (
        0.9444444444444444 - 0.7037037037037037
    )
    assert screening["endpoint"]["screens_off_endpoint"] == "1"
    adversary = {r["candidate_id"]: r for r in _rows("source_observable_adversary_summary.csv")}
    assert len(adversary) == 5
    assert {r["reliable_escape_hatch"] for r in adversary.values()} == {"0"}
    assert adversary["SADV62-2"]["uses_source_only_inputs"] == "1"
    assert float(adversary["SADV62-2"]["hit"]) == 0.5740740740740741
    assert adversary["SADV62-4"]["hit"] == ""


def test_c62_synthetic_rank_gauge_estimator_grid_keeps_c60_repair():
    rows = _rows("synthetic_rank_gauge_estimator_grid.csv")
    candidate = [r for r in rows if r["scenario"] == "candidate_specific_gauge"]
    common = [r for r in rows if r["scenario"] == "target_local_common_offset"]
    assert len(candidate) == 6
    assert len(common) == 5
    assert any(r["pair_flip_possible"] == "1" for r in candidate)
    assert {r["pair_flip_possible"] for r in common} == {"0"}
    assert {r["expected_behavior_pass"] for r in rows} == {"1"}
    assert float([r for r in candidate if r["gauge_scale"] == "0.0"][0]["source_error"]) < float(
        [r for r in candidate if r["gauge_scale"] == "2.0"][0]["source_error"]
    )


def test_c62_instrumentation_gate_and_red_team_artifacts_are_clean():
    gate = {r["need_id"]: r for r in _rows("instrumentation_gate.csv")}
    assert len(gate) == 7
    assert {r["authorized_in_c62"] for r in gate.values()} == {"0"}
    assert gate["IG1"]["gate_decision"] == "TRIAL_LEVEL_CACHE_NEEDED_BUT_NOT_AUTHORIZED"
    assert gate["IG6"]["gate_decision"] == c62.TRAINING_GATE
    red = {r["gate"]: r for r in _rows("red_team_failure_ledger.csv")}
    assert len(red) == 13
    assert {r["failed"] for r in red.values()} == {"0"}
    forbidden = _rows("forbidden_claim_scan.csv")
    assert len(forbidden) == 21
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    tests = _rows("test_command_manifest.csv")
    assert {r["status"] for r in tests} <= {"planned", "green"}


def test_c62_manifest_hashes_and_large_artifact_scan_pass():
    large = _rows("large_artifact_scan.csv")
    assert len(large) == 20
    assert {r["passed"] for r in large} == {"1"}
    manifest = _rows("artifact_manifest.csv")
    assert len(manifest) == 20
    for row in manifest:
        assert os.path.exists(row["path"])
        assert int(row["size_bytes"]) == os.path.getsize(row["path"])
        assert row["sha256"] == _sha256(row["path"])
        if row["artifact_class"] == "table":
            assert row["row_count"] != ""


def test_c62_run_recomputes_core_decision_without_writing():
    res = c62.run(test_status="unit")
    assert res["decision"]["primary"] == "C62-A_C61_ladder_reproduced"
    assert res["decision"]["training_gate"] == c62.TRAINING_GATE
    assert len(res["partition_cod_sensitivity_rows"]) == 45
    assert all(int(r["stable_order"]) for r in res["partition_cod_sensitivity_rows"])
