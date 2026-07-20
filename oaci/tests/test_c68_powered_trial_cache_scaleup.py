"""C68 powered trial-cache scale-up readiness tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c68_powered_trial_cache_scaleup as c68
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C68_POWERED_TRIAL_CACHE_SCALEUP.json"
TABLE_DIR = "oaci/reports/c68_tables"


def _summary() -> dict:
    with open(REPORT_JSON) as f:
        return json.load(f)


def _rows(name: str) -> list[dict]:
    with open(os.path.join(TABLE_DIR, name), newline="") as f:
        return list(csv.DictReader(f))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def test_c68_taxonomy_lock_and_noauth_gate():
    assert c68._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert set(c68.DECISIONS) == {
        "C68-A_c67_dual_mode_cache_contract_replayed",
        "C68-B_scaleup_plan_powered_and_manifested",
        "C68-C_reinference_only_scaleup_ready_but_not_authorized",
        "C68-D_reinference_only_scaleup_authorized_and_executed",
        "C68-E_scaled_trial_cache_integrity_validated",
        "C68-F_masked_view_contract_validated_at_scale",
        "C68-G_split_label_powered_diagnostic_completed_not_sufficiency",
        "C68-H_split_label_still_underpowered_or_unstable",
        "C68-I_sample_level_conditional_cs_feasible_at_scale",
        "C68-J_sample_level_conditional_cs_still_unstable_or_proxy_only",
        "C68-K_endpoint_oracle_boundary_preserved",
        "C68-L_trial_level_source_observable_escape_hatch_found",
        "C68-M_no_trial_level_source_observable_escape_hatch_found",
        "C68-N_larger_reinference_only_campaign_needed_but_not_authorized",
        "C68-O_new_training_still_not_justified",
        "C68-P_claim_or_availability_violation_found",
    }
    d = _summary()
    assert d["milestone"] == "C68"
    assert d["authorization_present"] is False
    assert d["diagnostic_only_non_deployable"] is True
    assert d["decision"]["primary"] == "C68-C_reinference_only_scaleup_ready_but_not_authorized"
    assert d["final_gate"] == "SCALEUP_READY_BUT_NOT_AUTHORIZED"
    assert d["decision"]["red_team_failure_count"] == 0
    for active in (
        "C68-A_c67_dual_mode_cache_contract_replayed",
        "C68-B_scaleup_plan_powered_and_manifested",
        "C68-C_reinference_only_scaleup_ready_but_not_authorized",
        "C68-K_endpoint_oracle_boundary_preserved",
        "C68-N_larger_reinference_only_campaign_needed_but_not_authorized",
        "C68-O_new_training_still_not_justified",
    ):
        assert active in d["decision"]["active"]
    for inactive in (
        "C68-D_reinference_only_scaleup_authorized_and_executed",
        "C68-E_scaled_trial_cache_integrity_validated",
        "C68-G_split_label_powered_diagnostic_completed_not_sufficiency",
        "C68-I_sample_level_conditional_cs_feasible_at_scale",
        "C68-L_trial_level_source_observable_escape_hatch_found",
        "C68-M_no_trial_level_source_observable_escape_hatch_found",
        "C68-P_claim_or_availability_violation_found",
    ):
        assert inactive in d["decision"]["inactive"]


def test_c68_authorization_function_requires_explicit_user_text_not_constant_presence():
    assert c68._auth_present("") is False
    assert c68._auth_present("planning only; do not execute") is False
    assert c68._auth_present(f"handoff says `{c68.AUTH_PHRASE}` is required") is False
    assert c68._auth_present(c68.AUTH_PHRASE) is True


def test_c68_replays_c67_dual_mode_and_c66_cache_hash():
    replay = {r["check"]: r for r in _rows("c68_c67_replay_dual_mode_ledger.csv")}
    assert replay["c67_commit"]["observed"] == "9f8c829"
    assert replay["c67_final_gate"]["observed"] == "C67_DUAL_MODE_MICROCACHE_VALID_BUT_UNDERPOWERED_FOR_SPLIT_LABEL_CS"
    assert replay["no_auth_forward"]["observed"] == "0"
    assert replay["no_auth_cache_rows"]["observed"] == "0"
    assert replay["authorized_forward"]["observed"] == "1"
    assert replay["authorized_cache_rows"]["observed"] == "3456"
    assert replay["authorized_cache_sha256"]["expected"] == replay["authorized_cache_sha256"]["observed"]
    assert replay["raw_cache_git_tracked"]["observed"] == "0"
    assert {r["passed"] for r in replay.values()} == {"1"}


def test_c68_replays_c65_frozen_universe_and_deduplicates_forward_units():
    universe = {r["metric"]: r for r in _rows("c68_frozen_universe_replay.csv")}
    assert universe["c65_logical_singleton_rows"]["value"] == "3804"
    assert universe["c65_unique_checkpoint_ids"]["value"] == "1268"
    assert universe["c65_physical_forward_units"]["value"] == "1268"
    assert universe["targets"]["value"] == "1;2;3;4;5;6;7;8;9"
    assert universe["seeds"]["value"] == "0;1;2"
    assert universe["levels"]["value"] == "0;1"
    assert universe["verified_pt_json_rows"]["value"] == "3804"
    assert {r["passed"] for r in universe.values()} == {"1"}


def test_c68_scaleup_ladder_is_power_plan_only_and_preserves_holdouts():
    ladder = {r["rung"]: r for r in _rows("c68_power_ladder.csv")}
    assert ladder["T0_micro_replay"]["independent_physical_forward_units"] == "6"
    assert ladder["T0_micro_replay"]["estimated_trial_rows"] == "3456"
    assert ladder["T1_pilot_scale"]["independent_physical_forward_units"] == "64"
    assert ladder["T2_medium_scale"]["independent_physical_forward_units"] == "216"
    assert ladder["T3_full_physical_dedup"]["independent_physical_forward_units"] == "1268"
    assert ladder["T3_full_physical_dedup"]["estimated_trial_rows"] == "730368"
    assert {r["claim_allowed_now"] for r in ladder.values()} == {"readiness_only_not_authorized"}

    plan = _rows("c68_scaleup_sampling_plan.csv")
    assert len(plan) == 58
    assert {r["authorized_to_execute"] for r in plan} == {"0"}
    assert {r["forward_attempted_in_c68"] for r in plan} == {"0"}
    assert {r["reserved_seed_used"] for r in plan} == {"0"}
    assert {r["bnci004_used"] for r in plan} == {"0"}
    assert all("performance" not in r["selection_rule"].replace("nonperformance", "") for r in plan)

    size = {r["rung"]: r for r in _rows("c68_expected_cache_size.csv")}
    assert size["T3_full_physical_dedup"]["external_only"] == "1"
    assert size["T3_full_physical_dedup"]["raw_cache_committable_to_git"] == "0"
    assert float(size["T3_full_physical_dedup"]["estimated_cache_mib"]) > 400.0


def test_c68_noauth_execution_gate_blocks_forward_training_gpu_and_raw_cache():
    gate = {r["gate"]: r for r in _rows("c68_noauth_execution_gate.csv")}
    assert gate["authorization_phrase_present"]["observed"] == "0"
    assert gate["new_forward_or_reinference"]["allowed"] == "0"
    assert gate["new_forward_or_reinference"]["observed"] == "0"
    assert gate["training_or_gradient_update"]["observed"] == "0"
    assert gate["gpu_use"]["observed"] == "0"
    assert gate["raw_cache_written"]["observed"] == "0"
    assert gate["selector_or_checkpoint_recommendation"]["observed"] == "0"
    assert {r["passed"] for r in gate.values()} == {"1"}


def test_c68_masking_dryrun_preserves_endpoint_oracle_boundary():
    views = {r["view"]: r for r in _rows("c68_masked_view_contract.csv")}
    assert views["source_only_view"]["label_visible_rows_c67"] == "0"
    assert views["source_only_view"]["prediction_visible_rows_c67"] == "0"
    assert views["source_only_view"]["selection_path_enforced"] == "1"
    assert views["same_label_oracle_view"]["available_at_selection_time"] == "0"
    assert views["same_label_oracle_view"]["diagnostic_only"] == "1"
    assert views["same_label_oracle_view"]["policy_boundary_only"] == "1"
    assert views["conditional_cs_diagnostic_view"]["policy_boundary_only"] == "1"
    assert {r["status"] for r in views.values()} == {"pass"}

    matrix = {r["column_family"]: r for r in _rows("c68_view_column_access_matrix.csv")}
    assert matrix["target_label"]["source_only_view"] == "masked"
    assert matrix["same_label_endpoint_scalar"]["source_only_view"] == "forbidden"
    assert matrix["target_joint_margin_raw"]["source_only_view"] == "forbidden"

    leakage = _rows("c68_label_leakage_redteam.csv")
    assert leakage
    assert {r["passed"] for r in leakage} == {"1"}


def test_c68_split_label_cs_and_source_adversary_are_plans_not_results():
    split = _rows("c68_split_label_power_summary.csv")
    assert split
    assert {r["status"] for r in split} == {"not_run_not_authorized"}
    assert {"few_label_sufficiency", "deployable_selector", "source_only_rescue"} <= {r["forbidden_claim"] for r in split}

    cs = _rows("c68_sample_level_cs_feasibility.csv")
    assert cs
    assert {r["status"] for r in cs} == {"not_run_not_authorized"}
    assert {r["full_cs_claim_allowed_now"] for r in cs} == {"0"}

    avail = {r["variable"]: r for r in _rows("c68_sample_level_cs_availability_ledger.csv")}
    assert avail["X1_source_only"]["available_at_selection_time"] == "1"
    assert avail["X1_source_only"]["uses_target_labels"] == "0"
    assert avail["X2_same_label_endpoint_scalar"]["available_at_selection_time"] == "0"
    assert avail["Y_heldout_eval_response"]["uses_eval_labels"] == "1"

    adv = _rows("c68_source_observable_adversary_plan.csv")
    assert adv
    assert {r["status"] for r in adv} == {"planned_not_run_not_authorized"}
    assert {r["escape_hatch_found"] for r in adv} == {"0"}


def test_c68_synthetic_writer_and_artifact_hygiene_are_clean():
    synth = _rows("c68_synthetic_cache_writer_dryrun.csv")
    assert len(synth) == 2
    assert {r["raw_eeg_cache"] for r in synth} == {"0"}
    assert {r["passed"] for r in synth} == {"1"}
    assert all(len(r["sha256"]) == 64 for r in synth)

    red = _rows("red_team_failure_ledger.csv")
    assert red
    assert {r["failed"] for r in red} == {"0"}

    forbidden = _rows("forbidden_claim_scan.csv")
    assert forbidden
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}

    large = _rows("large_artifact_scan.csv")
    assert large
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}

    manifest = _rows("artifact_manifest.csv")
    assert manifest
    for row in manifest:
        assert os.path.exists(row["path"])
        assert _sha256(row["path"]) == row["sha256"]
    assert "oaci/reports/C68_POWERED_TRIAL_CACHE_SCALEUP.md" in {r["path"] for r in manifest}
    assert "oaci/reports/C68_POWERED_TRIAL_CACHE_SCALEUP.json" in {r["path"] for r in manifest}

    summary = _summary()
    emitted_csvs = {
        os.path.splitext(name)[0]
        for name in os.listdir(TABLE_DIR)
        if name.endswith(".csv") and name not in {"large_artifact_scan.csv"}
    }
    assert emitted_csvs <= set(summary["table_row_counts"])
