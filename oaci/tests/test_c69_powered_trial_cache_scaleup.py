"""C69 powered trial-cache scale-up tests."""
from __future__ import annotations

import csv
import hashlib
import json
import os

from oaci.conditioned_ceiling_coverage import c69_powered_trial_cache_scaleup as c69
from oaci.conditioned_ceiling_coverage import schema as c49_schema


REPORT_JSON = "oaci/reports/C69_POWERED_TRIAL_CACHE_SCALEUP.json"
TABLE_DIR = "oaci/reports/c69_tables"


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


def test_c69_taxonomy_and_exact_cli_authorization_contract():
    assert c69._lock_config() == c49_schema.LOCKED_C19_CONFIG_HASH == "664007686afb520f"
    assert c69._auth_present("") is False
    assert c69._auth_present(f"handoff mentions {c69.AUTH_TOKEN}") is False
    assert c69._auth_present(f" {c69.AUTH_TOKEN} ") is True
    assert set(c69.DECISIONS) == {
        "C69-A_authorized_t1_reinference_cache_executed_and_manifested",
        "C69-B_authorized_t2_reinference_cache_executed_and_manifested",
        "C69-C_split_label_diagnostic_stable_but_not_sufficiency",
        "C69-D_cache_valid_but_split_label_still_underpowered",
        "C69-E_sample_level_conditional_cs_still_underpowered_or_unstable",
        "C69-F_sample_level_conditional_cs_feasible_but_diagnostic_only",
        "C69-G_endpoint_oracle_boundary_preserved",
        "C69-H_trial_level_source_observable_escape_hatch_found",
        "C69-I_no_trial_level_source_observable_escape_hatch_found",
        "C69-J_larger_t3_campaign_ready_but_not_authorized",
        "C69-K_reinference_blocked_by_abi_preprocess_or_data_contract",
        "C69-L_label_masking_or_availability_violation_found",
        "C69-M_new_training_still_not_justified",
        "C69-N_new_training_required_but_not_authorized",
        "C69-O_no_forward_readiness_only_due_missing_authorization",
    }

    d = _summary()
    assert d["milestone"] == "C69"
    assert d["diagnostic_only_non_deployable"] is True
    assert d["authorization_present"] is True
    assert d["decision"]["red_team_failure_count"] == 0
    assert d["key_numbers"]["t1_units"] == 64
    assert d["key_numbers"]["t2_units"] == 216
    assert d["key_numbers"]["t3_units_not_authorized"] == 1268
    assert "C69-A_authorized_t1_reinference_cache_executed_and_manifested" in d["decision"]["active"]
    assert "C69-B_authorized_t2_reinference_cache_executed_and_manifested" in d["decision"]["active"]
    assert "C69-G_endpoint_oracle_boundary_preserved" in d["decision"]["active"]
    assert "C69-J_larger_t3_campaign_ready_but_not_authorized" in d["decision"]["active"]
    assert "C69-M_new_training_still_not_justified" in d["decision"]["active"]


def test_c69_authorization_audit_and_stage_manifest():
    auth = {r["gate"]: r for r in _rows("c69_authorization_audit.csv")}
    assert auth["exact_cli_token"]["observed"] == "1"
    assert auth["protocol_text_authorization"]["observed"] == "0"
    assert auth["environment_variable_authorization"]["observed"] == "0"
    assert auth["t1_scope_authorized"]["passed"] == "1"
    assert auth["t2_scope_conditionally_authorized"]["passed"] == "1"
    assert auth["t3_scope_authorized"]["observed"] == "0"

    stage = {r["stage"]: r for r in _rows("c69_stage_manifest.csv")}
    assert stage["t1"]["executed"] == "1"
    assert stage["t1"]["success"] == "1"
    assert stage["t1"]["forward_units"] == "64"
    assert stage["t1"]["trial_rows"] == "36864"
    assert stage["t2"]["executed"] == "1"
    assert stage["t2"]["success"] == "1"
    assert stage["t2"]["forward_units"] == "216"
    assert stage["t2"]["trial_rows"] == "124416"
    assert stage["t3"]["authorized_to_execute"] == "0"
    assert stage["t3"]["executed"] == "0"


def test_c69_external_cache_manifests_are_content_addressed_and_external():
    for name, expected_rows in (("c69_cache_manifest_t1.csv", "36864"), ("c69_cache_manifest_t2.csv", "124416")):
        manifest = {r["cache_kind"]: r for r in _rows(name)}
        cache = manifest["minimal_logits_probs_metadata"]
        assert cache["exists"] == "1"
        assert cache["row_count"] == expected_rows
        assert cache["row_count"] == cache["manifest_row_count"]
        assert cache["git_tracked"] == "0"
        assert cache["sha256_match"] == "1"
        assert os.path.exists(cache["external_path"])
        assert _sha256(cache["external_path"]) == cache["sha256"]
        assert "cache_sha256_" in cache["external_path"]
        assert cache["sha256"][:16] in cache["external_path"]

        ext_manifest = manifest["manifest"]
        assert ext_manifest["exists"] == "1"
        assert ext_manifest["git_tracked"] == "0"
        assert ext_manifest["sha256_match"] == "1"
        assert os.path.exists(ext_manifest["external_path"])
        assert _sha256(ext_manifest["external_path"]) == ext_manifest["sha256"]


def test_c69_schema_and_masking_contracts_pass():
    sig = _rows("c69_schema_signature.csv")
    assert {r["stage"] for r in sig} == {"t1", "t2"}
    assert {r["required_minimum_present"] for r in sig} == {"1"}
    assert {r["status"] for r in sig} == {"pass"}

    views = {(r["stage"], r["view"]): r for r in _rows("c69_masked_view_contract.csv")}
    for stage, expected_rows in (("t1", "36864"), ("t2", "124416")):
        assert views[(stage, "source_only_view")]["sampled_rows"] == expected_rows
        assert views[(stage, "source_only_view")]["label_visible_rows"] == "0"
        assert views[(stage, "source_only_view")]["prediction_visible_rows"] == "0"
        assert views[(stage, "target_construction_view")]["eval_label_visible_rows"] == "0"
        assert views[(stage, "target_evaluation_view")]["construct_label_visible_rows"] == "0"
        assert views[(stage, "same_label_oracle_view")]["available_at_selection_time"] == "0"
        assert views[(stage, "same_label_oracle_view")]["policy_boundary_only"] == "1"
        assert views[(stage, "conditional_cs_diagnostic_view")]["policy_boundary_only"] == "1"
    assert {r["status"] for r in views.values()} == {"pass"}


def test_c69_split_label_cs_proxy_and_source_adversary_are_diagnostic_only():
    split = {r["stage"]: r for r in _rows("c69_split_label_summary.csv")}
    assert split["t1"]["independent_checkpoint_units"] == "64"
    assert split["t2"]["independent_checkpoint_units"] == "216"
    assert split["t1"]["few_label_sufficiency_claimed"] == "0"
    assert split["t2"]["few_label_sufficiency_claimed"] == "0"
    assert split["t2"]["status"] in {"stable_diagnostic_not_sufficiency", "valid_but_underpowered_or_unstable"}

    cs = {r["stage"]: r for r in _rows("c69_conditional_cs_summary.csv")}
    assert int(cs["t1"]["paired_eval_rows"]) > 18_000
    assert int(cs["t2"]["paired_eval_rows"]) > 64_000
    assert cs["t1"]["full_conditional_cs_claimed"] == "0"
    assert cs["t2"]["full_conditional_cs_claimed"] == "0"
    assert cs["t2"]["status"] in {"feasible_proxy_diagnostic_only", "underpowered_or_unstable"}

    adv = _rows("c69_source_adversary_summary.csv")
    assert adv
    assert {r["target_labels_used"] for r in adv} == {"0"}
    assert {r["source_domain_trial_logits_available"] for r in adv} == {"0"}
    assert {r["escape_hatch_found"] for r in adv} == {"0"}


def test_c69_endpoint_boundary_red_team_and_artifact_hygiene_pass():
    endpoint = {r["boundary"]: r for r in _rows("c69_endpoint_boundary_replay.csv")}
    assert endpoint["template_only_transfer"]["observed_hit"] == "0.7037037037037037"
    assert endpoint["template_only_transfer"]["beats_null"] == "0"
    assert endpoint["same_label_endpoint_scalar"]["observed_hit"] == "0.9444444444444444"
    assert endpoint["same_label_endpoint_scalar"]["beats_null"] == "1"
    assert endpoint["same_label_endpoint_scalar"]["available_at_selection_time"] == "0"
    assert endpoint["same_label_endpoint_scalar"]["diagnostic_only"] == "1"

    resources = _rows("c69_resource_runtime_summary.csv")
    assert {r["cpu_only"] for r in resources} == {"1"}
    assert {r["gpu_used"] for r in resources} == {"0"}
    assert {r["training_attempted"] for r in resources} == {"0"}

    red = _rows("red_team_failure_ledger.csv")
    assert red
    assert {r["failed"] for r in red} == {"0"}

    forbidden = _rows("c69_forbidden_claim_scan.csv")
    assert forbidden
    assert {r["affirmative_hits"] for r in forbidden} == {"0"}
    assert {r["passed"] for r in forbidden} == {"1"}

    large = _rows("c69_large_artifact_scan.csv")
    assert large
    assert {r["over_50mb"] for r in large} == {"0"}
    assert {r["passed"] for r in large} == {"1"}

    artifact = _rows("c69_artifact_manifest.csv")
    assert artifact
    for row in artifact:
        assert os.path.exists(row["path"])
        assert _sha256(row["path"]) == row["sha256"]
    assert "oaci/reports/C69_POWERED_TRIAL_CACHE_SCALEUP.md" in {r["path"] for r in artifact}
    assert "oaci/reports/C69_POWERED_TRIAL_CACHE_SCALEUP.json" in {r["path"] for r in artifact}
