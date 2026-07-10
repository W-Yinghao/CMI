"""Independent C74 artifact, leakage, and claim-boundary red-team gauntlet."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

import numpy as np

from . import c74_analysis as analysis
from . import c74_cache as cache
from . import c74_t2_source_wz_instrumentation as runner


REPORT_DIR = Path("oaci/reports")
TABLE_DIR = REPORT_DIR / "c74_tables"
RED_TEAM_REPORT = REPORT_DIR / "C74_RED_TEAM_VERIFICATION.md"


def _read_csv(name: str) -> list[dict]:
    with open(TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(name: str, rows: list[dict]) -> None:
    columns = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(TABLE_DIR / name, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: str | Path) -> str:
    return cache.sha256_file(path)


def _self_hash_valid(payload: dict) -> bool:
    expected = payload.get("manifest_sha256", "")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    observed = cache.sha256_text(json.dumps(body, sort_keys=True, separators=(",", ":")))
    return expected == observed


def _attempt_rows() -> list[dict]:
    return [
        {"attempt": 1, "slurm_job": "892076", "phase": "P0_pilot_initial", "status": "aborted_cancelled", "real_EEG_loaded": "partial", "model_forward_attempted": "at_least_3_tasks", "valid_unit_manifests": 0, "P1_started": 0, "superseded": 1, "reason": "shared_MNE_home_stale_lock_and_extra_numpy_torch_softmax_cross_implementation_gate"},
        {"attempt": 2, "slurm_job": "892090", "phase": "P0_pilot_repaired", "status": "passed", "real_EEG_loaded": 1, "model_forward_attempted": 1, "valid_unit_manifests": 54, "P1_started": 0, "superseded": 0, "reason": "per_job_MNE_home_and_protocol_aligned_softmax_identity"},
        {"attempt": 3, "slurm_job": "892103", "phase": "P0_payload_validation", "status": "passed", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 54, "P1_started": 0, "superseded": 0, "reason": "all_payloads_rehashed"},
        {"attempt": 4, "slurm_job": "892105", "phase": "P1_expansion", "status": "passed", "real_EEG_loaded": 1, "model_forward_attempted": 1, "valid_unit_manifests": 162, "P1_started": 1, "superseded": 0, "reason": "self_hashed_P0_gate_verified_by_every_task"},
        {"attempt": 5, "slurm_job": "892114", "phase": "P1_payload_validation", "status": "passed", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 162, "P1_started": 1, "superseded": 0, "reason": "all_payloads_rehashed"},
        {"attempt": 6, "slurm_job": "892115", "phase": "analysis_prepare", "status": "blocked_before_smoke", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 216, "P1_started": 1, "superseded": 1, "reason": "two_dataset_evidence_hashes_required_cross_node_drift_audit"},
        {"attempt": 7, "slurm_job": "892119;892121", "phase": "preprocess_replay_initial", "status": "failed_before_data_load", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 216, "P1_started": 1, "superseded": 1, "reason": "tensor_content_hash_import_path_repaired"},
        {"attempt": 8, "slurm_job": "892128;892129;892131", "phase": "cross_node_preprocess_replay", "status": "passed", "real_EEG_loaded": 1, "model_forward_attempted": "one_frozen_T2_checkpoint_pair", "valid_unit_manifests": 216, "P1_started": 1, "superseded": 0, "reason": "cross_node_input_and_downstream_drift_below_locked_tolerances"},
        {"attempt": 9, "slurm_job": "892135", "phase": "restricted_view_prepare", "status": "blocked_before_smoke", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 216, "P1_started": 1, "superseded": 1, "reason": "oracle_row_count_metadata_removed_from_restricted_manifest"},
        {"attempt": 10, "slurm_job": "892140", "phase": "smoke_analysis_initial", "status": "completed_then_superseded_by_red_team", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 216, "P1_started": 1, "superseded": 1, "reason": "cumulative_null_did_not_test_incremental_feature_block"},
        {"attempt": 11, "slurm_job": "892144", "phase": "smoke_analysis_nested_null", "status": "passed", "real_EEG_loaded": 0, "model_forward_attempted": 0, "valid_unit_manifests": 216, "P1_started": 1, "superseded": 0, "reason": "within_target_new_block_permutation_with_prior_blocks_fixed"},
    ]


def _check(checks: list[dict], name: str, passed: bool, observed, expected, blocking: bool = True) -> None:
    checks.append({
        "check": name, "passed": int(bool(passed)), "blocking": int(blocking),
        "observed": observed, "expected": expected,
    })


def run_red_team() -> dict:
    if (REPORT_DIR / "C74_T2_SOURCE_WZ_INSTRUMENTATION.md").exists():
        raise RuntimeError("C74 main report exists before red-team completion")
    protocol = cache.load_locked_protocol()
    checks: list[dict] = []
    attempts = _attempt_rows()
    _write_csv("execution_attempt_ledger.csv", attempts)

    _check(checks, "protocol_hash", cache.sha256_file(cache.PROTOCOL_PATH) == Path(cache.PROTOCOL_SHA_PATH).read_text().strip(), cache.sha256_file(cache.PROTOCOL_PATH), Path(cache.PROTOCOL_SHA_PATH).read_text().strip())
    protocol_commit_time = int(subprocess.check_output(["git", "show", "-s", "--format=%ct", "1f3ab88"], text=True).strip())
    all_manifests = analysis._all_manifests()
    earliest_cache_mtime = min(Path(shard["path"]).stat().st_mtime for manifest in all_manifests for shard in manifest["shards"])
    _check(checks, "protocol_commit_precedes_cache", protocol_commit_time < earliest_cache_mtime, protocol_commit_time, f"< {earliest_cache_mtime}")

    t2_ids, t3_ids = cache.locked_unit_sets()
    observed_ids = {manifest["checkpoint_id"] for manifest in all_manifests}
    _check(checks, "T2_exact_universe", observed_ids == t2_ids, len(observed_ids), 216)
    _check(checks, "T3_HO_zero_overlap", not (observed_ids & t3_ids), len(observed_ids & t3_ids), 0)
    locked_holdout = cache.read_csv(cache.T3_HO_MANIFEST_PATH)
    _check(checks, "T3_HO_generation_flags_zero", all(row["z_Wz_generated_in_C74"] == "0" for row in locked_holdout), sum(int(row["z_Wz_generated_in_C74"]) for row in locked_holdout), 0)

    for stage, expected_count, expected_gate in (
        ("P0_pilot", 54, "P0_PILOT_ALL_GATES_PASSED"),
        ("P1_expansion", 162, "P1_EXPANSION_ALL_GATES_PASSED"),
    ):
        gate = json.loads(cache.stage_gate_path(protocol, stage).read_text())
        _check(checks, f"{stage}_gate_self_hash", _self_hash_valid(gate), _self_hash_valid(gate), True)
        _check(checks, f"{stage}_gate", gate["validated_units"] == expected_count and gate["final_gate"] == expected_gate, f"{gate['validated_units']}:{gate['final_gate']}", f"{expected_count}:{expected_gate}")
        _check(checks, f"{stage}_payload_rehash_recorded", bool(gate["payloads_rehashed"]), gate["payloads_rehashed"], True)

    failed_payloads = 0
    identity_maxima = defaultdict_float()
    registered_identity_fields = {
        "Wz_plus_b_logits_max_abs", "Wz_plus_b_logits_max_relative",
        "softmax_probability_max_abs", "hook_z_max_abs",
        "repeat_logits_max_abs", "repeat_z_max_abs",
    }
    partition_failures = 0
    total_source_rows = 0
    total_target_rows = 0
    for manifest in all_manifests:
        try:
            verified = cache.verify_unit_manifest(
                cache.unit_directory(protocol, manifest["stage"], manifest["target_id"], manifest["unit_id"]) / "unit_manifest.json",
                rehash_payloads=True,
            )
            for shard in verified["shards"]:
                cache.verify_shard(shard, required_fields=runner.SHARD_SCHEMAS[shard["kind"]])
        except Exception:
            failed_payloads += 1
            continue
        total_source_rows += int(manifest["source_rows"])
        total_target_rows += int(manifest["target_unlabeled_rows"])
        for key, value in manifest["identity"].items():
            if key in registered_identity_fields:
                identity_maxima[key] = max(identity_maxima[key], abs(float(value)))
        construction = _load_label_shard(manifest, "target_construction_labels")
        evaluation = _load_label_shard(manifest, "target_evaluation_labels")
        construct_ids = set(map(str, construction["target_trial_id"]))
        eval_ids = set(map(str, evaluation["target_trial_id"]))
        if construct_ids & eval_ids or len(construct_ids | eval_ids) != 576:
            partition_failures += 1
    _check(checks, "independent_full_payload_rehash", failed_payloads == 0, failed_payloads, 0)
    _check(checks, "label_partition_all_units", partition_failures == 0, partition_failures, 0)
    _check(checks, "row_counts", total_source_rows == 995328 and total_target_rows == 124416, f"{total_source_rows}:{total_target_rows}", "995328:124416")
    _check(checks, "identity_exact", max(identity_maxima.values(), default=0.0) == 0.0, max(identity_maxima.values(), default=0.0), 0.0)
    _check(checks, "execution_guards", all(not manifest["execution"]["GPU_used"] and not manifest["execution"]["training_attempted"] and not manifest["execution"]["parameter_updates"] and manifest["execution"]["model_eval"] and not manifest["execution"]["gradients_enabled"] for manifest in all_manifests), "216 unit manifests", "CPU eval no-grad no-update")

    pointer_path = cache.run_root(protocol) / "views" / analysis.PRIMARY_SMOKE_POINTER
    pointer = json.loads(pointer_path.read_text())
    restricted_body = Path(pointer["path"]).read_text()
    _check(checks, "restricted_input_hash", cache.sha256_file(pointer["path"]) == pointer["sha256"], cache.sha256_file(pointer["path"]), pointer["sha256"])
    _check(checks, "oracle_absent_primary_input", "same_label_oracle" not in restricted_body, "same_label_oracle" in restricted_body, False)
    restricted = json.loads(restricted_body)
    expected_kinds = {"checkpoint_Wb", "strict_source_trial", "target_unlabeled_representation", "target_construction_labels", "target_evaluation_labels"}
    _check(checks, "restricted_view_set", len(restricted["units"]) == 216 and all({item["kind"] for item in unit["shards"]} == expected_kinds for unit in restricted["units"]), len(restricted["units"]), "216 units x 5 allowed views")

    comparison_path = cache.run_root(protocol) / "preprocessing_cross_node_replay" / "cross_node_preprocessing_comparison.json"
    comparison = cache.verify_unit_manifest(comparison_path, rehash_payloads=False)
    _check(checks, "cross_node_preprocessing_drift", comparison["passed"] and comparison["prediction_disagreements"] == 0, f"input_max={comparison['input_max_abs']};logit_max={comparison['logit_max_abs']};pred_disagree={comparison['prediction_disagreements']}", "within locked tolerances; zero prediction disagreements")

    state = json.loads((REPORT_DIR / "C74_T2_SOURCE_WZ_ANALYSIS_STATE.json").read_text())
    _check(checks, "analysis_information_boundary", not state["same_label_oracle_used_by_primary_smoke"] and not state["T3_HO_z_Wz_touched"] and not state["representation_mechanism_claimed"] and not state["strict_source_escape_hatch_claimed"], "oracle=0;T3=0;mechanism=0;escape=0", "all false")
    incremental = _read_csv("incremental_prediction_feasibility.csv")
    _check(checks, "nested_incremental_null_semantics", all(row["null_scheme"] == "permute_new_block_within_target_keep_prior_blocks_and_outcome_fixed" and "incremental_exceeds_null_p95" in row and "target_blocked_null_R2_p95" not in row for row in incremental), len(incremental), "5 rows with nested new-block null")
    nonconstruction = [row for row in incremental if row["model"] != "plus_construction_summaries"]
    _check(checks, "no_source_or_unlabeled_incremental_escape", all(row["incremental_exceeds_null_p95"] == "0" for row in nonconstruction), [row["model"] for row in nonconstruction if row["incremental_exceeds_null_p95"] != "0"], [])

    variance = _read_csv("target_common_candidate_projection_variance.csv")
    _check(checks, "projection_variance_accounting", len(variance) == 36 and max(abs(float(row["accounting_sum"]) - 1.0) for row in variance) < 1e-10, len(variance), "36 rows; sum=1")
    stability = _read_csv("projection_split_stability_smoke.csv")
    _check(checks, "projection_split_table", len(stability) == 36 and all(row["same_label_oracle_used"] == "0" for row in stability), len(stability), "36; oracle=0")
    counterfactual = _read_csv("projection_counterfactual_feasibility.csv")
    original = [row for row in counterfactual if row["counterfactual"] == "I0_original"]
    _check(checks, "counterfactual_identity", len(original) == 9 and all(float(row["pairwise_rank_flip_fraction"]) == 0 and int(row["top1_agreement"]) == 1 and float(row["original_Wz_plus_b_vs_stored_logits_max_abs"]) == 0 for row in original), len(original), "9 exact I0 controls")

    risks = _read_csv("risk_register.csv")
    _check(checks, "risk_register_no_blocker", all(row["blocking"] == "0" for row in risks), [row["risk"] for row in risks if row["blocking"] != "0"], [])
    semantics = _read_csv("c73_attribution_metric_semantics.csv")
    _check(checks, "C73_metric_semantics_reconciled", all(row["reconciled"] == "1" and row["schema_error"] == "0" for row in semantics), len(semantics), "all reconciled")

    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    _check(checks, "raw_cache_not_in_git", not any("oaci-c74-t2-source-wz" in path for path in tracked), 0, 0)
    report_paths = list(REPORT_DIR.glob("C74_*")) + list(TABLE_DIR.glob("*.csv"))
    max_size = max((path.stat().st_size for path in report_paths), default=0)
    _check(checks, "git_payload_hygiene", max_size < 50_000_000, max_size, "<50000000")
    slurm_source = Path("oaci/slurm_c74_t2_source_wz.sh").read_text()
    _check(checks, "authorization_direct_CLI", "--authorization-token C74_T2_SOURCE_WZ_REINFERENCE_AUTHORIZED" in slurm_source, "direct literal CLI", "direct literal CLI")
    _check(checks, "forbidden_dataset_and_seeds_absent", all(manifest["seed"] in {0, 1, 2} for manifest in all_manifests) and "BNCI2014_004" not in json.dumps(all_manifests), sorted({manifest["seed"] for manifest in all_manifests}), [0, 1, 2])
    _check(checks, "main_report_not_preexisting", True, False, False)

    passed = all(int(row["passed"]) or not int(row["blocking"]) for row in checks)
    _write_csv("red_team_checks.csv", checks)
    lines = [
        "# C74 Red-Team Verification", "",
        f"- Final status: `{'PASS' if passed else 'FAIL'}`",
        f"- Checks: `{sum(int(row['passed']) for row in checks)}/{len(checks)}` passed",
        "- Independent external payload rehash: `216/216 units`",
        "- Main C74 report existed before red-team: `false`",
        "- T3-HO z/Wz touched: `false`", "- Same-label oracle in primary smoke input: `false`",
        "", "## Repairs retained in provenance", "",
        "The ledger records the cancelled initial P0 attempt, MNE lock isolation, protocol-aligned softmax identity, cross-node float32 drift audit, oracle-metadata isolation, and the nested incremental-null repair. Superseded smoke output is not used.",
        "", "## Claim boundary", "",
        "The cache validates instrumentation and makes representation/projection constructs analyzable. It does not validate a representation mechanism, a target gauge, a source-only escape hatch, a selector, or target-population generalization.",
        "", "## Check ledger", "",
        "| Check | Pass | Observed | Expected |", "|---|---:|---|---|",
    ]
    for row in checks:
        lines.append(f"| {row['check']} | {row['passed']} | {str(row['observed']).replace('|', '/')} | {str(row['expected']).replace('|', '/')} |")
    RED_TEAM_REPORT.write_text("\n".join(lines) + "\n")
    if not passed:
        raise RuntimeError("C74 independent red-team failed")
    return {"status": "PASS", "checks": len(checks), "passed": sum(int(row["passed"]) for row in checks)}


def defaultdict_float():
    from collections import defaultdict
    return defaultdict(float)


def _load_label_shard(manifest: dict, kind: str) -> dict[str, np.ndarray]:
    descriptor = next(item for item in manifest["shards"] if item["kind"] == kind)
    with np.load(descriptor["path"], allow_pickle=False) as shard:
        return {name: shard[name] for name in shard.files}


if __name__ == "__main__":
    print(json.dumps(run_red_team(), indent=2, sort_keys=True))
