"""Independent red-team for the authorized C78 dual-mode milestone."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from . import c74_cache
from . import c78_authorized_collect as collect
from . import c78_authorized_common as common
from . import c78_seed3_instrumented_pilot as c78


REPORT = c78.REPORT_DIR / "C78_AUTHORIZED_RED_TEAM_VERIFICATION.md"
CANONICAL_REPORT = c78.REPORT_DIR / "C78_RED_TEAM_VERIFICATION.md"
CHECKS = c78.TABLE_DIR / "authorized_red_team_checks.csv"
REPAIRS = c78.TABLE_DIR / "authorized_red_team_repair_ledger.csv"


def _rows(name: str) -> list[dict[str, str]]:
    with open(c78.TABLE_DIR / name, newline="") as stream:
        return list(csv.DictReader(stream))


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    columns: list[str] = []
    for row in rows:
        for key in row:
            if key not in columns:
                columns.append(key)
    with open(path, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any, *, blocking: bool = True, note: str = "") -> None:
    checks.append({"check": name, "passed": int(bool(passed)), "blocking": int(blocking), "observed": observed, "expected": expected, "note": note})


def run_red_team() -> dict[str, Any]:
    state = json.loads(collect.STATE_PATH.read_text())
    lock = common.load_execution_lock()
    frozen = common.require_field_frozen(lock)
    instrument = common.verify_canonical_manifest(common.instrumentation_gate_path(lock))
    primary = common.verify_canonical_manifest(common.primary_input_gate_path(lock))
    labels = common.verify_canonical_manifest(common.label_view_gate_path(lock))
    checks: list[dict[str, Any]] = []

    current_result = json.loads((c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.json").read_text())
    _check(checks, "authorized_report_absent_before_red_team", current_result["schema_version"] == "c78_seed3_instrumented_pilot_no_auth_result_v1", current_result["schema_version"], "no-auth schema")
    _check(checks, "no_auth_baseline_commit", common.git("cat-file", "-t", common.NO_AUTH_RESULT_COMMIT) == "commit", common.NO_AUTH_RESULT_COMMIT, "git commit")
    _check(checks, "protocol_hash", lock["protocol_sha256"] == c78.PROTOCOL_SHA_PATH.read_text().strip(), lock["protocol_sha256"], c78.PROTOCOL_SHA_PATH.read_text().strip())
    _check(checks, "execution_lock_hash", c78.sha256_file(common.LOCK_PATH) == common.LOCK_SHA_PATH.read_text().strip(), c78.sha256_file(common.LOCK_PATH), common.LOCK_SHA_PATH.read_text().strip())
    _check(checks, "implementation_files_replay", all(c78.sha256_file(item["path"]) == item["sha256"] for item in lock["implementation_files"]), len(lock["implementation_files"]), 9)
    _check(checks, "authorization_exact_digest", lock["authorization"]["exact_token_sha256"] == "077241b125cb49982a41767a4e49d2894b930506bd9c65ecb7e2cb7fc915f06f", lock["authorization"]["exact_token_sha256"], "registered digest")

    _check(checks, "field_82_unique", frozen["unit_count"] == 82 and len({row["unit_id"] for row in frozen["units"]}) == 82, f"{frozen['unit_count']}/{len({row['unit_id'] for row in frozen['units']})}", "82/82")
    _check(checks, "field_regime_counts", frozen["ERM_anchor_count"] == 2 and frozen["OACI_trajectory_count"] == 80 and frozen["SRC_count"] == 0, f"{frozen['ERM_anchor_count']}/{frozen['OACI_trajectory_count']}/{frozen['SRC_count']}", "2/80/0")
    _check(checks, "fixed_retention_frozen", frozen["all_82_retention_decisions_frozen"] and not frozen["retention_uses_target_outcomes"], frozen["all_82_retention_decisions_frozen"], True)
    _check(checks, "source_only_training_subjects", frozen["execution"]["source_training_subjects"] == [1, 2, 3, 7, 8, 9], frozen["execution"]["source_training_subjects"], [1, 2, 3, 7, 8, 9])
    _check(checks, "training_target_rows_zero", frozen["execution"]["target_data_rows_loaded_during_training"] == 0 and frozen["execution"]["target_label_reads_during_training"] == 0, frozen["execution"], "target rows/labels zero")
    _check(checks, "training_source_audit_absent", frozen["execution"]["source_audit_subjects_loaded_during_training"] == [], frozen["execution"]["source_audit_subjects_loaded_during_training"], [])
    _check(checks, "seed4_BNCI004_absent", frozen["execution"]["seed4_access"] == frozen["execution"]["BNCI2014_004_access"] == 0, f"{frozen['execution']['seed4_access']}/{frozen['execution']['BNCI2014_004_access']}", "0/0")
    _check(checks, "source_loader_offline", frozen["source_loader"]["network_attempt_count"] == 0 and frozen["source_loader"]["rows"] == 3456, frozen["source_loader"], "0 network;3456 rows")

    checkpoint_rows = _rows("checkpoint_hash_replay.csv")
    _check(checks, "checkpoint_optimizer_replay_table", len(checkpoint_rows) == 82 and all(row["checkpoint_hash_match"] == row["optimizer_hash_match"] == "1" for row in checkpoint_rows), f"rows={len(checkpoint_rows)};pass={sum(r['checkpoint_hash_match']=='1' and r['optimizer_hash_match']=='1' for r in checkpoint_rows)}", "82/82")
    independent = []
    for unit in common.checkpoint_sidecars(lock):
        sidecar = common.verify_canonical_manifest(unit["sidecar_path"])
        independent.append(collect._verify_checkpoint_and_optimizer(sidecar))
    _check(checks, "independent_checkpoint_optimizer_rehash", len(independent) == 82 and all(row["checkpoint_hash_match"] and row["optimizer_hash_match"] for row in independent), sum(row["checkpoint_hash_match"] and row["optimizer_hash_match"] for row in independent), 82)

    cadence = _rows("checkpoint_cadence_audit.csv")
    genealogy = _rows("checkpoint_genealogy.csv")
    _check(checks, "two_level_cadence", len(cadence) == 2 and all(row["passed"] == "1" and row["ERM_actual"] == "1" and row["OACI_actual"] == "40" for row in cadence), cadence, "41 each level")
    _check(checks, "genealogy_complete", len(genealogy) == 82 and all(row["passed"] == "1" for row in genealogy), len(genealogy), 82)
    by_level = {level: [row for row in genealogy if row["level"] == str(level)] for level in (0, 1)}
    _check(checks, "ERM_OACI_asymmetry", all(sum(row["regime"] == "ERM" for row in rows) == 1 and sum(row["regime"] == "OACI" for row in rows) == 40 for rows in by_level.values()), {key: len(value) for key, value in by_level.items()}, "1+40 per level")

    _check(checks, "postfreeze_primary_gate", primary["created_after_field_freeze"] and primary["field_frozen_manifest_sha256"] == frozen["manifest_sha256"], primary["field_frozen_manifest_sha256"], frozen["manifest_sha256"])
    _check(checks, "primary_target_fields_no_labels", not primary["target_label_fields_present"] and not primary["label_view_gate_path_in_primary_descriptor"] and not primary["same_label_oracle_path_in_primary_descriptor"], primary, "all false")
    _check(checks, "target_input_schema", set(primary["target_unlabeled_input"]["fields"]) == {"X", "target_id", "target_trial_id"}, primary["target_unlabeled_input"]["fields"], ["X", "target_id", "target_trial_id"])
    _check(checks, "postfreeze_label_split", labels["created_after_field_freeze"] and labels["target_label_views"]["construction"]["row_count"] + labels["target_label_views"]["evaluation"]["row_count"] == 576, f"{labels['target_label_views']['construction']['row_count']}+{labels['target_label_views']['evaluation']['row_count']}", 576)
    _check(checks, "oracle_not_primary", not labels["available_to_primary_instrumentation"], labels["available_to_primary_instrumentation"], False)
    for name, descriptor in (("source_input", primary["strict_source_input"]), ("target_input", primary["target_unlabeled_input"]), ("construction", labels["target_label_views"]["construction"]), ("evaluation", labels["target_label_views"]["evaluation"]), ("oracle", labels["target_label_views"]["same_label_oracle"])):
        c74_cache.verify_shard(descriptor)
        _check(checks, f"physical_view_hash_{name}", True, descriptor["sha256"], "rehash pass")

    _check(checks, "instrumentation_82_unique", instrument["unit_count"] == instrument["unique_unit_count"] == 82, f"{instrument['unit_count']}/{instrument['unique_unit_count']}", "82/82")
    _check(checks, "instrumentation_row_counts", instrument["source_rows"] == 377856 and instrument["target_unlabeled_rows"] == 47232, f"{instrument['source_rows']}/{instrument['target_unlabeled_rows']}", "377856/47232")
    _check(checks, "instrumentation_identity_exact", all(float(instrument["identity"][key]) == 0 for key in ("Wz_plus_b_logits_max_abs", "softmax_max_abs", "repeat_max_abs", "hook_z_max_abs")) and instrument["identity"]["failed_units"] == 0, instrument["identity"], "all zero")
    isolation = instrument["physical_isolation"]
    isolation_pass = (
        isolation["target_unlabeled_contains_labels"] is False
        and isolation["instrumentation_received_label_gate_path"] is False
        and isolation["instrumentation_received_oracle_path"] is False
        and isolation["source_and_target_input_views_separate"] is True
        and isolation["construction_evaluation_oracle_separate"] is True
    )
    _check(checks, "instrumentation_physical_isolation", isolation_pass, isolation, "unsafe visibility false; physical separation true")
    unit_schema_pass = True
    unit_hash_pass = True
    for item in instrument["units"]:
        unit = common.verify_canonical_manifest(item["path"])
        for descriptor in unit["shards"]:
            c74_cache.verify_shard(descriptor)
            if descriptor["kind"] == "target_unlabeled_trial":
                unit_schema_pass &= not bool(set(descriptor["fields"]) & {"target_class_label", "split_role", "correctness", "target_bAcc", "joint_good"})
        unit_hash_pass &= unit["all_gates_passed"]
    _check(checks, "all_instrumentation_payload_hashes", unit_hash_pass, unit_hash_pass, True)
    _check(checks, "all_target_unlabeled_payload_schemas", unit_schema_pass, unit_schema_pass, True)

    feature = _rows("registered_feature_block_computability.csv")
    smoke = _rows("pilot_smoke_summary.csv")
    geometry = _rows("effective_multiplicity_top_gap_smoke.csv")
    endpoint = _rows("target_endpoint_smoke.csv")
    _check(checks, "feature_blocks_computable_not_tested", len(feature) == 5 and all(row["computable"] == "1" and row["scientific_test_in_C78"] == "0" for row in feature), f"rows={len(feature)}", "5;computable;not tested")
    _check(checks, "smoke_only_no_scientific_claim", len(smoke) == 6 and all(row["scientific_claim"] == "0" for row in smoke), len(smoke), 6)
    _check(checks, "endpoint_rows_postfreeze_diagnostic", len(endpoint) == 82 and all(row["postfreeze_diagnostic_only"] == "1" and row["checkpoint_recommendation"] == "0" for row in endpoint), len(endpoint), 82)
    _check(checks, "no_best_checkpoint_ID_emitted", len(geometry) == 2 and all(row["best_checkpoint_id_emitted"] == "0" and "checkpoint_id" not in row for row in geometry), geometry, "two aggregate rows;no IDs")
    _check(checks, "random_baseline_context", all(abs(float(row["uniform_random_top1"]) - 1 / 41) < 1e-15 for row in geometry), [row["uniform_random_top1"] for row in geometry], 1 / 41)

    trajectory = _rows("source_trajectory_sanity.csv")
    _check(checks, "trajectory_values_finite", len(trajectory) == 2 and all(row["finite"] == row["passed"] == "1" for row in trajectory), trajectory, "two finite levels")
    _check(checks, "trajectory_risk_stress_disclosed", all(row["OACI_risk_feasible_count"] == "23" and row["OACI_count"] == "40" for row in trajectory), [(row["OACI_risk_feasible_count"], row["OACI_count"]) for row in trajectory], "23/40 both", blocking=False, note="pipeline pass does not imply all checkpoints risk-feasible")
    _check(checks, "lambda_cap_stress_disclosed", all(float(row["lambda_max"]) == 20.0 for row in trajectory), [row["lambda_max"] for row in trajectory], [20.0, 20.0], blocking=False)
    _check(checks, "level1_surrogate_extreme_disclosed", float(trajectory[1]["train_surrogate_min"]) < -40, trajectory[1]["train_surrogate_min"], "<-40", blocking=False)

    runtime = _rows("training_runtime_ledger.csv")
    success = next(row for row in runtime if row["stage"] == "source_only_training")
    _check(checks, "measured_GPU_runtime", abs(float(success["GPU_hours_measured"]) - 0.5436387952168783) < 1e-12 and float(success["peak_GPU_memory_bytes"]) > 0, f"{success['GPU_hours_measured']};{success['peak_GPU_memory_bytes']}", "measured")
    _check(checks, "CPU_peak_RAM_unavailable_disclosed", success["peak_RAM_bytes"] == "unavailable_slurm_accounting_db_refused_connection", success["peak_RAM_bytes"], "unavailable disclosed", blocking=False)
    storage = _rows("actual_compute_storage_summary.csv")[0]
    _check(checks, "external_storage_measured", int(storage["actual_external_bytes"]) == 1798213676 and storage["measured_not_estimated"] == "1", storage["actual_external_bytes"], 1798213676)

    attempts = _rows("execution_attempt_ledger.csv")
    failed = next(row for row in attempts if row["job_id"] == "892830")
    _check(checks, "failed_GPU_canary_retained", failed["status"] == "blocked_before_data" and failed["real_data_rows"] == failed["checkpoint_count"] == "0", failed, "pre-data;zero rows/checkpoints")
    _check(checks, "successful_jobs_stderr_empty", all(Path(path).stat().st_size == 0 for path in [
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-train_892832.err",
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-views_892841.err",
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-instr-0_892843.err",
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-instr-1_892844.err",
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-aggregate_892845.err",
        "/projects/EEG-foundation-model/yinghao/oaci-c78-seed3-pilot/logs/c78-auth-collect_892848.err",
    ]), "all zero bytes", "all zero bytes")

    risks = _rows("risk_register.csv")
    _check(checks, "no_open_blocking_risk", not [row for row in risks if row["blocking_open"] == "1"], sum(row["blocking_open"] == "1" for row in risks), 0)
    _check(checks, "SRC_blocks_P2", any(row["risk"] == "SRC_not_exercised" and row["status"] == "blocks_P2" for row in risks), "SRC_not_exercised", "blocks_P2")
    _check(checks, "state_claims_all_negative", not any(state["claims"].values()), state["claims"], "all false")
    _check(checks, "state_gate_conservative", state["final_gate_candidate"] == "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD", state["final_gate_candidate"], "PILOT_VALID_SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD")
    _check(checks, "taxonomy_exact", state["taxonomy"]["primary_active"] == ["C78-A_seed3_OACI_ERM_pilot_executed_and_validated"] and "C78-S9_SRC_canary_required_before_full_field" in state["taxonomy"]["secondary_active"], state["taxonomy"], "C78-A + S9")

    tracked = common.git("ls-files").splitlines()
    raw = [path for path in tracked if path.startswith("oaci/") and Path(path).suffix in {".pt", ".pth", ".npz", ".npy", ".parquet"}]
    large = [(path, Path(path).stat().st_size) for path in tracked if Path(path).is_file() and Path(path).stat().st_size > c78.MAX_GIT_PAYLOAD]
    _check(checks, "no_raw_payload_in_git", not raw, raw, [])
    _check(checks, "no_tracked_payload_over_50MiB", not large, large, [])

    repairs = [
        {"item": "R1_no_auth_vs_authorized_provenance", "finding": "C78 has both a no-auth readiness baseline and a later exact-token execution", "resolution": "dual-mode ledger preserves commit 67bca01 and authorized jobs separately"},
        {"item": "R2_GPU_determinism_gate", "finding": "job 892830 failed before data because CuBLAS deterministic workspace was unset", "resolution": "prospective lock repaired with CUBLAS_WORKSPACE_CONFIG=:4096:8; job 892832 passed; failed attempt retained"},
        {"item": "R3_target_process_isolation", "finding": "the generic MOABB loader returns labels", "resolution": "training loaded source-train subjects only; target was provisioned only after FIELD_FROZEN and inference received an X/ID-only NPZ"},
        {"item": "R4_dummy_vs_real_identity", "finding": "P0 dummy identity was insufficient", "resolution": "authorized instrumentation checked 425088 real trial-unit rows over 82 checkpoints with all maxima zero"},
        {"item": "R5_trajectory_stress", "finding": "both levels have 23/40 source-risk-feasible OACI points, lambda reaches 20, and level-1 surrogate reaches -49.694", "resolution": "reported as pipeline smoke stress; no stability, replication, or control claim"},
        {"item": "R6_CPU_peak_RAM", "finding": "Slurm accounting DB refused the post-completion query", "resolution": "CPU peak RAM is marked unavailable; no estimate substituted; GPU peak/runtime and storage remain measured"},
        {"item": "R7_ERM_OACI_asymmetry", "finding": "ERM has one anchor while OACI has 40 trajectory points per level", "resolution": "all tables and report keep anchors and trajectories separate"},
        {"item": "R8_SRC_gap", "finding": "SRC execution/instrumentation path remains unexercised", "resolution": "full seed-3 expansion is not ready or authorized; final gate requires PM-reviewed SRC canary/path proof"},
        {"item": "R9_smoke_target_outcomes", "finding": "post-freeze bAcc/NLL/ECE could be misread as checkpoint selection", "resolution": "smoke emits no checkpoint ID, best flag, or recommendation and carries diagnostic-only fields"},
        {"item": "R10_isolation_boolean_semantics", "finding": "red-team job 892850 incorrectly required every isolation-ledger boolean to be true, including safe negative fields such as target_unlabeled_contains_labels=false", "resolution": "the rerun checks unsafe visibility fields are false and physical-separation fields are true; the failed review attempt is retained"},
    ]
    blocking_failures = [row for row in checks if row["blocking"] == 1 and row["passed"] == 0]
    _write_csv(CHECKS, checks)
    _write_csv(REPAIRS, repairs)
    status = "PASS" if not blocking_failures else "FAIL"
    text = (
        "# C78 Authorized Red-Team Verification\n\n"
        f"Final status: `{status}`\n\n"
        f"- Blocking checks passed: `{sum(row['blocking'] == 1 and row['passed'] == 1 for row in checks)}/{sum(row['blocking'] == 1 for row in checks)}`.\n"
        f"- Nonblocking disclosed stress/caveat checks: `{sum(row['blocking'] == 0 for row in checks)}`.\n"
        "- Authorized final report existed before this red-team: `false` (the canonical result was still the no-auth schema).\n"
        "- Field: `82/82`; checkpoint/optimizer independent replay: `82/82`; instrumentation failed units: `0`.\n"
        "- Real identity maxima (`Wz`, softmax, hook, repeat): `0 / 0 / 0 / 0`.\n"
        "- Training target rows/labels, seed 4, BNCI2014_004, SRC units: `0 / 0 / 0 / 0 / 0`.\n\n"
        "## Repairs and claim limits\n\n"
        + "\n".join(f"- **{row['item']}**: {row['finding']} Resolution: {row['resolution']}" for row in repairs)
        + "\n\n## Verdict boundary\n\n"
        "C78-A is supported only as an OACI+ERM training/instrumentation canary. C78 does not establish multi-regime replication, measurement-control replication, representation mechanism, source or target-unlabeled escape hatch, checkpoint control, or seed-level confirmation. SRC remains unexercised, so the 1,458-unit field is neither ready nor authorized.\n"
    )
    REPORT.write_text(text)
    CANONICAL_REPORT.write_text(text)
    if blocking_failures:
        raise RuntimeError(f"C78 authorized red-team blockers: {[row['check'] for row in blocking_failures]}")
    print(json.dumps({"status": status, "checks": len(checks), "blocking": sum(row["blocking"] == 1 for row in checks), "nonblocking": sum(row["blocking"] == 0 for row in checks), "repairs": len(repairs)}, sort_keys=True))
    return {"status": status, "checks": checks, "repairs": repairs}


if __name__ == "__main__":
    run_red_team()
