"""Independent adversarial verification for the C78 no-auth P0 result."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
from typing import Any

from . import c78_seed3_instrumented_pilot as c78


REPORT_PATH = c78.REPORT_DIR / "C78_RED_TEAM_VERIFICATION.md"
MAIN_REPORT = c78.REPORT_DIR / "C78_SEED3_INSTRUMENTED_PILOT.md"
CHECKS_PATH = c78.TABLE_DIR / "red_team_checks.csv"
REPAIR_PATH = c78.TABLE_DIR / "red_team_repair_ledger.csv"


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
        writer.writeheader()
        writer.writerows(rows)


def _check(checks: list[dict[str, Any]], name: str, passed: bool, observed: Any, expected: Any, *, blocking: bool = True, note: str = "") -> None:
    checks.append({
        "check": name, "passed": int(bool(passed)), "blocking": int(blocking),
        "observed": observed, "expected": expected, "note": note,
    })


def run_red_team() -> dict[str, Any]:
    if MAIN_REPORT.exists():
        raise RuntimeError("C78 main report existed before independent red-team")
    state = json.loads(c78.STATE_PATH.read_text())
    protocol, protocol_hash, token, token_path = c78.load_protocol()
    checks: list[dict[str, Any]] = []

    _check(checks, "main_report_absent_before_red_team", not MAIN_REPORT.exists(), MAIN_REPORT.exists(), False)
    _check(checks, "protocol_full_sha256", protocol_hash == c78.PROTOCOL_SHA_PATH.read_text().strip(), protocol_hash, c78.PROTOCOL_SHA_PATH.read_text().strip())
    _check(checks, "protocol_hash_is_full", len(protocol_hash) == 64, len(protocol_hash), 64)
    _check(checks, "unique_token_field", token_path == "authorization.exact_token", token_path, "authorization.exact_token")
    _check(checks, "token_not_empty", len(token) > 32, len(token), ">32")
    _check(checks, "prompt_not_authorization", protocol["authorization"]["prompt_text_is_authorization"] is False, protocol["authorization"]["prompt_text_is_authorization"], False)
    _check(checks, "environment_not_authorization", protocol["authorization"]["environment_is_authorization"] is False, protocol["authorization"]["environment_is_authorization"], False)
    _check(checks, "generic_authorization_rejected", not c78.authorization_matches("我显式授权", token), False, False)
    _check(checks, "whitespace_token_rejected", not c78.authorization_matches(token + "\n", token), False, False)

    anchor = c78._protocol_commit()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", anchor, c78.PARENT_RESULT_COMMIT], check=False).returncode == 0
    _check(checks, "protocol_anchor_exact", anchor.startswith(c78.PROTOCOL_ANCHOR_SHORT), anchor, c78.PROTOCOL_ANCHOR_SHORT)
    _check(checks, "protocol_anchor_precedes_C77_result", ancestor, ancestor, True)
    _check(checks, "state_protocol_identity", state["protocol_commit"] == anchor and state["protocol_sha256"] == protocol_hash, f"{state['protocol_commit']}:{state['protocol_sha256']}", f"{anchor}:{protocol_hash}")

    units = _rows("c78_unit_manifest.csv")
    _check(checks, "manifest_82_unique", len(units) == 82 and len({row["unit_id"] for row in units}) == 82, f"rows={len(units)};unique={len({r['unit_id'] for r in units})}", "82;82")
    _check(checks, "manifest_target4", {row["target"] for row in units} == {"4"}, sorted({row["target"] for row in units}), ["4"])
    _check(checks, "manifest_seed3", {row["seed"] for row in units} == {"3"}, sorted({row["seed"] for row in units}), ["3"])
    _check(checks, "manifest_levels_0_1", {row["level"] for row in units} == {"0", "1"}, sorted({row["level"] for row in units}), ["0", "1"])
    _check(checks, "manifest_dataset_only", {row["dataset"] for row in units} == {c78.DATASET}, sorted({row["dataset"] for row in units}), [c78.DATASET])
    _check(checks, "manifest_regimes_exact", {row["regime"] for row in units} == {"ERM", "OACI"}, sorted({row["regime"] for row in units}), ["ERM", "OACI"])
    _check(checks, "manifest_ERM_asymmetric", sum(row["regime"] == "ERM" for row in units) == 2 and all(row["role"] == "shared_stage1_final_anchor" for row in units if row["regime"] == "ERM"), sum(row["regime"] == "ERM" for row in units), 2)
    _check(checks, "manifest_OACI_complete", sum(row["regime"] == "OACI" for row in units) == 80, sum(row["regime"] == "OACI" for row in units), 80)
    _check(checks, "SRC_absent", not any(row["regime"] == "SRC" for row in units), 0, 0)
    _check(checks, "unit_execution_all_zero", all(row["executed"] == "0" and row["checkpoint_hash"] == "not_created_without_authorization" for row in units), sum(row["executed"] == "1" for row in units), 0)
    for level in ("0", "1"):
        epochs = [int(row["epoch"]) for row in units if row["level"] == level and row["regime"] == "OACI"]
        _check(checks, f"level_{level}_cadence", epochs == list(c78.OACI_EPOCHS), epochs, list(c78.OACI_EPOCHS))

    code = _rows("c78_code_config_hash_replay.csv")
    _check(checks, "code_config_seven_rows", len(code) == 7, len(code), 7)
    _check(checks, "historical_blobs_and_configs_exact", all(row["byte_exact"] == "1" for row in code), sum(row["byte_exact"] == "1" for row in code), 7)
    _check(checks, "historical_optimization_components", {row["component"] for row in code} >= {"ERM objective", "OACI objective", "training engine", "confirmatory manifest"}, sorted(row["component"] for row in code), "required four components")

    auth = _rows("c78_authorization_audit.csv")[0]
    _check(checks, "authorization_absent", auth["CLI_argument_present"] == auth["exact_match"] == auth["training_authorized"] == "0", auth, "all zero")
    _check(checks, "authorization_sources_not_scanned", auth["prompt_text_considered"] == auth["environment_considered"] == auth["substring_scan_performed"] == "0", auth, "all zero")
    _check(checks, "token_not_echoed_in_audit", token not in (c78.TABLE_DIR / "c78_authorization_audit.csv").read_text(), "plaintext absent", "plaintext absent")

    target = _rows("c78_target_isolation_preflight.csv")
    _check(checks, "target_isolation_preflight_complete", len(target) == 9 and all(row["passed"] == "1" for row in target), f"rows={len(target)};pass={sum(r['passed']=='1' for r in target)}", "9/9")
    _check(checks, "preflight_loaded_no_real_data", all(row["real_data_accessed"] == "0" for row in target), sum(row["real_data_accessed"] == "1" for row in target), 0)
    _check(checks, "runtime_isolation_checks_not_faked", sum(row["required_runtime_replay"] == "1" for row in target) == 3, sum(row["required_runtime_replay"] == "1" for row in target), 3)

    dummy = _rows("Wz_logit_dummy_ABI.csv")[0]
    _check(checks, "dummy_ABI_pass", dummy["passed"] == "1" and dummy["device"] == "cpu", dummy, "pass;cpu")
    _check(checks, "dummy_Wz_identity", float(dummy["Wz_plus_b_max_abs"]) <= 1e-6 and float(dummy["softmax_max_abs"]) <= 1e-7, f"{dummy['Wz_plus_b_max_abs']};{dummy['softmax_max_abs']}", "<=1e-6;<=1e-7")
    _check(checks, "dummy_repeat_identity", float(dummy["repeat_logit_max_abs"]) == 0 and float(dummy["repeat_z_max_abs"]) == 0, f"{dummy['repeat_logit_max_abs']};{dummy['repeat_z_max_abs']}", "0;0")
    _check(checks, "dummy_not_real_execution", dummy["real_EEG_rows_loaded"] == dummy["real_training_steps"] == dummy["CUDA_initialized"] == "0", dummy, "all zero")

    environment = _rows("c78_environment_preflight.csv")
    env = next(row for row in environment if "environment_prefix" in row)
    _check(checks, "locked_environment_hash", env["environment_hash_match"] == "1", env["conda_explicit_sha256"], env["expected_conda_explicit_sha256"])
    _check(checks, "P0_no_GPU_request", env["GPU_requested"] == "0" and env["CUDA_initialized"] == "0", f"{env['GPU_requested']};{env['CUDA_initialized']}", "0;0")
    partitions = [row for row in environment if row.get("partition")]
    _check(checks, "V100_snapshot_available", any(row["partition"] == "V100" and row["availability"] == "up" for row in partitions), partitions, "V100 up")
    _check(checks, "cpu_high_snapshot_available", any(row["partition"] == "cpu-high" and row["availability"] == "up" for row in partitions), partitions, "cpu-high up")
    _check(checks, "partition_snapshot_not_authorization", all(row["authorization"] == "0" for row in partitions), [row["authorization"] for row in partitions], "all zero")

    storage = _rows("c78_storage_preflight.csv")[0]
    _check(checks, "storage_capacity", storage["capacity_passed"] == "1", storage["free_GiB_snapshot"], f">={storage['required_temporary_reserve_GiB']}")
    _check(checks, "storage_write_not_faked", storage["write_probe_performed"] == "0", storage["write_probe_performed"], 0, note="actual atomic write gate remains mandatory after future authorization")

    views = {row["view"]: row for row in _rows("physical_view_manifest.csv")}
    _check(checks, "six_physical_views", len(views) == 6, len(views), 6)
    _check(checks, "strict_source_no_target", views["strict_source_trial_view"]["uses_target_rows"] == views["strict_source_trial_view"]["uses_target_labels"] == "0", views["strict_source_trial_view"], "no target")
    _check(checks, "target_unlabeled_no_labels", views["target_unlabeled_trial_view"]["uses_target_labels"] == views["target_unlabeled_trial_view"]["uses_evaluation_labels"] == "0", views["target_unlabeled_trial_view"], "no labels")
    _check(checks, "construction_no_evaluation", views["target_construction_view"]["uses_evaluation_labels"] == "0", views["target_construction_view"], "evaluation=0")
    _check(checks, "oracle_inaccessible_primary", views["same_label_oracle_view"]["available_to_training"] == "0" and views["same_label_oracle_view"]["status"] == "schema_locked_inaccessible_to_primary", views["same_label_oracle_view"], "inaccessible")
    _check(checks, "views_not_falsely_materialized", all("not_materialized" in row["status"] or "inaccessible" in row["status"] for row in views.values()), [row["status"] for row in views.values()], "planned only")

    attempts = _rows("execution_attempt_ledger.csv")
    zero_fields = ["training_attempted", "real_forward_attempted", "real_data_load_attempted", "GPU_requested", "GPU_initialized", "target_label_read", "seed3_execution_access", "seed4_access", "BNCI2014_004_access", "checkpoints_created", "raw_cache_rows"]
    _check(checks, "one_P0_attempt_only", len(attempts) == 1 and attempts[0]["stage"] == "P0_metadata_dummy_ABI", attempts, "one P0")
    _check(checks, "execution_attempt_all_real_actions_zero", all(attempts[0][field] == "0" for field in zero_fields), {field: attempts[0][field] for field in zero_fields}, "all zero")
    _check(checks, "state_execution_boundary_zero", all(value == 0 for value in state["execution_boundary"].values()), state["execution_boundary"], "all zero")
    _check(checks, "state_gate_no_auth", state["final_gate_candidate"] == "PILOT_READY_BUT_NOT_AUTHORIZED", state["final_gate_candidate"], "PILOT_READY_BUT_NOT_AUTHORIZED")
    _check(checks, "execution_taxonomy_not_faked", state["taxonomy"]["primary_active"] == [] and state["taxonomy"]["execution_taxonomy_not_evaluable"], state["taxonomy"], "no primary execution verdict")

    runtime = _rows("training_runtime_ledger.csv")[0]
    identity = _rows("Wz_logit_identity_summary.csv")[0]
    _check(checks, "runtime_explicit_zero_not_estimate", runtime["GPU_hours_measured"] == "0" and runtime["status"] == "not_executed_without_authorization", runtime, "zero/no execution")
    _check(checks, "real_identity_not_faked", identity["rows_checked"] == identity["units_checked"] == "0" and identity["passed"] == "0", identity, "zero rows; pending")

    seed4 = _rows("seed4_protection_audit.csv")[0]
    _check(checks, "seed4_untouched", all(seed4[key] == "0" for key in seed4 if key.startswith("seed4_") and key != "seed4_protection_audit"), seed4, "all zero")
    p2 = _rows("P2_expansion_gate.csv")[0]
    _check(checks, "SRC_gap_explicit", p2["SRC_engine_exercised"] == p2["full_seed3_authorized"] == "0" and p2["gate"] == "SRC_CANARY_REQUIRED_BEFORE_FULL_FIELD", p2, "SRC gap; P2 denied")

    risks = _rows("risk_register.csv")
    _check(checks, "risk_register_complete", len(risks) >= 25, len(risks), ">=25")
    _check(checks, "no_current_open_blocker", not [row for row in risks if row["blocking_open"] == "1"], sum(row["blocking_open"] == "1" for row in risks), 0)
    _check(checks, "future_runtime_gates_disclosed", sum(row["blocks_future_P1_if_unpassed"] == "1" for row in risks) >= 8, sum(row["blocks_future_P1_if_unpassed"] == "1" for row in risks), ">=8")

    c77_power = {row["gate"]: row for row in _read_external_csv(c78.REPORT_DIR / "c77_tables/power_and_false_positive_plan.csv")}
    multiplicity = float(c77_power["effective_multiplicity_reduces_actionability"]["observed"])
    _check(checks, "C77_multiplicity_caveat_replayed", abs(multiplicity - 0.0075) < 1e-12, multiplicity, 0.0075)

    tracked = c78._git("ls-files").splitlines()
    raw = [path for path in tracked if path.startswith("oaci/") and Path(path).suffix in {".pt", ".pth", ".npz", ".npy", ".parquet"}]
    large = [(path, Path(path).stat().st_size) for path in tracked if Path(path).is_file() and Path(path).stat().st_size > c78.MAX_GIT_PAYLOAD]
    _check(checks, "no_raw_payload_in_git", not raw, raw, [])
    _check(checks, "no_tracked_payload_over_50MiB", not large, large, [])

    failures = _rows("failure_reason_ledger.csv")
    _check(checks, "failure_ledger_reason_coded", len(failures) == 6 and all(row["reason"] for row in failures), len(failures), 6)
    _check(checks, "no_execution_claims", not any(state["claim_boundary"].values()), state["claim_boundary"], "all false")

    blocking_failures = [row for row in checks if row["blocking"] == 1 and row["passed"] == 0]
    repairs = [
        {"item": "R1_protocol_parent_semantics", "status": "repaired_before_compute", "finding": "initial implementation treated the C78 protocol anchor as a child of the later C77 result", "resolution": "replay now requires the anchor to be an ancestor of accepted C77 result 285ba1d; it correctly retains C76 as its lock-time parent"},
        {"item": "R2_authorization_phrase", "status": "guard_confirmed", "finding": "PM later wrote a generic explicit-authorization sentence without the exact CLI token", "resolution": "generic prompt text was rejected; training/forward/GPU/data counters remain zero"},
        {"item": "R3_execution_taxonomy", "status": "claim_narrowed", "finding": "a no-auth P0 cannot activate C78-A or C78-B-E", "resolution": "all primary execution taxonomy remains not evaluable; only readiness gate and boundary secondaries are reported"},
        {"item": "R4_runtime_identity", "status": "claim_narrowed", "finding": "dummy Wz identity cannot stand in for 82-unit real identity", "resolution": "dummy and real identity tables are separate; real rows/units checked remain explicitly zero"},
        {"item": "R5_SRC_coverage", "status": "boundary_enforced", "finding": "locked pilot has no SRC execution path", "resolution": "full seed-3 field remains blocked behind prospective SRC canary or exact-path proof and new PM review"},
        {"item": "R6_power_materiality", "status": "caveat_preserved", "finding": "C77 effective-multiplicity synthetic contrast is only 0.0075", "resolution": "C78 makes no H2/materiality claim and requires future power re-lock"},
        {"item": "R7_partition_row_normalization", "status": "repaired_after_blocking_red_team", "finding": "the first red-team pass treated the environment-summary row's empty partition cell as a scheduler row", "resolution": "partition checks now operate only on rows with a non-empty partition; job 892802 is retained as the blocking failed attempt"},
    ]
    _write_csv(CHECKS_PATH, checks)
    _write_csv(REPAIR_PATH, repairs)
    status = "PASS" if not blocking_failures else "FAIL"
    REPORT_PATH.write_text(
        "# C78 Red-Team Verification\n\n"
        f"Final status: `{status}`\n\n"
        f"- Blocking checks passed: `{sum(row['blocking'] == 1 and row['passed'] == 1 for row in checks)}/{sum(row['blocking'] == 1 for row in checks)}`.\n"
        "- Main C78 report existed before red-team: `false`.\n"
        "- Exact CLI authorization received: `false`.\n"
        "- Real training / EEG forward / data load / GPU request / checkpoint creation: `0 / 0 / 0 / 0 / 0`.\n"
        "- Planned manifest: `82` units (`2` ERM anchors + `80` OACI trajectory checkpoints).\n"
        "- Seed 4 / BNCI2014_004 / SRC pilot units: `0 / 0 / 0`.\n\n"
        "## Adversarial repairs\n\n"
        + "\n".join(f"- **{row['item']}**: {row['finding']} Resolution: {row['resolution']}" for row in repairs)
        + "\n\n## Claim boundary\n\n"
        "C78 P0 can establish only protocol, authorization, scope, historical-code, schema, environment, storage-capacity, and dummy-ABI readiness. It cannot establish that training or instrumentation succeeded, activate an execution taxonomy case, replicate measurement-control separation, test cross-regime transport, or support a selector or representation mechanism.\n"
    )
    if blocking_failures:
        raise RuntimeError(f"C78 red-team blockers: {[row['check'] for row in blocking_failures]}")
    print(json.dumps({"status": status, "checks": len(checks), "blocking": sum(row["blocking"] == 1 for row in checks), "repairs": len(repairs)}, sort_keys=True))
    return {"status": status, "checks": checks, "repairs": repairs}


def _read_external_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as stream:
        return list(csv.DictReader(stream))


if __name__ == "__main__":
    run_red_team()
