"""Independent pre-report red team for the authorized C78R canary."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
import subprocess
from typing import Any

from . import c78_authorized_common as c78_common
from . import c78r_collect as collect
from . import c78r_common as common
from . import c78r_seed3_src_canary as c78r


REPORT = c78r.REPORT_DIR / "C78R_RED_TEAM_VERIFICATION.md"
CHECKS = c78r.TABLE_DIR / "red_team_checks.csv"
REPAIRS = c78r.TABLE_DIR / "red_team_repair_ledger.csv"


def _rows(name: str) -> list[dict[str, str]]:
    return c78r.read_csv(c78r.TABLE_DIR / name)


def _check(rows, name, passed, observed, expected, *, blocking=True, note=""):
    rows.append({
        "check": name, "passed": int(bool(passed)), "blocking": int(blocking),
        "observed": observed, "expected": expected, "note": note,
    })


def _git_timestamp(commit: str) -> float:
    value = subprocess.check_output(["git", "show", "-s", "--format=%cI", commit], text=True).strip()
    return datetime.fromisoformat(value).timestamp()


def run() -> list[dict[str, Any]]:
    if not collect.STATE_PATH.exists():
        raise RuntimeError("C78R red-team requires collected execution state")
    state = json.loads(collect.STATE_PATH.read_text())
    protocol, protocol_sha, token = common.load_protocol()
    lock = common.load_execution_lock()
    field = common.require_field_frozen(lock)
    instrument = common.verify_manifest(common.instrumentation_gate_path(lock))
    checkpoints = _rows("SRC_checkpoint_manifest.csv")
    genealogy = _rows("SRC_checkpoint_genealogy.csv")
    cadence = _rows("SRC_checkpoint_cadence_audit.csv")
    trajectories = _rows("SRC_trajectory_trace_manifest.csv")
    compatibility = _rows("cross_regime_schema_compatibility.csv")
    isolation = _rows("SRC_target_isolation_runtime_audit.csv")[0]
    expansion = _rows("full_seed3_expansion_gate.csv")[0]
    seed4 = _rows("seed4_protection_audit.csv")[0]
    risks = _rows("risk_register.csv")
    main_report = c78r.REPORT_DIR / "C78R_SEED3_SRC_CANARY.md"
    checks = []

    _check(checks, "final_report_absent_before_red_team", not main_report.exists(), main_report.exists(), False)
    _check(checks, "protocol_sha", protocol_sha == c78r.PROTOCOL_SHA_PATH.read_text().strip(), protocol_sha, "SHA file")
    _check(checks, "protocol_commit", common.protocol_commit().startswith("99f710d"), common.protocol_commit(), "99f710d")
    _check(checks, "protocol_parent", protocol["parent_result_commit"] == c78r.PARENT_RESULT_COMMIT, protocol["parent_result_commit"], c78r.PARENT_RESULT_COMMIT)
    _check(checks, "exact_token_digest", lock["authorization"]["exact_token_sha256"] == common.sha256_text(token), lock["authorization"]["exact_token_sha256"], common.sha256_text(token))
    _check(checks, "authorization_received", lock["authorization"]["received"] is True, lock["authorization"]["received"], True)
    _check(checks, "protocol_before_GPU_preflight", _git_timestamp(lock["protocol_commit"]) < Path(common.campaign_root(lock) / "gates/GPU_PREFLIGHT.json").stat().st_mtime, lock["protocol_commit"], "before GPU preflight")
    _check(checks, "lock_before_GPU_preflight", _git_timestamp("750cb38") < Path(common.campaign_root(lock) / "gates/GPU_PREFLIGHT.json").stat().st_mtime, "750cb38", "before GPU preflight")
    _check(checks, "prelock_EEG_zero", lock["EEG_data_access_before_lock"] == 0, lock["EEG_data_access_before_lock"], 0)
    _check(checks, "prelock_GPU_zero", lock["GPU_submission_before_lock"] == 0, lock["GPU_submission_before_lock"], 0)
    _check(checks, "historical_byte_exact", all(int(row["byte_exact"]) == 1 for row in protocol["historical_hashes"]), [row["byte_exact"] for row in protocol["historical_hashes"]], "all 1")
    _check(checks, "implementation_hashes", all(c78r.sha256_file(item["path"]) == item["sha256"] for item in lock["implementation_files"]), len(lock["implementation_files"]), len(c78r.IMPLEMENTATION_FILES))
    _check(checks, "field_units", field["unit_count"] == field["SRC_count"] == 80, field["unit_count"], 80)
    _check(checks, "field_unique_units", len({row["unit_id"] for row in field["units"]}) == 80, len({row["unit_id"] for row in field["units"]}), 80)
    _check(checks, "levels_40_each", {int(level): sum(int(row["level"]) == level for row in field["units"]) for level in (0, 1)} == {0: 40, 1: 40}, {int(level): sum(int(row["level"]) == level for row in field["units"]) for level in (0, 1)}, {0: 40, 1: 40})
    _check(checks, "SRC_only", {row["regime"] for row in field["units"]} == {"SRC"}, {row["regime"] for row in field["units"]}, {"SRC"})
    _check(checks, "temperature_0_1", all(common.verify_manifest(row["sidecar_path"])["smooth_temperature"] == 0.1 for row in field["units"]), "80 sidecars", 0.1)
    _check(checks, "ERM_not_retrained", field["ERM_retrained_count"] == 0, field["ERM_retrained_count"], 0)
    _check(checks, "OACI_weights_not_accessed", field["OACI_weight_access_count"] == 0, field["OACI_weight_access_count"], 0)
    _check(checks, "two_read_only_anchors", len(field["read_only_C78_ERM_anchor_access"]) == 2 and all(row["read_only"] for row in field["read_only_C78_ERM_anchor_access"]), len(field["read_only_C78_ERM_anchor_access"]), 2)
    _check(checks, "source_subjects", field["execution"]["source_training_subjects"] == [1, 2, 3, 7, 8, 9], field["execution"]["source_training_subjects"], [1, 2, 3, 7, 8, 9])
    _check(checks, "target_rows_training_zero", field["execution"]["target_data_rows_loaded_during_training"] == 0, field["execution"]["target_data_rows_loaded_during_training"], 0)
    _check(checks, "target_labels_training_zero", field["execution"]["target_label_reads_during_training"] == 0, field["execution"]["target_label_reads_during_training"], 0)
    _check(checks, "source_audit_training_zero", not field["execution"]["source_audit_subjects_loaded_during_training"], field["execution"]["source_audit_subjects_loaded_during_training"], [])
    _check(checks, "target_outcome_retention_zero", not field["retention_uses_target_outcomes"], field["retention_uses_target_outcomes"], False)
    _check(checks, "target_outcome_retry_zero", not field["retry_selection_uses_target_outcomes"], field["retry_selection_uses_target_outcomes"], False)
    _check(checks, "checkpoint_hashes", len(checkpoints) == 80 and all(row["all_hashes_passed"] == "1" for row in checkpoints), len(checkpoints), 80)
    _check(checks, "genealogy", len(genealogy) == 80 and all(row["passed"] == "1" for row in genealogy), len(genealogy), 80)
    _check(checks, "cadence", len(cadence) == 2 and all(row["passed"] == "1" for row in cadence), cadence, "two 40-point levels")
    _check(checks, "trajectory_trace_manifest", len(trajectories) == 2 and all(row["passed"] == "1" and row["row_count"] == "40" and c78r.sha256_file(row["path"]) == row["sha256"] for row in trajectories), len(trajectories), "two hash-valid 40-row traces")
    _check(checks, "instrument_units", instrument["unit_count"] == instrument["unique_unit_count"] == 80, instrument["unit_count"], 80)
    _check(checks, "source_rows", instrument["source_rows"] == 368640, instrument["source_rows"], 368640)
    _check(checks, "target_unlabeled_rows", instrument["target_unlabeled_rows"] == 46080, instrument["target_unlabeled_rows"], 46080)
    _check(checks, "identity_no_failures", instrument["identity"]["failed_units"] == 0, instrument["identity"], "failed_units=0")
    _check(checks, "Wz_identity", float(instrument["identity"]["Wz_plus_b_logits_max_abs"]) <= 1e-6, instrument["identity"]["Wz_plus_b_logits_max_abs"], "<=1e-6")
    _check(checks, "softmax_identity", float(instrument["identity"]["softmax_max_abs"]) <= 1e-7, instrument["identity"]["softmax_max_abs"], "<=1e-7")
    _check(checks, "repeat_exact", float(instrument["identity"]["repeat_max_abs"]) == 0, instrument["identity"]["repeat_max_abs"], 0)
    _check(checks, "target_labels_absent_primary", instrument["physical_isolation"]["target_unlabeled_contains_labels"] is False, instrument["physical_isolation"], False)
    _check(checks, "oracle_absent_primary", instrument["physical_isolation"]["instrumentation_received_oracle_path"] is False, instrument["physical_isolation"], False)
    _check(checks, "views_linked_after_freeze", instrument["physical_isolation"]["C78_inputs_reused_read_only_after_SRC_freeze"] is True, instrument["physical_isolation"], True)
    _check(checks, "schema_compatibility", len(compatibility) == 3 and all(row["passed"] == "1" for row in compatibility), compatibility, "3 exact views")
    _check(checks, "isolation_table", isolation["passed"] == "1", isolation, "passed")
    _check(checks, "GPU_measured", float(field["execution"]["GPU_wall_hours"]) > 0 and field["execution"]["GPU_name"], field["execution"]["GPU_wall_hours"], ">0 measured")
    _check(checks, "storage_measured", int(state["resources"]["C78R_external_bytes"]) > 0, state["resources"]["C78R_external_bytes"], ">0")
    _check(checks, "phase_based_compute", "phase-level" in next(row for row in _rows("updated_full_seed3_compute_plan.csv") if row["phase"] == "TOTAL_48_PHASE_SCHEDULE")["measurement"], "phase-level", "phase-level")
    _check(checks, "expansion_technical_ready", expansion["technical_readiness"] == "1", expansion["technical_readiness"], 1)
    _check(checks, "expansion_not_authorized", expansion["full_seed3_authorized"] == "0", expansion["full_seed3_authorized"], 0)
    _check(checks, "remaining_scope", expansion["remaining_units"] == "1296" and expansion["remaining_training_phases"] == "48", expansion, "1296/48")
    _check(checks, "seed4_untouched", all(seed4[key] == "0" for key in ("seed4_data_access", "seed4_training_jobs", "seed4_checkpoints", "seed4_trial_caches", "seed4_outcome_reads")), seed4, "all zero")
    _check(checks, "BNCI004_untouched", field["execution"]["BNCI2014_004_access"] == 0, field["execution"]["BNCI2014_004_access"], 0)
    _check(checks, "no_blocking_risk", not any(row["blocking_open"] == "1" for row in risks), sum(row["blocking_open"] == "1" for row in risks), 0)
    _check(checks, "no_scientific_replication_claim", state["claims"]["multiregime_scientific_replication"] is False, state["claims"]["multiregime_scientific_replication"], False)
    _check(checks, "no_measurement_control_claim", state["claims"]["measurement_control_replication"] is False, state["claims"]["measurement_control_replication"], False)
    _check(checks, "no_SRC_transfer_claim", state["claims"]["SRC_transfer_claim"] is False, state["claims"]["SRC_transfer_claim"], False)
    _check(checks, "no_selector", state["claims"]["selector"] is False and state["claims"]["checkpoint_recommendation"] is False, state["claims"], "false")
    _check(checks, "no_manuscript", state["claims"]["manuscript"] is False, state["claims"]["manuscript"], False)
    tracked_large = subprocess.check_output("git ls-files -z | xargs -0 -r stat -c '%s %n'", shell=True, text=True)
    oversized = [line for line in tracked_large.splitlines() if int(line.split(" ", 1)[0]) > c78r.MAX_GIT_PAYLOAD]
    _check(checks, "no_large_git_payload", not oversized, oversized, [])
    c78_manifest = c78r.read_csv(c78r.REPORT_DIR / "c78_tables/artifact_manifest.csv")
    c78_replayed = all(Path(row["path"]).is_file() and c78r.sha256_file(row["path"]) == row["sha256"] for row in c78_manifest)
    _check(checks, "C78_artifacts_unchanged", c78_replayed, len(c78_manifest), "all hash replay")
    _check(checks, "final_gate", state["final_gate_candidate"] == "SRC_CANARY_EXECUTED_AND_VALIDATED_FULL_SEED3_READY_BUT_NOT_AUTHORIZED", state["final_gate_candidate"], "successful guarded gate")

    c78r.write_csv(CHECKS, checks)
    repairs = [
        {"item": "R1_frozen_ERM_initialization", "finding": "historical SRC requires ERM stage-2 parent while ERM retraining is forbidden", "resolution": "two C78 anchors are hash-locked read-only initialization dependencies; no OACI weight access or target outcome selection"},
        {"item": "R2_C78_view_reuse", "finding": "duplicating trial views would add storage and leakage surface", "resolution": "C78 physical views are linked read-only only after SRC FIELD_FROZEN; primary descriptor excludes label/oracle paths"},
        {"item": "R3_resource_phase_semantics", "finding": "C78 did not separately time ERM and OACI", "resolution": "compute projection uses measured ERM+OACI context cost plus measured SRC phase cost; no checkpoint-count runtime extrapolation"},
        {"item": "R4_storage_fixed_overhead", "finding": "initial collector mixed fixed and per-unit storage", "resolution": "checkpoint/optimizer/sidecar/trial-cache unit bytes and C78/C78R fixed overhead are separated; projection retains a 25% safety envelope"},
        {"item": "R5_single_target", "finding": "target 4 cannot support cross-regime science", "resolution": "C78R is technical compatibility only; all science/transport/actionability claims false"},
        {"item": "R6_full_expansion", "finding": "technical readiness could be confused with authorization", "resolution": "1,296 units and 48 phases remain explicitly unauthorized behind C78F"},
        {"item": "R7_CPU_peak_RAM", "finding": "Slurm accounting DB unavailable", "resolution": "reported unavailable without estimate; GPU/runtime/storage remain measured"},
    ]
    c78r.write_csv(REPAIRS, repairs)
    failures = [row for row in checks if row["blocking"] and not row["passed"]]
    REPORT.write_text(
        "# C78R Red-Team Verification\n\n"
        f"Final status: `{'PASS' if not failures else 'FAIL'}`\n\n"
        f"- Blocking checks: `{len([row for row in checks if row['blocking'] and row['passed']])}/{len([row for row in checks if row['blocking']])}`.\n"
        f"- Repairs/caveats recorded: `{len(repairs)}`.\n"
        "- Main C78R report existed before red-team: `false`.\n"
        "- C78R is a technical SRC execution/instrumentation canary only.\n"
        "- Full seed-3 expansion, seed 4, BNCI2014_004, selector output, and manuscript work remain unauthorized.\n"
    )
    if failures:
        raise RuntimeError(f"C78R red-team failed: {[row['check'] for row in failures]}")
    print(json.dumps({"status": "PASS", "blocking": len(checks), "repairs": len(repairs)}, sort_keys=True))
    return checks


if __name__ == "__main__":
    run()
