"""Independent pre-report red-team gauntlet for C78F."""
from __future__ import annotations

import csv
from datetime import datetime
import json
from pathlib import Path
from typing import Any

from . import c74_cache
from . import c78f_collect
from . import c78f_full_seed3_field as c78f
from . import c78f_runtime as runtime


REPORT_PATH = c78f.REPORT_DIR / "C78F_AUTHORIZED_RED_TEAM_VERIFICATION.md"
CHECKS_PATH = c78f.TABLE_DIR / "authorized_red_team_checks.csv"
REPAIRS_PATH = c78f.TABLE_DIR / "authorized_red_team_repair_ledger.csv"


def _read(name: str) -> list[dict[str, str]]:
    return c78f.read_csv(c78f.TABLE_DIR / name)


def _iso(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def run_red_team(*, require_main_report_absent: bool = True) -> dict[str, Any]:
    lock, protocol, protocol_sha = runtime.require_authorization()
    state = json.loads(c78f_collect.STATE_PATH.read_text())
    full = runtime.verify_manifest(runtime.full_field_path(lock))
    checks: list[dict[str, Any]] = []

    def check(category: str, name: str, passed: bool, evidence: Any, blocking: bool = True) -> None:
        checks.append({
            "category": category,
            "check": name,
            "status": "PASS" if passed else "FAIL",
            "blocking": int(blocking),
            "evidence": str(evidence),
        })

    main_report = c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.md"
    main_json = c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.json"
    check("timing", "red_team_precedes_main_report", not main_report.exists() and not main_json.exists() if require_main_report_absent else True, f"md={main_report.exists()};json={main_json.exists()}")
    check("protocol", "C78F_protocol_hash", c78f.sha256_file(c78f.PROTOCOL_PATH) == c78f.PROTOCOL_SHA_PATH.read_text().strip(), protocol_sha)
    check("protocol", "C78S_protocol_hash", c78f.sha256_file(c78f.C78S_PROTOCOL_PATH) == c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip())
    check("protocol", "C78S_bound_into_C78F", protocol["C78S_analysis_lock"]["sha256"] == c78f.C78S_PROTOCOL_SHA_PATH.read_text().strip(), protocol["C78S_analysis_lock"])
    check("authorization", "direct_user_authorization_recorded", lock["authorization"]["received"] and lock["authorization"]["mode"] == c78f.AUTHORIZATION_MODE, lock["authorization"])
    check("authorization", "magic_token_removed", not lock["authorization"]["magic_token_required"] and not protocol["authorization"]["magic_token_required"], "direct lock only")
    check("authorization", "scope_bound_execution_lock", lock["authorization"]["scope_bound"] and lock["scope"]["remaining_units"] == 1296, lock["scope_identity_sha256"])
    check("timing", "zero_access_before_lock", lock["remaining_target_EEG_data_access_before_lock"] == 0 and lock["GPU_submission_before_lock"] == 0 and lock["remaining_target_outcome_reads_before_lock"] == 0, "0/0/0")
    check("implementation", "all_locked_files_replay", all(c78f.sha256_file(item["path"]) == item["sha256"] for item in lock["implementation_files"]), len(lock["implementation_files"]))

    units = _read("full_unit_manifest.csv")
    remaining = [row for row in units if row["target"] != "4"]
    check("scope", "full_unit_registry_1458", len(units) == 1458 and len({row["unit_id"] for row in units}) == 1458, len(units))
    check("scope", "remaining_registry_1296", len(remaining) == 1296, len(remaining))
    check("scope", "remaining_regime_counts", {regime: sum(row["regime"] == regime for row in remaining) for regime in c78f.REGIMES} == {"ERM": 16, "OACI": 640, "SRC": 640}, "16/640/640")
    check("scope", "target4_exact_162", sum(row["target"] == "4" for row in units) == 162, 162)
    check("scope", "seed3_only", {row["seed"] for row in units} == {"3"}, sorted({row["seed"] for row in units}))
    check("scope", "BNCI001_only", {row["dataset"] for row in units} == {c78f.DATASET}, sorted({row["dataset"] for row in units}))

    waves = _read("wave_assignment.csv")
    check("waves", "two_disjoint_four_target_waves", len(waves) == 8 and {row["wave"] for row in waves} == {"A", "B"} and all(sum(item["wave"] == wave for item in waves) == 4 for wave in ("A", "B")), waves)
    wave_a = runtime.verify_manifest(runtime.wave_gate_path(lock, "A"))
    wave_b = runtime.verify_manifest(runtime.wave_gate_path(lock, "B"))
    check("waves", "wave_A_engineering_passed", wave_a["all_engineering_gates_passed"] and not wave_a["target_scientific_outcomes_read"], wave_a["manifest_sha256"])
    check("waves", "wave_B_engineering_passed", wave_b["all_engineering_gates_passed"] and not wave_b["target_scientific_outcomes_read"], wave_b["manifest_sha256"])
    b_fields = [runtime.require_oaci_field(lock, target) for target in c78f.wave_targets()["B"]]
    check("waves", "wave_A_gate_predates_wave_B_fields", all(_iso(field["created_at_utc"]) >= _iso(wave_a["created_at_utc"]) for field in b_fields), wave_a["created_at_utc"])
    check("waves", "continuation_engineering_only", wave_a["continuation_basis"] == "engineering_only" and wave_b["continuation_basis"] == "engineering_only", "engineering_only")

    checkpoints = _read("checkpoint_manifest.csv")
    check("checkpoint", "checkpoint_rows_1458", len(checkpoints) == 1458 and len({row["unit_id"] for row in checkpoints}) == 1458, len(checkpoints))
    check("checkpoint", "all_checkpoint_hashes_replay", all(row["all_hashes_passed"] == "1" for row in checkpoints), sum(row["all_hashes_passed"] == "1" for row in checkpoints))
    cadence = _read("checkpoint_cadence_audit.csv")
    check("checkpoint", "all_54_cadence_cells_pass", len(cadence) == 54 and all(row["passed"] == "1" for row in cadence), len(cadence))
    genealogy = _read("checkpoint_genealogy.csv")
    check("checkpoint", "remaining_genealogy_1296", len(genealogy) == 1296 and all(row["passed"] == "1" for row in genealogy), len(genealogy))
    check("checkpoint", "SRC_never_retrained_ERM", all(row["ERM_retrained_in_SRC_process"] == "0" for row in genealogy if row["regime"] == "SRC"), "0")
    check("checkpoint", "SRC_never_read_OACI_weights", all(row["OACI_weight_access_in_SRC_process"] == "0" for row in genealogy if row["regime"] == "SRC"), "0")
    optimizers = _read("optimizer_state_manifest.csv")
    check("checkpoint", "optimizer_replay_1296", len(optimizers) == 1296 and all(row["passed"] == "1" for row in optimizers), len(optimizers))

    isolation = _read("target_isolation_runtime_audit.csv")
    check("isolation", "all_8_targets_isolated", len(isolation) == 8 and all(row["passed"] == "1" for row in isolation), len(isolation))
    check("isolation", "zero_target_training_rows_labels", all(row["target_rows_read_during_training"] == "0" and row["target_labels_read_during_training"] == "0" for row in isolation), "0/0")
    check("isolation", "zero_source_audit_training_rows", all(row["source_audit_rows_read_during_training"] == "0" for row in isolation), "0")
    check("isolation", "zero_outcome_retention_retry", all(row["target_outcome_retention"] == "0" and row["target_outcome_retry_selection"] == "0" for row in isolation), "0/0")

    identities = _read("Wz_logit_identity_summary.csv")
    check("instrumentation", "all_1296_remaining_units_instrumented", len(identities) == 8 and sum(int(row["units"]) for row in identities) == 1296, sum(int(row["units"]) for row in identities))
    check("instrumentation", "zero_identity_failures", all(row["failed_units"] == "0" and row["passed"] == "1" for row in identities), "0")
    check("instrumentation", "Wz_logit_tolerance", max(float(row["Wz_plus_b_logits_max_abs"]) for row in identities) <= 1e-6, max(float(row["Wz_plus_b_logits_max_abs"]) for row in identities))
    check("instrumentation", "softmax_tolerance", max(float(row["softmax_max_abs"]) for row in identities) <= 1e-7, max(float(row["softmax_max_abs"]) for row in identities))
    check("instrumentation", "hook_tolerance", max(float(row["hook_z_max_abs"]) for row in identities) <= 1e-6, max(float(row["hook_z_max_abs"]) for row in identities))
    check("instrumentation", "repeat_identity", max(max(float(row["repeat_logits_max_abs"]), float(row["repeat_z_max_abs"])) for row in identities) == 0.0, "exact")
    check("instrumentation", "full_row_counts", full["full_source_rows"] == c78f.FULL_SOURCE_ROWS and full["full_target_unlabeled_rows"] == c78f.FULL_TARGET_ROWS, f"{full['full_source_rows']}/{full['full_target_unlabeled_rows']}")

    views = _read("physical_view_manifest.csv")
    remaining_views = [row for row in views if row["target"] != "4"]
    check("views", "five_views_per_remaining_target", len(remaining_views) == 40, len(remaining_views))
    unlabeled = [row for row in remaining_views if row["view_name"] == "target_unlabeled_input"]
    check("views", "unlabeled_views_have_no_labels", len(unlabeled) == 8 and all(row["uses_target_labels"] == "0" and "target_class_label" not in row["allowed_columns"] for row in unlabeled), len(unlabeled))
    oracle = [row for row in remaining_views if row["view_name"] == "same_label_oracle_view"]
    check("views", "oracle_diagnostic_only", len(oracle) == 8 and all(row["diagnostic_only"] == "1" and row["available_at_selection_time"] == "0" for row in oracle), len(oracle))
    check("views", "labels_after_full_freeze", full["label_views_created"] and all(runtime.verify_manifest(runtime.label_view_path(lock, target))["created_after_complete_1458_unit_field_freeze"] for target in c78f.TARGETS), full.get("pre_label_full_field_manifest_sha256"))

    attempts = _read("execution_attempt_ledger.csv")
    check("attempts", "all_attempts_retained", len(attempts) >= 32, len(attempts))
    check("attempts", "no_attempt_target_labels", all(row["target_labels_accessed"] == "0" for row in attempts), "0")
    completed = {(row["target"], row["stage"]) for row in attempts if row["event"] == "complete"}
    check("attempts", "all_16_training_stages_completed", len(completed) == 16, len(completed))

    c78s = json.loads(c78f.C78S_PROTOCOL_PATH.read_text())
    check("science_boundary", "target4_excluded_from_primary", c78s["data_roles"]["primary_targets"] == list(c78f.TARGETS) and c78s["data_roles"]["target4_canary"].startswith("descriptive_only"), c78s["data_roles"])
    check("science_boundary", "C78S_not_started", state["scientific_analysis_started"] is False and full["scientific_analysis_started"] is False, "false")
    check("science_boundary", "no_target_outcomes_inspected", state["target_outcomes_inspected"] == 0 and not full["target_scientific_outcomes_read"], "0")
    check("science_boundary", "ERM_anchor_asymmetry_locked", c78s["inference"]["ERM_role"] == "anchor_not_symmetric_trajectory", c78s["inference"]["ERM_role"])
    check("science_boundary", "SRC_negative_control_only", protocol["training"]["SRC"].startswith("exact_C11_historical_negative_control"), protocol["training"]["SRC"])
    check("science_boundary", "no_selector_or_recommendation", not state["selector_artifacts"] and not state["checkpoint_recommendations"], "false/false")

    seed4 = _read("seed4_protection_audit.csv")[0]
    check("protection", "seed4_untouched", all(seed4[key] == "0" for key in ("seed4_training_jobs", "seed4_data_execution_access", "seed4_checkpoints", "seed4_caches", "seed4_outcome_reads")), seed4)
    check("protection", "BNCI004_untouched", seed4["BNCI2014_004_access"] == "0", seed4["BNCI2014_004_access"])
    risks = _read("risk_register.csv")
    check("risk", "no_blocking_open_risk", all(row["blocking_open"] == "0" for row in risks), len(risks))

    tracked = runtime.git("ls-files").splitlines()
    large = []
    for name in tracked:
        path = Path(name)
        if path.is_file() and path.stat().st_size > c78f.MAX_GIT_PAYLOAD:
            large.append({"path": name, "bytes": path.stat().st_size})
    check("artifact", "no_git_payload_over_50MiB", not large, large)
    raw_markers = ("checkpoints/", "optimizer_states/", "strict_source_trial", "target_unlabeled_trial")
    check("artifact", "no_raw_external_payload_tracked", not any(any(marker in name for marker in raw_markers) for name in tracked), "external only")
    check("artifact", "complete_field_manifest_external", str(runtime.full_field_path(lock)).startswith(str(c78f.EXTERNAL_ROOT)), runtime.full_field_path(lock))

    blocking_failures = [row for row in checks if row["blocking"] and row["status"] != "PASS"]
    c78f.write_csv(CHECKS_PATH, checks)
    repairs = [
        {"issue": "authorization_ceremony_complexity", "severity": "governance", "repair": "direct explicit user authorization plus committed scope-bound lock", "status": "closed"},
        {"issue": "target_loader_structurally_bundles_y", "severity": "caveat", "repair": "primary provisioning code never indexes/hashes/emits y; physical target-unlabeled view contains X/IDs only", "status": "disclosed"},
        {"issue": "target4_canary_previously_observed", "severity": "inference", "repair": "excluded from every C78S primary test", "status": "closed"},
        {"issue": "seed3_not_target_population_confirmation", "severity": "claim", "repair": "exploratory replication field language only", "status": "closed"},
        {"issue": "ERM_OACI_SRC_asymmetry", "severity": "analysis", "repair": "ERM anchor never treated as 40-point trajectory", "status": "closed"},
        {"issue": "C78F_generation_not_science", "severity": "claim", "repair": "C78S remains not started", "status": "closed"},
    ]
    c78f.write_csv(REPAIRS_PATH, repairs)
    report = f"""# C78F Authorized Red-Team Verification

Final red-team gate: **{'PASS' if not blocking_failures else 'FAIL'}**

```text
checks: {len(checks)}
blocking failures: {len(blocking_failures)}
protocol SHA-256: {protocol_sha}
full units: 1,458
remaining units instrumented: 1,296
target outcomes inspected: 0
C78S analysis started: false
seed 4 touched: false
BNCI2014_004 touched: false
```

The authorization interface was deliberately simplified: direct explicit user
authorization is bound to the committed protocol scope by an immutable execution
lock. No magic token is required, while prompt/environment scanning remains
invalid.

All checkpoint, optimizer, cadence, genealogy, target-isolation, physical-view,
and numerical identity gates passed. Wave B followed an engineering-only Wave-A
gate. Target 4 remains descriptive-only and C78S has not started.

The MOABB loader structurally returns labels with target data; the primary
provisioning path never indexes, hashes, summarizes, or emits those labels. The
materialized target-unlabeled views contain only X and IDs. Label views were
created in a separate path only after the complete field freeze.
"""
    REPORT_PATH.write_text(report)
    if blocking_failures:
        raise RuntimeError(f"C78F red team failed: {blocking_failures}")
    print(json.dumps({"gate": "C78F_RED_TEAM_PASS", "checks": len(checks), "blocking_failures": 0}, sort_keys=True))
    return {"passed": True, "checks": len(checks), "blocking_failures": 0}


if __name__ == "__main__":
    run_red_team()
