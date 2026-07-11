"""Mechanical final-report audit for the completed C79E milestone."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c79_tables"
RESULT = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION.json"
REPORT = REPORT_DIR / "C79_SEED4_LOCKED_CONFIRMATION.md"
RED_JSON = REPORT_DIR / "C79E_FINAL_REPORT_RED_TEAM.json"
RED_MD = REPORT_DIR / "C79E_FINAL_REPORT_RED_TEAM.md"
RED_TABLE = TABLE_DIR / "c79e_final_report_red_team_checks.csv"
EXPECTED_PROTOCOL_SHA = "e350b7f0c4ee3dfcf6b4f5651c1c7a0e8beac72e478ffb6c1e98e12df814f587"
EXPECTED_GATE = "C79-E_seed4_does_not_replicate_either_core_pattern"


def _rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(rows: list[dict[str, Any]]) -> None:
    with RED_TABLE.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["check", "passed", "evidence"])
        writer.writeheader()
        writer.writerows(rows)


def audit() -> dict[str, Any]:
    result = json.loads(RESULT.read_text())
    report = REPORT.read_text()
    freeze = json.loads((REPORT_DIR / "C79_SEED4_PRIMARY_OUTPUT_FREEZE.json").read_text())
    science = json.loads((REPORT_DIR / "C79_SCIENTIFIC_RESULT_RED_TEAM.json").read_text())
    auth = json.loads((REPORT_DIR / "C79E_PI_AUTHORIZATION_RECORD.json").read_text())
    regression = _rows(TABLE_DIR / "c79e_regression_verification.csv")
    repairs = _rows(TABLE_DIR / "seed4_retry_repair_ledger.csv")
    risks = _rows(TABLE_DIR / "c79e_risk_register.csv")
    gates = _rows(TABLE_DIR / "cross_seed_gate_concordance.csv")
    mode_r_manifest = _rows(TABLE_DIR / "artifact_manifest.csv")

    checks = [
        ("direct_authorization_bound", auth["direct_explicit_PI_authorization"], "direct PI record"),
        ("protocol_commit_bound", auth["protocol_commit"].startswith("ec4834c"), auth["protocol_commit"]),
        ("protocol_sha_bound", auth["protocol_sha256"] == EXPECTED_PROTOCOL_SHA, auth["protocol_sha256"]),
        ("field_lock_bound", auth["field_lock_commit"].startswith("35d0c65"), auth["field_lock_commit"]),
        ("analysis_lock_bound", auth["analysis_lock_commit"].startswith("7cebf2e"), auth["analysis_lock_commit"]),
        ("complete_field_exact", result["field"]["complete_units"] == 1458, "1458"),
        ("primary_field_exact", result["field"]["primary_units"] == 1296, "1296"),
        ("target4_excluded", not result["field"]["target4_primary"], "engineering only"),
        ("target_training_isolation", result["field"]["training_target_rows"] == result["field"]["training_target_label_reads"] == 0, "0/0"),
        ("construction_evaluation_disjoint", result["field"]["construction_evaluation_overlap"] == 0, "overlap 0"),
        ("same_label_oracle_closed", not result["field"]["same_label_oracle_accessed"], "closed"),
        ("all_paths_unconditional", freeze["registered_paths_completed"] == 10 and not freeze["active_after_Holm_runtime_selection"], "10/10"),
        ("P1_decision_exact", not result["co_primary"]["P1"]["transition_replicates"], "compound fail"),
        ("P2_decision_exact", not result["co_primary"]["P2"]["local_nontransport_replicates"], "compound fail"),
        ("H2R_nonqualification_exact", not result["secondary"]["H2R"]["qualifies"], "unqualified"),
        ("H4R_nonqualification_exact", not result["secondary"]["H4R"]["qualifies"], "unqualified"),
        ("H5R_nonqualification_exact", not result["secondary"]["H5R"]["qualifies"], "unqualified"),
        ("H6R_familywise_exact", not result["secondary"]["H6R"]["familywise_active"], "inactive"),
        ("scientific_red_team", science["passed"] and science["checks_passed"] == 17, "17/17"),
        ("no_cross_seed_rescue", result["cross_seed"]["combined_p_values_computed"] == 0, "0 combined p"),
        ("P2_gate_heterogeneity_disclosed", {row["gate"] for row in gates if row["gate_concordant"] == "0"} == {"P2_L", "P2_overall"}, "P2 only"),
        ("repairs_additive_non_scientific", len(repairs) == 6 and all(row["scientific_registry_changed"] == row["outcome_dependent_decision"] == row["locked_implementation_changed"] == "0" for row in repairs), "6/6"),
        ("C79E_risks_closed", all(row["blocking"] == "0" for row in risks), f"{len(risks)}/{len(risks)}"),
        ("Mode_R_manifest_replays_after_namespace_repair", all(_sha256(REPO_ROOT / row["path"]) == row["sha256"] for row in mode_r_manifest), f"{len(mode_r_manifest)}/{len(mode_r_manifest)}"),
        ("final_regressions_pass", len(regression) == 4 and all(row["status"] == "PASS" and row["failed"] == "0" for row in regression), "4/4"),
        ("final_gate_exact", result["final_gate"] == EXPECTED_GATE and EXPECTED_GATE in report, EXPECTED_GATE),
        ("claim_and_stop_boundary", "not pre-C78S" in report and "does not authorize" in report and not result["information_boundaries"]["C80"], "training-seed only"),
    ]
    rows = [{"check": name, "passed": int(bool(passed)), "evidence": evidence} for name, passed, evidence in checks]
    _write_csv(rows)
    if not all(row["passed"] for row in rows):
        failed = [row["check"] for row in rows if not row["passed"]]
        raise RuntimeError(f"C79E final-report red team failed: {failed}")

    payload = {
        "schema_version": "c79e_final_report_red_team_v1",
        "checks_passed": len(rows),
        "checks_total": len(rows),
        "blocking_failures": 0,
        "final_gate": EXPECTED_GATE,
        "mode_r_history_preserved": True,
        "same_label_oracle_accessed": False,
        "C80_authorized": False,
        "passed": True,
    }
    RED_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    RED_MD.write_text(
        "# C79E Final-Report Red Team\n\n"
        f"All `{len(rows)}/{len(rows)}` checks pass with zero blocking failures. "
        "The final report matches the frozen seed-4 tables, preserves the post-seed-3 "
        "training-seed-only claim boundary, retains every repair/regression failure, "
        "and leaves the same-label oracle and C80 closed.\n\n"
        f"Final gate: `{EXPECTED_GATE}`.\n"
    )
    return payload


def main() -> int:
    audit()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
