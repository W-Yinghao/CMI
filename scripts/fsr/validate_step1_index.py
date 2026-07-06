#!/usr/bin/env python
"""FSR Step 2A — fail-closed validator for the Step-1 artifact index.

Pure validation. Reads results/fsr_artifact_index/artifact_index.csv, runs a fixed battery of
structural + policy checks, and writes results/fsr_phase2/schema_validation.json. Exits non-zero
if ANY check fails (fail-closed). No artifact is modified; no analysis is performed.

    python scripts/fsr/validate_step1_index.py

Checks (Step 2A spec):
  - parses with csv.reader AND pandas
  - row_count == 37
  - header == the exact 18-column schema
  - all required routes present
  - target_labels_used_for_fit in {NO, YES_FORBIDDEN, AUDIT_ONLY, UNKNOWN}
  - YES_FORBIDDEN only in the allowed retracted-legacy scope (LPC_CMI_legacy_boundary)
  - UNKNOWN fit-tag rows have non-empty notes
  - artifact_exists in {YES, NO}
  - status in the allowed vocabulary
  - no CMI-control-failure route is marked as a positive method
"""
from __future__ import annotations
import csv, json, sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
INDEX = REPO / "results" / "fsr_artifact_index" / "artifact_index.csv"
OUT = REPO / "results" / "fsr_phase2" / "schema_validation.json"

EXPECTED_COLS = [
    "source_branch", "source_sha", "component", "route", "dataset", "backbone", "method",
    "info_regime", "seeds", "folds_or_subjects", "representation_or_branch", "artifact_path",
    "artifact_exists", "metrics_available", "target_labels_used_for_fit",
    "target_labels_used_for_eval", "status", "notes",
]
EXPECTED_ROW_COUNT = 37

VALID_FIT = {"NO", "YES_FORBIDDEN", "AUDIT_ONLY", "UNKNOWN"}
VALID_EXISTS = {"YES", "NO"}
# Closed status vocabulary (exactly the kinds used in Step 1; extend deliberately, not silently).
ALLOWED_STATUS = {
    "FROZEN_PREMISE", "FROZEN_NEGATIVE", "FROZEN_DIAGNOSTIC", "FROZEN_CONTROL", "FROZEN_RESULT",
    "FROZEN_POSITIVE_BRANCHLOAD", "POSITIVE_NON_CMI", "BACKGROUND", "DESIGN_ONLY",
    "CLOSED_NEGATIVE", "PROTOCOL_ONLY", "PENDING",
}
# Statuses that assert a positive (non-negative) result.
POSITIVE_STATUS = {"POSITIVE_NON_CMI", "FROZEN_POSITIVE_BRANCHLOAD"}
# The only routes permitted to carry a positive status, and why.
ALLOWED_POSITIVE_ROUTES = {
    "TTA_Control_non_CMI": "explicitly NON-CMI target-unlabeled positive (walled off from CMI-control)",
    "FBCSP_LGG_branch_ablation": "L4 branch-load result, not a CMI/leakage positive",
}
# YES_FORBIDDEN is only allowed on this retracted-legacy route (scoped, disclosed).
ALLOWED_YES_FORBIDDEN_ROUTES = {"LPC_CMI_legacy_boundary"}

REQUIRED_ROUTES = [
    # frozen premises
    "CIGL_70_closure", "CMI_SYNTHESIS",
    # CMI-control cluster
    "CIGL", "FCIGL", "dCIGL", "MetaCMI", "CITA_lambda_1", "CITA_lambda_0.010", "TTA_Control_non_CMI",
    # TOS erasure cluster
    "TOS_mean_scatter", "TOS_LEACE", "TOS_INLP", "TOS_RLACE", "TOS_random_k", "TOS_refusal_gate",
    "TOS_global_LPC_collapse", "TOS_task_preserving_erasure", "TOS_capacity_factorial",
    # FBCSP branch cluster
    "FBCSP_LGG_branch_ablation", "FBCSP_LGG_gate_summary", "FBCSP_LGG_bottleneck_analysis",
    "FBCSP_LGG_graph_starvation", "CIGL_35_blueprint", "P6_spatial_CMI_scaffold",
    # OACI
    "OACI_selection_leakage_not_target", "OACI_source_audit_oracle_failure",
    "OACI_multivariate_weak_identifiability", "OACI_endpoint_estimability_limit",
    # ACAR
    "ACAR_paired_action_risk_design", "ACAR_v5_protocol_substrate_success", "ACAR_stage2b_dev_stop",
    # CSC
    "CSC_Z_only_unidentifiable", "CSC_dual_witness_candidate", "CSC_information_contract_boundary",
    # LPC legacy + H2CMI background
    "LPC_CMI_legacy_boundary",
    "PriorDecoupled_four_branch_protocol", "PriorDecoupled_geometry_vs_prevalence",
]


def main() -> int:
    checks: list[dict] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # -- parse with csv.reader --
    rows = []
    try:
        with open(INDEX, newline="") as fh:
            rows = list(csv.reader(fh))
        csv_ok = True
        csv_detail = f"parsed {len(rows)} lines (incl header)"
    except Exception as e:  # noqa: BLE001
        csv_ok, csv_detail = False, f"csv.reader failed: {e}"
    check("parse_csv_reader", csv_ok, csv_detail)

    # -- parse with pandas --
    try:
        import pandas as pd
        df = pd.read_csv(INDEX, dtype=str, keep_default_na=False)
        check("parse_pandas", True, f"pandas read {len(df)} data rows x {len(df.columns)} cols")
    except Exception as e:  # noqa: BLE001
        df = None
        check("parse_pandas", False, f"pandas failed: {e}")

    if not csv_ok or len(rows) < 2:
        _write(checks, False)
        return 1

    header, data = rows[0], rows[1:]

    check("header_exact_18col", header == EXPECTED_COLS,
          "OK" if header == EXPECTED_COLS else f"got {header}")
    check("row_count_37", len(data) == EXPECTED_ROW_COUNT, f"row_count={len(data)}")
    check("all_rows_18_fields", all(len(r) == len(EXPECTED_COLS) for r in data),
          "OK" if all(len(r) == len(EXPECTED_COLS) for r in data)
          else f"bad: {[(i + 2, len(r)) for i, r in enumerate(data) if len(r) != len(EXPECTED_COLS)]}")

    idx = {c: i for i, c in enumerate(header)} if header == EXPECTED_COLS else {c: i for i, c in enumerate(EXPECTED_COLS)}

    def col(r, name):
        return r[idx[name]] if idx[name] < len(r) else ""

    routes = [col(r, "route") for r in data]
    missing_routes = [x for x in REQUIRED_ROUTES if x not in routes]
    check("all_required_routes_present", not missing_routes,
          "OK" if not missing_routes else f"missing={missing_routes}")

    dup = sorted({x for x in routes if routes.count(x) > 1})
    check("routes_unique", not dup, "OK" if not dup else f"duplicate routes={dup}")

    bad_fit = [(col(r, "route"), col(r, "target_labels_used_for_fit")) for r in data
               if col(r, "target_labels_used_for_fit") not in VALID_FIT]
    check("fit_tag_in_vocab", not bad_fit, "OK" if not bad_fit else f"bad={bad_fit}")

    bad_yesforbidden = [col(r, "route") for r in data
                        if col(r, "target_labels_used_for_fit") == "YES_FORBIDDEN"
                        and col(r, "route") not in ALLOWED_YES_FORBIDDEN_ROUTES]
    check("yes_forbidden_scope", not bad_yesforbidden,
          "OK (only scoped legacy)" if not bad_yesforbidden else f"unexpected YES_FORBIDDEN={bad_yesforbidden}")

    bad_unknown = [col(r, "route") for r in data
                   if col(r, "target_labels_used_for_fit") == "UNKNOWN" and not col(r, "notes").strip()]
    check("unknown_rows_have_notes", not bad_unknown,
          "OK (no UNKNOWN rows)" if not bad_unknown else f"UNKNOWN without notes={bad_unknown}")

    bad_exists = [(col(r, "route"), col(r, "artifact_exists")) for r in data
                  if col(r, "artifact_exists") not in VALID_EXISTS]
    check("artifact_exists_in_vocab", not bad_exists, "OK" if not bad_exists else f"bad={bad_exists}")

    bad_status = [(col(r, "route"), col(r, "status")) for r in data
                  if col(r, "status") not in ALLOWED_STATUS]
    check("status_in_vocab", not bad_status, "OK" if not bad_status else f"bad={bad_status}")

    # No CMI-control-failure route marked positive; positive statuses only on allowed routes.
    bad_positive = [(col(r, "route"), col(r, "status")) for r in data
                    if col(r, "status") in POSITIVE_STATUS and col(r, "route") not in ALLOWED_POSITIVE_ROUTES]
    check("no_unexpected_positive_status", not bad_positive,
          "OK" if not bad_positive else f"unexpected positive={bad_positive}")

    # Any CMI_control component route that is a failure must NOT be positive.
    cmi_fail_positive = [col(r, "route") for r in data
                         if col(r, "component") == "CMI_control"
                         and col(r, "status") in POSITIVE_STATUS
                         and col(r, "route") != "TTA_Control_non_CMI"]
    check("no_cmi_control_failure_as_positive", not cmi_fail_positive,
          "OK (only TTA_Control_non_CMI is positive, explicitly non-CMI)"
          if not cmi_fail_positive else f"violations={cmi_fail_positive}")

    # No meaning-bearing field blank (route, status, fit, exists, artifact_path).
    blank = [(col(r, "route"), c) for r in data for c in
             ("route", "status", "target_labels_used_for_fit", "artifact_exists", "artifact_path")
             if not col(r, c).strip()]
    check("no_blank_key_fields", not blank, "OK" if not blank else f"blanks={blank}")

    all_pass = all(c["pass"] for c in checks)
    _write(checks, all_pass)
    print(("PASS" if all_pass else "FAIL") + f" — {sum(c['pass'] for c in checks)}/{len(checks)} checks")
    for c in checks:
        if not c["pass"]:
            print(f"  FAIL: {c['check']} — {c['detail']}")
    return 0 if all_pass else 1


def _write(checks, all_pass):
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "validator": "fsr/validate_step1_index.py",
        "index": str(INDEX.relative_to(REPO)),
        "expected_row_count": EXPECTED_ROW_COUNT,
        "expected_columns": EXPECTED_COLS,
        "all_pass": bool(all_pass),
        "n_checks": len(checks),
        "n_pass": sum(c["pass"] for c in checks),
        "checks": checks,
    }, indent=2) + "\n")


if __name__ == "__main__":
    sys.exit(main())
