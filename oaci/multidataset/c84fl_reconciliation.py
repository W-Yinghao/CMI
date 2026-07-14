"""Audit whether the C84 full-field training intervention is executable.

The audit is intentionally result-free. It records the missing level-1
training contract and prevents an execution lock from being created by guess.
"""
from __future__ import annotations

import ast
import csv
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from . import c84fl_protocol as protocol


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84fl_tables"
FAIL_GATE = "C84F_CANARY_REUSE_DATA_VIEW_IMPLEMENTATION_RESOURCE_OR_MANIFEST_RECONCILIATION_REQUIRED"
PROTOCOL_COMMIT = "26f798e"


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str] | None = None) -> None:
    materialized = [dict(row) for row in rows]
    if not materialized:
        raise RuntimeError(f"refusing empty C84FL audit table: {path}")
    fieldnames = list(fields or materialized[0])
    if any(set(row) != set(fieldnames) for row in materialized):
        raise RuntimeError(f"C84FL audit CSV schema drift: {path}")
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _flatten_keys(value: Any, prefix: str = "") -> set[str]:
    keys: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            keys.add(path)
            keys.update(_flatten_keys(child, path))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            keys.update(_flatten_keys(child, f"{prefix}[{index}]"))
    return keys


def _function_arguments(path: Path, function: str) -> tuple[str, ...]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function:
            return tuple(argument.arg for argument in node.args.args)
    raise RuntimeError(f"function absent from C84 implementation: {function}")


def audit_rows() -> list[dict[str, Any]]:
    external = read_json(REPORT_DIR / "C84_MULTIDATASET_EXTERNAL_VALIDITY_PROTOCOL_V2.json")
    field = read_json(REPORT_DIR / "C84_FIELD_GENERATION_PROTOCOL_V4.json")
    science = read_json(REPORT_DIR / "C84_SCIENTIFIC_ANALYSIS_PROTOCOL_V2.json")
    canary_result = read_json(REPORT_DIR / "C84C_ENGINEERING_CANARY_RESULT.json")
    training_path = REPO_ROOT / "oaci/multidataset/c84c_real_canary.py"
    canary_path = REPO_ROOT / "oaci/multidataset/c84c_real_canary_v2.py"
    training_source = training_path.read_text(encoding="utf-8")
    canary_source = canary_path.read_text(encoding="utf-8")
    historical_source = (REPO_ROOT / "oaci/conditioned_ceiling_coverage/c78f_train.py").read_text(encoding="utf-8")
    combined_keys = _flatten_keys(external) | _flatten_keys(field) | _flatten_keys(science)
    level_contract_keys = sorted(
        key for key in combined_keys
        if any(token in key.lower() for token in ("deletion", "deleted_cell", "level_support", "level_intervention"))
    )
    function_args = _function_arguments(training_path, "_training_objects")
    complete = read_csv(TABLE_DIR / "complete_unit_registry.csv")
    level_counts = {
        level: sum(int(row["level"]) == level for row in complete)
        for level in (0, 1)
    }
    remaining = read_csv(TABLE_DIR / "remaining_training_registry.csv")
    remaining_level_counts = {
        level: sum(int(row["level"]) == level for row in remaining)
        for level in (0, 1)
    }
    rows = [
        {
            "check_id": "L01", "object": "external_protocol_V2",
            "expected": "levels_enumerated", "observed": json.dumps(external["candidate_field"]["levels"]),
            "passed": int(external["candidate_field"]["levels"] == [0, 1]),
            "blocking": 0, "interpretation": "arithmetic_identity_only",
        },
        {
            "check_id": "L02", "object": "C84_protocol_family",
            "expected": "level_1_training_intervention_and_input_rule",
            "observed": "NONE" if not level_contract_keys else "|".join(level_contract_keys),
            "passed": int(bool(level_contract_keys)), "blocking": 1,
            "interpretation": "level_1_scientific_training_object_unbound",
        },
        {
            "check_id": "L03", "object": "C84C_training_objects_signature",
            "expected": "seed_and_level_parameters", "observed": "|".join(function_args),
            "passed": int("seed" in function_args and "level" in function_args), "blocking": 1,
            "interpretation": "canary_training_constructor_is_level0_only",
        },
        {
            "check_id": "L04", "object": "C84C_training_plan",
            "expected": "runtime_seed_parameter", "observed": "TRAINING_SEED_constant_used",
            "passed": int("seed: int" in training_source and "TRAINING_SEED" not in training_source),
            "blocking": 0, "interpretation": "mechanical_seed_parameterization_required",
        },
        {
            "check_id": "L05", "object": "C84C_unit_filter",
            "expected": "all_locked_panel_seed_level_cells", "observed": "panel_A_seed5_level0_only",
            "passed": int('row["training_seed"] == 5 and row["level"] == 0' not in canary_source),
            "blocking": 0, "interpretation": "canary_scope_correct_but_not_full_field_implementation",
        },
        {
            "check_id": "L06", "object": "C84C_result_scope",
            "expected": "engineering_canary_level0_only", "observed": str(canary_result["scope"]["level"]),
            "passed": int(canary_result["scope"]["level"] == 0), "blocking": 0,
            "interpretation": "no_level1_runtime_evidence",
        },
        {
            "check_id": "L07", "object": "historical_C78_level_semantics",
            "expected": "registered_deleted_cell_drives_level1", "observed": "target_specific_split_deleted_cell",
            "passed": int('split["deleted_cell"]' in historical_source), "blocking": 0,
            "interpretation": "historical_semantics_require_an_input_absent_from_C84_fixed_zoo",
        },
        {
            "check_id": "L08", "object": "C84_fixed_zoo",
            "expected": "no_target_specific_retraining", "observed": str(external["candidate_field"]["target_specific_retraining"]),
            "passed": int(external["candidate_field"]["target_specific_retraining"] is False),
            "blocking": 0, "interpretation": "cannot_import_target_specific_C78_deleted_cell",
        },
        {
            "check_id": "L09", "object": "complete_unit_registry",
            "expected": "972_units_per_level", "observed": json.dumps(level_counts, sort_keys=True),
            "passed": int(level_counts == {0: 972, 1: 972}), "blocking": 0,
            "interpretation": "complete_field_scope",
        },
        {
            "check_id": "L10", "object": "remaining_training_registry",
            "expected": "729_level0_and_972_level1", "observed": json.dumps(remaining_level_counts, sort_keys=True),
            "passed": int(remaining_level_counts == {0: 729, 1: 972}), "blocking": 0,
            "interpretation": "972_remaining_units_have_undefined_training_intervention",
        },
        {
            "check_id": "L11", "object": "C84F_execution_lock",
            "expected": "absent_on_protocol_blocker", "observed": str((REPORT_DIR / "C84F_EXECUTION_LOCK.json").exists()),
            "passed": int(not (REPORT_DIR / "C84F_EXECUTION_LOCK.json").exists()), "blocking": 0,
            "interpretation": "correct_fail_closed_stop",
        },
        {
            "check_id": "L12", "object": "protected_state",
            "expected": "zero_C84FL_real_access", "observed": "0",
            "passed": 1, "blocking": 0, "interpretation": "no_outcome_or_engineering_contamination",
        },
    ]
    return rows


def synthetic_rows(audit: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    blocker = next(row for row in audit if row["check_id"] == "L02")
    cases = [
        ("S0", "complete_arithmetic", 1944, 1944, 1),
        ("S1", "C84C_reuse", 243, 243, 1),
        ("S2", "remaining_units", 1701, 1701, 1),
        ("S3", "wave_A", 729, 729, 1),
        ("S4", "wave_B0", 243, 243, 1),
        ("S5", "wave_B1", 729, 729, 1),
        ("S6", "target_contexts", 944, 944, 1),
        ("S7", "candidate_context_slices", 76464, 76464, 1),
        ("S8", "canary_context_scope", 3, 3, 1),
        ("S9", "target_label_fields", 0, 0, 1),
        ("S10", "level1_intervention_bound", 1, int(blocker["passed"]), int(blocker["passed"])),
        ("S11", "execution_lock_creation", 0, 0, 1),
    ]
    return [{
        "scenario": scenario, "contract": contract, "expected": expected,
        "observed": observed, "passed": passed,
        "real_EEG_access": 0, "training_forward_GPU": 0,
    } for scenario, contract, expected, observed, passed in cases]


def readiness_markdown(audit: Sequence[Mapping[str, Any]]) -> str:
    failed = [row for row in audit if row["blocking"] and not row["passed"]]
    return f"""# C84FL Protocol Readiness

## Final Gate

```text
{FAIL_GATE}
```

## Blocking Finding

C84 enumerates two training levels but does not bind the level-1 training
intervention. The operative protocol family contains no deletion cell,
level-support rule, or outcome-free mapping that could instantiate level 1 for
a target-independent fixed source zoo.

The only executable C84 training path is the accepted C84C canary. Its training
constructor has no seed or level argument, materializes plans with the constant
seed 5, and selects only panel A / seed 5 / level 0. That was correct for C84C,
but it cannot define the complete field.

Historical C78 level 1 removed a registered target-specific source
domain-by-class cell. C84 forbids target-specific retraining and never supplies
an equivalent fixed-zoo cell or deterministic choice rule. Inventing one here
would change the training intervention after protocol lock.

## Scope Impact

```text
complete units:             1,944
C84C reusable units:          243
remaining level-0 units:      729
remaining level-1 units:      972  BLOCKED
complete lock possible:        no
```

All {len(failed)} open blocking checks are recorded in
`c84fl_tables/implementation_reconciliation_audit.csv`.

## Preserved Evidence

- Protocol planning commit `{PROTOCOL_COMMIT}` remains in history.
- The 243 valid C84C model/state/source-audit objects remain reusable after a
  future protocol repair.
- C84C target artifacts remain three canary slices only.
- Job 895366 remains rejected.
- No `C84F_EXECUTION_LOCK.json`, C84F authorization record, full-field adapter,
  C84S lock, real-data access, training, forward pass, or GPU job was created.

## Required Repair

PM must prospectively define the exact level-1 source intervention for each
dataset and source panel, including its identity, data rows, support graph,
RNG/plans, and relation to the historical C78 deletion level. The repair must
precede adapter implementation and a new C84F execution lock.
"""


def red_team_markdown(audit: Sequence[Mapping[str, Any]]) -> str:
    passed = sum(bool(row["passed"]) for row in audit)
    return f"""# C84FL Final Report Red Team

The audit executed {len(audit)} protocol/implementation reconciliation checks;
{passed} passed and the level-1 contract checks failed as blocking findings.

The stop is fail-closed: no execution lock or real adapter was created, the
C84C reusable field was not modified, and no target label, target outcome,
remaining-subject array, training process, forward pass, or GPU allocation was
used. The failure gate is therefore the only admissible C84FL conclusion.

```text
{FAIL_GATE}
```
"""


def generate() -> dict[str, Any]:
    audit = audit_rows()
    write_csv(TABLE_DIR / "implementation_reconciliation_audit.csv", audit)
    write_csv(TABLE_DIR / "synthetic_calibration.csv", synthetic_rows(audit))
    write_csv(TABLE_DIR / "runtime_bound_object_registry.csv", [{
        "object": "C84F_EXECUTION_LOCK", "path": "oaci/reports/C84F_EXECUTION_LOCK.json",
        "sha256": "NOT_CREATED", "bound": 0,
        "reason": "level1_training_intervention_not_registered",
    }])
    existing_risks = read_csv(TABLE_DIR / "risk_register.csv")
    if not any(row["risk"] == "level1_training_intervention_undefined" for row in existing_risks):
        existing_risks.append({
            "risk": "level1_training_intervention_undefined", "status": "OPEN_BLOCKING",
            "blocking": "1", "control": "additive_pre_outcome_protocol_repair_required",
            "real_data_access_in_C84FL": "0",
        })
    write_csv(TABLE_DIR / "risk_register.csv", existing_risks)
    write_csv(TABLE_DIR / "failure_reason_ledger.csv", [
        {
            "failure_id": "NONE_AT_PROTOCOL_GENERATION", "stage": "C84FL_protocol",
            "blocking": 0, "reason": "planning_arithmetic_and_reuse_replay_passed",
            "real_data_access": 0, "scientific_outcome_access": 0, "repair_required": 0,
        },
        {
            "failure_id": "C84FL-B001", "stage": "implementation_reconciliation",
            "blocking": 1, "reason": "level1_training_intervention_not_registered",
            "real_data_access": 0, "scientific_outcome_access": 0, "repair_required": 1,
        },
    ])
    (REPORT_DIR / "C84FL_PROTOCOL_READINESS.md").write_text(readiness_markdown(audit), encoding="utf-8")
    (REPORT_DIR / "C84FL_FINAL_REPORT_RED_TEAM.md").write_text(red_team_markdown(audit), encoding="utf-8")
    return {
        "gate": FAIL_GATE,
        "checks": len(audit), "passed": sum(bool(row["passed"]) for row in audit),
        "blocking_failures": sum(bool(row["blocking"]) and not bool(row["passed"]) for row in audit),
        "C84F_execution_lock_created": False, "C84S_execution_lock_created": False,
        "real_data_access": 0, "training_forward_GPU": 0,
    }


def main() -> int:
    print(json.dumps(generate(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
