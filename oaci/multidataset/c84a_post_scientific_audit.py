"""Build the read-only C84A post-scientific heterogeneity audit.

C84A consumes only frozen compact result tables, lifecycle receipts, and the
committed C80/C82/C83 evidence snapshots.  It does not import or invoke any
selector, Q0, inference, label-view, model, or field-array implementation.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable, Mapping, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci" / "reports"
TABLE_DIR = REPORT_DIR / "c84a_tables"
C84S_ROOT = Path("/projects/EEG-foundation-model/yinghao/oaci-c84s-analysis-v5")
C84S_RESULT_DIR = C84S_ROOT / "stage_c_scientific_result"
C80_TABLE_DIR = REPORT_DIR / "c80e_tables"
C82_TABLE_DIR = REPORT_DIR / "c82e_tables"
C83_TABLE_DIR = REPORT_DIR / "c83p_tables"

STATUS = "POST_C84S_EXPLORATORY_DESCRIPTIVE"
GATE = "C84-D_external_dataset_source_panel_seed_level_or_target_heterogeneous"
FRONTIER_TAG = "C84-L4"
SUCCESS_GATE = (
    "C84_POST_SCIENTIFIC_HETEROGENEITY_AND_TPAMI_THEORY_BRIDGE_"
    "AUDIT_COMPLETE_C85_PROTOCOL_REVIEW_REQUIRED"
)
FAILURE_GATE = (
    "C84_POST_SCIENTIFIC_RESULT_IDENTITY_HETEROGENEITY_OR_"
    "THEORY_BRIDGE_RECONCILIATION_REQUIRED"
)

DATASETS = ("Lee2019_MI", "Cho2017", "PhysionetMI")
TARGET_COUNTS = {"Lee2019_MI": 22, "Cho2017": 20, "PhysionetMI": 76}
PRIMARY_METHODS = ("U5", "U7", "U11", "U13", "U14", "U15")
METHOD_NAMES = {
    "S1": "Strict source",
    "U5": "NuclearNorm",
    "U7": "ATC",
    "U11": "MaNo",
    "U13": "COTT",
    "U14": "SND",
    "U15": "Agreement-on-the-Line",
    "Q0_B1": "Q0 B=1",
    "Q0_FULL": "Q0 FULL",
}
PRIMARY_BUDGETS = ("1", "2", "4", "8", "FULL")

EXPECTED_IDENTITIES = {
    "C84S_analysis_lock": (
        REPORT_DIR / "C84S_ANALYSIS_EXECUTION_LOCK_V5.json",
        "030be9c9ebac401ca9e7ae5e51bb1ce99b592faceac00fac8781070420b0b846",
    ),
    "C84S_authorization": (
        REPORT_DIR / "C84S_V5_PI_AUTHORIZATION_RECORD.json",
        "3446e3562a8dd5db51c9f56a03765bf040f9678ee527ea13a4cf75e63dd575e1",
    ),
    "C84S_selection_freeze": (
        C84S_ROOT / "stage_b_selection_freeze" / "C84S_SELECTION_FREEZE_MANIFEST_V3.json",
        "30ad539c8758a15701a582f0391671682107beb694860c9c531856425f2c7df4",
    ),
    "C84S_scientific_result": (
        C84S_RESULT_DIR / "C84S_RESULT.json",
        "5590f85c3552ec0176a015e34296059a950dd2c5853a51aa140657cf53d79ee7",
    ),
    "C84S_result_manifest": (
        C84S_RESULT_DIR / "C84S_RESULT_ARTIFACT_MANIFEST.json",
        "516ae135125d66233c9ee87aa71e5b40941fcb9140a63c036f58b40fce11a2b5",
    ),
}

LIFECYCLE_FILES = {
    "authorization_consumption": C84S_ROOT / "authorization_consumed.json",
    "lifecycle": C84S_ROOT / "C84S_V5_LIFECYCLE_ATTEMPT.json",
    "stage_A_attempt": C84S_ROOT / "stage_a_immutable_replay.stage_a_replay_attempt.json",
    "stage_A_receipt": C84S_ROOT / "stage_a_replay_subprocess_receipt.json",
    "stage_B_attempt": C84S_ROOT / "stage_b_selection_freeze.stage_b_attempt.json",
    "stage_B_receipt": C84S_ROOT / "stage_b_v3_subprocess_receipt.json",
    "stage_C_attempt": C84S_ROOT / "stage_c_scientific_result.stage_c_attempt.json",
    "stage_C_receipt": C84S_ROOT / "stage_c_v3_subprocess_receipt.json",
}


class C84AAuditError(RuntimeError):
    """Raised when a frozen identity or read-only audit contract drifts."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(path: Path) -> Any:
    return json.loads(path.read_text())


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise C84AAuditError(f"refusing to write empty C84A table: {path.name}")
    fields = list(rows[0])
    if any(list(row) != fields for row in rows):
        raise C84AAuditError(f"schema drift while writing {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=True) + "\n")


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, check=check,
        capture_output=True, text=True,
    )


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _source_key(**values: Any) -> str:
    return "|".join(f"{key}={value}" for key, value in values.items())


def _tag(
    row: Mapping[str, Any], *, sources: Iterable[Path], source_keys: Iterable[str],
) -> dict[str, Any]:
    return {
        **row,
        "analysis_status": STATUS,
        "source_artifacts": "|".join(_relative(path) for path in sources),
        "source_row_keys": "||".join(source_keys),
        "confirmatory_gate_changed": 0,
    }


def _as_float(value: str | int | float | None) -> float | None:
    if value in (None, "", "NA", "NONE", "NOT_FROZEN"):
        return None
    return float(value)


def _as_bool(value: Any) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "pass"}


def _lookup(
    rows: Sequence[Mapping[str, str]], **key: Any,
) -> dict[str, str]:
    matches = [
        dict(row) for row in rows
        if all(str(row.get(field)) == str(value) for field, value in key.items())
    ]
    if len(matches) != 1:
        raise C84AAuditError(f"expected one row for {key}, found {len(matches)}")
    return matches[0]


def _mean(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        raise C84AAuditError("mean requires at least one value")
    return sum(materialized) / len(materialized)


def _quantile(values: Iterable[float], probability: float) -> float:
    ordered = sorted(values)
    if not ordered:
        raise C84AAuditError("quantile requires at least one value")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _manifest_tables() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    manifest_path = C84S_RESULT_DIR / "C84S_RESULT_ARTIFACT_MANIFEST.json"
    manifest = read_json(manifest_path)
    artifacts = manifest.get("artifacts", [])
    if manifest.get("table_count") != 18 or len(artifacts) != 18:
        raise C84AAuditError("C84S result manifest must contain exactly 18 tables")
    if len({row["path"] for row in artifacts}) != 18:
        raise C84AAuditError("duplicate C84S result table path")
    return manifest, artifacts


def _result_table(name: str) -> Path:
    _, artifacts = _manifest_tables()
    names = {row["path"] for row in artifacts}
    if name not in names:
        raise C84AAuditError(f"unregistered C84S result table requested: {name}")
    path = C84S_RESULT_DIR / name
    if path.parent != C84S_RESULT_DIR or path.suffix != ".csv":
        raise C84AAuditError(f"unsafe C84S table path: {path}")
    return path


def build_identity_replays() -> dict[str, list[dict[str, Any]]]:
    identity_rows: list[dict[str, Any]] = []
    for object_id, (path, expected) in EXPECTED_IDENTITIES.items():
        observed = sha256_file(path)
        identity_rows.append(_tag({
            "object_id": object_id,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "bytes": path.stat().st_size,
            "status": "PASS" if observed == expected else "FAIL",
        }, sources=[path], source_keys=["WHOLE_FILE_SHA256"]))

    result = read_json(C84S_RESULT_DIR / "C84S_RESULT.json")
    report = read_json(REPORT_DIR / "C84S_OVERALL_REPORT.json")
    c84s_final_commit = "2821c7099fc979672c4675e8c9ae54aa41ecd535"
    if git("merge-base", "--is-ancestor", c84s_final_commit, "HEAD", check=False).returncode != 0:
        raise C84AAuditError("C84S final commit is not an ancestor of the C84A worktree")
    identity_checks = {
        "C84S_final_commit": c84s_final_commit,
        "primary_gate": GATE,
        "label_frontier_tag": FRONTIER_TAG,
        "method_context_rows": "18432",
        "selection_contexts": "944",
    }
    observed_checks = {
        "C84S_final_commit": git("rev-parse", c84s_final_commit).stdout.strip(),
        "primary_gate": result["primary_gate"],
        "label_frontier_tag": result["label_frontier_tag"],
        "method_context_rows": str(result["method_context_rows"]),
        "selection_contexts": str(report["selection_freeze"]["contexts"]),
    }
    for object_id, expected in identity_checks.items():
        observed = observed_checks[object_id]
        identity_rows.append(_tag({
            "object_id": object_id,
            "expected_sha256": expected,
            "observed_sha256": observed,
            "bytes": "NA",
            "status": "PASS" if observed == expected else "FAIL",
        }, sources=[C84S_RESULT_DIR / "C84S_RESULT.json", REPORT_DIR / "C84S_OVERALL_REPORT.json"],
            source_keys=[object_id]))

    manifest, artifacts = _manifest_tables()
    table_rows: list[dict[str, Any]] = []
    for item in artifacts:
        path = _result_table(item["path"])
        observed_hash = sha256_file(path)
        observed_rows = len(read_csv(path))
        passed = observed_hash == item["sha256"] and observed_rows == int(item["rows"])
        table_rows.append(_tag({
            "table": item["path"],
            "expected_sha256": item["sha256"],
            "observed_sha256": observed_hash,
            "expected_rows": item["rows"],
            "observed_rows": observed_rows,
            "status": "PASS" if passed else "FAIL",
        }, sources=[C84S_RESULT_DIR / "C84S_RESULT_ARTIFACT_MANIFEST.json", path],
            source_keys=[_source_key(path=item["path"])]))

    lifecycle = read_json(LIFECYCLE_FILES["lifecycle"])
    stage_expectations = {
        "Stage_A_immutable_replay": ("COMPLETE", 0, 0),
        "Stage_B": ("COMPLETE", 0, 0),
        "Stage_C": ("COMPLETE", 1, 18432),
    }
    attempt_by_stage = {
        "Stage_A_immutable_replay": read_json(LIFECYCLE_FILES["stage_A_attempt"]),
        "Stage_B": read_json(LIFECYCLE_FILES["stage_B_attempt"]),
        "Stage_C": read_json(LIFECYCLE_FILES["stage_C_attempt"]),
    }
    lifecycle_rows: list[dict[str, Any]] = []
    for stage, (expected_status, evaluation_access, result_rows) in stage_expectations.items():
        attempt = attempt_by_stage[stage]
        observed_eval = int(attempt.get("evaluation_descriptor_received", 0))
        observed_rows = int(attempt.get("result", {}).get("method_context_rows", 0))
        if stage == "Stage_C":
            observed_rows = int(attempt["result"]["method_context_rows"])
        passed = (
            attempt["status"] == expected_status
            and observed_eval == evaluation_access
            and observed_rows == result_rows
        )
        lifecycle_rows.append(_tag({
            "stage": stage,
            "expected_status": expected_status,
            "observed_status": attempt["status"],
            "expected_evaluation_descriptor": evaluation_access,
            "observed_evaluation_descriptor": observed_eval,
            "expected_scientific_rows": result_rows,
            "observed_scientific_rows": observed_rows,
            "status": "PASS" if passed else "FAIL",
        }, sources=[LIFECYCLE_FILES["lifecycle"], next(
            path for key, path in LIFECYCLE_FILES.items()
            if key == {"Stage_A_immutable_replay": "stage_A_attempt", "Stage_B": "stage_B_attempt", "Stage_C": "stage_C_attempt"}[stage]
        )], source_keys=[stage]))
    if lifecycle["status"] != "COMPLETE":
        raise C84AAuditError("C84S lifecycle is not complete")

    counters = read_json(REPORT_DIR / "C84S_OVERALL_REPORT.json")["protected_counters"]
    expected_counters = {
        "stage_A_label_loader_calls": 0,
        "target_label_rows_reloaded": 0,
        "construction_label_access": 1,
        "evaluation_label_access": 1,
        "selector_score_contexts": 944,
        "scientific_result_rows": 18432,
        "training": 0,
        "forward": 0,
        "GPU": 0,
        "same_label_oracle": 0,
        "C85_authorized": False,
    }
    counter_rows = [_tag({
        "counter": key,
        "expected": int(value) if isinstance(value, bool) else value,
        "observed": int(counters[key]) if isinstance(counters[key], bool) else counters[key],
        "status": "PASS" if counters[key] == value else "FAIL",
    }, sources=[REPORT_DIR / "C84S_OVERALL_REPORT.json"], source_keys=[f"protected_counters/{key}"])
        for key, value in expected_counters.items()]

    verification = report["verification"]
    regression_rows: list[dict[str, Any]] = []
    for audit_id, values in (
        ("scientific_red_team", verification["scientific_red_team"]),
        ("final_report_red_team", verification["final_report_red_team"]),
    ):
        regression_rows.append(_tag({
            "audit_id": audit_id,
            "job": "NA",
            "passed": values["passed"],
            "failed": values["total"] - values["passed"],
            "skipped": 0,
            "deselected": 0,
            "stderr_bytes": 0,
            "status": "PASS" if values["passed"] == values["total"] else "FAIL",
        }, sources=[REPORT_DIR / "C84S_OVERALL_REPORT.json"], source_keys=[f"verification/{audit_id}"]))
    for suite, values in verification["regressions"].items():
        regression_rows.append(_tag({
            "audit_id": suite,
            "job": values["job"],
            "passed": values["passed"],
            "failed": values["failed"],
            "skipped": values["skipped"],
            "deselected": values["deselected"],
            "stderr_bytes": values["stderr_bytes"],
            "status": "PASS" if values["failed"] == 0 and values["stderr_bytes"] == 0 else "FAIL",
        }, sources=[REPORT_DIR / "C84S_OVERALL_REPORT.json"], source_keys=[f"verification/regressions/{suite}"]))

    return {
        "c84s_identity_replay.csv": identity_rows,
        "result_table_manifest_replay.csv": table_rows,
        "lifecycle_stage_replay.csv": lifecycle_rows,
        "protected_counter_replay.csv": counter_rows,
        "regression_redteam_replay.csv": regression_rows,
    }


def _component_pass_and_distances(row: Mapping[str, str], target_count: int) -> dict[str, Any]:
    minimum = math.ceil(0.75 * target_count)
    q1_mean = float(row["Q1_mean"])
    q1_p = float(row["Q1_pvalue"])
    q1_fav = int(row["Q1_favorable_targets"])
    q1_worst = float(row["Q1_worst_target"])
    q1_panel = _as_bool(row["panel_seed_Q1_all_directional"])
    q2_mean = float(row["Q2_mean_excess"])
    q2_upper = float(row["Q2_simultaneous_upper"])
    q2_p = float(row["Q2_pvalue"])
    q2_count = int(row["Q2_within_margin_targets"])
    q2_worst = float(row["Q2_worst_excess"])
    q2_panel = _as_bool(row["panel_seed_Q2_all_within_margin"])
    return {
        "target_count": target_count,
        "minimum_75pct_targets": minimum,
        "Q1_mean": q1_mean,
        "Q1_mean_pass": int(q1_mean >= 0.05),
        "Q1_mean_distance": q1_mean - 0.05,
        "Q1_maxT_p": q1_p,
        "Q1_maxT_pass": int(q1_p <= 0.05),
        "Q1_maxT_distance": 0.05 - q1_p,
        "Q1_favorable_targets": q1_fav,
        "Q1_favorable_pass": int(q1_fav >= minimum),
        "Q1_favorable_distance": q1_fav - minimum,
        "Q1_worst_target": q1_worst,
        "Q1_worst_pass": int(q1_worst >= -0.10),
        "Q1_worst_distance": q1_worst + 0.10,
        "Q1_panel_seed_pass": int(q1_panel),
        "Q1_panel_seed_distance": 0 if q1_panel else -1,
        "Q2_mean_excess": q2_mean,
        "Q2_mean_pass": int(q2_mean <= 0.05),
        "Q2_mean_distance": 0.05 - q2_mean,
        "Q2_simultaneous_upper": q2_upper,
        "Q2_upper_pass": int(q2_upper <= 0.05),
        "Q2_upper_distance": 0.05 - q2_upper,
        "Q2_maxT_p": q2_p,
        "Q2_maxT_pass": int(q2_p <= 0.05),
        "Q2_maxT_distance": 0.05 - q2_p,
        "Q2_within_margin_targets": q2_count,
        "Q2_target_count_pass": int(q2_count >= minimum),
        "Q2_target_count_distance": q2_count - minimum,
        "Q2_worst_excess": q2_worst,
        "Q2_worst_pass": int(q2_worst <= 0.20),
        "Q2_worst_distance": 0.20 - q2_worst,
        "Q2_panel_seed_pass": int(q2_panel),
        "Q2_panel_seed_distance": 0 if q2_panel else -1,
    }


def _nearest_failure(components: Mapping[str, tuple[bool, float, float]]) -> tuple[str, float | str]:
    failures = [
        (name, abs(distance) / scale)
        for name, (passed, distance, scale) in components.items()
        if not passed
    ]
    if not failures:
        return "NONE", "NA"
    return min(failures, key=lambda item: (item[1], item[0]))


def _context_effects(
    method_context: Sequence[Mapping[str, str]], dataset: str, method: str, level: int | None = None,
) -> list[dict[str, Any]]:
    selected = [
        row for row in method_context
        if row["dataset"] == dataset and row["method_id"] == method
        and (level is None or int(row["level"]) == level)
    ]
    index = {
        (row["dataset"], row["target_subject_id"], row["panel"], row["training_seed"], row["level"], row["method_id"]): row
        for row in method_context
    }
    effects = []
    for row in selected:
        base = (row["dataset"], row["target_subject_id"], row["panel"], row["training_seed"], row["level"])
        source = index[(*base, "S1")]
        q0 = index[(*base, "Q0_B1")]
        effects.append({
            "dataset": dataset,
            "target_subject_id": row["target_subject_id"],
            "panel": row["panel"],
            "training_seed": int(row["training_seed"]),
            "level": int(row["level"]),
            "Q1_effect": float(source["standardized_regret"]) - float(row["standardized_regret"]),
            "Q2_excess": float(row["standardized_regret"]) - float(q0["standardized_regret"]),
        })
    return effects


def build_gate_matrices() -> dict[str, list[dict[str, Any]]]:
    dataset_path = _result_table("dataset_Q1_Q2.csv")
    level_path = _result_table("level_specific_Q1_Q2.csv")
    target_path = _result_table("target_level_method_effects.csv")
    panel_path = _result_table("panel_seed_stability.csv")
    context_path = _result_table("method_context_decisions.csv")
    dataset_rows = read_csv(dataset_path)
    level_rows = read_csv(level_path)
    target_rows = read_csv(target_path)
    panel_rows = read_csv(panel_path)
    context_rows = read_csv(context_path)

    full_rows: list[dict[str, Any]] = []
    boundary_rows: list[dict[str, Any]] = []
    for row in dataset_rows:
        dataset, method = row["dataset"], row["method_id"]
        parts = _component_pass_and_distances(row, TARGET_COUNTS[dataset])
        q1_nearest = _nearest_failure({
            "Q1_mean": (bool(parts["Q1_mean_pass"]), parts["Q1_mean_distance"], 0.05),
            "Q1_maxT": (bool(parts["Q1_maxT_pass"]), parts["Q1_maxT_distance"], 0.05),
            "Q1_favorable": (bool(parts["Q1_favorable_pass"]), parts["Q1_favorable_distance"], parts["minimum_75pct_targets"]),
            "Q1_worst": (bool(parts["Q1_worst_pass"]), parts["Q1_worst_distance"], 0.10),
            "Q1_panel_seed": (bool(parts["Q1_panel_seed_pass"]), parts["Q1_panel_seed_distance"], 1.0),
        })
        q2_nearest = _nearest_failure({
            "Q2_mean": (bool(parts["Q2_mean_pass"]), parts["Q2_mean_distance"], 0.05),
            "Q2_upper": (bool(parts["Q2_upper_pass"]), parts["Q2_upper_distance"], 0.05),
            "Q2_maxT": (bool(parts["Q2_maxT_pass"]), parts["Q2_maxT_distance"], 0.05),
            "Q2_target_count": (bool(parts["Q2_target_count_pass"]), parts["Q2_target_count_distance"], parts["minimum_75pct_targets"]),
            "Q2_worst": (bool(parts["Q2_worst_pass"]), parts["Q2_worst_distance"], 0.20),
            "Q2_panel_seed": (bool(parts["Q2_panel_seed_pass"]), parts["Q2_panel_seed_distance"], 1.0),
        })
        full_rows.append(_tag({
            "dataset": dataset,
            "method_id": method,
            "method_name": METHOD_NAMES[method],
            **parts,
            "Q1_pass_frozen": int(_as_bool(row["Q1_pass"])),
            "Q1_nearest_failing_component": q1_nearest[0],
            "Q1_nearest_failing_normalized_distance": q1_nearest[1],
            "Q2_pass_frozen": int(_as_bool(row["Q2_pass"])),
            "Q2_nearest_failing_component": q2_nearest[0],
            "Q2_nearest_failing_normalized_distance": q2_nearest[1],
        }, sources=[dataset_path], source_keys=[_source_key(dataset=dataset, method_id=method)]))

        component_definitions = {
            "Q1_mean": (parts["Q1_mean_pass"], parts["Q1_mean_distance"], 0.05),
            "Q1_maxT": (parts["Q1_maxT_pass"], parts["Q1_maxT_distance"], 0.05),
            "Q1_favorable": (parts["Q1_favorable_pass"], parts["Q1_favorable_distance"], parts["minimum_75pct_targets"]),
            "Q1_worst": (parts["Q1_worst_pass"], parts["Q1_worst_distance"], 0.10),
            "Q1_panel_seed": (parts["Q1_panel_seed_pass"], parts["Q1_panel_seed_distance"], 1.0),
            "Q2_mean": (parts["Q2_mean_pass"], parts["Q2_mean_distance"], 0.05),
            "Q2_upper": (parts["Q2_upper_pass"], parts["Q2_upper_distance"], 0.05),
            "Q2_maxT": (parts["Q2_maxT_pass"], parts["Q2_maxT_distance"], 0.05),
            "Q2_target_count": (parts["Q2_target_count_pass"], parts["Q2_target_count_distance"], parts["minimum_75pct_targets"]),
            "Q2_worst": (parts["Q2_worst_pass"], parts["Q2_worst_distance"], 0.20),
            "Q2_panel_seed": (parts["Q2_panel_seed_pass"], parts["Q2_panel_seed_distance"], 1.0),
        }
        for component, (passed, distance, scale) in component_definitions.items():
            if passed:
                continue
            normalized = abs(float(distance)) / float(scale)
            boundary_rows.append(_tag({
                "scope": "FULL_PANEL",
                "dataset": dataset,
                "level": "ALL",
                "method_id": method,
                "component": component,
                "signed_distance": distance,
                "normalized_deficit": normalized,
                "boundary_class": "NEAR" if normalized <= 0.10 else ("MODERATE" if normalized <= 0.50 else "MATERIAL"),
                "frozen_decision": row["Q1_pass"] if component.startswith("Q1") else row["Q2_pass"],
            }, sources=[dataset_path], source_keys=[_source_key(dataset=dataset, method_id=method)]))

    level_output: list[dict[str, Any]] = []
    for row in level_rows:
        dataset, method, level = row["dataset"], row["method_id"], int(row["level"])
        target_level = [
            item for item in target_rows
            if item["dataset"] == dataset and item["method_id"] == method and int(item["level"]) == level
        ]
        if len(target_level) != TARGET_COUNTS[dataset]:
            raise C84AAuditError(f"level target coverage drift for {dataset}/{method}/L{level}")
        q1_values = [float(item["Q1_effect"]) for item in target_level]
        q2_values = [float(item["Q2_excess"]) for item in target_level]
        minimum = math.ceil(0.75 * TARGET_COUNTS[dataset])
        context_effects = _context_effects(context_rows, dataset, method, level)
        cell_q1 = {
            f"{panel}{seed}": _mean(item["Q1_effect"] for item in context_effects if item["panel"] == panel and item["training_seed"] == seed)
            for panel in ("A", "B") for seed in (5, 6)
        }
        cell_q2 = {
            f"{panel}{seed}": _mean(item["Q2_excess"] for item in context_effects if item["panel"] == panel and item["training_seed"] == seed)
            for panel in ("A", "B") for seed in (5, 6)
        }
        q1_mean = float(row["Q1_mean"])
        q1_p = float(row["Q1_pvalue"])
        q1_fav = sum(value > 0 for value in q1_values)
        q1_worst = min(q1_values)
        q2_mean = float(row["Q2_mean_excess"])
        q2_p = float(row["Q2_pvalue"])
        q2_count = sum(value <= 0.05 for value in q2_values)
        q2_worst = max(q2_values)
        q1_positive_cells = sum(value > 0 for value in cell_q1.values())
        q2_within_cells = sum(value <= 0.05 for value in cell_q2.values())
        level_failures = {
            "Q1_mean": (q1_mean >= 0.05, q1_mean - 0.05, 0.05),
            "Q1_maxT": (q1_p <= 0.05, 0.05 - q1_p, 0.05),
            "Q1_favorable_descriptive": (q1_fav >= minimum, q1_fav - minimum, minimum),
            "Q1_worst_descriptive": (q1_worst >= -0.10, q1_worst + 0.10, 0.10),
            "Q1_panel_seed_descriptive": (q1_positive_cells >= 3, q1_positive_cells - 3, 3),
            "Q2_mean": (q2_mean <= 0.05, 0.05 - q2_mean, 0.05),
            "Q2_maxT": (q2_p <= 0.05, 0.05 - q2_p, 0.05),
            "Q2_target_count_descriptive": (q2_count >= minimum, q2_count - minimum, minimum),
            "Q2_worst_descriptive": (q2_worst <= 0.20, 0.20 - q2_worst, 0.20),
            "Q2_panel_seed_descriptive": (q2_within_cells >= 3, q2_within_cells - 3, 3),
        }
        q1_nearest = _nearest_failure({key: value for key, value in level_failures.items() if key.startswith("Q1")})
        q2_nearest = _nearest_failure({key: value for key, value in level_failures.items() if key.startswith("Q2")})
        level_output.append(_tag({
            "dataset": dataset,
            "level": level,
            "method_id": method,
            "method_name": METHOD_NAMES[method],
            "target_count": TARGET_COUNTS[dataset],
            "minimum_75pct_targets": minimum,
            "Q1_mean": q1_mean,
            "Q1_mean_pass": int(q1_mean >= 0.05),
            "Q1_mean_distance": q1_mean - 0.05,
            "Q1_maxT_p": q1_p,
            "Q1_maxT_pass": int(q1_p <= 0.05),
            "Q1_maxT_distance": 0.05 - q1_p,
            "Q1_favorable_targets_descriptive": q1_fav,
            "Q1_favorable_pass_descriptive": int(q1_fav >= minimum),
            "Q1_favorable_distance_descriptive": q1_fav - minimum,
            "Q1_worst_target_descriptive": q1_worst,
            "Q1_worst_pass_descriptive": int(q1_worst >= -0.10),
            "Q1_worst_distance_descriptive": q1_worst + 0.10,
            "Q1_panel_seed_cell_means_descriptive": json.dumps(cell_q1, sort_keys=True, separators=(",", ":")),
            "Q1_positive_panel_seed_cells_descriptive": q1_positive_cells,
            "Q1_panel_seed_at_least_3_positive_descriptive": int(q1_positive_cells >= 3),
            "Q1_panel_seed_distance_descriptive": q1_positive_cells - 3,
            "Q1_pass_frozen": int(_as_bool(row["Q1_pass"])),
            "Q1_nearest_available_failing_component": q1_nearest[0],
            "Q1_nearest_available_failing_normalized_distance": q1_nearest[1],
            "Q2_mean_excess": q2_mean,
            "Q2_mean_pass": int(q2_mean <= 0.05),
            "Q2_mean_distance": 0.05 - q2_mean,
            "Q2_simultaneous_upper": "NOT_FROZEN_AT_LEVEL_SCOPE",
            "Q2_upper_component_status": "NOT_RECOMPUTED",
            "Q2_maxT_p": q2_p,
            "Q2_maxT_pass": int(q2_p <= 0.05),
            "Q2_maxT_distance": 0.05 - q2_p,
            "Q2_within_margin_targets_descriptive": q2_count,
            "Q2_target_count_pass_descriptive": int(q2_count >= minimum),
            "Q2_target_count_distance_descriptive": q2_count - minimum,
            "Q2_worst_excess_descriptive": q2_worst,
            "Q2_worst_pass_descriptive": int(q2_worst <= 0.20),
            "Q2_worst_distance_descriptive": 0.20 - q2_worst,
            "Q2_panel_seed_cell_means_descriptive": json.dumps(cell_q2, sort_keys=True, separators=(",", ":")),
            "Q2_within_margin_panel_seed_cells_descriptive": q2_within_cells,
            "Q2_panel_seed_at_least_3_within_margin_descriptive": int(q2_within_cells >= 3),
            "Q2_panel_seed_distance_descriptive": q2_within_cells - 3,
            "Q2_pass_frozen": int(_as_bool(row["Q2_pass"])),
            "Q2_nearest_available_failing_component": q2_nearest[0],
            "Q2_nearest_available_failing_normalized_distance": q2_nearest[1],
            "level_heterogeneity_frozen": int(_as_bool(row["level_heterogeneity"])),
        }, sources=[level_path, target_path, context_path], source_keys=[
            _source_key(dataset=dataset, level=level, method_id=method),
            _source_key(dataset=dataset, method_id=method, level=level),
        ]))

        for component, (passed, distance, scale) in level_failures.items():
            if passed:
                continue
            normalized = abs(float(distance)) / float(scale)
            boundary_rows.append(_tag({
                "scope": "LEVEL_SPECIFIC_DESCRIPTIVE_COMPONENT",
                "dataset": dataset,
                "level": level,
                "method_id": method,
                "component": component,
                "signed_distance": distance,
                "normalized_deficit": normalized,
                "boundary_class": "NEAR" if normalized <= 0.10 else ("MODERATE" if normalized <= 0.50 else "MATERIAL"),
                "frozen_decision": row["Q1_pass"] if component.startswith("Q1") else row["Q2_pass"],
            }, sources=[level_path, target_path], source_keys=[_source_key(dataset=dataset, level=level, method_id=method)]))

    boundary_rows.sort(key=lambda item: (float(item["normalized_deficit"]), item["dataset"], item["method_id"], str(item["level"])))
    return {
        "full_panel_gate_component_matrix.csv": full_rows,
        "level_specific_gate_component_matrix.csv": level_output,
        "near_boundary_gate_failures.csv": boundary_rows,
    }


def _full_target_effects(
    rows: Sequence[Mapping[str, str]], dataset: str, method: str,
) -> list[dict[str, Any]]:
    selected = [
        row for row in rows
        if row["dataset"] == dataset and row["method_id"] == method
    ]
    by_target: dict[str, dict[int, dict[str, str]]] = {}
    for row in selected:
        by_target.setdefault(row["target_subject_id"], {})[int(row["level"])] = dict(row)
    if len(by_target) != TARGET_COUNTS[dataset] or any(set(levels) != {0, 1} for levels in by_target.values()):
        raise C84AAuditError(f"target-level coverage drift for {dataset}/{method}")
    output = []
    for target, levels in by_target.items():
        output.append({
            "target_subject_id": target,
            "level0_Q1_effect": float(levels[0]["Q1_effect"]),
            "level1_Q1_effect": float(levels[1]["Q1_effect"]),
            "primary_Q1_effect": _mean(float(levels[level]["Q1_effect"]) for level in (0, 1)),
            "level0_Q2_excess": float(levels[0]["Q2_excess"]),
            "level1_Q2_excess": float(levels[1]["Q2_excess"]),
            "primary_Q2_excess": _mean(float(levels[level]["Q2_excess"]) for level in (0, 1)),
        })
    return output


def build_cott_audit() -> dict[str, list[dict[str, Any]]]:
    target_path = _result_table("target_level_method_effects.csv")
    dataset_path = _result_table("dataset_Q1_Q2.csv")
    level_path = _result_table("level_specific_Q1_Q2.csv")
    loto_path = _result_table("leave_one_target_out.csv")
    panel_path = _result_table("panel_seed_stability.csv")
    target_rows = read_csv(target_path)
    dataset_rows = read_csv(dataset_path)
    level_rows = read_csv(level_path)
    loto_rows = read_csv(loto_path)
    panel_rows = read_csv(panel_path)

    distribution: list[dict[str, Any]] = []
    influence: list[dict[str, Any]] = []
    separation: list[dict[str, Any]] = []
    recurrence: list[dict[str, Any]] = []
    for dataset in DATASETS:
        effects = _full_target_effects(target_rows, dataset, "U13")
        ordered = sorted(effects, key=lambda row: (row["primary_Q1_effect"], row["target_subject_id"]))
        full_mean = _mean(row["primary_Q1_effect"] for row in effects)
        dataset_gate = _lookup(dataset_rows, dataset=dataset, method_id="U13")
        level0 = _lookup(level_rows, dataset=dataset, level=0, method_id="U13")
        level1 = _lookup(level_rows, dataset=dataset, level=1, method_id="U13")
        panel = _lookup(panel_rows, dataset=dataset, method_id="U13")
        loto_index = {
            row["left_out_target"]: row
            for row in loto_rows if row["dataset"] == dataset
        }
        if set(loto_index) != {row["target_subject_id"] for row in effects}:
            raise C84AAuditError(f"LOTO target identity drift for {dataset}")
        for rank, row in enumerate(ordered, start=1):
            target = row["target_subject_id"]
            loto = loto_index[target]
            distribution.append(_tag({
                "dataset": dataset,
                "target_subject_id": target,
                "adverse_rank_1_is_worst": rank,
                "primary_Q1_effect": row["primary_Q1_effect"],
                "level0_Q1_effect": row["level0_Q1_effect"],
                "level1_Q1_effect": row["level1_Q1_effect"],
                "primary_Q2_excess": row["primary_Q2_excess"],
                "favorable": int(row["primary_Q1_effect"] > 0),
                "registered_Q1_floor_breached": int(row["primary_Q1_effect"] < -0.10),
                "LOTO_category_frozen": loto["LOTO_category"],
                "LOTO_category_changed": int(loto["LOTO_category"] != loto["full_category"]),
            }, sources=[target_path, loto_path], source_keys=[
                _source_key(dataset=dataset, target_subject_id=target, method_id="U13", levels="0|1"),
                _source_key(dataset=dataset, left_out_target=target),
            ]))
            without = _mean(
                other["primary_Q1_effect"] for other in effects
                if other["target_subject_id"] != target
            )
            influence.append(_tag({
                "dataset": dataset,
                "left_out_target": target,
                "target_Q1_effect": row["primary_Q1_effect"],
                "full_panel_mean_Q1_effect": full_mean,
                "descriptive_mean_without_target": without,
                "descriptive_mean_shift_after_omission": without - full_mean,
                "full_category_frozen": loto["full_category"],
                "LOTO_category_frozen": loto["LOTO_category"],
                "category_changed_frozen": int(loto["LOTO_category"] != loto["full_category"]),
                "same_method_preserved_frozen": loto["same_method_preserved"],
            }, sources=[target_path, loto_path], source_keys=[
                _source_key(dataset=dataset, target_subject_id=target, method_id="U13", levels="0|1"),
                _source_key(dataset=dataset, left_out_target=target),
            ]))

        values = [row["primary_Q1_effect"] for row in effects]
        floor_breaches = [row for row in effects if row["primary_Q1_effect"] < -0.10]
        separation.append(_tag({
            "dataset": dataset,
            "target_count": len(values),
            "mean": full_mean,
            "median": _quantile(values, 0.50),
            "q25": _quantile(values, 0.25),
            "q10": _quantile(values, 0.10),
            "minimum": min(values),
            "mean_minus_minimum": full_mean - min(values),
            "favorable_targets": sum(value > 0 for value in values),
            "adverse_targets": sum(value < 0 for value in values),
            "registered_floor_breach_targets": len(floor_breaches),
            "registered_floor_breach_target_ids": "|".join(sorted((row["target_subject_id"] for row in floor_breaches), key=lambda value: int(value))),
            "tail_interpretation": (
                "single_target_near_boundary_floor_failure" if len(floor_breaches) == 1 and min(values) >= -0.12
                else "material_multi_or_deep_target_tail_failure"
            ),
        }, sources=[target_path, dataset_path], source_keys=[
            _source_key(dataset=dataset, method_id="U13", aggregation="levels_then_targets"),
        ]))

        preserved = sum(
            int(row["LOTO_category"] == row["full_category"])
            for row in loto_rows if row["dataset"] == dataset
        )
        recurrence.append(_tag({
            "dataset": dataset,
            "Q1_mean_frozen": dataset_gate["Q1_mean"],
            "Q1_maxT_p_frozen": dataset_gate["Q1_pvalue"],
            "Q1_favorable_targets_frozen": dataset_gate["Q1_favorable_targets"],
            "Q1_worst_target_frozen": dataset_gate["Q1_worst_target"],
            "Q1_pass_frozen": int(_as_bool(dataset_gate["Q1_pass"])),
            "Q2_pass_frozen": int(_as_bool(dataset_gate["Q2_pass"])),
            "level0_Q1_pass_frozen": int(_as_bool(level0["Q1_pass"])),
            "level1_Q1_pass_frozen": int(_as_bool(level1["Q1_pass"])),
            "level_heterogeneity_frozen": int(_as_bool(level0["level_heterogeneity"])),
            "panel_seed_Q1_all_directional_frozen": int(_as_bool(panel["Q1_all_directional"])),
            "LOTO_category_preserved": preserved,
            "LOTO_total": TARGET_COUNTS[dataset],
            "pattern": "POSITIVE_AVERAGE_NON_ROBUST_TAIL",
        }, sources=[dataset_path, level_path, panel_path, loto_path], source_keys=[
            _source_key(dataset=dataset, method_id="U13"),
        ]))

    influence.sort(key=lambda row: (row["dataset"], -abs(float(row["descriptive_mean_shift_after_omission"]))))
    return {
        "cott_target_effect_distribution.csv": distribution,
        "cott_target_influence.csv": influence,
        "cott_average_tail_separation.csv": separation,
        "cott_cross_cohort_recurrence.csv": recurrence,
    }


def build_mano_audit() -> dict[str, list[dict[str, Any]]]:
    selected_path = _result_table("selected_utility_summary.csv")
    gain_path = _result_table("source_relative_regret_gain.csv")
    topk_path = _result_table("topk_decision_summary.csv")
    measurement_path = _result_table("measurement_vs_decision.csv")
    dataset_path = _result_table("dataset_Q1_Q2.csv")
    regime_path = _result_table("selected_regime_distribution.csv")
    level_path = _result_table("level_specific_Q1_Q2.csv")
    panel_path = _result_table("panel_seed_stability.csv")
    context_path = _result_table("method_context_decisions.csv")
    selected_rows = read_csv(selected_path)
    gain_rows = read_csv(gain_path)
    topk_rows = read_csv(topk_path)
    measurement_rows = read_csv(measurement_path)
    dataset_rows = read_csv(dataset_path)
    regime_rows = read_csv(regime_path)
    level_rows = read_csv(level_path)
    panel_rows = read_csv(panel_path)
    context_rows = read_csv(context_path)

    profile: list[dict[str, Any]] = []
    for dataset in DATASETS:
        selected = _lookup(selected_rows, dataset=dataset, method_id="U11")
        gain = _lookup(gain_rows, dataset=dataset, method_id="U11")
        topk = _lookup(topk_rows, dataset=dataset, method_id="U11")
        measurement = _lookup(measurement_rows, dataset=dataset, method_id="U11")
        decision = _lookup(dataset_rows, dataset=dataset, method_id="U11")
        level0 = _lookup(level_rows, dataset=dataset, level=0, method_id="U11")
        level1 = _lookup(level_rows, dataset=dataset, level=1, method_id="U11")
        panel = _lookup(panel_rows, dataset=dataset, method_id="U11")
        regimes = [row for row in regime_rows if row["dataset"] == dataset and row["method_id"] == "U11"]
        dominant = max(regimes, key=lambda row: float(row["fraction"]))
        contexts = [row for row in context_rows if row["dataset"] == dataset and row["method_id"] == "U11"]
        b1_index = {
            (row["target_subject_id"], row["panel"], row["training_seed"], row["level"]): row
            for row in context_rows if row["dataset"] == dataset and row["method_id"] == "B1"
        }
        exact_b1 = sum(
            float(row["standardized_regret"]) == float(b1_index[(row["target_subject_id"], row["panel"], row["training_seed"], row["level"])]["standardized_regret"])
            and float(row["selected_utility"]) == float(b1_index[(row["target_subject_id"], row["panel"], row["training_seed"], row["level"])]["selected_utility"])
            for row in contexts
        )
        profile.append(_tag({
            "dataset": dataset,
            "method_id": "U11",
            "method_name": "MaNo",
            "mean_regret": selected["mean_regret"],
            "mean_selected_utility": selected["mean_selected_utility"],
            "mean_source_relative_regret_gain_frozen": gain["mean_source_relative_regret_gain"],
            "top1": topk["mean_top1"],
            "top5": topk["mean_top5"],
            "top10": topk["mean_top10"],
            "mean_Spearman": measurement["mean_Spearman"],
            "mean_Kendall": measurement["mean_Kendall"],
            "mean_pairwise_ordering_accuracy": measurement["mean_pairwise_ordering_accuracy"],
            "Q1_pass": int(_as_bool(decision["Q1_pass"])),
            "Q2_pass": int(_as_bool(decision["Q2_pass"])),
            "Q1_mean": decision["Q1_mean"],
            "Q1_worst_target": decision["Q1_worst_target"],
            "dominant_selected_regime": dominant["selected_regime"],
            "dominant_regime_fraction": dominant["fraction"],
            "level0_Q1_pass": int(_as_bool(level0["Q1_pass"])),
            "level1_Q1_pass": int(_as_bool(level1["Q1_pass"])),
            "panel_seed_Q1_all_directional": int(_as_bool(panel["Q1_all_directional"])),
            "selected_regret_at_most_0_05_fraction_descriptive": sum(float(row["standardized_regret"]) <= 0.05 for row in contexts) / len(contexts),
            "selected_regret_at_most_0_10_fraction_descriptive": sum(float(row["standardized_regret"]) <= 0.10 for row in contexts) / len(contexts),
            "selected_regret_at_most_0_20_fraction_descriptive": sum(float(row["standardized_regret"]) <= 0.20 for row in contexts) / len(contexts),
            "exact_B1_utility_and_regret_context_fraction_descriptive": exact_b1 / len(contexts),
            "near_optimal_action_density_identified": 0,
            "action_density_limitation": "all_candidate_utility_geometry_not_in_allowed_compact_tables",
        }, sources=[selected_path, gain_path, topk_path, measurement_path, dataset_path, regime_path, level_path, panel_path, context_path],
            source_keys=[_source_key(dataset=dataset, method_id="U11")]))

    separation: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for method in PRIMARY_METHODS:
            selected = _lookup(selected_rows, dataset=dataset, method_id=method)
            topk = _lookup(topk_rows, dataset=dataset, method_id=method)
            measurement = _lookup(measurement_rows, dataset=dataset, method_id=method)
            decision = _lookup(dataset_rows, dataset=dataset, method_id=method)
            separation.append(_tag({
                "dataset": dataset,
                "method_id": method,
                "method_name": METHOD_NAMES[method],
                "global_rank_Spearman": measurement["mean_Spearman"],
                "global_rank_Kendall": measurement["mean_Kendall"],
                "pairwise_ordering_accuracy": measurement["mean_pairwise_ordering_accuracy"],
                "top1": topk["mean_top1"],
                "top5": topk["mean_top5"],
                "top10": topk["mean_top10"],
                "mean_regret": selected["mean_regret"],
                "Q1_pass": int(_as_bool(decision["Q1_pass"])),
                "Q2_pass": int(_as_bool(decision["Q2_pass"])),
                "measurement_substitutes_for_regret": 0,
            }, sources=[selected_path, topk_path, measurement_path, dataset_path],
                source_keys=[_source_key(dataset=dataset, method_id=method)]))

    return {
        "mano_cross_dataset_decision_profile.csv": profile,
        "rank_topk_regret_separation.csv": separation,
    }


def _budget_method(budget: str) -> str:
    return "Q0_FULL" if budget == "FULL" else f"Q0_B{budget}"


def build_label_frontier_audit() -> dict[str, list[dict[str, Any]]]:
    frontier_path = _result_table("label_budget_frontier.csv")
    budget_path = _result_table("label_budget_context.csv")
    context_path = _result_table("method_context_decisions.csv")
    frontier_rows = read_csv(frontier_path)
    budget_rows = read_csv(budget_path)
    context_rows = read_csv(context_path)
    matrix: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    interaction: list[dict[str, Any]] = []

    context_index = {
        (row["dataset"], row["target_subject_id"], row["panel"], row["training_seed"], row["level"], row["method_id"]): row
        for row in context_rows
    }
    for dataset in DATASETS:
        minimum = math.ceil(0.75 * TARGET_COUNTS[dataset])
        dataset_frontier = [row for row in frontier_rows if row["dataset"] == dataset]
        direct_by_budget = {row["budget"]: _as_bool(row["direct_qualification"]) for row in dataset_frontier}
        for row in dataset_frontier:
            budget = row["budget"]
            target_values = [
                item for item in budget_rows
                if item["dataset"] == dataset and item["budget"] == budget
            ]
            if len(target_values) != TARGET_COUNTS[dataset]:
                raise C84AAuditError(f"budget target coverage drift for {dataset}/B{budget}")
            panel_seed: dict[str, float] = {}
            method = _budget_method(budget)
            for panel in ("A", "B"):
                for seed in (5, 6):
                    effects = []
                    for target in {item["target_subject_id"] for item in target_values}:
                        for level in ("0", "1"):
                            base = (dataset, target, panel, str(seed), level)
                            source = context_index[(*base, "S1")]
                            q0 = context_index[(*base, method)]
                            effects.append(float(source["standardized_regret"]) - float(q0["standardized_regret"]))
                    panel_seed[f"{panel}{seed}"] = _mean(effects)
            mean_effect = float(row["mean_effect"])
            pvalue = float(row["maxT_pvalue"])
            favorable = sum(float(item["primary_effect"]) > 0 for item in target_values)
            worst = min(float(item["primary_effect"]) for item in target_values)
            worst_targets = "|".join(sorted(
                (item["target_subject_id"] for item in target_values if float(item["primary_effect"]) == worst),
                key=lambda value: int(value),
            ))
            panel_pass = sum(value > 0 for value in panel_seed.values()) >= 3
            component_pass = {
                "mean": mean_effect >= 0.05,
                "maxT": pvalue <= 0.05,
                "favorable": favorable >= minimum,
                "worst": worst >= -0.10,
                "panel_seed": panel_pass,
            }
            derived_direct = all(component_pass.values())
            if derived_direct != direct_by_budget[budget]:
                raise C84AAuditError(f"frontier direct gate replay mismatch for {dataset}/B{budget}")
            matrix.append(_tag({
                "dataset": dataset,
                "scope": "PRIMARY_EIGHT_CONTEXT_TARGET_AGGREGATE",
                "level": "ALL",
                "budget": budget,
                "mean_source_relative_effect": mean_effect,
                "mean_component_pass": int(component_pass["mean"]),
                "mean_distance": mean_effect - 0.05,
                "maxT_pvalue_frozen": pvalue,
                "maxT_component_pass": int(component_pass["maxT"]),
                "maxT_distance": 0.05 - pvalue,
                "favorable_targets_descriptive_replay": favorable,
                "favorable_component_pass": int(component_pass["favorable"]),
                "favorable_distance": favorable - minimum,
                "worst_target_effect_descriptive_replay": worst,
                "worst_target_ids_descriptive_replay": worst_targets,
                "worst_component_pass": int(component_pass["worst"]),
                "worst_distance": worst + 0.10,
                "panel_seed_cell_means_descriptive_replay": json.dumps(panel_seed, sort_keys=True, separators=(",", ":")),
                "panel_seed_component_pass": int(panel_pass),
                "direct_qualification_frozen": int(direct_by_budget[budget]),
                "larger_budget_closure_frozen": int(_as_bool(row["closure_qualification"])),
                "Bstar_frozen": row["Bstar"],
            }, sources=[frontier_path, budget_path, context_path], source_keys=[
                _source_key(dataset=dataset, budget=budget),
            ]))

            current_index = PRIMARY_BUDGETS.index(budget)
            larger = PRIMARY_BUDGETS[current_index:]
            failed_larger = [value for value in larger if not direct_by_budget[value]]
            failures = [name for name, passed in component_pass.items() if not passed]
            closure_rows.append(_tag({
                "dataset": dataset,
                "budget": budget,
                "direct_qualification_frozen": int(direct_by_budget[budget]),
                "direct_failing_components": "|".join(failures) if failures else "NONE",
                "larger_or_equal_registered_budgets": "|".join(larger),
                "larger_or_equal_direct_failures": "|".join(failed_larger) if failed_larger else "NONE",
                "closure_qualification_frozen": int(_as_bool(row["closure_qualification"])),
                "closure_failure_reason": (
                    "DIRECT_COMPONENT_FAILURE" if not direct_by_budget[budget]
                    else ("LARGER_BUDGET_FAILURE" if failed_larger else "NONE")
                ),
            }, sources=[frontier_path, budget_path, context_path], source_keys=[_source_key(dataset=dataset, budget=budget)]))

            level0_values = [float(item["level0_effect"]) for item in target_values]
            level1_values = [float(item["level1_effect"]) for item in target_values]
            interaction.append(_tag({
                "dataset": dataset,
                "budget": budget,
                "level0_mean_effect_descriptive": _mean(level0_values),
                "level1_mean_effect_descriptive": _mean(level1_values),
                "level1_minus_level0_mean_effect_descriptive": _mean(level1_values) - _mean(level0_values),
                "level0_favorable_targets_descriptive": sum(value > 0 for value in level0_values),
                "level1_favorable_targets_descriptive": sum(value > 0 for value in level1_values),
                "level0_worst_target_descriptive": min(level0_values),
                "level1_worst_target_descriptive": min(level1_values),
                "level_specific_maxT_status": "NOT_FROZEN_NOT_RECOMPUTED",
                "level0_Bstar_frozen": row["level0_Bstar"],
                "level1_Bstar_frozen": row["level1_Bstar"],
                "registered_heterogeneity_frozen": row["registered_heterogeneity"],
            }, sources=[frontier_path, budget_path], source_keys=[_source_key(dataset=dataset, budget=budget)]))

    return {
        "label_frontier_component_matrix.csv": matrix,
        "label_frontier_closure_failures.csv": closure_rows,
        "level_label_complexity_interaction.csv": interaction,
    }


def build_transport_audit() -> dict[str, list[dict[str, Any]]]:
    c82_result_path = C82_TABLE_DIR / "seed_specific_method_results.csv"
    c82_gate_path = C82_TABLE_DIR / "seed_method_Q1_Q2.csv"
    c82_measure_path = C82_TABLE_DIR / "measurement_vs_decision_separation.csv"
    c82_loto_path = C82_TABLE_DIR / "leave_one_target_method_stability.csv"
    c80_seed_paths = {
        3: C80_TABLE_DIR / "seed3_budget_frontier.csv",
        4: C80_TABLE_DIR / "seed4_budget_frontier.csv",
    }
    c80_topk_path = C80_TABLE_DIR / "topk_coverage_summary.csv"
    c80_loto_path = C80_TABLE_DIR / "leave_one_target_out_sensitivity.csv"
    c84_selected_path = _result_table("selected_utility_summary.csv")
    c84_topk_path = _result_table("topk_decision_summary.csv")
    c84_measure_path = _result_table("measurement_vs_decision.csv")
    c84_gate_path = _result_table("dataset_Q1_Q2.csv")
    c84_level_path = _result_table("level_specific_Q1_Q2.csv")
    c84_loto_path = _result_table("leave_one_target_out.csv")
    c84_frontier_path = _result_table("label_budget_frontier.csv")

    c82_results = read_csv(c82_result_path)
    c82_gates = read_csv(c82_gate_path)
    c82_measure = read_csv(c82_measure_path)
    c84_selected = read_csv(c84_selected_path)
    c84_topk = read_csv(c84_topk_path)
    c84_measure = read_csv(c84_measure_path)
    c84_gates = read_csv(c84_gate_path)
    c84_levels = read_csv(c84_level_path)
    c84_loto = read_csv(c84_loto_path)
    c84_frontier = read_csv(c84_frontier_path)
    transport: list[dict[str, Any]] = []

    def append_transport(
        *, milestone: str, cohort: str, method: str, mean_regret: Any,
        direction: str, registered_p_or_gate: Any, q1_status: str,
        worst_target: Any, q2_status: str, top1: Any, top5: Any, top10: Any,
        rank_association: Any, repeated_factor_effect: str, loto_status: str,
        epistemic_role: str, sources: Sequence[Path], keys: Sequence[str],
    ) -> None:
        transport.append(_tag({
            "source_milestone": milestone,
            "cohort_or_seed": cohort,
            "method_id": method,
            "method_name": METHOD_NAMES[method],
            "mean_standardized_regret": mean_regret,
            "mean_direction_vs_S1": direction,
            "registered_p_or_gate": registered_p_or_gate,
            "Q1_or_source_gate_status": q1_status,
            "worst_target_or_tail_status": worst_target,
            "Q2_status": q2_status,
            "top1": top1,
            "top5": top5,
            "top10": top10,
            "rank_association": rank_association,
            "level_or_seed_effect": repeated_factor_effect,
            "LOTO_status": loto_status,
            "epistemic_role": epistemic_role,
        }, sources=sources, source_keys=keys))

    for seed in (3, 4):
        for method in ("S1", *PRIMARY_METHODS):
            result = _lookup(c82_results, seed=seed, method_id=method)
            if method == "S1":
                append_transport(
                    milestone="C82", cohort=f"BNCI2014_001_seed{seed}", method=method,
                    mean_regret=result["mean_standardized_regret"], direction="COMPARATOR",
                    registered_p_or_gate="NOT_APPLICABLE", q1_status="STRICT_SOURCE_COMPARATOR",
                    worst_target="NOT_APPLICABLE", q2_status="NOT_APPLICABLE",
                    top1=result["top1"], top5=result["top5"], top10=result["top10"],
                    rank_association=_lookup(c82_measure, seed=seed, method_id=method)["mean_spearman"],
                    repeated_factor_effect="paired_training_seed_factor; no population replication",
                    loto_status="global_C82_method_aware_rule_only",
                    epistemic_role="frozen_same_field_strict_source_comparator",
                    sources=[c82_result_path, c82_measure_path, c82_loto_path],
                    keys=[_source_key(seed=seed, method_id=method)],
                )
                continue
            gate = _lookup(c82_gates, seed=seed, method_id=method)
            measurement = _lookup(c82_measure, seed=seed, method_id=method)
            direction = "POSITIVE" if float(gate["mean_regret_improvement_vs_source"]) > 0 else (
                "NEGATIVE" if float(gate["mean_regret_improvement_vs_source"]) < 0 else "ZERO"
            )
            append_transport(
                milestone="C82", cohort=f"BNCI2014_001_seed{seed}", method=method,
                mean_regret=result["mean_standardized_regret"], direction=direction,
                registered_p_or_gate=gate["Q1_maxT_p"],
                q1_status="PASS" if _as_bool(gate["Q1_pass"]) else "FAIL",
                worst_target=gate["Q1_worst_target"],
                q2_status="PASS" if _as_bool(gate["Q2_pass"]) else "FAIL",
                top1=result["top1"], top5=result["top5"], top10=result["top10"],
                rank_association=measurement["mean_spearman"],
                repeated_factor_effect="seed-specific row; compare paired seed rows without pooled p-value",
                loto_status="global_C82_method_aware_rule_7_of_16; not per-method panel ledger",
                epistemic_role="post_C81_outcome_access_same_field_recovery",
                sources=[c82_result_path, c82_gate_path, c82_measure_path, c82_loto_path],
                keys=[_source_key(seed=seed, method_id=method)],
            )

        source_regret = float(_lookup(c82_results, seed=seed, method_id="S1")["mean_standardized_regret"])
        c80_rows = read_csv(c80_seed_paths[seed])
        c80_topk = read_csv(c80_topk_path)
        for method, budget in (("Q0_B1", "1"), ("Q0_FULL", "FULL")):
            frontier = _lookup(c80_rows, seed=seed, budget=budget)
            topk = _lookup(c80_topk, seed=seed, budget=budget)
            effect = float(frontier["mean_regret_reduction_vs_source"])
            append_transport(
                milestone="C80", cohort=f"BNCI2014_001_seed{seed}", method=method,
                mean_regret=frontier["expected_standardized_regret"],
                direction="POSITIVE" if effect > 0 else ("NEGATIVE" if effect < 0 else "ZERO"),
                registered_p_or_gate=frontier["maxT_p"],
                q1_status="PASS" if _as_bool(frontier["direct_qualification"]) else "FAIL",
                worst_target=frontier["catastrophic_target"], q2_status="NOT_APPLICABLE",
                top1=topk["top1"], top5=topk["top5"], top10=topk["top10"],
                rank_association="NOT_FROZEN_AS_COMPARABLE_SPEARMAN",
                repeated_factor_effect=f"source_regret={source_regret}; paired seed frontier",
                loto_status="frontier_LOTO_sensitive_2_to_4_envelope",
                epistemic_role="construction_label_source_relative_policy",
                sources=[c80_seed_paths[seed], c80_topk_path, c80_loto_path],
                keys=[_source_key(seed=seed, budget=budget)],
            )

    for dataset in DATASETS:
        for method in ("S1", *PRIMARY_METHODS, "Q0_B1", "Q0_FULL"):
            selected = _lookup(c84_selected, dataset=dataset, method_id=method)
            topk = _lookup(c84_topk, dataset=dataset, method_id=method)
            measurement = _lookup(c84_measure, dataset=dataset, method_id=method)
            if method in PRIMARY_METHODS:
                gate = _lookup(c84_gates, dataset=dataset, method_id=method)
                effect = float(gate["Q1_mean"])
                q1_status = "PASS" if _as_bool(gate["Q1_pass"]) else "FAIL"
                q2_status = "PASS" if _as_bool(gate["Q2_pass"]) else "FAIL"
                p_or_gate = gate["Q1_pvalue"]
                worst = gate["Q1_worst_target"]
                levels = [row for row in c84_levels if row["dataset"] == dataset and row["method_id"] == method]
                repeated = (
                    f"level0_Q1={_lookup(levels, level=0)['Q1_pass']};"
                    f"level1_Q1={_lookup(levels, level=1)['Q1_pass']};"
                    f"level_heterogeneity={_lookup(levels, level=0)['level_heterogeneity']}"
                )
            elif method == "S1":
                effect = 0.0
                q1_status = "STRICT_SOURCE_COMPARATOR"
                q2_status = "NOT_APPLICABLE"
                p_or_gate = "NOT_APPLICABLE"
                worst = "NOT_APPLICABLE"
                repeated = "panel_seed_level_repeated_factor_comparator"
            else:
                budget = "1" if method == "Q0_B1" else "FULL"
                frontier = _lookup(c84_frontier, dataset=dataset, budget=budget)
                effect = float(frontier["mean_effect"])
                q1_status = "PASS" if _as_bool(frontier["direct_qualification"]) else "FAIL"
                q2_status = "NOT_APPLICABLE"
                p_or_gate = frontier["maxT_pvalue"]
                budget_targets = [
                    row for row in read_csv(_result_table("label_budget_context.csv"))
                    if row["dataset"] == dataset and row["budget"] == budget
                ]
                worst = min(float(row["primary_effect"]) for row in budget_targets)
                repeated = f"level0_Bstar={frontier['level0_Bstar']};level1_Bstar={frontier['level1_Bstar']}"
            direction = "COMPARATOR" if method == "S1" else (
                "POSITIVE" if effect > 0 else ("NEGATIVE" if effect < 0 else "ZERO")
            )
            preserved = sum(
                row["LOTO_category"] == row["full_category"]
                for row in c84_loto if row["dataset"] == dataset
            )
            append_transport(
                milestone="C84", cohort=dataset, method=method,
                mean_regret=selected["mean_regret"], direction=direction,
                registered_p_or_gate=p_or_gate, q1_status=q1_status,
                worst_target=worst, q2_status=q2_status,
                top1=topk["mean_top1"], top5=topk["mean_top5"], top10=topk["mean_top10"],
                rank_association=measurement["mean_Spearman"] or "NULL_NOT_APPLICABLE",
                repeated_factor_effect=repeated,
                loto_status=f"category_preserved_{preserved}_of_{TARGET_COUNTS[dataset]}",
                epistemic_role="harmonized_binary_MI_external_cohort_frozen_policy",
                sources=[c84_selected_path, c84_topk_path, c84_measure_path, c84_gate_path, c84_level_path, c84_loto_path, c84_frontier_path],
                keys=[_source_key(dataset=dataset, method_id=method)],
            )

    axes = [
        ("dataset", "Lee=C|Cho=A|Physionet=C", "sufficient_for_C84_D", "independent_cohorts; no pooled p-value"),
        ("training_seed", "C82 seed3=B and seed4=C; C84 cell gates retained", "active", "paired training factor, not population replication"),
        ("source_support_level", "Lee COTT level0 pass and level1 fail; Cho Bstar FULL versus 4", "active", "effect modifier; mechanism not identified"),
        ("source_panel", "registered four-cell direction gates replayed", "method_and_dataset_specific", "repeated factor, not independent N"),
        ("target_composition", "LOTO 21/22|20/20|76/76", "mostly_stable", "Lee target 8 changes C to A; not sole cause of C84-D"),
        ("decision_objective", "regret|top-k|rank association diverge", "active", "measurement does not substitute for actionability"),
        ("registered_policy", "Q0 budget frontier absent|8|absent", "active", "information value is not identified by one restricted policy"),
    ]
    heterogeneity = [_tag({
        "axis": axis,
        "frozen_or_descriptive_evidence": evidence,
        "status": status,
        "interpretation_boundary": boundary,
    }, sources=[REPORT_DIR / "C84S_OVERALL_REPORT.json", C82_TABLE_DIR / "cross_seed_method_identity_stability.csv"],
        source_keys=[axis]) for axis, evidence, status, boundary in axes]
    return {
        "c82_c84_method_transport_matrix.csv": transport,
        "heterogeneity_axis_matrix.csv": heterogeneity,
    }


def build_theory_and_next_experiment_tables() -> dict[str, list[dict[str, Any]]]:
    evidence_source = REPORT_DIR / "C84S_OVERALL_REPORT.json"
    geometry_specs = [
        ("I0", "no target observations", "fixed defaults", "baseline action geometry", "does not identify unrestricted optimal risk"),
        ("IS", "source audit labels and predictions", "S1 fixed score", "source competence can fail to transport", "S1 is one registered policy, not all source-information policies"),
        ("IU", "target-unlabeled predictions/features", "U5|U11|U14 fixed formulas", "cohort-specific decision behavior", "no Blackwell ordering against labels"),
        ("ISU", "source audit plus target-unlabeled outputs", "U7|U13|U15 fixed formulas", "COTT positive-average/tail-failure pattern", "Q1 failure is not universal method failure"),
        ("ILc", "independent construction labels", "Q0 fixed metric aggregation", "Bstar absent|8|absent", "registered-policy outcome is not unrestricted label value"),
        ("IOr", "held evaluation labels", "B5 denominator only", "evaluation ceiling", "never a deployable selection policy"),
    ]
    geometry = [_tag({
        "information_class": info,
        "information_experiment": experiment,
        "registered_policy_class_Delta": policy,
        "observed_action_geometry": action,
        "identification_boundary": boundary,
        "unrestricted_optimal_risk": "NOT_IDENTIFIED",
        "registered_policy_risk": "OBSERVED_ONLY_FOR_FROZEN_POLICIES",
        "policy_approximation_gap": "NOT_IDENTIFIED",
    }, sources=[evidence_source, C83_TABLE_DIR / "claim_contract.csv"], source_keys=[info])
        for info, experiment, policy, action, boundary in geometry_specs]

    theory_specs = [
        ("T1", "Blackwell/Le Cam experiment comparison", "registered policy outcomes only", "common state/action/loss experiment and dominance assumptions", "new untouched cohort or analytically specified experiment", "high", "formal deficiency or non-dominance result", "prospective policy-independent comparison"),
        ("T2", "partial identification and minimax regret", "target-level effect distributions and frozen tails", "uncertainty set and admissible policy class", "untouched target population", "high", "identified set or minimax bound", "pre-registered robust-risk evaluation"),
        ("T3", "average risk versus worst-target/CVaR", "positive means with adverse tails", "tail functional and target exchangeability", "untouched cohorts with sufficient targets", "high", "risk-functional comparison", "fixed mean/worst/CVaR gates"),
        ("T4", "near-tie effective multiplicity", "top-k and regret separate", "candidate utility geometry and tie scale", "frozen field may support a future separately governed audit", "medium", "effective action-set definition", "prospective geometry summary without selector retuning"),
        ("T5", "active label acquisition", "passive Q0 is cohort/level dependent", "acquisition cost and adaptive stopping rule", "new untouched construction labels or dataset", "high", "value-of-information policy bound", "pre-registered active-versus-passive comparison"),
        ("T6", "heterogeneous architecture/model zoo", "current fixed zoo only", "architecture distribution and comparability", "new untouched zoo and targets", "medium", "transport condition across zoos", "prospective heterogeneous-zoo field"),
        ("T7", "additional datasets/paradigms", "three binary-MI cohorts", "task and acquisition harmonization", "new untouched cohorts/paradigms", "medium", "scope conditions for external transport", "prospective multi-paradigm validation"),
    ]
    theory = [_tag({
        "gap_id": gap_id,
        "scientific_question": question,
        "current_evidence": current,
        "missing_assumption": assumption,
        "required_untouched_population": population,
        "risk_of_outcome_informed_design": "HIGH; C84A is post-outcome and cannot validate the proposal",
        "theoretical_deliverable": theoretical,
        "prospective_empirical_deliverable": empirical,
        "priority": priority,
        "C85_authorized": 0,
    }, sources=[evidence_source], source_keys=[gap_id])
        for gap_id, question, current, assumption, population, priority, theoretical, empirical in theory_specs]

    experiment_specs = [
        ("E1", "robust target-risk policy", "distinguish average from tail control", "T2|T3", "new untouched target cohort", "HIGH", "do_not_start_without_new_protocol"),
        ("E2", "active label acquisition", "test policy utilization rather than label presence", "T5", "new untouched construction-label stream", "HIGH", "do_not_start_without_new_protocol"),
        ("E3", "candidate geometry audit", "measure near-tie multiplicity and action density", "T4", "separately governed frozen-field audit or untouched field", "MEDIUM", "protocol_review_first"),
        ("E4", "heterogeneous model zoo", "test transport beyond one architecture trajectory", "T6", "new models and untouched targets", "MEDIUM", "resource_and_protocol_review_first"),
        ("E5", "additional paradigms", "scope binary-MI conclusions", "T7", "new untouched datasets", "MEDIUM", "harmonization_protocol_first"),
        ("E6", "formal information-experiment comparison", "separate information from policy approximation", "T1", "analytical model plus prospective cohort", "HIGH", "theory_contract_first"),
    ]
    experiments = [_tag({
        "proposal_id": proposal_id,
        "direction": direction,
        "scientific_question": question,
        "linked_theory_gaps": gaps,
        "required_untouched_population": population,
        "priority": priority,
        "decision": decision,
        "current_authorization": "NOT_AUTHORIZED",
        "outcome_informed_design_guard": "C84 findings may motivate but may not validate the next protocol",
    }, sources=[evidence_source], source_keys=[proposal_id])
        for proposal_id, direction, question, gaps, population, priority, decision in experiment_specs]

    return {
        "information_policy_action_geometry_matrix.csv": geometry,
        "theory_gap_registry.csv": theory,
        "next_experiment_decision_matrix.csv": experiments,
    }


def build_claim_contract() -> list[dict[str, Any]]:
    source = REPORT_DIR / "C84S_OVERALL_REPORT.json"
    specs = [
        ("A1", "SUPPORTED", "COTT has positive mean Q1 effects in all three external cohorts but fails the registered Q1 tail robustness rule in each.", "C84S frozen Q1 rows plus C84A target-effect summaries", "Post-C84S descriptive synthesis; Q1 remains failed.", "COTT universal success or failure"),
        ("A2", "SUPPORTED", "Lee COTT is a single-target near-boundary floor failure; Cho has one deeper floor breach; Physionet has nine floor breaches.", "cott_average_tail_separation.csv", "Order statistics are exploratory and do not change Q1.", "One target explains all C84 heterogeneity"),
        ("A3", "SUPPORTED", "Cho MaNo passes Q1 and Q2 despite near-zero global rank association and weak exact-best localization.", "mano_cross_dataset_decision_profile.csv", "Cohort-specific fixed-policy decision result.", "MaNo universal external validity"),
        ("A4", "SUPPORTED", "Cho MaNo's frozen selections are entirely concentrated in the ERM regime.", "selected_regime_distribution.csv", "Association only; no causal mechanism or candidate-density claim.", "ERM concentration explains the result causally"),
        ("A5", "SUPPORTED", "Passive Q0 actionability is cohort- and source-support-level dependent under the registered policy.", "label_frontier_component_matrix.csv", "Restricted-policy statement; C84-L4 remains fixed.", "Labels are less informative than unlabeled data"),
        ("A6", "SUPPORTED", "Source-support level is an observed effect modifier for Lee COTT and Cho label complexity.", "level_specific_gate_component_matrix.csv|level_label_complexity_interaction.csv", "No mechanism is identified.", "Support deletion improves selection in general"),
        ("A7", "SUPPORTED", "LOTO category is preserved in 21/22 Lee, 20/20 Cho, and 76/76 Physionet panels.", "leave_one_target_out.csv", "Target composition is not the sole basis for C84-D.", "LOTO proves population transport"),
        ("A8", "SUPPORTED", "Observed policy risk does not identify Blackwell dominance or the unrestricted value of information.", "C84A_TPAMI_DECISION_THEORY_BRIDGE.md", "Formal distinction, not a proved comparison theorem.", "C84 proves a Blackwell or minimax theorem"),
        ("F1", "FORBIDDEN", "COTT succeeds or fails universally.", "NONE", "Outside dataset, field and fixed-policy scope.", "ANY_ASSERTION"),
        ("F2", "FORBIDDEN", "MaNo has universal external validity.", "NONE", "Cho-specific result only.", "ANY_ASSERTION"),
        ("F3", "FORBIDDEN", "Labels are less informative than unlabeled data.", "NONE", "Restricted non-nested policies do not identify information ordering.", "ANY_ASSERTION"),
        ("F4", "FORBIDDEN", "Lee or Physionet labels have no value.", "NONE", "Only registered Q0 policy actionability failed.", "ANY_ASSERTION"),
        ("F5", "FORBIDDEN", "One target causes all heterogeneity.", "NONE", "Dataset and level heterogeneity independently suffice.", "ANY_ASSERTION"),
        ("F6", "FORBIDDEN", "C85 or manuscript work is authorized.", "NONE", "C84A ends at protocol review required.", "ANY_ASSERTION"),
    ]
    return [_tag({
        "claim_id": claim_id,
        "status": status,
        "claim_text": claim,
        "evidence": evidence,
        "required_qualifier": qualifier,
        "forbidden_expansion": forbidden,
        "manuscript_authorized": 0,
        "C85_authorized": 0,
    }, sources=[source], source_keys=[claim_id])
        for claim_id, status, claim, evidence, qualifier, forbidden in specs]


def _format_float(value: Any, digits: int = 6) -> str:
    return f"{float(value):.{digits}f}"


def build_decision_theory_bridge() -> str:
    return """# C84A TPAMI Decision-Theory Bridge

## Status And Boundary

This is a post-C84S read-only synthesis. Every empirical summary is
`POST_C84S_EXPLORATORY_DESCRIPTIVE`. It neither changes C84-D/C84-L4 nor proves
an information-ordering, minimax, or transport theorem.

## Information, Policy, And Risk

Let `E` denote an information experiment, `D(E)` the unrestricted set of
decision rules measurable under that experiment, `Delta(E)` the frozen
registered policy class, and `R(delta; E)` the risk under the common action and
loss definition. Distinguish:

```text
unrestricted optimal risk:
  R*(E) = inf_{delta in D(E)} R(delta; E)

registered-policy optimal risk:
  R_Delta(E) = inf_{delta in Delta(E)} R(delta; E)

policy approximation/optimization gap:
  G_Delta(E) = R_Delta(E) - R*(E) >= 0
```

If one experiment Blackwell-dominates another and the state, action and loss
spaces are common, unrestricted optimal risk cannot increase. C84S does not
estimate those unrestricted infima. It evaluates particular pre-registered,
non-nested policies: fixed zero-label formulas and the fixed Q0 construction-
label policy. Therefore an observed row in which COTT has lower regret than Q0
B=1 or Q0 FULL cannot establish that unlabeled observations are more
informative than labels.

The observed ordering can arise because the registered policy classes differ,
because a policy uses its information imperfectly, because optimization is
restricted, or because the robust target-level gate differs from mean risk.
Q0 failure can therefore be evidence about registered policy actionability and
tail robustness without being evidence that the labels carry no information.

## Action Geometry

The compact result separates four objects:

1. Global ranking fidelity: Spearman, Kendall, and pairwise ordering.
2. Top-tail localization: top-1, top-5, and top-10.
3. Selected-action regret: loss from the frozen selected candidate.
4. Robust qualification: mean, multiplicity, target-count, tail, panel/seed,
   level, and LOTO gates.

Cho MaNo illustrates this separation: near-zero global rank association and
low top-1 coexist with a Q1/Q2-qualified selected action. The allowed compact
tables show that its selections concentrate in the ERM regime, but they do not
contain the full candidate utility geometry needed to identify near-optimal
action density or a causal mechanism.

## What Remains Unidentified

- Blackwell or Le Cam deficiency between the information experiments.
- The unrestricted value of construction labels.
- The policy approximation gap of Q0, COTT, or MaNo.
- A minimax or CVaR-optimal selector.
- Causal effects of source-support deletion on candidate geometry.
- Transport beyond the three harmonized binary-MI cohorts and frozen zoo.

The evidence-to-theory mapping is frozen in
`c84a_tables/information_policy_action_geometry_matrix.csv`; prospective gaps
and empirical requirements are in `theory_gap_registry.csv` and
`next_experiment_decision_matrix.csv`. No next experiment is authorized.
"""


def build_audit_report(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> tuple[str, dict[str, Any]]:
    cott = {row["dataset"]: row for row in tables["cott_average_tail_separation.csv"]}
    recurrence = {row["dataset"]: row for row in tables["cott_cross_cohort_recurrence.csv"]}
    mano = {row["dataset"]: row for row in tables["mano_cross_dataset_decision_profile.csv"]}
    frontier = {
        (row["dataset"], row["budget"]): row
        for row in tables["label_frontier_component_matrix.csv"]
    }
    result = read_json(C84S_RESULT_DIR / "C84S_RESULT.json")
    report = read_json(REPORT_DIR / "C84S_OVERALL_REPORT.json")
    gate_table_rows = [
        "| Dataset | Method | Q1 | Q1 nearest failure | Q2 | Q2 nearest failure |",
        "|---|---|---|---|---|---|",
    ]
    for row in tables["full_panel_gate_component_matrix.csv"]:
        gate_table_rows.append(
            f"| {row['dataset']} | {row['method_id']} {row['method_name']} | "
            f"{'PASS' if row['Q1_pass_frozen'] else 'FAIL'} | {row['Q1_nearest_failing_component']} | "
            f"{'PASS' if row['Q2_pass_frozen'] else 'FAIL'} | {row['Q2_nearest_failing_component']} |"
        )
    gate_table = "\n".join(gate_table_rows)
    frontier_table_rows = [
        "| Dataset | Budget | Mean effect | max-T p | Favorable | Worst (target) | Direct | Closure |",
        "|---|---|---:|---:|---:|---:|---|---|",
    ]
    for row in tables["label_frontier_component_matrix.csv"]:
        frontier_table_rows.append(
            f"| {row['dataset']} | {row['budget']} | {_format_float(row['mean_source_relative_effect'])} | "
            f"{float(row['maxT_pvalue_frozen']):.6g} | {row['favorable_targets_descriptive_replay']}/{TARGET_COUNTS[row['dataset']]} | "
            f"{_format_float(row['worst_target_effect_descriptive_replay'])} ({row['worst_target_ids_descriptive_replay']}) | "
            f"{'PASS' if row['direct_qualification_frozen'] else 'FAIL'} | "
            f"{'PASS' if row['larger_budget_closure_frozen'] else 'FAIL'} |"
        )
    frontier_table = "\n".join(frontier_table_rows)

    markdown = f"""# C84A Post-Scientific External Heterogeneity Audit

## Final Audit Gate

```text
{SUCCESS_GATE}
```

C84A is a read-only, post-outcome synthesis. It preserves the confirmatory C84S
gate `{GATE}` and frontier tag `{FRONTIER_TAG}`. All new order statistics,
component distances and cross-cycle matrices are explicitly `{STATUS}`.

## Frozen Result Replay

- C84S lock, authorization, selection freeze, scientific result, and result
  manifest replay exactly.
- All 18 registered tables replay by SHA-256 and row count.
- Coverage remains 944 contexts and 18,432 method-context rows.
- Stage A/B/C lifecycle and protected counters replay without mismatch.
- Historical C84S scientific/final red teams and accepted regressions replay.

No EEG array, direct label view, target logit/source array, selector, Q0 builder,
Stage B/C callable, inference engine, model checkpoint, or oracle was opened.

## Audit Semantics

The full-panel matrix copies frozen max-T p-values and simultaneous Q2 bounds,
then replays each registered threshold as a signed distance. The level table
copies its frozen Q1/Q2 pass and p-values and computes only descriptive target
and panel/seed summaries from frozen result rows. A level-specific Q2
simultaneous upper bound was not frozen; it is marked
`NOT_FROZEN_AT_LEVEL_SCOPE` and is not reconstructed. Quantiles use linear
interpolation at `(n-1)p`. No permutation stream, max-T statistic, confidence
bound, selector score, or candidate choice is recomputed.

## Complete Zero-Label Gate Map

{gate_table}

The nearest-failure label is a navigation aid, not a replacement gate. It uses
normalized distance only among failed registered components; all raw signed
distances and every pass flag remain in
`full_panel_gate_component_matrix.csv`. The level matrix preserves unavailable
components rather than inventing finite values.

## COTT: Positive Average, Non-Robust Tail

| Dataset | Mean | Median | Lower decile | Worst | Floor breaches | Frozen Q1 | Frozen Q2 |
|---|---:|---:|---:|---:|---:|---|---|
| Lee | {_format_float(cott['Lee2019_MI']['mean'])} | {_format_float(cott['Lee2019_MI']['median'])} | {_format_float(cott['Lee2019_MI']['q10'])} | {_format_float(cott['Lee2019_MI']['minimum'])} | {cott['Lee2019_MI']['registered_floor_breach_targets']} | FAIL | PASS |
| Cho | {_format_float(cott['Cho2017']['mean'])} | {_format_float(cott['Cho2017']['median'])} | {_format_float(cott['Cho2017']['q10'])} | {_format_float(cott['Cho2017']['minimum'])} | {cott['Cho2017']['registered_floor_breach_targets']} | FAIL | PASS |
| Physionet | {_format_float(cott['PhysionetMI']['mean'])} | {_format_float(cott['PhysionetMI']['median'])} | {_format_float(cott['PhysionetMI']['q10'])} | {_format_float(cott['PhysionetMI']['minimum'])} | {cott['PhysionetMI']['registered_floor_breach_targets']} | FAIL | PASS |

Lee has two adverse COTT targets, but its only registered floor breach is target 8 at
`{_format_float(cott['Lee2019_MI']['minimum'])}`, only 0.007873 below the -0.10
floor. Omitting target 8 is the sole Lee LOTO category change, C to A. Cho also
has two adverse targets; its only floor breach is target 3 at `{_format_float(cott['Cho2017']['minimum'])}` and is
materially deeper, although Cho remains category A through U11/MaNo.
Physionet has 19 adverse targets, including nine floor-breaching targets
(`{cott['PhysionetMI']['registered_floor_breach_target_ids']}`), with a minimum
of `{_format_float(cott['PhysionetMI']['minimum'])}`. Its 76/76 LOTO categories
remain C. Thus Lee is a single-target near-boundary COTT failure, while the
Physionet tail is distributed; target composition is not the sole explanation
of C84-D.

COTT's mean direction is positive and its frozen Q2 decision passes in all
three cohorts. It nevertheless fails Q1 in every cohort. This is a recurrent
average-case/non-robust-tail pattern, not a Q1 success and not a universal
failure statement.

## MaNo: Decision Without Global Rank Fidelity

Cho U11/MaNo has mean regret `{_format_float(mano['Cho2017']['mean_regret'])}`,
Q1 mean `{_format_float(mano['Cho2017']['Q1_mean'])}`, frozen Q1/Q2 PASS, and
worst target `{_format_float(mano['Cho2017']['Q1_worst_target'])}`. Its mean
Spearman is `{float(mano['Cho2017']['mean_Spearman']):.6f}` and top-1/top-5/top-10
are `{float(mano['Cho2017']['top1']):.4f}` / `{float(mano['Cho2017']['top5']):.4f}` /
`{float(mano['Cho2017']['top10']):.4f}`.

All Cho MaNo selections are in the ERM regime, and its frozen selected utility
and regret exactly match B1 in all Cho contexts. This is a descriptive
selection-concentration fact. It does not identify why the policy works, nor
does it establish a dense near-optimal candidate region: the allowed compact
tables do not contain all 81 candidate utilities per context.

## Passive-Label Frontier

- Lee has no B*. B=8 and FULL satisfy the mean and frozen max-T components but
  fail the registered worst-target floor; lower budgets fail additional
  components. A lower mean Q0 FULL regret is therefore insufficient for the
  robust compound gate.
- Cho has B*=8 because B=8 and FULL directly qualify and satisfy larger-budget
  closure. Its frozen level frontiers differ: level 0 is FULL and level 1 is 4.
- Physionet has no B*. At B=8 and FULL the mean component is positive, but
  max-T, favorable-target and worst-target components fail.

The policy comparison is nonmonotone in observed regret under fixed,
non-nested registered policies. It is not a Blackwell comparison. The exact
decision-theory boundary is in `C84A_TPAMI_DECISION_THEORY_BRIDGE.md`.

### Complete Primary-Budget Component Replay

{frontier_table}

The favorable counts, worst rows, and panel/seed cell means are exact
descriptive reductions of frozen C84S result rows. The displayed max-T p-values,
direct decisions, closure decisions, and B* values are copied from the frozen
frontier table; none is re-estimated.

## Heterogeneity Synthesis

C82 and C84 together show heterogeneity across training seed, dataset,
source-support level, target composition and decision objective. No
cross-dataset p-value is introduced. Method identity is retained row by row in
`c82_c84_method_transport_matrix.csv`.

The transport matrix contains 45 identity-preserving rows: nine methods across
BNCI2014_001 seeds 3/4 and the three C84 cohorts. C80 supplies the frozen Q0
rows for the BNCI field; C82 supplies the strict-source and zero-label rows;
C84S supplies the external-cohort rows. Missing or semantically incomparable
measurements are labelled as such. The matrix performs no pooling and creates
no cross-study p-value.

LOTO category preservation is {recurrence['Lee2019_MI']['LOTO_category_preserved']}/22,
{recurrence['Cho2017']['LOTO_category_preserved']}/20, and
{recurrence['PhysionetMI']['LOTO_category_preserved']}/76. Lee's level-specific
COTT disagreement and the A/C/C dataset categories already suffice for C84-D,
so the single Lee LOTO category change does not carry the full heterogeneity
interpretation.

## Theory And Next Evidence

The highest-priority gaps are a formal information-experiment comparison, a
robust target-risk formulation, and a prospective active-versus-passive label
policy. Each requires assumptions and untouched populations not supplied by
C84A. The decision matrix is advisory only: C85, active acquisition, new data,
new model zoos, and manuscript work remain unauthorized.

## Claim Boundary

Supported: COTT has cross-cohort positive-average value with registered tail
failure; MaNo has Cho-specific decision value without global ranking fidelity;
passive-label actionability and source-support effects are nonuniform under
fixed policies.

Not supported: universal method success/failure, universal external validity,
an information-theoretic ordering of labels and unlabeled outputs, a proved
Blackwell/minimax theorem, a general benefit of support deletion, or any C85 or
manuscript authorization.
"""

    machine = {
        "schema_version": "c84a_post_scientific_heterogeneity_audit_v1",
        "status": "COMPLETE",
        "final_gate": SUCCESS_GATE,
        "analysis_status": STATUS,
        "confirmatory_gate_changed": False,
        "immutable_C84S_primary_gate": result["primary_gate"],
        "immutable_C84S_label_frontier_tag": result["label_frontier_tag"],
        "independent_confirmation": False,
        "new_pvalues": 0,
        "new_scientific_gate": 0,
        "inputs": {
            "result_sha256": EXPECTED_IDENTITIES["C84S_scientific_result"][1],
            "manifest_sha256": EXPECTED_IDENTITIES["C84S_result_manifest"][1],
            "selection_freeze_sha256": EXPECTED_IDENTITIES["C84S_selection_freeze"][1],
            "frozen_tables": 18,
            "method_context_rows": 18432,
        },
        "cott": {
            dataset: {
                "mean": row["mean"],
                "median": row["median"],
                "q10": row["q10"],
                "worst": row["minimum"],
                "floor_breach_targets": row["registered_floor_breach_targets"],
                "floor_breach_target_ids": row["registered_floor_breach_target_ids"],
                "Q1_pass_frozen": bool(recurrence[dataset]["Q1_pass_frozen"]),
                "Q2_pass_frozen": bool(recurrence[dataset]["Q2_pass_frozen"]),
            } for dataset, row in cott.items()
        },
        "mano": {
            dataset: {
                "mean_regret": row["mean_regret"],
                "mean_Spearman": row["mean_Spearman"],
                "top1": row["top1"],
                "top5": row["top5"],
                "top10": row["top10"],
                "Q1_pass": bool(row["Q1_pass"]),
                "Q2_pass": bool(row["Q2_pass"]),
                "dominant_regime": row["dominant_selected_regime"],
                "dominant_regime_fraction": row["dominant_regime_fraction"],
                "action_density_identified": False,
            } for dataset, row in mano.items()
        },
        "label_frontier": {
            "Bstar": {
                dataset: report["dataset_results"][dataset]["Bstar"]
                for dataset in DATASETS
            },
            "level0_Bstar": {
                dataset: report["dataset_results"][dataset]["level0_Bstar"]
                for dataset in DATASETS
            },
            "level1_Bstar": {
                dataset: report["dataset_results"][dataset]["level1_Bstar"]
                for dataset in DATASETS
            },
            "Lee_FULL_failure_components": next(
                row["direct_failing_components"]
                for row in tables["label_frontier_closure_failures.csv"]
                if row["dataset"] == "Lee2019_MI" and row["budget"] == "FULL"
            ),
            "Physionet_FULL_failure_components": next(
                row["direct_failing_components"]
                for row in tables["label_frontier_closure_failures.csv"]
                if row["dataset"] == "PhysionetMI" and row["budget"] == "FULL"
            ),
        },
        "read_boundary": {
            "EEG_arrays": 0,
            "direct_label_roots": 0,
            "target_logits_or_source_arrays": 0,
            "selector_or_Q0_callable": 0,
            "inference_callable": 0,
            "training_forward_GPU_oracle": 0,
        },
        "authorization": {
            "C85": False,
            "active_acquisition": False,
            "new_dataset_or_model_zoo": False,
            "manuscript": False,
        },
        "table_registry": {},
    }
    return markdown, machine


def validate_tables(tables: Mapping[str, Sequence[Mapping[str, Any]]]) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []

    def check(check_id: str, passed: bool, evidence: Any) -> None:
        checks.append({
            "check_id": check_id,
            "passed": int(bool(passed)),
            "blocking": int(not passed),
            "evidence": str(evidence),
        })

    check("identity_replay", all(row["status"] == "PASS" for row in tables["c84s_identity_replay.csv"]), "10/10")
    check("result_manifest_replay", all(row["status"] == "PASS" for row in tables["result_table_manifest_replay.csv"]), "18/18")
    check("result_table_count", len(tables["result_table_manifest_replay.csv"]) == 18, 18)
    check("result_table_rows", sum(int(row["observed_rows"]) for row in tables["result_table_manifest_replay.csv"]) > 0, "all nonempty")
    check("lifecycle_replay", all(row["status"] == "PASS" for row in tables["lifecycle_stage_replay.csv"]), "Stage A/B/C")
    check("protected_counters", all(row["status"] == "PASS" for row in tables["protected_counter_replay.csv"]), "11/11")
    check("historical_verification", all(row["status"] == "PASS" for row in tables["regression_redteam_replay.csv"]), "6/6")
    check("full_gate_rows", len(tables["full_panel_gate_component_matrix.csv"]) == 18, 18)
    check("level_gate_rows", len(tables["level_specific_gate_component_matrix.csv"]) == 36, 36)
    check("cott_target_rows", len(tables["cott_target_effect_distribution.csv"]) == 118, 118)
    cott_summary = {row["dataset"]: row for row in tables["cott_average_tail_separation.csv"]}
    check("cott_Lee_worst", abs(float(cott_summary["Lee2019_MI"]["minimum"]) + 0.10787324378695262) < 1e-14, cott_summary["Lee2019_MI"]["minimum"])
    check("cott_Lee_target", cott_summary["Lee2019_MI"]["registered_floor_breach_target_ids"] == "8", "target 8")
    check("cott_Cho_target", cott_summary["Cho2017"]["registered_floor_breach_target_ids"] == "3", "target 3")
    check("cott_Physionet_floor_count", int(cott_summary["PhysionetMI"]["registered_floor_breach_targets"]) == 9, 9)
    changed = [row for row in tables["cott_target_influence.csv"] if int(row["category_changed_frozen"])]
    check("cott_LOTO_category_change", len(changed) == 1 and changed[0]["dataset"] == "Lee2019_MI" and changed[0]["left_out_target"] == "8", "Lee target 8 only")
    mano = {row["dataset"]: row for row in tables["mano_cross_dataset_decision_profile.csv"]}
    check("mano_Cho_Q1_Q2", int(mano["Cho2017"]["Q1_pass"]) == 1 and int(mano["Cho2017"]["Q2_pass"]) == 1, "PASS/PASS")
    check("mano_Cho_ERM", mano["Cho2017"]["dominant_selected_regime"] == "ERM" and float(mano["Cho2017"]["dominant_regime_fraction"]) == 1.0, "160/160")
    check("mano_action_density_unidentified", all(int(row["near_optimal_action_density_identified"]) == 0 for row in mano.values()), "not in compact tables")
    check("frontier_rows", len(tables["label_frontier_component_matrix.csv"]) == 15, 15)
    front = {(row["dataset"], row["budget"]): row for row in tables["label_frontier_component_matrix.csv"]}
    check("Lee_FULL_tail_failure", int(front[("Lee2019_MI", "FULL")]["worst_component_pass"]) == 0, "worst component")
    check("Cho_B8_qualification", int(front[("Cho2017", "8")]["direct_qualification_frozen"]) == 1, "B*=8")
    check("Physionet_FULL_failures", int(front[("PhysionetMI", "FULL")]["maxT_component_pass"]) == 0 and int(front[("PhysionetMI", "FULL")]["worst_component_pass"]) == 0, "maxT/tail")
    check("transport_rows", len(tables["c82_c84_method_transport_matrix.csv"]) == 45, 45)
    check("theory_gaps", len(tables["theory_gap_registry.csv"]) == 7, 7)
    check("next_experiments_unauthorized", all(row["current_authorization"] == "NOT_AUTHORIZED" for row in tables["next_experiment_decision_matrix.csv"]), "all")

    for name, rows in tables.items():
        if name == "C84A_CLAIM_CONTRACT.csv":
            continue
        check(f"tag_{name}", all(row.get("analysis_status") == STATUS and int(row.get("confirmatory_gate_changed", 1)) == 0 for row in rows), len(rows))
        check(f"source_keys_{name}", all(row.get("source_artifacts") and row.get("source_row_keys") for row in rows), len(rows))

    tree = ast.parse(Path(__file__).read_text())
    imported = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.append(node.module or "")
    forbidden_import_fragments = ("c84s_select", "c84s_q0", "c84s_inference", "c84sr3_stage", "torch", "numpy", "mne", "moabb")
    check("static_no_scientific_import", not any(any(fragment in name for fragment in forbidden_import_fragments) for name in imported), imported)
    call_names: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_names.append(node.func.id)
        elif isinstance(node.func, ast.Attribute):
            parts = [node.func.attr]
            current = node.func.value
            while isinstance(current, ast.Attribute):
                parts.append(current.attr)
                current = current.value
            if isinstance(current, ast.Name):
                parts.append(current.id)
            call_names.append(".".join(reversed(parts)))
    check("no_array_loader", not any(name in {"numpy.load", "np.load", "torch.load"} for name in call_names), "no array/checkpoint load call")
    check("no_new_pvalue_engine", not any("rademacher" in name.lower() or "permutation" in name.lower() for name in call_names), "frozen p-values copied only")
    check("immutable_gate", read_json(C84S_RESULT_DIR / "C84S_RESULT.json")["primary_gate"] == GATE, GATE)
    check("immutable_frontier", read_json(C84S_RESULT_DIR / "C84S_RESULT.json")["label_frontier_tag"] == FRONTIER_TAG, FRONTIER_TAG)
    check("git_payload_hygiene", not any(path.stat().st_size > 50 * 1024 * 1024 for path in TABLE_DIR.glob("*") if path.is_file()), "no generated file >50 MiB")
    return checks


def build(output_report_dir: Path = REPORT_DIR) -> dict[str, Any]:
    output_table_dir = output_report_dir / "c84a_tables"
    table_groups = (
        build_identity_replays(),
        build_gate_matrices(),
        build_cott_audit(),
        build_mano_audit(),
        build_label_frontier_audit(),
        build_transport_audit(),
        build_theory_and_next_experiment_tables(),
    )
    tables: dict[str, list[dict[str, Any]]] = {}
    for group in table_groups:
        overlap = set(tables).intersection(group)
        if overlap:
            raise C84AAuditError(f"duplicate output table names: {sorted(overlap)}")
        tables.update(group)
    claims = build_claim_contract()
    tables["C84A_CLAIM_CONTRACT.csv"] = claims

    checks = validate_tables(tables)
    failed = [row for row in checks if not row["passed"]]
    if failed:
        raise C84AAuditError(f"C84A validation failed: {failed}")

    output_table_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in tables.items():
        destination = output_report_dir / name if name == "C84A_CLAIM_CONTRACT.csv" else output_table_dir / name
        write_csv(destination, rows)

    bridge = build_decision_theory_bridge()
    (output_report_dir / "C84A_TPAMI_DECISION_THEORY_BRIDGE.md").write_text(bridge)
    markdown, machine = build_audit_report(tables)
    for name in sorted(key for key in tables if key != "C84A_CLAIM_CONTRACT.csv"):
        path = output_table_dir / name
        machine["table_registry"][name] = {
            "rows": len(tables[name]),
            "sha256": sha256_file(path),
        }
    machine["claim_contract"] = {
        "rows": len(claims),
        "sha256": sha256_file(output_report_dir / "C84A_CLAIM_CONTRACT.csv"),
    }
    machine["validation"] = {
        "checks": len(checks),
        "passed": len(checks),
        "failed": 0,
    }
    md_path = output_report_dir / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.md"
    json_path = output_report_dir / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.json"
    md_path.write_text(markdown)
    write_json(json_path, machine)
    sha_path = output_report_dir / "C84A_POST_SCIENTIFIC_HETEROGENEITY_AUDIT.sha256"
    sha_path.write_text(
        f"{sha256_file(md_path)}  {md_path.name}\n"
        f"{sha256_file(json_path)}  {json_path.name}\n"
    )

    red_team_lines = [
        "# C84A Final Report Red Team", "",
        f"Final gate: `{SUCCESS_GATE}`", "",
        "| Check | Pass | Evidence |", "|---|---:|---|",
    ]
    red_team_lines.extend(
        f"| {row['check_id']} | {row['passed']} | {row['evidence']} |"
        for row in checks
    )
    red_team_lines.extend([
        "", "All checks passed. C84-D and C84-L4 remain immutable. C85 and manuscript work remain unauthorized.", "",
    ])
    (output_report_dir / "C84A_FINAL_REPORT_RED_TEAM.md").write_text("\n".join(red_team_lines))
    return {"tables": tables, "checks": checks, "report": machine}


def _main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "validate"))
    parser.add_argument("--output-report-dir", type=Path, default=REPORT_DIR)
    args = parser.parse_args(argv)
    result = build(args.output_report_dir)
    if args.command == "validate":
        print(json.dumps({
            "status": "PASS",
            "checks": len(result["checks"]),
            "gate": result["report"]["final_gate"],
        }, sort_keys=True))
    else:
        print(json.dumps({
            "status": "COMPLETE",
            "tables": len(result["tables"]),
            "gate": result["report"]["final_gate"],
        }, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
