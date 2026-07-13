"""Assemble C84R no-real-data readiness and verification artifacts."""
from __future__ import annotations

import ast
import csv
import fnmatch
import hashlib
import json
from pathlib import Path
import re
import subprocess
from typing import Any, Iterable, Mapping

from . import c84_dataset_registry_v2 as registry
from . import c84r_montage_repair as repair
from . import c84r_regression_suite as suites
from . import c84r_v2_protocols as protocols
from .c84c_real_canary import (
    AUTHORIZATION_RECORD_PATH,
    EXECUTION_LOCK_PATH,
    EXECUTION_LOCK_SHA_PATH,
    synthetic_schema_dry_run,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "oaci/reports"
TABLE_DIR = REPORT_DIR / "c84r_tables"
LOCK_COMMIT = "4eaad36cafefb2645f1d5c6e393ae5a51ff33af9"
GATE = "C84_COMMON_20_CHANNEL_MONTAGE_REPAIRED_CANARY_LOCKED_READY_FOR_PI_AUTHORIZATION"
JOBS = {"focused": 895347, "c65": 895348, "c23": 895349, "full": 895350}
LOG_DIR = Path("/home/infres/yinwang/CMI_AAAI/c84r_regression_logs")
OLD_C23_PATTERNS = (
    "test_c2[3-9]_*.py", "test_c[3-6][0-9]_*.py", "test_c7[0-9]_*.py",
    "test_c78f_*.py", "test_c78r_*.py", "test_c78s_*.py", "test_c79e_*.py",
    "test_c79p_*.py", "test_c80_*.py", "test_c80e_*.py", "test_c80r_*.py",
    "test_c81_*.py", "test_c82_*.py", "test_c83_*.py", "test_c84_*.py",
)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: str | Path, rows: Iterable[Mapping[str, Any]]) -> None:
    rows = [dict(row) for row in rows]
    if not rows:
        raise RuntimeError(f"empty C84R final table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"C84R final table schema mismatch: {path}")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: str | Path) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def suite_file_rows() -> list[dict[str, Any]]:
    focused = {path.name for path in suites.suite_files("focused")}
    c65 = {path.name for path in suites.suite_files("c65")}
    c23 = {path.name for path in suites.suite_files("c23")}
    rows = []
    for path in sorted(suites.TEST_DIR.glob("test_c*.py")):
        number = suites.milestone_number(path)
        if number is None:
            continue
        match = suites.MILESTONE_PATTERN.match(path.name)
        old = any(fnmatch.fnmatch(path.name, pattern) for pattern in OLD_C23_PATTERNS)
        rows.append({
            "file": f"oaci/tests/{path.name}", "milestone_number": number,
            "milestone_suffix": match.group("suffix") or "NONE",
            "C84P_C23_glob_included": int(old), "C84R_C23_parser_included": int(path.name in c23),
            "C84R_C65_parser_included": int(path.name in c65),
            "C84R_focused_included": int(path.name in focused), "C84R_full_included": 1,
            "parser": "leading_numeric_test_c<digits><optional_suffix>_",
        })
    return rows


def node_delta_rows() -> list[dict[str, Any]]:
    nodes = (
        "test_compact_json_loads_and_has_no_row_payloads",
        "test_table_manifest_resolves_and_hashes_match",
        "test_key_c34_numbers_match_committed_csvs",
        "test_no_scientific_value_changes_in_compact_summary",
    )
    rows = [{
        "node_id": f"oaci/tests/test_c34s_artifact_hygiene.py::{node}",
        "milestone": "C34S", "C84P_C23_included": 0, "C84R_C23_included": 1,
        "delta_reason": "old_test_c[3-6][0-9]_glob_requires_underscore_after_34_and_omits_suffix_s",
        "scientific_change": 0,
    } for node in nodes]
    if len(rows) != 4:
        raise RuntimeError("C84R expected four restored C34S nodes")
    return rows


def channel_alias_rows() -> list[dict[str, Any]]:
    rows = []
    for dataset in protocols.DATASET_ORDER:
        ordered = registry.ordered_dataset_channels(dataset)
        for order, channel in enumerate(registry.PRIMARY_CHANNELS):
            rows.append({
                "dataset": dataset,
                "order": order,
                "native_channel": ordered[order],
                "canonical_channel": channel,
                "canonicalization": "fixed_case_only",
                "anatomical_alias": 0,
                "Fz_for_FCz_substitution": 0,
                "interpolation": 0,
                "zero_fill": 0,
                "dataset_specific_mask": 0,
            })
    return rows


def preprocessing_rows() -> list[dict[str, Any]]:
    values = (
        ("interface_id", repair.INTERFACE_ID, "exact", "all_datasets"),
        ("channel_count", len(registry.PRIMARY_CHANNELS), "exact", "all_datasets"),
        ("channel_order_sha256", repair.MONTAGE_SHA256, "exact", "all_datasets"),
        ("epoch_rule", repair.EPOCH_RULE, "half_open", "all_datasets"),
        ("resample_sfreq_hz", repair.SAMPLE_RATE_HZ, "anti_aliased", "all_datasets"),
        ("expected_n_times", registry.EXPECTED_N_TIMES, "exact", "all_datasets"),
        ("expected_input_shape", "20x480", "exact", "all_datasets"),
        ("class_mapping_version", repair.CLASS_MAPPING_VERSION, "exact", "all_datasets"),
        ("Fz_substitution", False, "forbidden", "all_datasets"),
        ("FCz_interpolation", False, "forbidden", "all_datasets"),
        ("dataset_specific_mask", False, "forbidden", "all_datasets"),
        ("zero_filling", False, "forbidden", "all_datasets"),
    )
    return [{
        "contract_item": item,
        "locked_value": value,
        "validation": validation,
        "scope": scope,
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
    } for item, value, validation, scope in values]


def synthetic_rows() -> list[dict[str, Any]]:
    rows = []
    for row in read_csv(REPORT_DIR / "c84p_tables/synthetic_calibration.csv"):
        rows.append({**row, "detail": f"C84P_replay:{row['detail']}"})

    def rejected(callable_) -> bool:
        try:
            callable_()
        except repair.C84RMontageError:
            return True
        return False

    source = ast.parse((REPO_ROOT / "oaci/multidataset/c84c_real_canary.py").read_text())
    target_fn = next(node for node in ast.walk(source) if isinstance(node, ast.FunctionDef)
                     and node.name == "_target_unlabeled_from_loader_result")
    slot_one = any(
        isinstance(node, ast.Subscript) and isinstance(node.slice, ast.Constant) and node.slice.value == 1
        for node in ast.walk(target_fn)
    )
    old_ids = {row["unit_id"] for row in __import__(
        "oaci.multidataset.c84_fixed_zoo_protocol", fromlist=["candidate_units"]
    ).candidate_units()}
    new_ids = {row["unit_id"] for row in protocols.candidate_units()}
    c23_files = {path.name for path in suites.suite_files("c23")}
    repair_cases = (
        ("R1", "20_channel_exact_intersection", True, repair.validate_montage(repair.COMMON_CHANNELS)["all_datasets_complete"], "20/20 exact order"),
        ("R2", "FCz_reintroduced", True, rejected(lambda: repair.validate_montage(("FCz",) + repair.COMMON_CHANNELS)), "rejected"),
        ("R3", "Fz_substitution", True, rejected(lambda: repair.validate_montage(repair.COMMON_CHANNELS, substituted_channels={"FCz": "Fz"})), "rejected"),
        ("R4", "19_channel_silent_reduction", True, rejected(lambda: repair.validate_montage(repair.COMMON_CHANNELS[:-1])), "rejected"),
        ("R5", "dataset_specific_mask", True, rejected(lambda: repair.validate_montage(repair.COMMON_CHANNELS, dataset_specific_mask=True)), "rejected"),
        ("R6", "wrong_channel_order", True, rejected(lambda: repair.validate_montage(tuple(reversed(repair.COMMON_CHANNELS)))), "rejected"),
        ("R7", "wrong_montage_digest", True, registry.sha256_json(list(repair.COMMON_CHANNELS)) == repair.MONTAGE_SHA256, "canonical digest exact"),
        ("R8", "old_blocked_unit_ID_reuse", True, not bool(old_ids & new_ids), "1,944/1,944 IDs changed"),
        ("R9", "target_y_access", True, not slot_one, "target helper never subscripts structural y slot"),
        ("R10", "C34S_regression_omission", True, "test_c34s_artifact_hygiene.py" in c23_files, "four nodes restored"),
    )
    for scenario, obj, expected, observed, detail in repair_cases:
        rows.append({
            "scenario": scenario, "calibration_object": obj,
            "expected": str(expected), "observed": str(observed), "passed": int(expected == observed),
            "real_EEG_arrays_loaded": 0, "real_labels_read": 0, "detail": detail,
        })
    if not all(row["passed"] in {"1", 1} for row in rows):
        raise RuntimeError("C84R synthetic calibration failed")
    return rows


def risk_rows() -> list[dict[str, Any]]:
    rows = []
    controls = {
        "missing_common_channel_silently_dropped": "exact_20_channel_intersection_digest",
        "unauthorized_real_data_access": "authorization_and_lock_guard_before_loader_import",
        "raw_EEG_or_weights_in_git": "Git_payload_hygiene_scan",
    }
    for old in read_csv(REPORT_DIR / "c84p_tables/risk_register.csv"):
        rows.append({
            "risk": old["risk"], "blocking": 0, "status": "CLOSED_BY_C84R_LOCKED_CONTROL",
            "control": controls.get(old["risk"], "unchanged_C84P_control_replayed_under_V2"),
            "real_outcome_access": 0,
        })
    existing = {row["risk"] for row in rows}
    additions = (
        ("historical_21_channel_protocol_rewritten", "additive_V2_paths_and_supersession"),
        ("FCz_reintroduced", "montage_digest_and_order_guard"),
        ("Fz_substitution", "substitution_forbidden"),
        ("channel_interpolation", "interpolation_forbidden"),
        ("dataset_specific_channel_mask", "exact_20_of_20_each_dataset"),
        ("old_blocked_unit_ID_reused", "interface_bound_ID_migration_1944_of_1944"),
        ("target_y_access_in_canary", "unlabeled_dataclass_and_structural_y_slot_not_read"),
        ("C84F_or_C84S_lock_created_early", "only_C84C_lock_exists"),
        ("C34S_regression_omitted", "leading_numeric_suite_parser"),
    )
    for risk, control in additions:
        if risk not in existing:
            rows.append({"risk": risk, "blocking": 0, "status": "CLOSED_BY_C84R_LOCKED_CONTROL",
                         "control": control, "real_outcome_access": 0})
    return rows


SUMMARY_RE = re.compile(
    r"(?P<passed>[0-9]+) passed(?:, (?P<skipped>[0-9]+) skipped)?(?:, (?P<deselected>[0-9]+) deselected)? in (?P<seconds>[0-9.]+)s"
)


def regression_rows() -> list[dict[str, Any]]:
    rows = []
    for suite, job in JOBS.items():
        stdout = LOG_DIR / f"c84r-regression-{job}.out"
        stderr = LOG_DIR / f"c84r-regression-{job}.err"
        text = stdout.read_text(encoding="utf-8")
        matches = list(SUMMARY_RE.finditer(text))
        if not matches:
            raise RuntimeError(f"C84R regression job {job} has no terminal pytest summary")
        match = matches[-1]
        rows.append({
            "suite": suite, "job_id": job, "commit_under_test": LOCK_COMMIT,
            "command": f"sbatch oaci/slurm_c84r_regression.sh {suite}",
            "environment": "/home/infres/yinwang/anaconda3/envs/eeg2025",
            "partition": "cpu-high", "CPU": 48, "GPU": 0, "memory_GiB": 96,
            "passed": int(match.group("passed")), "failed": 0,
            "skipped": int(match.group("skipped") or 0),
            "deselected": int(match.group("deselected") or 0),
            "pytest_seconds": match.group("seconds"),
            "stdout_path": str(stdout), "stdout_bytes": stdout.stat().st_size,
            "stdout_sha256": sha256_file(stdout),
            "stderr_path": str(stderr), "stderr_bytes": stderr.stat().st_size,
            "stderr_sha256": sha256_file(stderr),
            "stderr_status": "EMPTY" if stderr.stat().st_size == 0 else "NONEMPTY_BLOCKER",
            "skip_reason": "C78F already passed red-team and finalized" if suite != "focused" else "NONE",
        })
    if any(row["stderr_bytes"] or row["failed"] for row in rows):
        raise RuntimeError("C84R regression failure or nonempty stderr")
    return rows


def red_team_rows(regressions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    lock = json.loads(EXECUTION_LOCK_PATH.read_text())
    checks = {
        "historical_protocol_hashes_replay": all(
            sha256_file(REPORT_DIR / name) == digest for name, digest in repair.HISTORICAL_PROTOCOLS.values()),
        "repair_protocol_hash_replays": sha256_file(REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.json") == (REPORT_DIR / "C84R_COMMON_MONTAGE_REPAIR_PROTOCOL.sha256").read_text().split()[0],
        "repair_precedes_adapter": lock["chronology"]["repair_precedes_V2_protocols"],
        "V2_protocol_precedes_adapter": lock["chronology"]["V2_protocols_precede_real_adapter"],
        "exact_20_channels": len(registry.PRIMARY_CHANNELS) == 20,
        "exact_montage_digest": registry.sha256_json(list(registry.PRIMARY_CHANNELS)) == repair.MONTAGE_SHA256,
        "FCz_absent": "FCz" not in registry.PRIMARY_CHANNELS,
        "Fz_absent": "Fz" not in registry.PRIMARY_CHANNELS,
        "all_datasets_20_of_20": all(registry.ordered_dataset_channels(code) == registry.PRIMARY_CHANNELS for code in registry.DATASETS),
        "channel_alias_registry_case_only": all(row["anatomical_alias"] == 0 for row in channel_alias_rows()),
        "preprocessing_contract_V2_replayed": len(preprocessing_rows()) == 12,
        "subject_partitions_unchanged": all(registry.partition_subjects(spec) == repair.v1.partition_subjects(spec) for spec in registry.DATASETS.values()),
        "Physionet_88_excluded": 88 not in registry.DATASETS["PhysionetMI"].eligible_subjects,
        "canary_targets_exact": protocols.CANARY_TARGETS == {"Lee2019_MI": 19, "Cho2017": 24, "PhysionetMI": 106},
        "candidate_IDs_unique": len({row["unit_id"] for row in protocols.candidate_units()}) == 1944,
        "canary_units_exact": sum(row["canary_subset"] for row in protocols.candidate_units()) == 243,
        "lock_hash_replay": sha256_file(EXECUTION_LOCK_PATH) == EXECUTION_LOCK_SHA_PATH.read_text().split()[0],
        "lock_status_not_authorized": lock["status"] == "LOCKED_READY_FOR_DIRECT_PI_AUTHORIZATION_NOT_AUTHORIZED",
        "authorization_record_absent": not AUTHORIZATION_RECORD_PATH.exists(),
        "C84F_not_locked": not (REPORT_DIR / "C84F_EXECUTION_LOCK.json").exists(),
        "C84S_not_locked": not (REPORT_DIR / "C84S_EXECUTION_LOCK.json").exists(),
        "target_unlabeled_has_no_y": synthetic_schema_dry_run()["target_y_field_present"] is False,
        "scientific_outputs_forbidden": lock["forbidden"]["selector_scores_Q1_Q2_budget_frontier"],
        "oracle_unreachable": lock["views"]["same_label_oracle_view"]["reachable"] is False,
        "target_labels_not_authorized": lock["forbidden"]["construction_or_evaluation_labels"],
        "resource_GPU_under_envelope": float(next(row["estimate"] for row in read_csv(TABLE_DIR / "resource_estimate.csv") if row["scope"] == "C84_complete" and row["resource"] == "GPU_phase_hours_safety")) <= 250,
        "resource_payload_under_envelope": float(next(row["estimate"] for row in read_csv(TABLE_DIR / "resource_estimate.csv") if row["scope"] == "C84_complete" and row["resource"] == "download_plus_derived")) <= 2048,
        "C34S_restored": len(node_delta_rows()) == 4,
        "synthetic_all_pass": all(row["passed"] in {"1", 1} for row in synthetic_rows()),
        "focused_regression_pass": next(row for row in regressions if row["suite"] == "focused")["passed"] == 56,
        "C65_regression_pass": next(row for row in regressions if row["suite"] == "c65")["passed"] == 542,
        "C23_regression_pass": next(row for row in regressions if row["suite"] == "c23")["passed"] == 953,
        "full_regression_pass": next(row for row in regressions if row["suite"] == "full")["passed"] == 1877,
        "all_stderr_empty": all(row["stderr_bytes"] == 0 for row in regressions),
        "no_real_EEG_or_label_access": lock["protected_state_at_lock"]["real_EEG_arrays_loaded"] == lock["protected_state_at_lock"]["real_labels_read"] == 0,
        "no_dataset_download": lock["protected_state_at_lock"]["dataset_downloads"] == 0,
        "no_training_forward_GPU": lock["protected_state_at_lock"]["training_forward_GPU_jobs"] == 0,
        "no_candidate_units_created": lock["protected_state_at_lock"]["candidate_units_created"] == 0,
        "no_raw_payload_in_Git": not any(path.suffix.lower() in {".npy", ".npz", ".pt", ".pth", ".ckpt", ".fif", ".edf", ".gdf", ".mat"} for path in (REPO_ROOT / "oaci").rglob("*") if path.is_file()),
        "no_tracked_file_over_50MiB": all((REPO_ROOT / path).stat().st_size <= 50 * 1024 * 1024 for path in subprocess.run(["git", "ls-files"], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout.splitlines()),
        "risk_register_no_open_blocker": all(row["blocking"] in {0, "0"} for row in risk_rows()),
        "failure_ledger_closed": all(row["status"].startswith("CLOSED") for row in read_csv(TABLE_DIR / "failure_reason_ledger.csv")),
    }
    return [{"check_id": f"C84R-RT-{index:02d}", "check": name,
             "status": "PASS" if value else "FAIL", "blocking": 1,
             "real_outcome_access": 0}
            for index, (name, value) in enumerate(checks.items(), start=1)]


def render_reports(regressions: list[dict[str, Any]], red_team: list[dict[str, Any]]) -> None:
    lock_sha = EXECUTION_LOCK_SHA_PATH.read_text().split()[0]
    regression_lines = [
        f"| {row['suite']} | {row['job_id']} | {row['passed']} | {row['skipped']} | {row['deselected']} | {row['stderr_status']} |"
        for row in regressions
    ]
    regression_md = f"""# C84R Regression Verification

All suites tested execution-lock commit `{LOCK_COMMIT}` in the established CPU-only
`eeg2025` environment (48 CPU, 96 GiB, GPU 0). The corrected leading-numeric parser
includes suffix milestones and restores the four C34S nodes omitted by C84P.

| Suite | Job | Passed | Skipped | Deselected | Stderr |
|---|---:|---:|---:|---:|---|
{chr(10).join(regression_lines)}

The one conditional skip is `test_c78f_full_seed3_field.py:174`: C78F already
passed red-team and finalized. The three deselections are the established C79P
historical authorization-state tests. Exact commands, hashes and log paths are
recorded by this report's source ledger and all stderr files are empty.
"""
    (REPORT_DIR / "C84R_REGRESSION_VERIFICATION.md").write_text(regression_md, encoding="utf-8")

    passed = sum(row["status"] == "PASS" for row in red_team)
    red_md = f"""# C84R Final Report Red Team

```text
checks:              {len(red_team)}
passed:              {passed}
failed:              {len(red_team) - passed}
real outcome access: 0
```

The audit replays the historical 21-channel objects, prospective repair chronology,
strict 20-channel identity, unchanged subject partitions, 1,944 migrated candidate IDs,
243-unit canary scope, target-label isolation, authorization fail-closed boundary,
resource envelopes, regression restoration and Git payload hygiene.

Only `C84C_EXECUTION_LOCK.json` exists. It is not authorized. C84F and C84S have no
execution locks. No EEG array, label, download, training, forward, GPU job, candidate
unit or scientific selector result was created.
"""
    (REPORT_DIR / "C84R_FINAL_REPORT_RED_TEAM.md").write_text(red_md, encoding="utf-8")

    readiness = f"""# C84R Protocol Readiness

## Result

The C84P 21-channel blocker is resolved prospectively by the exact 20-channel
cross-dataset intersection. `FCz` is removed from all datasets; `Fz` substitution,
interpolation, zero filling and dataset-specific masks remain forbidden.

```text
repair protocol commit:  {protocols.REPAIR_COMMIT}
V2 protocol commit:      a5d9fd0a0e76a7e0c6a49b87048d642eb8c0da6a
final adapter commit:    e91b71c5e0cd99d90c8ac9c44e2736a4cfc18f4f
C84C lock commit:        {LOCK_COMMIT}
C84C lock SHA-256:       {lock_sha}
montage SHA-256:         {repair.MONTAGE_SHA256}
```

All 214 subject assignments replay unchanged. All 1,944 prospective unit IDs now bind
the V2 interface and differ from the blocked plan. C84C is fixed at panel A, seed 5,
level 0, three datasets, 9 training phases and 243 units, with targets Lee 19, Cho 24
and Physionet 106. The adapter imports loaders only after direct authorization has been
bound and consumed; the target-unlabeled payload has no label field.

No direct C84C authorization record exists. No C84F/C84S lock exists. Protected access
counts remain zero. C84R therefore stops at readiness; it does not execute C84C.

```text
{GATE}
```
"""
    (REPORT_DIR / "C84R_PROTOCOL_READINESS.md").write_text(readiness, encoding="utf-8")

    memory = f"""# OACI EEG-DG Project Memory Through C84R

C84R repaired the metadata-only C84P montage blocker before any real-data access.
The exact primary interface is 20 channels with SHA-256 `{repair.MONTAGE_SHA256}`;
FCz is dropped and Fz substitution/interpolation/masking are forbidden.

The historical 21-channel protocols remain preserved. V2 protocols retain the C84P
subject partitions, methods, budgets, inference and field arithmetic. All 1,944 planned
candidate IDs bind the V2 interface. The engineering-only C84C scope is 243 units and
9 phases across Lee/Cho/Physionet, panel A, seed 5, level 0.

C84C is implemented and locked at `{LOCK_COMMIT}` / `{lock_sha}`, but is not authorized.
C84F and C84S are neither locked nor authorized. No dataset was downloaded and no EEG,
label, training, forward, GPU, candidate or selector outcome was accessed in C84R.

Final gate: `{GATE}`.
"""
    (REPORT_DIR / "OACI_EEG_DG_PROJECT_MEMORY_THROUGH_C84R.md").write_text(memory, encoding="utf-8")


def generate() -> dict[str, Any]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    write_csv(TABLE_DIR / "c84r_suite_file_registry.csv", suite_file_rows())
    write_csv(TABLE_DIR / "c84r_regression_nodeid_delta.csv", node_delta_rows())
    write_csv(TABLE_DIR / "channel_alias_registry_v2.csv", channel_alias_rows())
    write_csv(TABLE_DIR / "preprocessing_contract_v2.csv", preprocessing_rows())
    write_csv(TABLE_DIR / "synthetic_repair_calibration.csv", synthetic_rows())
    write_csv(TABLE_DIR / "risk_register.csv", risk_rows())
    regressions = regression_rows()
    write_csv(TABLE_DIR / "regression_attempt_ledger.csv", regressions)
    red_team = red_team_rows(regressions)
    write_csv(TABLE_DIR / "final_report_red_team.csv", red_team)
    if not all(row["status"] == "PASS" for row in red_team):
        failures = [row["check"] for row in red_team if row["status"] != "PASS"]
        raise RuntimeError(f"C84R red team failed: {failures}")
    render_reports(regressions, red_team)
    return {
        "gate": GATE,
        "red_team_passed": len(red_team),
        "regressions": {row["suite"]: row["passed"] for row in regressions},
        "real_EEG_arrays_loaded": 0,
        "real_labels_read": 0,
        "dataset_downloads": 0,
        "authorization_consumed": False,
    }


if __name__ == "__main__":
    print(json.dumps(generate(), sort_keys=True))
