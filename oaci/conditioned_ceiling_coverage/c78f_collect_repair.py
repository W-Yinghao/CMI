"""Prospective additive repair for C78F collector descriptor row semantics.

The locked collector expected ``rows`` while the frozen c74 descriptor ABI uses
``row_count``.  This module does not modify any file hashed by the C78F execution
lock and performs no training, forward pass, label read, or scientific analysis.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

from . import c74_cache
from . import c78f_collect
from . import c78f_full_seed3_field as c78f
from . import c78f_runtime as runtime


PROTOCOL_PATH = c78f.REPORT_DIR / "C78F_COLLECTOR_REPAIR_PROTOCOL.json"
PROTOCOL_SHA_PATH = c78f.REPORT_DIR / "C78F_COLLECTOR_REPAIR_PROTOCOL.sha256"
RED_TEAM_PATH = c78f.REPORT_DIR / "C78F_COLLECTOR_REPAIR_RED_TEAM.md"
REPAIR_TABLE = c78f.TABLE_DIR / "collector_repair_ledger.csv"
REPAIR_FILES = (
    "oaci/conditioned_ceiling_coverage/c78f_collect_repair.py",
    "oaci/tests/test_c78f_collect_repair.py",
)
FAILED_COLLECTOR_JOB = "893052"


def build_protocol() -> dict[str, Any]:
    lock = runtime.load_execution_lock()
    original = next(item for item in lock["implementation_files"] if item["path"].endswith("c78f_collect.py"))
    return {
        "schema_version": "c78f_collector_repair_protocol_v1",
        "created_at_utc": c78f.utc_now(),
        "parent_protocol_sha256": lock["protocol_sha256"],
        "parent_execution_lock_sha256": c78f.sha256_file(runtime.LOCK_PATH),
        "failed_job_id": FAILED_COLLECTOR_JOB,
        "failure_stage": "engineering_collector_physical_view_table",
        "failure_signature": "KeyError: rows",
        "root_cause": "c74_cache descriptor ABI uses row_count; collector requested rows",
        "repair": "map descriptor row_count to compact table rows without touching payloads",
        "scope": {
            "training": 0,
            "forward": 0,
            "GPU": 0,
            "target_label_reads": 0,
            "target_metrics": 0,
            "scientific_analysis": 0,
            "checkpoint_or_cache_writes": 0,
            "compact_report_table_rewrite_only": True,
        },
        "original_locked_collector": original,
        "original_collector_unchanged": c78f.sha256_file(original["path"]) == original["sha256"],
        "repair_files": [
            {"path": name, "sha256": c78f.sha256_file(name), "size_bytes": Path(name).stat().st_size}
            for name in REPAIR_FILES
        ],
        "required_preconditions": {
            "full_field_frozen": True,
            "label_views_isolated": True,
            "C78S_not_started": True,
            "main_report_absent": True,
        },
    }


def lock_protocol() -> dict[str, Any]:
    protocol = build_protocol()
    c78f.write_json(PROTOCOL_PATH, protocol)
    digest = c78f.sha256_file(PROTOCOL_PATH)
    PROTOCOL_SHA_PATH.write_text(digest + "\n")
    return {"protocol_sha256": digest, "failed_job": FAILED_COLLECTOR_JOB}


def load_protocol() -> tuple[dict[str, Any], str]:
    expected = PROTOCOL_SHA_PATH.read_text().strip()
    observed = c78f.sha256_file(PROTOCOL_PATH)
    if observed != expected:
        raise RuntimeError("C78F collector-repair protocol drift")
    protocol = json.loads(PROTOCOL_PATH.read_text())
    for item in protocol["repair_files"]:
        if c78f.sha256_file(item["path"]) != item["sha256"]:
            raise RuntimeError(f"C78F collector-repair implementation drift: {item['path']}")
    return protocol, observed


def protocol_commit() -> str:
    commit = runtime.git("log", "-1", "--format=%H", "--", str(PROTOCOL_PATH))
    if not commit or runtime.git("merge-base", "--is-ancestor", commit, "HEAD") != "":
        raise RuntimeError("C78F collector-repair protocol is not a committed ancestor")
    return commit


def red_team() -> dict[str, Any]:
    protocol, digest = load_protocol()
    lock = runtime.load_execution_lock()
    full = runtime.verify_manifest(runtime.full_field_path(lock))
    checks = {
        "original_collector_unchanged": c78f.sha256_file(protocol["original_locked_collector"]["path"]) == protocol["original_locked_collector"]["sha256"],
        "all_locked_implementations_unchanged": all(c78f.sha256_file(item["path"]) == item["sha256"] for item in lock["implementation_files"]),
        "repair_files_hashed": all(c78f.sha256_file(item["path"]) == item["sha256"] for item in protocol["repair_files"]),
        "full_field_frozen": full["full_seed3_units"] == 1458,
        "label_views_isolated": full["label_views_created"] is True,
        "science_not_started": full["scientific_analysis_started"] is False,
        "target_outcomes_not_read": full["target_scientific_outcomes_read"] is False,
        "main_report_absent": not (c78f.REPORT_DIR / "C78F_FULL_SEED3_FIELD.md").exists(),
        "repair_no_training_forward_GPU": all(protocol["scope"][key] == 0 for key in ("training", "forward", "GPU")),
        "repair_no_label_or_metric_read": protocol["scope"]["target_label_reads"] == 0 and protocol["scope"]["target_metrics"] == 0,
    }
    failed = [name for name, passed in checks.items() if not passed]
    c78f.write_csv(c78f.TABLE_DIR / "collector_repair_red_team_checks.csv", [
        {"check": name, "status": "PASS" if passed else "FAIL", "blocking": 1}
        for name, passed in checks.items()
    ])
    RED_TEAM_PATH.write_text(f"""# C78F Collector Repair Red-Team

Gate: **{'PASS' if not failed else 'FAIL'}**

```text
repair protocol SHA-256: {digest}
checks: {len(checks)}
failures: {len(failed)}
failed collector job retained: {FAILED_COLLECTOR_JOB}
training/forward/GPU: 0/0/0
target labels/metrics: 0/0
```

The repair is additive and changes only the compact descriptor row-key mapping
from `rows` to the frozen ABI field `row_count`. All execution-locked training
and instrumentation files remain byte-identical.
""")
    if failed:
        raise RuntimeError(f"C78F collector-repair red team failed: {failed}")
    return {"passed": True, "checks": len(checks), "protocol_sha256": digest}


def _fixed_view_rows(lock: dict[str, Any]):
    views = []
    schema = []
    forbidden = c78f_collect.c78f_instrument_forbidden()
    for target in c78f.TARGETS:
        primary = runtime.verify_manifest(runtime.primary_view_path(lock, target))
        labels = runtime.verify_manifest(runtime.label_view_path(lock, target))
        instrument = runtime.verify_manifest(runtime.instrumentation_path(lock, target))
        descriptors = [
            ("strict_source_input", primary["strict_source_input"], 0, 0, 0, "C78F_instrumentation"),
            ("target_unlabeled_input", primary["target_unlabeled_input"], 0, 0, 0, "C78F_instrumentation"),
            ("target_construction_view", labels["target_label_views"]["construction"], 1, 0, 1, "future_C78S_router"),
            ("target_evaluation_view", labels["target_label_views"]["evaluation"], 1, 1, 1, "future_C78S_router"),
            ("same_label_oracle_view", labels["target_label_views"]["same_label_oracle"], 1, 1, 1, "future_C78S_oracle_only"),
        ]
        for name, descriptor, uses_labels, uses_eval, diagnostic, consumer in descriptors:
            c74_cache.verify_shard(descriptor)
            fields = list(descriptor["fields"])
            views.append({
                "target": target, "view_name": name, "path": descriptor["path"],
                "sha256": descriptor["sha256"], "rows": descriptor["row_count"],
                "allowed_columns": json.dumps(fields),
                "forbidden_columns": json.dumps(sorted(forbidden if "unlabeled" in name else [])),
                "uses_target_labels": uses_labels, "uses_evaluation_labels": uses_eval,
                "available_at_selection_time": int(name in {"strict_source_input", "target_unlabeled_input"}),
                "diagnostic_only": diagnostic, "consumer_command": consumer, "physically_separate": 1,
            })
            schema.append({"target": target, "view_name": name, "fields": json.dumps(fields), "field_count": len(fields), "schema_passed": 1})
        schema.append({"target": target, "view_name": "instrumented_outputs", "fields": "registered_Wb_source_target_unlabeled", "field_count": 3, "schema_passed": int(instrument["all_gates_passed"])})
    views.append({"target": 4, "view_name": "C78_C78R_parent_views", "path": "committed_C78_and_C78R_physical_view_manifests", "sha256": c78f.sha256_file(c78f.REPORT_DIR / "c78_tables/physical_view_manifest.csv"), "rows": 162, "allowed_columns": "parent_committed", "forbidden_columns": "parent_committed", "uses_target_labels": 0, "uses_evaluation_labels": 0, "available_at_selection_time": 0, "diagnostic_only": 1, "consumer_command": "C78S_target4_descriptive_only", "physically_separate": 1})
    return views, schema


def execute_repair() -> dict[str, Any]:
    protocol, digest = load_protocol()
    commit = protocol_commit()
    red_checks = c78f.read_csv(c78f.TABLE_DIR / "collector_repair_red_team_checks.csv")
    if not red_checks or any(row["status"] != "PASS" for row in red_checks):
        raise RuntimeError("C78F collector repair requires a passing committed red team")
    c78f_collect._view_rows = _fixed_view_rows
    state = c78f_collect.collect()
    attempts = c78f.read_csv(c78f.TABLE_DIR / "execution_attempt_ledger.csv")
    attempts.extend([
        {"event": "failed", "stage": "engineering_collector", "job_id": FAILED_COLLECTOR_JOB, "target": "all", "wave": "A+B", "time_utc": protocol["created_at_utc"], "failure_stage": "physical_view_descriptor_rows_key", "EEG_data_loaded": 0, "training_started": 0, "target_labels_accessed": 0, "retry_reason": "registered_schema_key_repair", "replacement_job_id": os.environ.get("SLURM_JOB_ID", "unknown"), "final_status": "failed_retained"},
        {"event": "complete", "stage": "engineering_collector_repair", "job_id": os.environ.get("SLURM_JOB_ID", "unknown"), "target": "all", "wave": "A+B", "time_utc": c78f.utc_now(), "failure_stage": "", "EEG_data_loaded": 0, "training_started": 0, "target_labels_accessed": 0, "retry_reason": "", "replacement_job_id": "", "final_status": "completed"},
    ])
    c78f.write_csv(c78f.TABLE_DIR / "execution_attempt_ledger.csv", attempts)
    c78f.write_csv(REPAIR_TABLE, [{
        "repair_protocol_commit": commit,
        "repair_protocol_sha256": digest,
        "failed_job_id": FAILED_COLLECTOR_JOB,
        "replacement_job_id": os.environ.get("SLURM_JOB_ID", "unknown"),
        "root_cause": protocol["root_cause"],
        "repair": protocol["repair"],
        "training": 0, "forward": 0, "GPU": 0,
        "target_labels": 0, "target_metrics": 0,
        "original_locked_collector_unchanged": 1,
        "status": "closed",
    }])
    print(json.dumps({"gate": "C78F_COLLECTOR_REPAIR_COMPLETE", "failed_job_retained": FAILED_COLLECTOR_JOB, "units": state["scope"]["full_units"]}, sort_keys=True))
    return state


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="c78f_collect_repair")
    parser.add_argument("command", choices=("lock-protocol", "red-team", "execute"))
    args = parser.parse_args(argv)
    if args.command == "lock-protocol":
        print(json.dumps(lock_protocol(), sort_keys=True))
    elif args.command == "red-team":
        print(json.dumps(red_team(), sort_keys=True))
    else:
        execute_repair()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
