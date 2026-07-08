"""C34 compact report-payload helpers.

The row-level C34 payload lives in c34_tables CSV artifacts. The JSON report is
kept as a compact summary plus table manifest so the current tree does not carry
a monolithic row dump.
"""
from __future__ import annotations

import csv
import hashlib
import os


TABLES = (
    "endpoint_utility_registry.csv",
    "selected_vs_continuous_better_pairs.csv",
    "continuous_local_regret_by_trajectory.csv",
    "local_source_gradient_alignment.csv",
    "source_objective_component_conflict.csv",
    "gauge_jump_local_regret.csv",
    "target_unlabeled_local_regret.csv",
    "binary_vs_continuous_boundary_status.csv",
    "local_random_baseline_continuous.csv",
    "no_selector_artifact_gate.csv",
    "c34_case_taxonomy.csv",
)


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def csv_row_count(path):
    with open(path, newline="") as f:
        return max(sum(1 for _ in csv.reader(f)) - 1, 0)


def table_manifest(table_dir):
    rows = []
    for name in TABLES:
        path = os.path.join(table_dir, name)
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        rows.append({
            "table": name,
            "path": os.path.join("c34_tables", name),
            "bytes": os.path.getsize(path),
            "rows": csv_row_count(path),
            "sha256": sha256_file(path),
        })
    return rows


def _gate_summary(table_dir):
    path = os.path.join(table_dir, "no_selector_artifact_gate.csv")
    with open(path, newline="") as f:
        return {r["check"]: (r["passed"] == "True") for r in csv.DictReader(f)}


def compact_payload(res, table_dir):
    selected = res["selected_pairs"]["summary"]
    direction = res["source_direction"]["summary"]
    components = res["source_objective_components"]["summary"]
    gauge = res["gauge_locality"]["summary"]
    boundary = res["binary_vs_continuous_boundary"]["summary"]
    return {
        "artifact": "C34_CONTINUOUS_LOCAL_REGRET_AUDIT.compact",
        "schema_version": "C34S-compact-v1",
        "payload_policy": "row_level_payload_in_c34_tables_csv",
        "verdict": {
            "primary_cases": res["taxonomy"]["cases"],
            "not_established": ["M1_binary_margin_artifact", "M6_target_gauge_jump_drives_local_regret"],
            "diagnostic_only_non_deployable": bool(res["diagnostic_only_non_deployable"]),
        },
        "config": {
            "config_hash": res["config_hash"],
            "mode": res["mode"],
            "n_rows": res["n_rows"],
            "primary_margin": res["primary_margin"],
            "robust_margin": res["robust_margin"],
        },
        "taxonomy": res["taxonomy"],
        "gates": _gate_summary(table_dir),
        "key_aggregates": {
            "selected_pair_regret": selected,
            "source_direction": direction,
            "source_objective_components": components,
            "gauge_locality": gauge,
            "binary_vs_continuous_boundary": boundary,
            "endpoint_summary": res["endpoint_summary"],
            "robust_selected_pairs_summary": res["robust_selected_pairs"]["summary"],
        },
        "table_manifest": table_manifest(table_dir),
        "target_unlabeled": res["target_unlabeled"],
    }
