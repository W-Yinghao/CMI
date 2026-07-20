"""Read-only C42 artifact loader."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..rank_gauge import artifact_loader as c30_loader
from . import schema


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def read_json(path):
    with open(path) as f:
        return json.load(f)


def as_float(v, default=math.nan):
    try:
        if v == "":
            return default
        return float(v)
    except (TypeError, ValueError):
        return default


def as_int(v, default=0):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def finite(v) -> bool:
    try:
        return math.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def finite_mean(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.mean(vals)) if vals else None


def finite_median(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.median(vals)) if vals else None


def trajectory_key(row):
    return (row["seed"], row["target"], row["level"], row["regime"])


def candidate_key(row):
    return (row["seed"], row["target"], row["level"], row["regime"], row["candidate_order"])


def _metric(rows, name):
    for r in rows:
        if r.get("metric") == name:
            return as_float(r.get("value"))
    return None


def _axis(rows, name):
    for r in rows:
        if r.get("axis") == name:
            return as_float(r.get("auc"))
    return None


def load_tables():
    return {
        "c30_rank_gauge": read_csv(os.path.join(schema.C30_TABLE_DIR, "rank_gauge_decomposition.csv")),
        "c30_source_rank": read_csv(os.path.join(schema.C30_TABLE_DIR, "source_rank_vs_target_competence.csv")),
        "c32_random": read_csv(os.path.join(schema.C32_TABLE_DIR, "trajectory_random_baseline.csv")),
        "c35_preference": read_csv(os.path.join(schema.C35_TABLE_DIR, "preference_robust_case_audit.csv")),
        "c37_exact": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_vs_better_exact_ucl.csv")),
        "c38_gauge": read_csv(os.path.join(schema.C38_TABLE_DIR, "leakage_vs_target_gauge_conflict.csv")),
        "c40_summary": read_json("oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"),
        "c41_summary": read_json("oaci/reports/C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json"),
        "c41_registry": read_csv(os.path.join(schema.C41_TABLE_DIR, "candidate_objective_field_registry.csv")),
        "c41_alignment": read_csv(os.path.join(schema.C41_TABLE_DIR, "leakage_target_rank_alignment.csv")),
    }


def _c30_score_rows():
    rows = {}
    for r in c30_loader.load():
        if r["mode"] != "in_regime":
            continue
        key = (str(r["seed"]), str(r["target"]), str(r["level"]), r["regime"], str(r["order"]))
        rows[key] = r
    return rows


def build_candidate_registry(tables):
    score_rows = _c30_score_rows()
    rows = []
    for r in tables["c41_registry"]:
        key = candidate_key(r)
        s = score_rows.get(key)
        if s is None:
            continue
        row = {
            **r,
            "source_rank_score": as_float(s["score"]),
            "c19_robust_core_score": as_float(s["score"]),
            "target_utility_oracle_score": as_float(r["target_utility_score"]),
        }
        for k in ("selection_leakage_point", "audit_leakage_point", "R_src", "target_utility_score",
                  "target_bacc_delta", "target_nll_delta", "target_ece_delta"):
            row[k] = as_float(row[k])
        for k in ("primary_joint_good", "pareto_good", "preference_robust_better_candidate", "selected_oaci"):
            row[k] = as_int(row[k])
        rows.append(row)
    rows.sort(key=lambda x: (int(x["seed"]), int(x["target"]), int(x["level"]), x["regime"],
                             int(x["candidate_order"])))
    return rows


def context():
    tables = load_tables()
    registry = build_candidate_registry(tables)
    by_traj = defaultdict(list)
    for r in registry:
        by_traj[r["trajectory_id"]].append(r)
    selected = {}
    for tid, rs in by_traj.items():
        ss = [r for r in rs if int(r["selected_oaci"]) == 1]
        if len(ss) == 1:
            selected[tid] = ss[0]
    summary = {
        "c30_source_rank_auc": _axis(tables["c30_rank_gauge"], "within_target_rank(score)"),
        "c30_pooled_auc": _axis(tables["c30_rank_gauge"], "pooled_gauge(score)"),
        "c30_gauge_centered_auc": _axis(tables["c30_rank_gauge"], "gauge_centered_pooled"),
        "c41_selection_leakage_auc": tables["c41_summary"]["objective_field_comparison_summary"]["selection_leakage_auc"],
        "c41_audit_leakage_auc": tables["c41_summary"]["objective_field_comparison_summary"]["audit_leakage_auc"],
    }
    return {"tables": tables, "registry": registry, "by_traj": dict(by_traj), "selected": selected,
            "summary": summary}
