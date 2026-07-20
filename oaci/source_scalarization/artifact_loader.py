"""Read-only C43 artifact loader."""
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


def candidate_key(row):
    return (row["seed"], row["target"], row["level"], row["regime"], row["candidate_order"])


def load_tables():
    return {
        "c30_rank_gauge": read_csv(os.path.join(schema.C30_TABLE_DIR, "rank_gauge_decomposition.csv")),
        "c41_registry": read_csv(os.path.join(schema.C41_TABLE_DIR, "candidate_objective_field_registry.csv")),
        "c42_top1": read_csv(os.path.join(schema.C42_TABLE_DIR, "diagnostic_top1_by_score.csv")),
        "c42_conflict": read_csv(os.path.join(schema.C42_TABLE_DIR, "leakage_vs_rank_conflict.csv")),
        "c42_summary": read_json("oaci/reports/C42_SOURCE_RANK_ACTIONABILITY_AUDIT.json"),
    }


def _c30_sidecar():
    out = {}
    for r in c30_loader.load():
        if r["mode"] != "in_regime":
            continue
        key = (str(r["seed"]), str(r["target"]), str(r["level"]), r["regime"], str(r["order"]))
        out[key] = r
    return out


def build_candidate_registry(tables):
    sidecar = _c30_sidecar()
    rows = []
    for r in tables["c41_registry"]:
        s = sidecar.get(candidate_key(r))
        if s is None:
            continue
        row = dict(r)
        row["source_rank_score"] = as_float(s["score"])
        row["c19_robust_core_score"] = as_float(s["score"])
        row["train_surrogate"] = as_float(s["train_surrogate"])
        for key, value in s.items():
            if key.startswith("feat__"):
                row[key] = as_float(value)
        for k in ("selection_leakage_point", "audit_leakage_point", "R_src", "balanced_err",
                  "source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece",
                  "source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
                  "target_utility_score", "target_bacc_delta", "target_nll_delta", "target_ece_delta"):
            row[k] = as_float(row.get(k))
        for k in ("primary_joint_good", "pareto_good", "preference_robust_better_candidate", "selected_oaci"):
            row[k] = as_int(row.get(k))
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
    return {"tables": tables, "registry": registry, "by_traj": dict(by_traj), "selected": selected}
