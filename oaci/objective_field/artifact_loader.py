"""Read-only C41 loaders over candidate-level objective-field artifacts."""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

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
    return (row["seed"], row["target"], row["level"], row["regime"], row.get("candidate_order", row.get("order")))


def load_tables():
    c36 = {
        "registry": read_csv(os.path.join(schema.C36_TABLE_DIR, "selector_trace_registry.csv")),
    }
    c34 = {
        "endpoint": read_csv(os.path.join(schema.C34_TABLE_DIR, "endpoint_utility_registry.csv")),
        "selected_pairs": read_csv(os.path.join(schema.C34_TABLE_DIR, "selected_vs_continuous_better_pairs.csv")),
        "target_unlabeled": read_csv(os.path.join(schema.C34_TABLE_DIR, "target_unlabeled_local_regret.csv")),
    }
    c35 = {
        "utility_simplex": read_csv(os.path.join(schema.C35_TABLE_DIR, "utility_simplex_regret_by_pair.csv")),
        "preference_robust": read_csv(os.path.join(schema.C35_TABLE_DIR, "preference_robust_case_audit.csv")),
    }
    c37 = {
        "exact": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_vs_better_exact_ucl.csv")),
    }
    c38 = {
        "gauge": read_csv(os.path.join(schema.C38_TABLE_DIR, "leakage_vs_target_gauge_conflict.csv")),
    }
    c40 = {
        "summary": read_json("oaci/reports/C40_LEAKAGE_POINT_DRIFT_FORENSICS.json"),
    }
    c30 = {
        "rank_vs_target": read_csv(os.path.join(schema.C30_TABLE_DIR, "source_rank_vs_target_competence.csv")),
        "rank_baseline": read_csv(os.path.join(schema.C30_TABLE_DIR, "source_rank_permutation_baseline.csv")),
        "rank_gauge": read_csv(os.path.join(schema.C30_TABLE_DIR, "rank_gauge_decomposition.csv")),
    }
    return {"c36": c36, "c34": c34, "c35": c35, "c37": c37, "c38": c38, "c40": c40, "c30": c30}


def _preference_robust_better_keys(tables):
    keys = set()
    for r in tables["c35"]["utility_simplex"]:
        if (r["comparison"] == "nearest_continuous_better" and r["scaling"] == "raw" and
                r["utility_cone_category"] == "preference_robust_regret"):
            # C35 stores selected/better orders in pair_id rather than separate
            # candidate columns: seed|target|level|regime|comparison|selected|better.
            parts = r["pair_id"].split("|")
            if len(parts) >= 7:
                keys.add((r["seed"], r["target"], r["level"], r["regime"], parts[-1]))
    return keys


def build_candidate_registry(tables):
    endpoint = {candidate_key({**r, "candidate_order": r["order"]}): r for r in tables["c34"]["endpoint"]}
    robust_better = _preference_robust_better_keys(tables)
    rows = []
    for r in tables["c36"]["registry"]:
        if r["is_erm"] == "1":
            continue
        e = endpoint.get(candidate_key(r))
        if e is None:
            continue
        key = candidate_key(r)
        utility = as_float(e["continuous_joint_min_margin"])
        row = {
            "candidate_id": r["candidate_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "trajectory_id": "|".join([r["seed"], r["target"], r["level"], r["regime"]]),
            "candidate_order": r["candidate_order"],
            "epoch": r["epoch"],
            "selected_oaci": r["selected_oaci"],
            "feasible": r["feasible"],
            "selection_leakage_point": as_float(r["selection_leakage_point"]),
            "selection_leakage_ucl": as_float(r["actual_selector_score_ucl"]),
            "selection_leakage_ucl_available": as_int(r["actual_selector_score_available"]),
            "audit_leakage_point": as_float(r["audit_leakage_point"]),
            "R_src": as_float(r["R_src"]),
            "balanced_err": as_float(r["balanced_err"]),
            "source_guard_worst_bacc": as_float(r["source_guard_worst_bacc"]),
            "source_guard_worst_nll": as_float(r["source_guard_worst_nll"]),
            "source_guard_worst_ece": as_float(r["source_guard_worst_ece"]),
            "source_audit_worst_bacc": as_float(r["source_audit_worst_bacc"]),
            "source_audit_worst_nll": as_float(r["source_audit_worst_nll"]),
            "source_audit_worst_ece": as_float(r["source_audit_worst_ece"]),
            "target_bacc_delta": as_float(e["target_bacc_delta"]),
            "target_nll_delta": as_float(e["target_nll_delta"]),
            "target_ece_delta": as_float(e["target_ece_delta"]),
            "target_bacc_z": as_float(e["target_bacc_z"]),
            "target_nll_z": as_float(e["target_nll_z"]),
            "target_ece_z": as_float(e["target_ece_z"]),
            "continuous_joint_min_margin": utility,
            "endpoint_vector_norm_regret": as_float(e["endpoint_vector_norm_regret"]),
            "pareto_distance": as_float(e["pareto_distance"]),
            "dominated_hypervolume_regret": as_float(e["dominated_hypervolume_regret"]),
            "primary_joint_good": as_int(e["primary_joint_good"]),
            "pareto_good": int(abs(as_float(e["pareto_distance"])) <= 1e-12),
            "preference_robust_better_candidate": int(key in robust_better),
            "target_utility_score": utility,
            "target_utility_finite": int(finite(utility)),
            "target_labels_diagnostic_only": 1,
        }
        rows.append(row)
    rows.sort(key=lambda x: (int(x["seed"]), int(x["target"]), int(x["level"]), x["regime"],
                             int(x["candidate_order"])))
    return rows


def context():
    tables = load_tables()
    registry = build_candidate_registry(tables)
    by_traj = {}
    for r in registry:
        by_traj.setdefault(r["trajectory_id"], []).append(r)
    by_candidate = {(r["seed"], r["target"], r["level"], r["regime"], r["candidate_order"]): r
                    for r in registry}
    by_pair = {
        "c37_exact": {r["pair_id"]: r for r in tables["c37"]["exact"]},
        "c38_gauge": {r["pair_id"]: r for r in tables["c38"]["gauge"]},
    }
    return {"tables": tables, "registry": registry, "by_traj": by_traj,
            "by_candidate": by_candidate, "by_pair": by_pair}
