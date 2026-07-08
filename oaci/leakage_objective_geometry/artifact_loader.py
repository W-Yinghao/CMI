"""Read-only C38 loaders over C37/C36/C35/C34/C27/C29 artifacts."""
from __future__ import annotations

import csv
import json
import math
import os

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


def pref_from_delta(delta, eps, *, positive_prefers="selected", negative_prefers="better"):
    if not finite(delta):
        return "unavailable"
    delta = float(delta)
    if delta > eps:
        return positive_prefers
    if delta < -eps:
        return negative_prefers
    return "flat"


def pair_key(seed, target, level, selected_order, better_order):
    return "|".join(map(str, (seed, target, level, selected_order, better_order)))


def load_tables():
    c37 = {
        "exact": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_vs_better_exact_ucl.csv")),
        "p0": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_ucl_identity_gate.csv")),
        "better": read_csv(os.path.join(schema.C37_TABLE_DIR, "better_candidate_ucl_recovery.csv")),
        "manifest": read_csv(os.path.join(schema.C37_TABLE_DIR, "selector_trace_recovery_manifest.csv")),
        "source_pareto_after": read_csv(os.path.join(schema.C37_TABLE_DIR, "source_pareto_after_ucl_recovery.csv")),
    }
    c36 = {
        "trace": read_csv(os.path.join(schema.C36_TABLE_DIR, "selected_vs_better_selector_trace.csv")),
        "inversion": read_csv(os.path.join(schema.C36_TABLE_DIR, "selection_audit_inversion.csv")),
        "source_pareto": read_csv(os.path.join(schema.C36_TABLE_DIR, "source_pareto_status.csv")),
        "availability": read_csv(os.path.join(schema.C36_TABLE_DIR, "selector_trace_availability.csv")),
    }
    c35 = {
        "preference_robust": read_csv(os.path.join(schema.C35_TABLE_DIR, "preference_robust_case_audit.csv")),
        "endpoint_vectors": read_csv(os.path.join(schema.C35_TABLE_DIR, "endpoint_vector_registry.csv")),
    }
    c34 = {
        "pairs": read_csv(os.path.join(schema.C34_TABLE_DIR, "selected_vs_continuous_better_pairs.csv")),
        "source_components": read_csv(os.path.join(schema.C34_TABLE_DIR, "source_objective_component_conflict.csv")),
    }
    c27 = {
        "factor_registry": read_csv(os.path.join(schema.C27_TABLE_DIR, "logit_factor_registry.csv")),
        "class_confidence": read_csv(os.path.join(schema.C27_TABLE_DIR,
                                                  "class_conditioned_confidence_features.csv")),
    }
    c29 = {
        "rep_availability": read_csv(os.path.join(schema.C29_TABLE_DIR, "rep_head_artifact_availability.csv")),
        "target_rep_geometry": read_csv(os.path.join(schema.C29_TABLE_DIR, "target_representation_geometry.csv")),
    }
    return {"c37": c37, "c36": c36, "c35": c35, "c34": c34, "c27": c27, "c29": c29}


def context():
    tables = load_tables()
    by_pair = {
        "c36_trace": {r["pair_id"]: r for r in tables["c36"]["trace"]},
        "c36_inversion": {r["pair_id"]: r for r in tables["c36"]["inversion"]},
        "c36_source_pareto": {r["pair_id"]: r for r in tables["c36"]["source_pareto"]},
        "c37_source_pareto_after": {r["pair_id"]: r for r in tables["c37"]["source_pareto_after"]},
        "c35_preference": {r["pair_id"]: r for r in tables["c35"]["preference_robust"]},
        "c35_endpoint": {r["pair_id"]: r for r in tables["c35"]["endpoint_vectors"]
                         if r["comparison"] == schema.ROBUST_COMPARISON},
        "c34_pair": {
            "|".join([r["seed"], r["target"], r["level"], r["regime"], r["comparison"],
                      r["selected_order"], r["candidate_order"]]): r
            for r in tables["c34"]["pairs"]
        },
    }
    return {"tables": tables, "by_pair": by_pair}


def c34_for_exact(row, ctx):
    return ctx["by_pair"]["c34_pair"].get(row["pair_id"], {})
