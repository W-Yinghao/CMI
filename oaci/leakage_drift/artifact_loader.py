"""Read-only C40 loaders over committed C39/C38/C37 artifacts."""
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


def pref_from_delta(delta, eps=schema.POINT_SIGN_EPS):
    if not finite(delta):
        return "unavailable"
    delta = float(delta)
    if delta > eps:
        return "selected"
    if delta < -eps:
        return "better"
    return "flat"


def load_tables():
    c39 = {
        "summary": read_json("oaci/reports/C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json"),
        "identity": read_csv(os.path.join(schema.C39_TABLE_DIR, "selected_atom_identity_gate.csv")),
        "availability": read_csv(os.path.join(schema.C39_TABLE_DIR, "atom_recovery_availability.csv")),
        "point_atoms": read_csv(os.path.join(schema.C39_TABLE_DIR, "selected_vs_better_point_atoms.csv")),
        "concentration": read_csv(os.path.join(schema.C39_TABLE_DIR, "atom_concentration_summary.csv")),
        "support": read_csv(os.path.join(schema.C39_TABLE_DIR, "support_cell_artifact_audit.csv")),
        "audit_stability": read_csv(os.path.join(schema.C39_TABLE_DIR, "selection_audit_atom_stability.csv")),
        "gauge": read_csv(os.path.join(schema.C39_TABLE_DIR, "atom_target_gauge_conflict.csv")),
        "bootstrap": read_csv(os.path.join(schema.C39_TABLE_DIR, "bootstrap_atom_diagnostics.csv")),
    }
    c38 = {
        "ucl": read_csv(os.path.join(schema.C38_TABLE_DIR, "ucl_point_width_decomposition.csv")),
    }
    c37 = {
        "exact": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_vs_better_exact_ucl.csv")),
    }
    return {"c39": c39, "c38": c38, "c37": c37}


def context():
    tables = load_tables()
    by_pair = {
        "c39_concentration": {r["pair_id"]: r for r in tables["c39"]["concentration"]},
        "c39_support": {r["pair_id"]: r for r in tables["c39"]["support"]},
        "c39_audit_stability": {r["pair_id"]: r for r in tables["c39"]["audit_stability"]},
        "c39_gauge": {r["pair_id"]: r for r in tables["c39"]["gauge"]},
        "c37_exact": {r["pair_id"]: r for r in tables["c37"]["exact"]},
        "c38_ucl": {r["pair_id"]: r for r in tables["c38"]["ucl"]},
    }
    return {"tables": tables, "by_pair": by_pair}


def selection_identity_rows(ctx):
    return [r for r in ctx["tables"]["c39"]["identity"] if r["split"] == "selection"]


def audit_identity_rows(ctx):
    return [r for r in ctx["tables"]["c39"]["identity"] if r["split"] == "source_audit"]


def identity_by_job(rows):
    return {r["job_key"]: r for r in rows}


def pair_candidate_keys(pair):
    split = "selection"
    selected_key = "|".join([pair["seed"], pair["target"], pair["level"],
                             pair["selected_order"], "selected", split])
    better_key = "|".join([pair["seed"], pair["target"], pair["level"],
                           pair["better_order"], "better", split])
    return selected_key, better_key
