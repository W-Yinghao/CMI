"""Read-only C45 loader over inherited source-objective artifacts."""
from __future__ import annotations

import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..source_scalarization import artifact_loader as c43_loader
from ..source_scalarization import objective_registry as c43_objectives
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


def finite_quantile(values, q):
    vals = [float(v) for v in values if finite(v)]
    return float(np.quantile(vals, q)) if vals else None


def entropy01(values):
    vals = [int(float(v)) for v in values if finite(v)]
    if not vals:
        return None
    p = float(np.mean(vals))
    if p <= 1e-12 or p >= 1.0 - 1e-12:
        return 0.0
    return float(-(p * math.log2(p) + (1.0 - p) * math.log2(1.0 - p)))


def row_key(row):
    return (str(row["seed"]), str(row["target"]), str(row["level"]), row["regime"], str(row["candidate_order"]))


def trajectory_key(row):
    return row["trajectory_id"]


def _target_joint_margin(row):
    acc = as_float(row.get("target_bacc_delta"))
    nll = as_float(row.get("target_nll_delta"))
    ece = as_float(row.get("target_ece_delta"))
    if not all(finite(v) for v in (acc, nll, ece)):
        return math.nan
    return float(min(acc - schema.PRIMARY_MARGIN, max(nll, ece) - schema.PRIMARY_MARGIN))


def inherited_source_registry(ctx):
    rows = []
    for r in c43_objectives.registry(ctx)["rows"]:
        source_complete = (
            int(r["target_field"]) == 0 and
            int(r["proxy_used"]) == 0 and
            int(r["n_available"]) == int(r["n_candidate_rows"])
        )
        out = dict(r)
        out["used_for_c45_primary_distance"] = int(source_complete)
        out["source_only_objective"] = int(source_complete)
        rows.append(out)
    return rows


def source_specs(ctx, families=None):
    famset = set(families or [])
    specs = []
    for r in inherited_source_registry(ctx):
        if not int(r["used_for_c45_primary_distance"]):
            continue
        if famset and r["family"] not in famset:
            continue
        specs.append({"objective": r["objective"], "family": r["family"], "orientation": r["orientation"]})
    return specs


def oriented_value(row, spec):
    v = as_float(row.get(spec["objective"]))
    return -v if spec["orientation"] == "lower" else v


def _attach_c45_fields(rows):
    out = []
    for i, r in enumerate(rows):
        nr = dict(r)
        nr["source_idx"] = i
        nr["candidate_order_int"] = as_int(nr.get("candidate_order"))
        nr["target_joint_margin_raw"] = _target_joint_margin(nr)
        nr["target_labels_diagnostic_only"] = 1
        out.append(nr)
    return out


def context():
    ctx = c43_loader.context()
    registry = _attach_c45_fields(ctx["registry"])
    by_traj = defaultdict(list)
    by_target = defaultdict(list)
    by_regime = defaultdict(list)
    for r in registry:
        by_traj[trajectory_key(r)].append(r)
        by_target[str(r["target"])].append(r)
        by_regime[r["regime"]].append(r)
    for d in (by_traj, by_target, by_regime):
        for k in d:
            d[k] = sorted(d[k], key=lambda x: x["source_idx"])
    ctx.update({
        "registry": registry,
        "by_traj": dict(by_traj),
        "by_target": dict(by_target),
        "by_regime": dict(by_regime),
        "c45_source_registry": inherited_source_registry(ctx),
        "tables": {
            **ctx["tables"],
            "c44_summary": read_json("oaci/reports/C44_SOURCE_PARETO_DEGENERACY_AUDIT.json"),
            "c44_geometry": read_csv(os.path.join(schema.C44_TABLE_DIR, "source_objective_geometry_summary.csv")),
        },
    })
    return ctx
