"""Read-only C49 loader over C48/C47/C45 artifacts."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

from ..conditioned_local_ceiling import artifact_loader as c48_loader
from . import schema

read_csv = c48_loader.read_csv
read_json = c48_loader.read_json
as_float = c48_loader.as_float
as_int = c48_loader.as_int
finite = c48_loader.finite
finite_mean = c48_loader.finite_mean
finite_median = c48_loader.finite_median
finite_quantile = c48_loader.finite_quantile
entropy01 = c48_loader.entropy01
group_rows = c48_loader.group_rows


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _add_stability_maps(ctx):
    by_seed = defaultdict(list)
    by_level = defaultdict(list)
    for r in ctx["registry"]:
        by_seed[str(r["seed"])].append(r)
        by_level[str(r["level"])].append(r)
    for d in (by_seed, by_level):
        for k in d:
            d[k] = sorted(d[k], key=lambda x: int(x["source_idx"]))
    ctx["by_seed"] = dict(by_seed)
    ctx["by_level"] = dict(by_level)
    return ctx


def context():
    ctx = c48_loader.context()
    ctx = _add_stability_maps(ctx)
    ctx["c48_summary"] = read_json("oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.json")
    ctx["c48_best_rows"] = read_csv_dicts(f"{schema.C48_TABLE_DIR}/local_ceiling_best_by_scope.csv")
    ctx["c47_best_rows"] = read_csv_dicts(f"{schema.C47_TABLE_DIR}/group_actionability_best_by_scope.csv")
    return ctx


def c47_actual_top1(ctx, group_scope, label):
    rows = [
        r for r in ctx["c47_best_rows"]
        if r["best_kind"] == "best_strict_source" and r["group_scope"] == group_scope and
        r["label"] == label and int(float(r["top_k"])) == 1
    ]
    if not rows:
        return math.nan
    return as_float(rows[0].get("mean_any_hit"))


def stability_groups(ctx, grouping):
    if grouping == "target":
        return ctx["by_target"]
    if grouping == "seed":
        return ctx["by_seed"]
    if grouping == "level":
        return ctx["by_level"]
    if grouping == "trajectory":
        return ctx["by_traj"]
    if grouping == "regime":
        return ctx["by_regime"]
    raise ValueError(grouping)
