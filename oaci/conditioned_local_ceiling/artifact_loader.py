"""Read-only C48 loader over C47/C45 artifacts."""
from __future__ import annotations

import csv
import json
import math

from ..conditioned_actionability import artifact_loader as c47_loader
from . import schema

read_csv = c47_loader.read_csv
read_json = c47_loader.read_json
as_float = c47_loader.as_float
as_int = c47_loader.as_int
finite = c47_loader.finite
finite_mean = c47_loader.finite_mean
finite_median = c47_loader.finite_median
finite_quantile = c47_loader.finite_quantile
entropy01 = c47_loader.entropy01
source_specs = c47_loader.source_specs
oriented_value = c47_loader.oriented_value
group_rows = c47_loader.group_rows
comb = c47_loader.comb


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def context():
    ctx = c47_loader.context()
    ctx["c47_summary"] = read_json("oaci/reports/C47_CONDITIONED_SOURCE_SPACE_ACTIONABILITY.json")
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
