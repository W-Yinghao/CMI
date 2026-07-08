"""Read-only C47 loader."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict

import numpy as np

from ..source_nonidentifiability import artifact_loader as c45_loader

read_csv = c45_loader.read_csv
read_json = c45_loader.read_json
as_float = c45_loader.as_float
as_int = c45_loader.as_int
finite = c45_loader.finite
finite_mean = c45_loader.finite_mean
finite_median = c45_loader.finite_median
finite_quantile = c45_loader.finite_quantile
entropy01 = c45_loader.entropy01
source_specs = c45_loader.source_specs
oriented_value = c45_loader.oriented_value


def finite_max(values):
    vals = [float(v) for v in values if finite(v)]
    return float(max(vals)) if vals else None


def finite_min(values):
    vals = [float(v) for v in values if finite(v)]
    return float(min(vals)) if vals else None


def read_csv_dicts(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _add_group_maps(ctx):
    by_target_seed = defaultdict(list)
    by_target_level = defaultdict(list)
    for r in ctx["registry"]:
        by_target_seed[f"{r['target']}|{r['seed']}"].append(r)
        by_target_level[f"{r['target']}|{r['level']}"].append(r)
    for d in (by_target_seed, by_target_level):
        for k in d:
            d[k] = sorted(d[k], key=lambda x: int(x["source_idx"]))
    ctx["by_target_seed"] = dict(by_target_seed)
    ctx["by_target_level"] = dict(by_target_level)
    return ctx


def context():
    ctx = c45_loader.context()
    ctx = _add_group_maps(ctx)
    ctx["c46_summary"] = read_json("oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json")
    return ctx


def group_rows(ctx, scope):
    if scope == "global":
        return {"global": ctx["registry"]}
    if scope == "within_target":
        return ctx["by_target"]
    if scope == "within_trajectory":
        return ctx["by_traj"]
    if scope == "within_target_seed":
        return ctx["by_target_seed"]
    if scope == "within_target_level":
        return ctx["by_target_level"]
    if scope == "within_regime":
        return ctx["by_regime"]
    raise ValueError(scope)


def comb(n, k):
    if k < 0 or k > n:
        return 0
    return math.comb(int(n), int(k))
