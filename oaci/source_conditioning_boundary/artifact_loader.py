"""Read-only C46 loader over C45 source-space artifacts."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from ..source_nonidentifiability import artifact_loader as c45_loader
from . import schema

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


def finite_var(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.var(vals)) if vals else None


def finite_range(values):
    vals = [float(v) for v in values if finite(v)]
    return float(max(vals) - min(vals)) if vals else None


def endpoint_z_norm(row):
    vals = [as_float(row.get("target_bacc_z")), as_float(row.get("target_nll_z")),
            as_float(row.get("target_ece_z"))]
    return float(np.linalg.norm(vals)) if all(finite(v) for v in vals) else math.nan


def endpoint_z_trace_variance(rows):
    mat = []
    for r in rows:
        vals = [as_float(r.get("target_bacc_z")), as_float(r.get("target_nll_z")),
                as_float(r.get("target_ece_z"))]
        if all(finite(v) for v in vals):
            mat.append(vals)
    if not mat:
        return None
    arr = np.asarray(mat, dtype=float)
    return float(np.var(arr[:, 0]) + np.var(arr[:, 1]) + np.var(arr[:, 2]))


def context():
    ctx = c45_loader.context()
    by_seed = defaultdict(list)
    by_level = defaultdict(list)
    by_target_regime = defaultdict(list)
    for r in ctx["registry"]:
        r["endpoint_z_norm"] = endpoint_z_norm(r)
        by_seed[str(r["seed"])].append(r)
        by_level[str(r["level"])].append(r)
        by_target_regime[f"{r['target']}|{r['regime']}"].append(r)
    for d in (by_seed, by_level, by_target_regime):
        for k in d:
            d[k] = sorted(d[k], key=lambda x: int(x["source_idx"]))
    ctx.update({
        "by_seed": dict(by_seed),
        "by_level": dict(by_level),
        "by_target_regime": dict(by_target_regime),
        "c45_summary": read_json("oaci/reports/C45_SOURCE_EQUIVALENCE_NONIDENTIFIABILITY.json"),
    })
    return ctx


def group_rows(ctx, grouping):
    if grouping == "global":
        return {"global": ctx["registry"]}
    if grouping == "target":
        return ctx["by_target"]
    if grouping == "trajectory":
        return ctx["by_traj"]
    if grouping == "seed":
        return ctx["by_seed"]
    if grouping == "level":
        return ctx["by_level"]
    if grouping == "regime":
        return ctx["by_regime"]
    if grouping == "target_regime":
        return ctx["by_target_regime"]
    raise ValueError(grouping)
