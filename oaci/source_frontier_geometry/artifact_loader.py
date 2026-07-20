"""Read-only C44 loader inherited from C43."""
from __future__ import annotations

import csv
import json
import math
import os

import numpy as np

from ..source_scalarization import artifact_loader as c43_loader
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


def context():
    ctx = c43_loader.context()
    ctx["tables"]["c43_summary"] = read_json("oaci/reports/C43_SOURCE_OBJECTIVE_SCALARIZATION_FRONTIER.json")
    ctx["tables"]["c43_objectives"] = read_csv(os.path.join(schema.C43_TABLE_DIR, "source_objective_registry.csv"))
    return ctx
