"""Shared utilities for conditioned coverage diagnostic audits."""
from __future__ import annotations

import csv
import math

import numpy as np

from . import artifact_loader as al
from . import schema


DEFAULT_NEGATION_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic")


def lock_config(milestone: str) -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"{milestone} requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def read_csv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, cols):
    def clean(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def as_float(x, default=math.nan):
    return al.as_float(x, default)


def finite_values(vals):
    return [float(v) for v in vals if al.finite(v)]


def finite_mean(vals):
    vals = finite_values(vals)
    return float(np.mean(vals)) if vals else math.nan


def finite_median(vals):
    vals = finite_values(vals)
    return float(np.median(vals)) if vals else math.nan


def finite_quantile(vals, q):
    vals = finite_values(vals)
    return float(np.quantile(vals, q)) if vals else math.nan


def enrichment(hit, base):
    return hit / base if al.finite(hit) and al.finite(base) and float(base) > 0 else math.nan


def query_id(row):
    return f"c50q_{int(row['source_idx']):04d}"


def row_group_key(row, group_type):
    if group_type == "target":
        return str(row["target"])
    if group_type == "trajectory":
        return row.get("trajectory_id", row.get("trajectory"))
    if group_type == "seed":
        return str(row["seed"])
    if group_type == "level":
        return str(row["level"])
    if group_type == "regime":
        return row["regime"]
    if group_type == "conditioned_key":
        return str(row["target"])
    raise ValueError(group_type)


def fmt3(x):
    return "n/a" if not al.finite(x) else f"{float(x):.3f}"


def guard_forbidden(text, substrings, *, negation_cues=DEFAULT_NEGATION_CUES, window=180, label="audit"):
    low = text.lower()
    for s in substrings:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - window):i] for cue in negation_cues):
                raise ValueError(f"forbidden affirmative {label} claim near: {s}")
            i += len(s)
