"""Read-only C39 loaders over C38/C37/C36/C35/C34 and Phase-A replay stores."""
from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass

import numpy as np

from ..selector_trace_recovery import artifact_loader as c37_loader
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


@dataclass(frozen=True)
class CandidateReplayJob:
    """Internal replay instruction. The model hash is never emitted to reports/tables."""

    job_key: str
    seed: str
    target: str
    level: str
    regime: str
    pair_key: str
    pair_id: str
    selected_order: str
    better_order: str
    candidate_order: str
    candidate_role: str
    candidate_id: str
    expected_point: float
    split: str
    model_hash: str


def load_tables():
    c37 = {
        "exact": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_vs_better_exact_ucl.csv")),
        "p0": read_csv(os.path.join(schema.C37_TABLE_DIR, "selected_ucl_identity_gate.csv")),
        "manifest": read_csv(os.path.join(schema.C37_TABLE_DIR, "selector_trace_recovery_manifest.csv")),
    }
    c36 = {
        "trace": read_csv(os.path.join(schema.C36_TABLE_DIR, "selected_vs_better_selector_trace.csv")),
        "inversion": read_csv(os.path.join(schema.C36_TABLE_DIR, "selection_audit_inversion.csv")),
    }
    c35 = {
        "endpoint": read_csv(os.path.join(schema.C35_TABLE_DIR, "endpoint_vector_registry.csv")),
    }
    c34 = {
        "pairs": read_csv(os.path.join(schema.C34_TABLE_DIR, "selected_vs_continuous_better_pairs.csv")),
    }
    c38 = {
        "ucl": read_csv(os.path.join(schema.C38_TABLE_DIR, "ucl_point_width_decomposition.csv")),
        "gauge": read_csv(os.path.join(schema.C38_TABLE_DIR, "leakage_vs_target_gauge_conflict.csv")),
    }
    return {"c37": c37, "c36": c36, "c35": c35, "c34": c34, "c38": c38}


def context():
    tables = load_tables()
    pairs = tables["c37"]["exact"]
    regimes = sorted({r["regime"] for r in pairs})
    trace = c37_loader.load_c10_trace(regimes)
    by_pair = {
        "c36_trace": {r["pair_id"]: r for r in tables["c36"]["trace"]},
        "c36_inversion": {r["pair_id"]: r for r in tables["c36"]["inversion"]},
        "c38_ucl": {r["pair_id"]: r for r in tables["c38"]["ucl"]},
        "c38_gauge": {r["pair_id"]: r for r in tables["c38"]["gauge"]},
        "c35_endpoint": {r["pair_id"]: r for r in tables["c35"]["endpoint"]
                         if r["comparison"] == schema.ROBUST_COMPARISON},
        "c34_pair": {
            "|".join([r["seed"], r["target"], r["level"], r["regime"], r["comparison"],
                      r["selected_order"], r["candidate_order"]]): r
            for r in tables["c34"]["pairs"]
        },
    }
    return {
        "tables": tables,
        "pairs": pairs,
        "trace": trace,
        "ctx_cache": c37_loader.ContextCache(trace),
        "by_pair": by_pair,
    }


def unique_pair_rows(pair_rows):
    """Collapse the S0/S2/S3 duplicate local pair rows to unique replay work units."""
    seen = {}
    for r in pair_rows:
        key = (r["seed"], r["target"], r["level"], r["selected_order"], r["better_order"])
        seen.setdefault(key, r)
    return [seen[k] for k in sorted(seen, key=lambda x: (int(x[0]), int(x[1]), int(x[2]),
                                                         int(x[3]), int(x[4])))]


def _job_key(row, role, split):
    order = row["selected_order"] if role == "selected" else row["better_order"]
    return "|".join([row["seed"], row["target"], row["level"], order, role, split])


def candidate_jobs(ctx, *, split="selection"):
    if split not in ("selection", "source_audit"):
        raise ValueError(f"unknown split {split}")
    out = []
    for r in unique_pair_rows(ctx["pairs"]):
        for role in ("selected", "better"):
            order = r["selected_order"] if role == "selected" else r["better_order"]
            point_col = "selected_point" if role == "selected" else "better_point"
            cand = ctx["trace"]["by_key"][(r["seed"], r["target"], r["level"], r["regime"], order)]
            expected_point = as_float(r[point_col]) if split == "selection" else math.nan
            out.append(CandidateReplayJob(
                job_key=_job_key(r, role, split),
                seed=r["seed"],
                target=r["target"],
                level=r["level"],
                regime=r["regime"],
                pair_key=r["pair_key"],
                pair_id=r["pair_id"],
                selected_order=r["selected_order"],
                better_order=r["better_order"],
                candidate_order=order,
                candidate_role=role,
                candidate_id=cand["candidate_id"],
                expected_point=expected_point,
                split=split,
                model_hash=cand["model_hash"],
            ))
    return out


def _feature_key_map(ctx, kind):
    ctx.load()
    attr = f"_c39_{kind.replace(':', '_')}_feature_keys"
    if not hasattr(ctx, attr):
        setattr(ctx, attr, {getattr(k, "model_hash", None): k for stored_kind, k in ctx.store._d
                            if stored_kind == kind})
    return getattr(ctx, attr)


def feature_by_hash(ctx, model_hash, *, split="selection"):
    kind = "feat:source_train" if split == "selection" else "feat:source_audit"
    keys = _feature_key_map(ctx, kind)
    key = keys.get(model_hash)
    if key is None:
        raise KeyError(f"{kind} feature missing for candidate")
    return ctx.store.lookup(kind, key)


def support_and_fold(ctx, *, split="selection"):
    ctx.load()
    if split == "selection":
        return ctx.support_graph, ctx.fold_plan
    audit = getattr(ctx.fold.fold_scope, "source_audit", None)
    if audit is None or getattr(audit, "status", None) != "estimable":
        raise ValueError("source-audit scope is not estimable")
    return audit.support_graph, audit.fold_plan


def unit_context(ctx, job: CandidateReplayJob):
    return ctx["ctx_cache"].get(job.seed, job.target, job.level, job.regime).load()


def c34_for_pair(pair_id, ctx):
    return ctx["by_pair"]["c34_pair"].get(pair_id, {})


def finite_mean(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.mean(vals)) if vals else None


def finite_sum(values):
    vals = [float(v) for v in values if finite(v)]
    return float(np.sum(vals)) if vals else 0.0
