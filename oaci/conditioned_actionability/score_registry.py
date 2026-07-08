"""Frozen C47 source score registry."""
from __future__ import annotations

import os

from . import artifact_loader as al
from . import schema


def _c43_best_scalarization():
    rows = al.read_csv(os.path.join(schema.C43_TABLE_DIR, "best_hindsight_scalarization_ceiling.csv"))
    best = rows[0]
    grid = {
        r["scalarization_id"]: r
        for r in al.read_csv(os.path.join(schema.C43_TABLE_DIR, "scalarization_grid_registry.csv"))
    }
    return best["scalarization_id"], grid[best["scalarization_id"]]


def registry(ctx):
    best_id, best = _c43_best_scalarization()
    rows = [
        {
            "score": "selection_leakage",
            "family": "leakage",
            "field": "selection_leakage_point",
            "orientation": "lower",
            "source_only": 1,
            "hindsight_diagnostic_only": 0,
            "target_label_used": 0,
            "diagnostic_ceiling": 0,
            "note": "C41 selection leakage point",
        },
        {
            "score": "R_src",
            "family": "source_risk",
            "field": "R_src",
            "orientation": "lower",
            "source_only": 1,
            "hindsight_diagnostic_only": 0,
            "target_label_used": 0,
            "diagnostic_ceiling": 0,
            "note": "source risk",
        },
        {
            "score": "C30_source_rank",
            "family": "source_rank",
            "field": "source_rank_score",
            "orientation": "higher",
            "source_only": 1,
            "hindsight_diagnostic_only": 0,
            "target_label_used": 0,
            "diagnostic_ceiling": 0,
            "note": "C30 source-rank score",
        },
        {
            "score": "C19_robust_core",
            "family": "source_rank",
            "field": "c19_robust_core_score",
            "orientation": "higher",
            "source_only": 1,
            "hindsight_diagnostic_only": 0,
            "target_label_used": 0,
            "diagnostic_ceiling": 0,
            "note": "frozen C19 robust-core score",
        },
        {
            "score": "C43_best_hindsight_scalarization",
            "family": "fixed_hindsight_scalarization",
            "field": best_id,
            "orientation": "higher",
            "source_only": 1,
            "hindsight_diagnostic_only": 1,
            "target_label_used": 0,
            "diagnostic_ceiling": 0,
            "note": "C43 fixed best hindsight scalarization weights, diagnostic-only",
        },
        {
            "score": "target_utility_oracle_ceiling",
            "family": "target_grouped_ceiling",
            "field": "target_utility_score",
            "orientation": "higher",
            "source_only": 0,
            "hindsight_diagnostic_only": 1,
            "target_label_used": 1,
            "diagnostic_ceiling": 1,
            "note": "target-label diagnostic oracle ceiling",
        },
    ]
    n = len(ctx["registry"])
    for r in rows:
        if r["score"] == "C43_best_hindsight_scalarization":
            available = n
        else:
            available = sum(1 for c in ctx["registry"] if al.finite(c.get(r["field"])))
        r["n_candidate_rows"] = n
        r["n_available"] = available
        r["availability_fraction"] = available / n if n else None
    return {"rows": rows, "best_scalarization": best}


def _oriented(row, spec):
    v = al.as_float(row.get(spec["field"]))
    return -v if spec["orientation"] == "lower" else v


def _minmax(values):
    vals = [float(v) for v in values if al.finite(v)]
    if not vals:
        return None, None
    return min(vals), max(vals)


def _norm(v, mn, mx):
    if mn is None or mx is None:
        return 0.0
    return 0.5 if abs(mx - mn) <= 1e-12 else (float(v) - mn) / (mx - mn)


def c43_best_scores(rows, best):
    specs = {
        "leakage": ("selection_leakage_point", "lower", float(best["weight_leakage"])),
        "source_rank": ("source_rank_score", "higher", float(best["weight_source_rank"])),
        "source_risk": ("R_src", "lower", float(best["weight_source_risk"])),
        "audit_leakage": ("audit_leakage_point", "lower", float(best["weight_audit_leakage"])),
    }
    raw = {}
    stats = {}
    for name, (field, orient, weight) in specs.items():
        if weight <= 0:
            continue
        vals = []
        for r in rows:
            v = al.as_float(r.get(field))
            vals.append(-v if orient == "lower" else v)
        stats[name] = _minmax(vals)
        raw[name] = vals
    scores = {}
    for i, r in enumerate(rows):
        total = 0.0
        for name, (_, _, weight) in specs.items():
            if weight <= 0:
                continue
            mn, mx = stats[name]
            total += weight * _norm(raw[name][i], mn, mx)
        scores[id(r)] = float(total)
    return scores


def score_values(rows, spec, best_scalarization):
    if spec["score"] == "C43_best_hindsight_scalarization":
        return c43_best_scores(rows, best_scalarization)
    return {id(r): _oriented(r, spec) for r in rows}
