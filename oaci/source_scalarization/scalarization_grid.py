"""Frozen source-only scalarization grid."""
from __future__ import annotations

from . import schema


def _fmt(x):
    return f"{x:.1f}"


def simplex2(a, b):
    for i in range(11):
        wa = i / 10.0
        yield {a: wa, b: 1.0 - wa}


def simplex3(a, b, c):
    for i in range(11):
        for j in range(11 - i):
            k = 10 - i - j
            yield {a: i / 10.0, b: j / 10.0, c: k / 10.0}


def build_grid():
    rows = []
    for obj in ("leakage", "source_rank", "source_risk", "audit_leakage"):
        rows.append({
            "scalarization_id": f"single__{obj}",
            "grid_family": "single_objective",
            "weight_leakage": 1.0 if obj == "leakage" else 0.0,
            "weight_source_rank": 1.0 if obj == "source_rank" else 0.0,
            "weight_source_risk": 1.0 if obj == "source_risk" else 0.0,
            "weight_audit_leakage": 1.0 if obj == "audit_leakage" else 0.0,
            "grid_step": schema.SCALARIZATION_GRID_STEP,
            "source_only": 1,
            "hindsight_diagnostic_only": 1,
        })
    for a, b, fam in (("leakage", "source_rank", "leakage_source_rank"),
                      ("leakage", "source_risk", "leakage_source_risk"),
                      ("source_rank", "source_risk", "rank_source_risk")):
        for w in simplex2(a, b):
            rows.append({
                "scalarization_id": f"{fam}__{a}_{_fmt(w[a])}__{b}_{_fmt(w[b])}",
                "grid_family": fam,
                "weight_leakage": w.get("leakage", 0.0),
                "weight_source_rank": w.get("source_rank", 0.0),
                "weight_source_risk": w.get("source_risk", 0.0),
                "weight_audit_leakage": 0.0,
                "grid_step": schema.SCALARIZATION_GRID_STEP,
                "source_only": 1,
                "hindsight_diagnostic_only": 1,
            })
    for w in simplex3("leakage", "source_rank", "source_risk"):
        rows.append({
            "scalarization_id": (
                f"leakage_rank_risk__leakage_{_fmt(w['leakage'])}__rank_{_fmt(w['source_rank'])}"
                f"__risk_{_fmt(w['source_risk'])}"),
            "grid_family": "leakage_source_rank_source_risk",
            "weight_leakage": w["leakage"],
            "weight_source_rank": w["source_rank"],
            "weight_source_risk": w["source_risk"],
            "weight_audit_leakage": 0.0,
            "grid_step": schema.SCALARIZATION_GRID_STEP,
            "source_only": 1,
            "hindsight_diagnostic_only": 1,
        })
    return {"rows": rows, "summary": {"n_scalarizations": len(rows), "grid_step": schema.SCALARIZATION_GRID_STEP}}


def weights(row):
    return {
        "leakage": float(row["weight_leakage"]),
        "source_rank": float(row["weight_source_rank"]),
        "source_risk": float(row["weight_source_risk"]),
        "audit_leakage": float(row["weight_audit_leakage"]),
    }
