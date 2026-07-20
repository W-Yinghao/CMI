"""Frozen C43 source-objective registry."""
from __future__ import annotations

from . import artifact_loader as al


OBJECTIVES = (
    ("selection_leakage_point", "leakage", "lower", 1, 1, "C41 selection leakage point"),
    ("audit_leakage_point", "leakage", "lower", 1, 0, "source-audit leakage point"),
    ("R_src", "source_risk", "lower", 1, 1, "source risk"),
    ("train_surrogate", "source_risk", "lower", 1, 0, "training surrogate"),
    ("balanced_err", "source_risk", "lower", 1, 0, "source balanced error"),
    ("source_guard_worst_bacc", "source_endpoint", "higher", 0, 0, "source guard bAcc"),
    ("source_guard_worst_nll", "source_endpoint", "lower", 1, 0, "source guard NLL"),
    ("source_guard_worst_ece", "source_endpoint", "lower", 1, 0, "source guard ECE"),
    ("source_audit_worst_bacc", "source_endpoint", "higher", 0, 0, "source audit bAcc"),
    ("source_audit_worst_nll", "source_endpoint", "lower", 1, 0, "source audit NLL"),
    ("source_audit_worst_ece", "source_endpoint", "lower", 1, 0, "source audit ECE"),
    ("c19_robust_core_score", "source_rank", "higher", 0, 0, "frozen C19 robust-core score"),
    ("source_rank_score", "source_rank", "higher", 1, 1, "C30 source-rank interpretation of frozen score"),
    ("feat__source_guard_entropy", "source_calibration_softness", "lower", 0, 0, "guard entropy"),
    ("feat__source_guard_confidence", "source_calibration_softness", "higher", 0, 0, "guard confidence"),
    ("feat__source_guard_margin", "source_calibration_softness", "higher", 0, 0, "guard margin"),
    ("feat__source_guard_logit_norm", "source_calibration_softness", "lower", 0, 0, "guard logit norm"),
    ("feat__source_guard_conf_on_wrong", "source_calibration_softness", "lower", 0, 0, "guard wrong-confidence"),
)

SCALAR_OBJECTIVE_MAP = {
    "leakage": ("selection_leakage_point", "lower"),
    "source_rank": ("source_rank_score", "higher"),
    "source_risk": ("R_src", "lower"),
    "audit_leakage": ("audit_leakage_point", "lower"),
}


def registry(ctx):
    rows = []
    n = len(ctx["registry"])
    for field, family, orientation, used_pareto, used_scalar, note in OBJECTIVES:
        available = sum(1 for r in ctx["registry"] if field in r and al.finite(r.get(field)))
        rows.append({
            "objective": field,
            "family": family,
            "orientation": orientation,
            "n_candidate_rows": n,
            "n_available": available,
            "availability_fraction": available / n if n else None,
            "used_for_source_pareto": used_pareto,
            "used_for_scalarization_grid": used_scalar,
            "target_field": 0,
            "proxy_used": 0,
            "note": note,
        })
    summary = {r["objective"]: r for r in rows}
    return {"rows": rows, "summary": summary}


def oriented_value(row, field, orientation):
    v = float(row[field])
    return -v if orientation == "lower" else v


def scalar_objective_specs():
    return {name: {"field": field, "orientation": orientation}
            for name, (field, orientation) in SCALAR_OBJECTIVE_MAP.items()}
