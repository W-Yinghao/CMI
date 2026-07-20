"""C36 selector-mechanics taxonomy."""
from __future__ import annotations

from . import schema


def _case_rows(cases, evidence):
    return [{"case": c, "established": int(c in cases), "evidence": evidence.get(c, "")}
            for c in schema.ALL_CASES]


def classify(feasibility, source_pareto, inversion, plateau, trace_availability):
    fs = feasibility["summary"]
    ps = source_pareto["summary"]
    inv = inversion["summary"]
    pl = plateau["summary"]
    cases = []
    evidence = {}

    if fs["risk_gate_regret_fraction"] and fs["risk_gate_regret_fraction"] >= 0.10:
        cases.append(schema.S1)
    evidence[schema.S1] = f"risk_gate_regret_fraction={fs['risk_gate_regret_fraction']}"

    if fs["leakage_objective_regret_fraction"] and fs["leakage_objective_regret_fraction"] >= 0.50:
        cases.append(schema.S2)
    evidence[schema.S2] = ("selection leakage point component prefers selected in "
                           f"{fs['leakage_objective_regret_fraction']} of robust pairs; actual UCL deltas unavailable")

    if fs["source_endpoint_regret_fraction"] and fs["source_endpoint_regret_fraction"] >= 0.50:
        cases.append(schema.S3)
    evidence[schema.S3] = f"source_endpoint_regret_fraction={fs['source_endpoint_regret_fraction']}"

    if inv["selection_to_audit_inversion_rate"] and inv["selection_to_audit_inversion_rate"] >= 0.25:
        cases.append(schema.S4)
    evidence[schema.S4] = f"selection_to_audit_inversion_rate={inv['selection_to_audit_inversion_rate']}"

    if ps["source_pareto_conflict_fraction"] and ps["source_pareto_conflict_fraction"] >= 0.50:
        cases.append(schema.S5)
    evidence[schema.S5] = f"source_pareto_conflict_fraction={ps['source_pareto_conflict_fraction']}"

    if pl["actual_selector_plateau_available"] and pl.get("actual_selector_plateau_fraction", 0) >= 0.25:
        cases.append(schema.S6)
    evidence[schema.S6] = "actual selector UCL plateau unavailable; point plateau is component-only"

    if trace_availability["pair_ucl"]["pairwise_selector_score_delta_available"]:
        active = pl.get("actual_selector_active_selected_margin_fraction", 0)
        if active >= 0.25:
            cases.append(schema.S7)
        evidence[schema.S7] = f"actual_selector_active_margin_fraction={active}"
    else:
        evidence[schema.S7] = "blocked: better candidate per-candidate selector UCL unavailable"

    if ps["better_source_dominates_fraction"] and ps["better_source_dominates_fraction"] >= 0.10:
        cases.append(schema.S8)
    evidence[schema.S8] = f"better_source_dominates_fraction={ps['better_source_dominates_fraction']}"

    if fs["trace_unavailable_fraction"] and fs["trace_unavailable_fraction"] > 0:
        cases.append(schema.S9)
    evidence[schema.S9] = f"trace_unavailable_fraction={fs['trace_unavailable_fraction']}"

    return {"cases": cases, "case_rows": _case_rows(cases, evidence), "evidence": evidence}

