"""C37 taxonomy."""
from __future__ import annotations

from . import schema


def classify(ordering_summary, plateau_summary, reconcile_summary, source_pareto_summary, p0_summary, recovery_summary):
    cases = []
    ev = {}
    if not p0_summary["p0_pass"] or not recovery_summary.get("all_recovered"):
        cases.append(schema.T7)
    ev[schema.T7] = f"p0_pass={p0_summary['p0_pass']}; all_recovered={recovery_summary.get('all_recovered')}"

    if recovery_summary.get("all_recovered"):
        if ordering_summary["ucl_prefers_selected_fraction"] >= 0.50:
            cases.append(schema.T1)
        if ordering_summary["ucl_prefers_better_fraction"] >= 0.25:
            cases.append(schema.T2)
        if ordering_summary["point_ucl_disagreement_fraction"] >= 0.25:
            cases.append(schema.T3)
        if plateau_summary["ucl_plateau_fraction"] >= 0.25:
            cases.append(schema.T4)
        if reconcile_summary["selection_audit_inversion_exact_rate"] >= 0.25:
            cases.append(schema.T5)
        if source_pareto_summary["source_pareto_conflict_fraction"] >= 0.50:
            cases.append(schema.T6)
        if schema.T1 in cases and plateau_summary["ucl_plateau_fraction"] < 0.25:
            cases.append(schema.T8)
        if schema.T2 in cases and schema.T8 not in cases:
            cases.append(schema.T9)
    ev.update({
        schema.T1: f"ucl_prefers_selected_fraction={ordering_summary.get('ucl_prefers_selected_fraction')}",
        schema.T2: f"ucl_prefers_better_fraction={ordering_summary.get('ucl_prefers_better_fraction')}",
        schema.T3: f"point_ucl_disagreement_fraction={ordering_summary.get('point_ucl_disagreement_fraction')}",
        schema.T4: f"ucl_plateau_fraction={plateau_summary.get('ucl_plateau_fraction')}",
        schema.T5: f"selection_audit_inversion_exact_rate={reconcile_summary.get('selection_audit_inversion_exact_rate')}",
        schema.T6: f"source_pareto_conflict_fraction={source_pareto_summary.get('source_pareto_conflict_fraction')}",
        schema.T8: "exact UCL must clearly favor selected despite robust target utility",
        schema.T9: "exact UCL does not support selected-favoring misdirection",
    })
    return {"cases": cases, "case_rows": [{"case": c, "established": int(c in cases), "evidence": ev.get(c, "")}
                                         for c in schema.ALL_CASES], "evidence": ev}

