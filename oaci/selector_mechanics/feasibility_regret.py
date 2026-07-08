"""C36 feasibility-regret decomposition."""
from __future__ import annotations

import numpy as np


def _rate(rows, key):
    return float(np.mean([int(bool(r.get(key))) for r in rows])) if rows else None


def decompose(pair_rows):
    rows = []
    for r in pair_rows:
        risk_gate = int(bool(int(r["selected_feasible"])) and not bool(int(r["better_feasible"])))
        leakage_conflict = int(r["selection_leakage_point_prefers"] == "selected" and not risk_gate)
        source_endpoint = int(r["source_endpoint_majority_prefers"] == "selected")
        inversion = int(r["selection_leakage_point_prefers"] == "selected" and
                        r["audit_leakage_point_prefers"] in ("better", "flat"))
        trace_unavailable = int(not bool(r["actual_selector_score_delta_available"]))
        out = {
            "pair_id": r["pair_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "risk_gate_regret": risk_gate,
            "leakage_objective_regret": leakage_conflict,
            "source_endpoint_regret": source_endpoint,
            "selection_audit_inversion": inversion,
            "tie_break_regret": 0,
            "trace_unavailable": trace_unavailable,
            "selection_leakage_point_prefers": r["selection_leakage_point_prefers"],
            "audit_leakage_point_prefers": r["audit_leakage_point_prefers"],
            "source_endpoint_majority_prefers": r["source_endpoint_majority_prefers"],
            "actual_selector_relation": r["actual_selector_relation"],
        }
        rows.append(out)
    summary = {
        "n_pairs": len(rows),
        "risk_gate_regret_fraction": _rate(rows, "risk_gate_regret"),
        "leakage_objective_regret_fraction": _rate(rows, "leakage_objective_regret"),
        "source_endpoint_regret_fraction": _rate(rows, "source_endpoint_regret"),
        "selection_audit_inversion_fraction": _rate(rows, "selection_audit_inversion"),
        "tie_break_regret_fraction": _rate(rows, "tie_break_regret"),
        "trace_unavailable_fraction": _rate(rows, "trace_unavailable"),
    }
    leakage_rows = [{
        "pair_id": r["pair_id"],
        "seed": r["seed"],
        "target": r["target"],
        "level": r["level"],
        "regime": r["regime"],
        "selection_leakage_point_delta_better_minus_selected": r["selection_leakage_point_delta_better_minus_selected"],
        "selection_leakage_point_prefers": r["selection_leakage_point_prefers"],
        "audit_leakage_point_delta_better_minus_selected": r["audit_leakage_point_delta_better_minus_selected"],
        "audit_leakage_point_prefers": r["audit_leakage_point_prefers"],
        "actual_selector_ucl_delta_available": r["actual_selector_score_delta_available"],
    } for r in pair_rows]
    return {"rows": rows, "summary": summary, "leakage_rows": leakage_rows}

