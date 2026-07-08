"""C36 selected-vs-preference-robust-better selector trace rows."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def _pref_lower(delta, eps=schema.POINT_FLAT_EPS):
    if not al.finite(delta):
        return "unavailable"
    if float(delta) > eps:
        return "selected"
    if float(delta) < -eps:
        return "better"
    return "flat"


def _pref_oriented(delta, eps=schema.SOURCE_ENDPOINT_EPS):
    if not al.finite(delta):
        return "unavailable"
    if float(delta) > eps:
        return "better"
    if float(delta) < -eps:
        return "selected"
    return "flat"


def _endpoint_majority(selected, better, prefix):
    oriented = [
        al.as_float(better.get(f"{prefix}_worst_bacc")) - al.as_float(selected.get(f"{prefix}_worst_bacc")),
        al.as_float(selected.get(f"{prefix}_worst_nll")) - al.as_float(better.get(f"{prefix}_worst_nll")),
        al.as_float(selected.get(f"{prefix}_worst_ece")) - al.as_float(better.get(f"{prefix}_worst_ece")),
    ]
    finite_vals = [v for v in oriented if al.finite(v)]
    if not finite_vals:
        return "unavailable", oriented
    pos = sum(v > schema.SOURCE_ENDPOINT_EPS for v in finite_vals)
    neg = sum(v < -schema.SOURCE_ENDPOINT_EPS for v in finite_vals)
    if pos > neg:
        return "better", oriented
    if neg > pos:
        return "selected", oriented
    return "flat", oriented


def _source_endpoint_majority(selected, better):
    sg_pref, sg = _endpoint_majority(selected, better, "source_guard")
    sa_pref, sa = _endpoint_majority(selected, better, "source_audit")
    vals = sg + sa
    finite_vals = [v for v in vals if al.finite(v)]
    if not finite_vals:
        return "unavailable"
    pos = sum(v > schema.SOURCE_ENDPOINT_EPS for v in finite_vals)
    neg = sum(v < -schema.SOURCE_ENDPOINT_EPS for v in finite_vals)
    if pos > neg:
        return "better"
    if neg > pos:
        return "selected"
    return "flat"


def build_selected_pair_trace(pairs, trace):
    rows = []
    for p in pairs:
        s = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])]
        b = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])]
        sel_delta = al.as_float(b.get("selection_leakage_point")) - al.as_float(s.get("selection_leakage_point"))
        audit_delta = al.as_float(b.get("audit_leakage_point")) - al.as_float(s.get("audit_leakage_point"))
        sg_pref, sg_vec = _endpoint_majority(s, b, "source_guard")
        sa_pref, sa_vec = _endpoint_majority(s, b, "source_audit")
        row = {
            "pair_id": p["pair_id"],
            "seed": p["seed"],
            "target": p["target"],
            "level": p["level"],
            "regime": p["regime"],
            "selected_order": p["selected_order"],
            "better_order": p["candidate_order"],
            "selected_candidate_id": s["candidate_id"],
            "better_candidate_id": b["candidate_id"],
            "selected_is_actual_oaci": int(bool(s.get("selected_oaci"))),
            "better_is_actual_oaci": int(bool(b.get("selected_oaci"))),
            "selected_feasible": s.get("feasible"),
            "better_feasible": b.get("feasible"),
            "selected_R_src": s.get("R_src"),
            "better_R_src": b.get("R_src"),
            "R_src_delta_better_minus_selected": al.as_float(b.get("R_src")) - al.as_float(s.get("R_src")),
            "selected_risk_slack_to_tau": s.get("risk_slack_to_tau"),
            "better_risk_slack_to_tau": b.get("risk_slack_to_tau"),
            "risk_slack_delta_better_minus_selected": (
                al.as_float(b.get("risk_slack_to_tau")) - al.as_float(s.get("risk_slack_to_tau"))),
            "actual_selector_score_name": s.get("actual_selector_score_name"),
            "selected_actual_selector_ucl": s.get("actual_selector_score_ucl"),
            "better_actual_selector_ucl": b.get("actual_selector_score_ucl"),
            "actual_selector_score_delta_available": 0,
            "actual_selector_rank_delta_available": 0,
            "actual_selector_relation": "selected_chosen_by_actual_ucl_rule_better_ucl_unavailable",
            "selection_leakage_point_delta_better_minus_selected": sel_delta,
            "selection_leakage_point_prefers": _pref_lower(sel_delta),
            "audit_leakage_point_delta_better_minus_selected": audit_delta,
            "audit_leakage_point_prefers": _pref_lower(audit_delta),
            "source_guard_endpoint_prefers": sg_pref,
            "source_audit_endpoint_prefers": sa_pref,
            "source_endpoint_majority_prefers": _source_endpoint_majority(s, b),
            "source_guard_bacc_delta": sg_vec[0],
            "source_guard_nll_improve_delta": sg_vec[1],
            "source_guard_ece_improve_delta": sg_vec[2],
            "source_audit_bacc_delta": sa_vec[0],
            "source_audit_nll_improve_delta": sa_vec[1],
            "source_audit_ece_improve_delta": sa_vec[2],
            "target_bacc_delta": al.as_float(p.get("target_bacc_delta")),
            "target_nll_delta": al.as_float(p.get("target_nll_delta")),
            "target_ece_delta": al.as_float(p.get("target_ece_delta")),
            "fraction_weights_alt_beats_selected": al.as_float(p.get("fraction_weights_alt_beats_selected")),
            "mean_regret_over_simplex": al.as_float(p.get("mean_regret_over_simplex")),
            "utility_cone_category": p.get("utility_cone_category"),
            "pareto_status": p.get("pareto_status"),
            "target_endpoint_prefers": "better",
            "trace_complete_for_point_components": int(
                all(al.finite(x) for x in (sel_delta, audit_delta, al.as_float(s.get("R_src")),
                                           al.as_float(b.get("R_src"))))),
        }
        rows.append(row)
    return rows
