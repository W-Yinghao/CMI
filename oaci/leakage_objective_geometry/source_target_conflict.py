"""Source-rational / target-wrong leakage geometry decomposition."""
from __future__ import annotations

from statistics import mean

from . import artifact_loader as al


def _risk_pref(delta):
    d = al.as_float(delta)
    if not al.finite(d):
        return "unavailable"
    if d < -1e-12:
        return "better"
    if d > 1e-12:
        return "selected"
    return "flat"


def _case(row, trace):
    endpoint = trace["source_endpoint_majority_prefers"]
    audit = trace["audit_leakage_point_prefers"]
    risk = _risk_pref(trace["R_src_delta_better_minus_selected"])
    if endpoint == "selected":
        return "source_endpoint_conflict"
    if endpoint == "better" and audit == "better" and risk in ("better", "flat"):
        return "pure_selection_leakage_conflict"
    if endpoint == "better":
        return "leakage_endpoint_conflict"
    return "leakage_endpoint_flat_or_mixed"


def audit(ctx, ucl_rows):
    rows = []
    for ur in ucl_rows:
        tr = ctx["by_pair"]["c36_trace"][ur["pair_id"]]
        sp = ctx["by_pair"]["c37_source_pareto_after"][ur["pair_id"]]
        target_pref = ur["target_endpoint_prefers"]
        source_rational = int(
            ur["ucl_prefers"] == "selected" and target_pref == "better" and
            sp["source_pareto_status"] in ("source_pareto_incomparable", "selected_source_dominates_better"))
        case = _case(ur, tr)
        rows.append({
            "pair_id": ur["pair_id"],
            "seed": ur["seed"],
            "target": ur["target"],
            "level": ur["level"],
            "regime": ur["regime"],
            "selection_ucl_prefers": ur["ucl_prefers"],
            "selection_point_prefers": ur["point_prefers"],
            "source_audit_leakage_prefers": tr["audit_leakage_point_prefers"],
            "source_endpoint_majority_prefers": tr["source_endpoint_majority_prefers"],
            "source_audit_endpoint_prefers": tr["source_audit_endpoint_prefers"],
            "source_pareto_status": sp["source_pareto_status"],
            "source_pareto_conflict": sp["source_pareto_conflict"],
            "R_src_prefers": _risk_pref(tr["R_src_delta_better_minus_selected"]),
            "target_endpoint_prefers": target_pref,
            "source_rational_target_wrong": source_rational,
            "leakage_source_target_conflict_class": case,
        })
    n = len(rows)
    summary = {
        "n_pairs": n,
        "source_rational_target_wrong_count": sum(r["source_rational_target_wrong"] for r in rows),
        "source_rational_target_wrong_fraction": mean(
            [r["source_rational_target_wrong"] for r in rows]) if rows else None,
        "source_pareto_conflict_fraction": mean(
            [al.as_int(r["source_pareto_conflict"]) for r in rows]) if rows else None,
        "source_endpoint_conflict_count": sum(
            1 for r in rows if r["leakage_source_target_conflict_class"] == "source_endpoint_conflict"),
        "leakage_endpoint_conflict_count": sum(
            1 for r in rows if r["leakage_source_target_conflict_class"] == "leakage_endpoint_conflict"),
        "pure_selection_leakage_conflict_count": sum(
            1 for r in rows if r["leakage_source_target_conflict_class"] == "pure_selection_leakage_conflict"),
        "source_endpoint_majority_prefers_selected_count": sum(
            1 for r in rows if r["source_endpoint_majority_prefers"] == "selected"),
        "source_endpoint_majority_prefers_better_count": sum(
            1 for r in rows if r["source_endpoint_majority_prefers"] == "better"),
        "source_endpoint_majority_flat_count": sum(
            1 for r in rows if r["source_endpoint_majority_prefers"] == "flat"),
    }
    return {"rows": rows, "summary": summary}

