"""Selection leakage vs source-audit leakage local inversion."""
from __future__ import annotations

from statistics import mean

from . import artifact_loader as al


def audit(ctx, ucl_rows):
    ucl = {r["pair_id"]: r for r in ucl_rows}
    rows = []
    for inv in ctx["tables"]["c36"]["inversion"]:
        ur = ucl[inv["pair_id"]]
        tr = ctx["by_pair"]["c36_trace"][inv["pair_id"]]
        rows.append({
            "pair_id": inv["pair_id"],
            "seed": inv["seed"],
            "target": inv["target"],
            "level": inv["level"],
            "regime": inv["regime"],
            "selection_ucl_prefers": ur["ucl_prefers"],
            "selection_point_prefers": ur["point_prefers"],
            "source_audit_leakage_prefers": inv["audit_leakage_prefers"],
            "source_audit_endpoint_prefers": inv["source_audit_endpoint_prefers"],
            "source_endpoint_majority_prefers": tr["source_endpoint_majority_prefers"],
            "target_endpoint_prefers": inv["target_endpoint_prefers"],
            "selection_point_delta_better_minus_selected": tr["selection_leakage_point_delta_better_minus_selected"],
            "audit_point_delta_better_minus_selected": tr["audit_leakage_point_delta_better_minus_selected"],
            "selection_to_audit_inversion": inv["selection_to_audit_inversion"],
            "audit_to_target_inversion": inv["audit_to_target_inversion"],
            "selection_ucl_to_audit_inversion": int(
                ur["ucl_prefers"] in ("selected", "better") and
                inv["audit_leakage_prefers"] in ("selected", "better") and
                ur["ucl_prefers"] != inv["audit_leakage_prefers"]),
            "local_leakage_target_conflict": inv["local_leakage_target_conflict"],
        })
    n = len(rows)
    summary = {
        "n_pairs": n,
        "selection_to_audit_inversion_rate": mean(
            [al.as_int(r["selection_to_audit_inversion"]) for r in rows]) if rows else None,
        "selection_ucl_to_audit_inversion_rate": mean(
            [al.as_int(r["selection_ucl_to_audit_inversion"]) for r in rows]) if rows else None,
        "audit_to_target_inversion_rate": mean(
            [al.as_int(r["audit_to_target_inversion"]) for r in rows]) if rows else None,
        "local_leakage_target_conflict_rate": mean(
            [al.as_int(r["local_leakage_target_conflict"]) for r in rows]) if rows else None,
        "audit_prefers_selected_count": sum(1 for r in rows if r["source_audit_leakage_prefers"] == "selected"),
        "audit_prefers_better_count": sum(1 for r in rows if r["source_audit_leakage_prefers"] == "better"),
    }
    return {"rows": rows, "summary": summary}

