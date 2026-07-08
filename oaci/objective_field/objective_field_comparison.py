"""Compare available objective fields against target utility."""
from __future__ import annotations

from . import artifact_loader as al


def _c30_value(ctx, metric):
    for r in ctx["tables"]["c30"]["rank_gauge"]:
        if r["axis"] == metric:
            return al.as_float(r["auc"])
    return None


def _c30_baseline(ctx, feature):
    for r in ctx["tables"]["c30"]["rank_baseline"]:
        if r["feature"] == feature:
            return al.as_float(r["within_target_auc"])
    return None


def compare(ctx, alignment):
    rows = []
    for field, s in sorted(alignment["summary"].items()):
        rows.append({
            "field": field,
            "scope": "candidate_level_c41",
            "n_trajectories": s["n_trajectories"],
            "target_utility_mean_auc": s["mean_pairwise_auc"],
            "target_utility_median_auc": s["median_pairwise_auc"],
            "rank_strength_abs": abs(float(s["mean_pairwise_auc"]) - 0.5),
            "candidate_level_available": 1,
            "non_source_only": 0,
            "proxy_used": 0,
        })
    c30_rank = _c30_value(ctx, "within_target_rank(score)")
    if c30_rank is not None:
        rows.append({
            "field": "C30_source_rank_score",
            "scope": "aggregate_only_c30",
            "n_trajectories": "",
            "target_utility_mean_auc": c30_rank,
            "target_utility_median_auc": "",
            "rank_strength_abs": abs(c30_rank - 0.5),
            "candidate_level_available": 0,
            "non_source_only": 0,
            "proxy_used": 0,
        })
    for feature in ("feat__selection_leakage_point", "feat__audit_leakage_point", "R_src"):
        auc = _c30_baseline(ctx, feature)
        if auc is not None:
            rows.append({
                "field": f"C30_aggregate_{feature}",
                "scope": "aggregate_only_c30",
                "n_trajectories": "",
                "target_utility_mean_auc": auc,
                "target_utility_median_auc": "",
                "rank_strength_abs": abs(auc - 0.5),
                "candidate_level_available": 0,
                "non_source_only": 0,
                "proxy_used": 0,
            })
    rows.sort(key=lambda r: -al.as_float(r["rank_strength_abs"], -1))
    best = rows[0] if rows else {}
    sel = next((r for r in rows if r["field"] == "selection_leakage_point"), {})
    audit = next((r for r in rows if r["field"] == "audit_leakage_point"), {})
    c30 = next((r for r in rows if r["field"] == "C30_source_rank_score"), {})
    summary = {
        "best_field": best.get("field"),
        "selection_leakage_auc": sel.get("target_utility_mean_auc"),
        "audit_leakage_auc": audit.get("target_utility_mean_auc"),
        "c30_source_rank_auc": c30.get("target_utility_mean_auc"),
        "c30_rank_better_than_selection_leakage": (
            al.as_float(c30.get("target_utility_mean_auc")) > al.as_float(sel.get("target_utility_mean_auc"))),
    }
    return {"rows": rows, "summary": summary}


def source_audit_vs_selection(alignment):
    sel = alignment["summary"].get("selection_leakage_point", {})
    aud = alignment["summary"].get("audit_leakage_point", {})
    rows = [{
        "metric": "mean_pairwise_auc",
        "selection_leakage": sel.get("mean_pairwise_auc"),
        "source_audit_leakage": aud.get("mean_pairwise_auc"),
        "audit_minus_selection": al.as_float(aud.get("mean_pairwise_auc")) - al.as_float(sel.get("mean_pairwise_auc")),
        "source_audit_substantially_better": 0,
    }, {
        "metric": "median_pairwise_auc",
        "selection_leakage": sel.get("median_pairwise_auc"),
        "source_audit_leakage": aud.get("median_pairwise_auc"),
        "audit_minus_selection": al.as_float(aud.get("median_pairwise_auc")) - al.as_float(sel.get("median_pairwise_auc")),
        "source_audit_substantially_better": 0,
    }]
    return {"rows": rows, "summary": {"audit_mean_auc_minus_selection": rows[0]["audit_minus_selection"],
                                       "source_audit_no_better": rows[0]["audit_minus_selection"] <= 0.02}}
