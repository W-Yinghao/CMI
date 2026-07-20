"""Conditional target variance and label entropy for C46."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def _metrics(rows):
    return {
        "n_rows": len(rows),
        "target_utility_variance": al.finite_var([r["target_utility_score"] for r in rows]),
        "joint_good_entropy": al.entropy01([r["primary_joint_good"] for r in rows]),
        "pareto_good_entropy": al.entropy01([r["pareto_good"] for r in rows]),
        "preference_robust_entropy": al.entropy01([r["preference_robust_better_candidate"] for r in rows]),
        "target_gauge_variance": al.finite_var([r["target_joint_margin_raw"] for r in rows]),
        "endpoint_vector_trace_variance": al.endpoint_z_trace_variance(rows),
        "target_utility_range": al.finite_range([r["target_utility_score"] for r in rows]),
    }


def _weighted_mean(group_metrics, key):
    num = den = 0.0
    for g in group_metrics:
        if al.finite(g.get(key)) and g["n_rows"]:
            num += float(g[key]) * g["n_rows"]
            den += g["n_rows"]
    return num / den if den else None


def audit(ctx):
    global_metrics = _metrics(ctx["registry"])
    rows = []
    group_rows = []
    for grouping in schema.VARIANCE_GROUPINGS:
        groups = al.group_rows(ctx, grouping)
        metrics = []
        for gid, rs in sorted(groups.items()):
            m = {"grouping": grouping, "group_id": gid, **_metrics(rs), "target_labels_diagnostic_only": 1}
            metrics.append(m)
            group_rows.append(m)
        row = {
            "grouping": grouping,
            "n_groups": len(metrics),
            "mean_group_size": al.finite_mean([m["n_rows"] for m in metrics]),
            "weighted_target_utility_variance": _weighted_mean(metrics, "target_utility_variance"),
            "weighted_joint_good_entropy": _weighted_mean(metrics, "joint_good_entropy"),
            "weighted_pareto_good_entropy": _weighted_mean(metrics, "pareto_good_entropy"),
            "weighted_preference_robust_entropy": _weighted_mean(metrics, "preference_robust_entropy"),
            "weighted_target_gauge_variance": _weighted_mean(metrics, "target_gauge_variance"),
            "weighted_endpoint_vector_trace_variance": _weighted_mean(metrics, "endpoint_vector_trace_variance"),
            "global_target_utility_variance": global_metrics["target_utility_variance"],
            "global_joint_good_entropy": global_metrics["joint_good_entropy"],
            "global_target_gauge_variance": global_metrics["target_gauge_variance"],
            "global_endpoint_vector_trace_variance": global_metrics["endpoint_vector_trace_variance"],
            "target_labels_diagnostic_only": 1,
        }
        row["target_utility_variance_over_global"] = (
            row["weighted_target_utility_variance"] / global_metrics["target_utility_variance"]
            if global_metrics["target_utility_variance"] else None)
        row["joint_entropy_over_global"] = (
            row["weighted_joint_good_entropy"] / global_metrics["joint_good_entropy"]
            if global_metrics["joint_good_entropy"] else None)
        row["target_gauge_variance_over_global"] = (
            row["weighted_target_gauge_variance"] / global_metrics["target_gauge_variance"]
            if global_metrics["target_gauge_variance"] else None)
        row["endpoint_trace_variance_over_global"] = (
            row["weighted_endpoint_vector_trace_variance"] / global_metrics["endpoint_vector_trace_variance"]
            if global_metrics["endpoint_vector_trace_variance"] else None)
        rows.append(row)
    return {"rows": rows, "group_rows": group_rows, "summary": {r["grouping"]: r for r in rows}}
