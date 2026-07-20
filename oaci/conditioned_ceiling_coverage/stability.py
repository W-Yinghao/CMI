"""C49 cross-group stability for a fixed local-Bayes setup."""
from __future__ import annotations

import math

import numpy as np

from . import artifact_loader as al
from . import coverage_curve
from . import schema


def _metric_for_group(space, group, setup):
    labels = np.asarray([int(r[setup["label"]]) for r in group], dtype=int)
    base = float(np.mean(labels)) if len(labels) else math.nan
    spec = {
        "neighborhood": setup["neighborhood"],
        "neighborhood_kind": setup["neighborhood_kind"],
        "neighborhood_value": setup["neighborhood_value"],
    }
    purity, counts, _ = coverage_curve._purity_counts(space, group, spec, labels, base)
    covered = counts >= int(setup["min_neighbor_count"])
    hit, _, _, _ = coverage_curve._expected_hit_from_mask(purity, labels, covered)
    covered_base = float(np.mean(labels[covered])) if np.any(covered) else math.nan
    return {
        "n_candidates": len(group),
        "n_covered_candidates": int(np.sum(covered)),
        "coverage": float(np.mean(covered)) if len(covered) else math.nan,
        "empty_fraction": float(np.mean(counts == 0)) if len(counts) else math.nan,
        "mean_neighbor_count": float(np.mean(counts)) if len(counts) else math.nan,
        "covered_base_rate": covered_base,
        "local_bayes_top1_expected_hit": hit,
        "local_bayes_top1_enrichment": hit / covered_base
        if al.finite(hit) and al.finite(covered_base) and covered_base > 0 else math.nan,
        "local_bayes_gain_vs_covered_random": hit - covered_base
        if al.finite(hit) and al.finite(covered_base) else math.nan,
    }


def audit(ctx, spaces, setup):
    if not setup:
        return {"rows": [], "summary_rows": []}
    rows = []
    space = spaces[setup["source_space"]]
    for grouping in schema.STABILITY_GROUPINGS:
        for group_key, group in sorted(al.stability_groups(ctx, grouping).items()):
            m = _metric_for_group(space, group, setup)
            rows.append({
                "stability_grouping": grouping,
                "group_key": group_key,
                "source_space": setup["source_space"],
                "neighborhood": setup["neighborhood"],
                "neighborhood_kind": setup["neighborhood_kind"],
                "neighborhood_value": setup["neighborhood_value"],
                "min_neighbor_count": setup["min_neighbor_count"],
                "label": setup["label"],
                **m,
                "target_labels_diagnostic_only": 1,
            })
    summary = []
    for grouping in schema.STABILITY_GROUPINGS:
        rs = [r for r in rows if r["stability_grouping"] == grouping]
        hits = [float(r["local_bayes_top1_expected_hit"]) for r in rs
                if al.finite(r.get("local_bayes_top1_expected_hit"))]
        cov = [float(r["coverage"]) for r in rs if al.finite(r.get("coverage"))]
        enr = [float(r["local_bayes_top1_enrichment"]) for r in rs
               if al.finite(r.get("local_bayes_top1_enrichment"))]
        summary.append({
            "stability_grouping": grouping,
            "n_groups": len(rs),
            "n_evaluable_groups": len(hits),
            "mean_hit": float(np.mean(hits)) if hits else math.nan,
            "median_hit": float(np.median(hits)) if hits else math.nan,
            "min_hit": min(hits) if hits else math.nan,
            "max_hit": max(hits) if hits else math.nan,
            "std_hit": float(np.std(hits)) if hits else math.nan,
            "mean_coverage": float(np.mean(cov)) if cov else math.nan,
            "min_coverage": min(cov) if cov else math.nan,
            "mean_enrichment": float(np.mean(enr)) if enr else math.nan,
            "min_enrichment": min(enr) if enr else math.nan,
            "mean_neighbor_count": al.finite_mean([r["mean_neighbor_count"] for r in rs]),
            "worst_group_key_by_hit": min(
                [r for r in rs if al.finite(r.get("local_bayes_top1_expected_hit"))],
                key=lambda r: float(r["local_bayes_top1_expected_hit"]),
            )["group_key"] if hits else "",
            "target_labels_diagnostic_only": 1,
        })
    return {"rows": rows, "summary_rows": summary}
