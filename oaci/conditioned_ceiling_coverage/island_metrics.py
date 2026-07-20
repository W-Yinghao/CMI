"""Shared local-island morphology metrics."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import audit_utils as au


def local_bayes_hit(rows):
    covered = [r for r in rows if int(r["covered"])]
    if not covered:
        return math.nan, math.nan, 0
    max_p = max(float(r["neighbor_positive_rate"]) for r in covered if al.finite(r["neighbor_positive_rate"]))
    tied = [r for r in covered if abs(float(r["neighbor_positive_rate"]) - max_p) <= 1e-12]
    return float(np.mean([int(r["query_positive_label"]) for r in tied])), max_p, len(tied)


def group_fragmentation(island_rows, group_types, coverage_gate, hit_gate, enrichment_gate):
    out = []
    for group_type in group_types:
        buckets = defaultdict(list)
        for r in island_rows:
            buckets[au.row_group_key(r, group_type)].append(r)
        for group_key, rows in sorted(buckets.items()):
            n = len(rows)
            n_covered = sum(int(r["covered"]) for r in rows)
            coverage = n_covered / n if n else math.nan
            labels = [int(r["query_positive_label"]) for r in rows]
            base = float(np.mean(labels)) if labels else math.nan
            hit, max_p, tie_count = local_bayes_hit(rows)
            lift = hit - base if al.finite(hit) and al.finite(base) else math.nan
            enrich = au.enrichment(hit, base)
            neighbor_counts = [int(r["neighbors_n"]) for r in rows]
            out.append({
                "group_type": group_type,
                "group_key": group_key,
                "n_queries": n,
                "n_covered": n_covered,
                "coverage": coverage,
                "hit_rate_if_covered": hit,
                "base_rate": base,
                "absolute_lift": lift,
                "enrichment": enrich,
                "max_neighbor_positive_rate": max_p,
                "local_bayes_tie_count": tie_count,
                "mean_neighbor_count": au.finite_mean(neighbor_counts),
                "median_neighbor_count": au.finite_median(neighbor_counts),
                "empty_fraction": float(np.mean([int(r["neighbors_n"]) == 0 for r in rows])) if rows else math.nan,
                "min_neighbor_count": min(neighbor_counts) if neighbor_counts else math.nan,
                "max_neighbor_count": max(neighbor_counts) if neighbor_counts else math.nan,
                "actionability_pass": int(
                    al.finite(coverage) and coverage >= coverage_gate and
                    al.finite(hit) and hit >= hit_gate and
                    al.finite(enrich) and enrich >= enrichment_gate
                ),
                "target_labels_diagnostic_only": 1,
            })
    return out
