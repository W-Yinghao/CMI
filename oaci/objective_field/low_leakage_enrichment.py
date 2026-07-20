"""C41 low-leakage enrichment against trajectory-conditioned baselines."""
from __future__ import annotations

import math

from . import artifact_loader as al


SELECTIONS = (
    ("top1", 1),
    ("top3", 3),
    ("top5", 5),
    ("top_decile", 0.10),
    ("bottom_leakage_quartile", 0.25),
)
LABELS = ("primary_joint_good", "pareto_good", "preference_robust_better_candidate")


def _draw_count(n, spec):
    if isinstance(spec, int):
        return min(spec, n)
    return max(1, int(math.ceil(n * float(spec))))


def _hypergeom_upper_tail(N, K, n, k):
    if n <= 0:
        return 1.0
    denom = math.comb(N, n)
    hi = min(K, n)
    total = 0
    for x in range(k, hi + 1):
        if n - x <= N - K:
            total += math.comb(K, x) * math.comb(N - K, n - x)
    return total / denom if denom else 1.0


def audit(ctx):
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        ordered = sorted([r for r in rs if al.finite(r["selection_leakage_point"])],
                         key=lambda r: (float(r["selection_leakage_point"]), int(r["candidate_order"])))
        N = len(ordered)
        if N == 0:
            continue
        for sel_name, spec in SELECTIONS:
            n = _draw_count(N, spec)
            chosen = ordered[:n]
            for label in LABELS:
                K = sum(int(r[label]) for r in ordered)
                k = sum(int(r[label]) for r in chosen)
                baseline = K / N if N else 0.0
                hit = k / n if n else 0.0
                p = _hypergeom_upper_tail(N, K, n, k)
                rows.append({
                    "trajectory_id": tid,
                    "seed": seed,
                    "target": target,
                    "level": level,
                    "regime": regime,
                    "selection_rule": sel_name,
                    "label": label,
                    "n_candidates": N,
                    "n_selected_by_low_leakage": n,
                    "trajectory_label_count": K,
                    "selected_label_count": k,
                    "hit_rate": hit,
                    "trajectory_random_baseline": baseline,
                    "enrichment_ratio": hit / baseline if baseline > 0 else "",
                    "hypergeom_p_value": p,
                    "bonferroni_p_value": min(1.0, p * len(SELECTIONS) * len(LABELS)),
                    "target_labels_diagnostic_only": 1,
                })
    summary_rows = []
    for sel_name, _ in SELECTIONS:
        for label in LABELS:
            rs = [r for r in rows if r["selection_rule"] == sel_name and r["label"] == label]
            ratios = [al.as_float(r["enrichment_ratio"]) for r in rs]
            summary_rows.append({
                "selection_rule": sel_name,
                "label": label,
                "n_trajectories": len(rs),
                "mean_hit_rate": al.finite_mean([r["hit_rate"] for r in rs]),
                "mean_random_baseline": al.finite_mean([r["trajectory_random_baseline"] for r in rs]),
                "mean_enrichment_ratio": al.finite_mean(ratios),
                "median_enrichment_ratio": al.finite_median(ratios),
                "significant_enriched_trajectories_bonferroni": sum(
                    1 for r in rs if al.as_float(r["bonferroni_p_value"]) < 0.05),
            })
    summary = {(r["selection_rule"], r["label"]): r for r in summary_rows}
    return {"rows": rows, "summary_rows": summary_rows, "summary": summary}

