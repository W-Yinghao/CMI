"""Family-reduced source-space ambiguity comparison for C45."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema, source_space


def _nearest_within_trajectory(ctx, space):
    rows = []
    for _, cs in sorted(ctx["by_traj"].items()):
        idx = [int(r["source_idx"]) for r in cs]
        mat = space["z"][idx, :]
        dist = np.sqrt(((mat[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2))
        np.fill_diagonal(dist, np.inf)
        for i, row in enumerate(cs):
            j = int(np.argmin(dist[i]))
            metrics = source_space.pair_metrics(row, cs[j], space, metric_distance=float(dist[i, j]))
            rows.append(metrics)
    return rows


def _random_baseline(ctx, space):
    rows = []
    for _, cs in sorted(ctx["by_traj"].items()):
        for i, row in enumerate(cs):
            j = (i + max(1, len(cs) // 2)) % len(cs)
            if j == i:
                continue
            dist = source_space.distance(space, int(row["source_idx"]), int(cs[j]["source_idx"]))
            rows.append(source_space.pair_metrics(row, cs[j], space, metric_distance=dist))
    return rows


def audit(ctx):
    out = []
    for name, families in schema.FAMILY_SPACES:
        space = source_space.build_space(ctx, families=families or None)
        radii = source_space.epsilon_radii(ctx, space)
        radius = radii[schema.SOURCE_EQUIVALENT_Q]
        nearest = _nearest_within_trajectory(ctx, space)
        baseline = _random_baseline(ctx, space)
        eq = [r for r in nearest if r["source_distance_primary"] <= radius]
        out.append({
            "space": name,
            "families": ";".join(families) if families else "all",
            "n_objectives": len(space["specs"]),
            "q10_radius": radius,
            "mean_nearest_source_distance": al.finite_mean([r["source_distance_primary"] for r in nearest]),
            "nearest_joint_good_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in nearest]),
            "nearest_pareto_good_disagreement_rate": al.finite_mean(
                [r["pareto_good_disagreement"] for r in nearest]),
            "nearest_target_divergent_rate": al.finite_mean([r["target_divergent"] for r in nearest]),
            "nearest_mean_target_utility_gap": al.finite_mean([r["target_utility_gap"] for r in nearest]),
            "source_equivalent_q10_fraction": len(eq) / len(nearest) if nearest else None,
            "source_equivalent_q10_target_divergent_rate": al.finite_mean(
                [r["target_divergent"] for r in eq]),
            "source_equivalent_q10_joint_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in eq]),
            "baseline_joint_good_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in baseline]),
            "baseline_target_divergent_rate": al.finite_mean([r["target_divergent"] for r in baseline]),
            "joint_disagreement_reduction_vs_baseline": (
                al.finite_mean([r["joint_good_disagreement"] for r in baseline]) -
                al.finite_mean([r["joint_good_disagreement"] for r in nearest])
            ),
            "distance_metrics_frozen": 1,
        })
    return {"rows": out, "summary": {r["space"]: r for r in out}}
