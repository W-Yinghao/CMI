"""C45 epsilon-radius conditional target variance."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema, source_space


def _var(vals):
    vals = [float(v) for v in vals if al.finite(v)]
    return float(np.var(vals)) if vals else None


def _range(vals):
    vals = [float(v) for v in vals if al.finite(v)]
    return float(max(vals) - min(vals)) if vals else None


def _trajectory_baseline(rows):
    return {
        "baseline_target_utility_variance": _var([r["target_utility_score"] for r in rows]),
        "baseline_joint_good_entropy": al.entropy01([r["primary_joint_good"] for r in rows]),
        "baseline_pareto_good_entropy": al.entropy01([r["pareto_good"] for r in rows]),
        "baseline_preference_robust_entropy": al.entropy01([r["preference_robust_better_candidate"] for r in rows]),
        "baseline_target_gauge_variance": _var([r["target_joint_margin_raw"] for r in rows]),
    }


def _neighborhood_metrics(rows, idxs):
    ns = [rows[i] for i in idxs]
    return {
        "n_neighbors_including_self": len(ns),
        "target_utility_variance": _var([r["target_utility_score"] for r in ns]),
        "target_utility_range": _range([r["target_utility_score"] for r in ns]),
        "joint_good_entropy": al.entropy01([r["primary_joint_good"] for r in ns]),
        "pareto_good_entropy": al.entropy01([r["pareto_good"] for r in ns]),
        "preference_robust_entropy": al.entropy01([r["preference_robust_better_candidate"] for r in ns]),
        "target_gauge_variance": _var([r["target_joint_margin_raw"] for r in ns]),
        "contains_joint_good_and_bad": int(
            len({int(r["primary_joint_good"]) for r in ns}) > 1),
    }


def audit(ctx, space=None):
    space = space or source_space.build_space(ctx)
    radii = source_space.epsilon_radii(ctx, space)
    traj_rows = []
    summary_rows = []
    for q, radius in radii.items():
        per_candidate = []
        for tid, rows in sorted(ctx["by_traj"].items()):
            idx = [int(r["source_idx"]) for r in rows]
            mat = space["z"][idx, :]
            dist = np.sqrt(((mat[:, None, :] - mat[None, :, :]) ** 2).sum(axis=2))
            base = _trajectory_baseline(rows)
            local = []
            for i in range(len(rows)):
                nidx = [j for j in range(len(rows)) if dist[i, j] <= radius]
                m = _neighborhood_metrics(rows, nidx)
                local.append(m)
                per_candidate.append({**m, **base})
            seed, target, level, regime = tid.split("|")
            traj_rows.append({
                "epsilon_quantile": q,
                "epsilon_radius": radius,
                "trajectory_id": tid,
                "seed": seed,
                "target": target,
                "level": level,
                "regime": regime,
                "mean_neighbors_including_self": al.finite_mean(
                    [r["n_neighbors_including_self"] for r in local]),
                "mean_target_utility_variance": al.finite_mean(
                    [r["target_utility_variance"] for r in local]),
                "mean_target_utility_range": al.finite_mean([r["target_utility_range"] for r in local]),
                "mean_joint_good_entropy": al.finite_mean([r["joint_good_entropy"] for r in local]),
                "mean_pareto_good_entropy": al.finite_mean([r["pareto_good_entropy"] for r in local]),
                "mean_preference_robust_entropy": al.finite_mean(
                    [r["preference_robust_entropy"] for r in local]),
                "mean_target_gauge_variance": al.finite_mean([r["target_gauge_variance"] for r in local]),
                "joint_good_cohabitation_rate": al.finite_mean(
                    [r["contains_joint_good_and_bad"] for r in local]),
                **base,
                "target_labels_diagnostic_only": 1,
            })
        summary_rows.append({
            "epsilon_quantile": q,
            "epsilon_radius": radius,
            "mean_neighbors_including_self": al.finite_mean(
                [r["n_neighbors_including_self"] for r in per_candidate]),
            "mean_target_utility_variance": al.finite_mean(
                [r["target_utility_variance"] for r in per_candidate]),
            "mean_target_utility_range": al.finite_mean([r["target_utility_range"] for r in per_candidate]),
            "mean_joint_good_entropy": al.finite_mean([r["joint_good_entropy"] for r in per_candidate]),
            "mean_pareto_good_entropy": al.finite_mean([r["pareto_good_entropy"] for r in per_candidate]),
            "mean_preference_robust_entropy": al.finite_mean(
                [r["preference_robust_entropy"] for r in per_candidate]),
            "mean_target_gauge_variance": al.finite_mean([r["target_gauge_variance"] for r in per_candidate]),
            "joint_good_cohabitation_rate": al.finite_mean(
                [r["contains_joint_good_and_bad"] for r in per_candidate]),
            "baseline_target_utility_variance": al.finite_mean(
                [r["baseline_target_utility_variance"] for r in per_candidate]),
            "baseline_joint_good_entropy": al.finite_mean(
                [r["baseline_joint_good_entropy"] for r in per_candidate]),
            "target_utility_variance_over_baseline": (
                al.finite_mean([r["target_utility_variance"] for r in per_candidate]) /
                al.finite_mean([r["baseline_target_utility_variance"] for r in per_candidate])
            ),
            "joint_entropy_over_baseline": (
                al.finite_mean([r["joint_good_entropy"] for r in per_candidate]) /
                al.finite_mean([r["baseline_joint_good_entropy"] for r in per_candidate])
            ),
        })
    return {"rows": summary_rows, "trajectory_rows": traj_rows,
            "summary": {f"q{int(r['epsilon_quantile'] * 100):02d}": r for r in summary_rows}}
