"""Nearest source-neighbor ambiguity under frozen conditioning scopes."""
from __future__ import annotations

import numpy as np

from ..source_nonidentifiability import source_space
from . import artifact_loader as al
from . import schema


def _scope_candidates(ctx, row, scope):
    if scope == "within_trajectory":
        rows = ctx["by_traj"][row["trajectory_id"]]
    elif scope == "within_target":
        rows = ctx["by_target"][str(row["target"])]
    elif scope == "within_seed":
        rows = ctx["by_seed"][str(row["seed"])]
    elif scope == "within_level":
        rows = ctx["by_level"][str(row["level"])]
    elif scope == "within_regime":
        rows = ctx["by_regime"][row["regime"]]
    elif scope == "cross_target":
        rows = [r for r in ctx["registry"] if str(r["target"]) != str(row["target"])]
    elif scope == "cross_regime":
        rows = [r for r in ctx["registry"] if r["regime"] != row["regime"]]
    else:
        raise ValueError(scope)
    idx = int(row["source_idx"])
    return [r for r in rows if int(r["source_idx"]) != idx]


def _nearest(row, candidates, space):
    i = int(row["source_idx"])
    idxs = np.asarray([int(r["source_idx"]) for r in candidates], dtype=int)
    diff = space["z"][idxs, :] - space["z"][i, :]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    k = int(np.argmin(dist))
    return candidates[k], float(dist[k])


def _baseline(row, candidates, space):
    ordered = sorted(candidates, key=lambda r: int(r["source_idx"]))
    if not ordered:
        return None, None
    pos = 0
    for k, r in enumerate(ordered):
        if int(r["source_idx"]) > int(row["source_idx"]):
            pos = k
            break
    nn = ordered[(pos + max(1, len(ordered) // 2)) % len(ordered)]
    dist = source_space.distance(space, int(row["source_idx"]), int(nn["source_idx"]))
    return nn, dist


def _row(scope, row, nn, dist, space, relation, radius):
    m = source_space.pair_metrics(row, nn, space, metric_distance=dist)
    return {
        "scope": scope,
        "seed": row["seed"],
        "target": row["target"],
        "level": row["level"],
        "regime": row["regime"],
        "candidate_order": row["candidate_order"],
        "neighbor_seed": nn["seed"],
        "neighbor_target": nn["target"],
        "neighbor_level": nn["level"],
        "neighbor_regime": nn["regime"],
        "neighbor_order": nn["candidate_order"],
        "relation": relation,
        "same_target": int(str(row["target"]) == str(nn["target"])),
        "same_trajectory": int(row["trajectory_id"] == nn["trajectory_id"]),
        "same_regime": int(row["regime"] == nn["regime"]),
        "source_equivalent_q10": int(m["source_distance_primary"] <= radius),
        **m,
    }


def _summarize(rows):
    out = []
    for scope in schema.CONDITIONING_SCOPES:
        rs = [r for r in rows if r["scope"] == scope and r["relation"] == "nearest_source"]
        bs = [r for r in rows if r["scope"] == scope and r["relation"] == "scope_conditioned_baseline"]
        eq = [r for r in rs if int(r["source_equivalent_q10"])]
        out.append({
            "scope": scope,
            "n_rows": len(rs),
            "mean_source_distance": al.finite_mean([r["source_distance_primary"] for r in rs]),
            "mean_target_utility_gap": al.finite_mean([r["target_utility_gap"] for r in rs]),
            "target_divergent_rate": al.finite_mean([r["target_divergent"] for r in rs]),
            "joint_good_disagreement_rate": al.finite_mean([r["joint_good_disagreement"] for r in rs]),
            "pareto_good_disagreement_rate": al.finite_mean([r["pareto_good_disagreement"] for r in rs]),
            "target_gauge_gap_mean": al.finite_mean([r["target_gauge_gap"] for r in rs]),
            "source_equivalent_q10_fraction": len(eq) / len(rs) if rs else None,
            "source_equivalent_q10_target_divergent_rate": al.finite_mean(
                [r["target_divergent"] for r in eq]),
            "source_equivalent_q10_joint_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in eq]),
            "baseline_target_divergent_rate": al.finite_mean([r["target_divergent"] for r in bs]),
            "baseline_joint_good_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in bs]),
            "nearest_over_baseline_divergence": (
                al.finite_mean([r["target_divergent"] for r in rs]) /
                al.finite_mean([r["target_divergent"] for r in bs])
            ),
        })
    return out


def audit(ctx, space=None):
    space = space or source_space.build_space(ctx)
    radius = source_space.epsilon_radii(ctx, space)[schema.SOURCE_EQUIVALENT_Q]
    rows = []
    for row in ctx["registry"]:
        for scope in schema.CONDITIONING_SCOPES:
            candidates = _scope_candidates(ctx, row, scope)
            if not candidates:
                continue
            nn, dist = _nearest(row, candidates, space)
            rows.append(_row(scope, row, nn, dist, space, "nearest_source", radius))
            b, bdist = _baseline(row, candidates, space)
            if b is not None:
                rows.append(_row(scope, row, b, bdist, space, "scope_conditioned_baseline", radius))
    summary_rows = _summarize(rows)
    return {"rows": rows, "summary_rows": summary_rows,
            "summary": {r["scope"]: r for r in summary_rows},
            "q10_radius": radius, "space": space}
