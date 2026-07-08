"""Frozen C45 source-objective spaces and distances."""
from __future__ import annotations

import math

import numpy as np

from . import artifact_loader as al
from . import schema


def _rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    denom = max(len(vals) - 1, 1)
    return np.asarray([r / denom for r in ranks], dtype=float)


def _zscore(mat):
    mu = mat.mean(axis=0)
    sd = mat.std(axis=0)
    sd[sd <= 1e-12] = 1.0
    return (mat - mu) / sd


def oriented_matrix(rows, specs):
    return np.asarray([[al.oriented_value(r, s) for s in specs] for r in rows], dtype=float)


def build_space(ctx, families=None):
    specs = al.source_specs(ctx, families=families)
    if not specs:
        raise ValueError("C45 source space has no inherited source objectives")
    n = len(ctx["registry"])
    z = np.zeros((n, len(specs)), dtype=float)
    ranks = np.zeros((n, len(specs)), dtype=float)
    for _, rows in sorted(ctx["by_traj"].items()):
        idx = [int(r["source_idx"]) for r in rows]
        mat = oriented_matrix(rows, specs)
        z[np.asarray(idx), :] = _zscore(mat)
        for j in range(mat.shape[1]):
            ranks[np.asarray(idx), j] = _rankdata(mat[:, j])
    families_by_col = {}
    for j, s in enumerate(specs):
        families_by_col.setdefault(s["family"], []).append(j)
    return {"specs": specs, "z": z, "rank": ranks, "families_by_col": families_by_col}


def distance(space, i, j, metric=schema.PRIMARY_DISTANCE):
    if metric == schema.PRIMARY_DISTANCE:
        diff = space["z"][i] - space["z"][j]
        return float(np.linalg.norm(diff))
    if metric == schema.RANK_DISTANCE:
        return float(np.mean(np.abs(space["rank"][i] - space["rank"][j])))
    if metric == schema.FAMILY_BLOCK_DISTANCE:
        vals = []
        for cols in space["families_by_col"].values():
            diff = space["z"][i, cols] - space["z"][j, cols]
            vals.append(float(np.linalg.norm(diff) / math.sqrt(max(len(cols), 1))))
        return float(np.mean(vals)) if vals else math.nan
    raise ValueError(metric)


def pair_metrics(a, b, space, metric_distance=None):
    i, j = int(a["source_idx"]), int(b["source_idx"])
    primary = distance(space, i, j, schema.PRIMARY_DISTANCE)
    rank = distance(space, i, j, schema.RANK_DISTANCE)
    block = distance(space, i, j, schema.FAMILY_BLOCK_DISTANCE)
    target_gap = abs(float(a["target_utility_score"]) - float(b["target_utility_score"]))
    joint_disagree = int(int(a["primary_joint_good"]) != int(b["primary_joint_good"]))
    pareto_disagree = int(int(a["pareto_good"]) != int(b["pareto_good"]))
    pref_disagree = int(int(a["preference_robust_better_candidate"]) != int(b["preference_robust_better_candidate"]))
    gauge_gap = abs(float(a["target_joint_margin_raw"]) - float(b["target_joint_margin_raw"]))
    raw = np.asarray([float(a["target_bacc_delta"]) - float(b["target_bacc_delta"]),
                      float(a["target_nll_delta"]) - float(b["target_nll_delta"]),
                      float(a["target_ece_delta"]) - float(b["target_ece_delta"])], dtype=float)
    z = np.asarray([float(a["target_bacc_z"]) - float(b["target_bacc_z"]),
                    float(a["target_nll_z"]) - float(b["target_nll_z"]),
                    float(a["target_ece_z"]) - float(b["target_ece_z"])], dtype=float)
    divergent = int(
        target_gap >= schema.TARGET_UTILITY_LARGE_GAP or
        joint_disagree or pareto_disagree or pref_disagree or
        gauge_gap >= schema.GAUGE_JUMP_EPS or
        float(np.linalg.norm(z)) >= schema.ENDPOINT_Z_LARGE_GAP
    )
    return {
        "source_distance": primary if metric_distance is None else metric_distance,
        "source_distance_primary": primary,
        "source_distance_rank_l1": rank,
        "source_distance_family_block": block,
        "target_utility_gap": target_gap,
        "joint_good_disagreement": joint_disagree,
        "pareto_good_disagreement": pareto_disagree,
        "preference_robust_disagreement": pref_disagree,
        "target_gauge_gap": gauge_gap,
        "endpoint_vector_gap_raw": float(np.linalg.norm(raw)),
        "endpoint_vector_gap_z": float(np.linalg.norm(z)),
        "target_divergent": divergent,
        "target_labels_diagnostic_only": 1,
    }


def within_trajectory_pair_distances(ctx, space):
    vals = []
    for rows in ctx["by_traj"].values():
        for a_i in range(len(rows)):
            for b_i in range(a_i + 1, len(rows)):
                vals.append(distance(space, int(rows[a_i]["source_idx"]), int(rows[b_i]["source_idx"])))
    return vals


def epsilon_radii(ctx, space):
    vals = within_trajectory_pair_distances(ctx, space)
    return {q: al.finite_quantile(vals, q) for q in schema.EPSILON_QUANTILES}
