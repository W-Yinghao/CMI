"""C45 source-nearest target-divergent witness audit."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema, source_space


def _row_id(r):
    return {
        "seed": r["seed"], "target": r["target"], "level": r["level"], "regime": r["regime"],
        "candidate_order": r["candidate_order"],
    }


def _eligible(ctx, row, scope):
    idx = int(row["source_idx"])
    if scope == "within_trajectory":
        rows = ctx["by_traj"][row["trajectory_id"]]
    elif scope == "within_target":
        rows = ctx["by_target"][str(row["target"])]
    elif scope == "cross_target":
        rows = [r for r in ctx["registry"] if str(r["target"]) != str(row["target"])]
    elif scope == "same_regime":
        rows = ctx["by_regime"][row["regime"]]
    else:
        raise ValueError(scope)
    return [r for r in rows if int(r["source_idx"]) != idx]


def _nearest(row, candidates, space):
    i = int(row["source_idx"])
    idxs = np.asarray([int(r["source_idx"]) for r in candidates], dtype=int)
    diff = space["z"][idxs, :] - space["z"][i, :]
    dist = np.sqrt(np.sum(diff * diff, axis=1))
    best = int(np.argmin(dist))
    return candidates[best], float(dist[best])


def _baseline_neighbor(row, candidates):
    ordered = sorted(candidates, key=lambda r: int(r["source_idx"]))
    if not ordered:
        return None
    pos = 0
    for k, r in enumerate(ordered):
        if int(r["source_idx"]) > int(row["source_idx"]):
            pos = k
            break
    return ordered[(pos + max(1, len(ordered) // 2)) % len(ordered)]


def _pair_row(scope, row, nn, dist, space, radii, relation):
    metrics = source_space.pair_metrics(row, nn, space, metric_distance=dist)
    out = {
        "scope": scope,
        **_row_id(row),
        "neighbor_seed": nn["seed"],
        "neighbor_target": nn["target"],
        "neighbor_level": nn["level"],
        "neighbor_regime": nn["regime"],
        "neighbor_order": nn["candidate_order"],
        "neighbor_relation": relation,
        "same_target": int(str(row["target"]) == str(nn["target"])),
        "same_trajectory": int(row["trajectory_id"] == nn["trajectory_id"]),
    }
    out.update(metrics)
    for q, radius in radii.items():
        out[f"source_equivalent_q{int(q * 100):02d}"] = int(metrics["source_distance_primary"] <= radius)
    return out


def _summarize(rows, baseline_rows):
    out = {}
    for scope in schema.NEAREST_SCOPES:
        rs = [r for r in rows if r["scope"] == scope]
        brs = [r for r in baseline_rows if r["scope"] == scope]
        eq = [r for r in rs if int(r["source_equivalent_q10"])]
        out[scope] = {
            "n_rows": len(rs),
            "mean_source_distance": al.finite_mean([r["source_distance_primary"] for r in rs]),
            "mean_target_utility_gap": al.finite_mean([r["target_utility_gap"] for r in rs]),
            "joint_good_disagreement_rate": al.finite_mean([r["joint_good_disagreement"] for r in rs]),
            "pareto_good_disagreement_rate": al.finite_mean([r["pareto_good_disagreement"] for r in rs]),
            "preference_robust_disagreement_rate": al.finite_mean(
                [r["preference_robust_disagreement"] for r in rs]),
            "target_divergent_rate": al.finite_mean([r["target_divergent"] for r in rs]),
            "source_equivalent_q10_fraction": len(eq) / len(rs) if rs else None,
            "source_equivalent_q10_target_divergent_rate": al.finite_mean(
                [r["target_divergent"] for r in eq]),
            "source_equivalent_q10_joint_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in eq]),
            "baseline_joint_good_disagreement_rate": al.finite_mean(
                [r["joint_good_disagreement"] for r in brs]),
            "baseline_target_divergent_rate": al.finite_mean([r["target_divergent"] for r in brs]),
            "baseline_mean_target_utility_gap": al.finite_mean([r["target_utility_gap"] for r in brs]),
        }
    return out


def source_equivalent_pairs(ctx, space, radius):
    rows = []
    for _, cs in sorted(ctx["by_traj"].items()):
        for a_i in range(len(cs)):
            for b_i in range(a_i + 1, len(cs)):
                a, b = cs[a_i], cs[b_i]
                dist = source_space.distance(space, int(a["source_idx"]), int(b["source_idx"]))
                if dist > radius:
                    continue
                metrics = source_space.pair_metrics(a, b, space, metric_distance=dist)
                if not metrics["target_divergent"]:
                    continue
                rows.append({
                    **_row_id(a),
                    "neighbor_order": b["candidate_order"],
                    "source_equivalent_radius_q10": radius,
                    **metrics,
                })
    rows.sort(key=lambda r: (r["source_distance_primary"], -r["target_utility_gap"], r["seed"],
                             r["target"], r["level"], r["regime"], int(r["candidate_order"])))
    return rows


def within_trajectory_pair_baselines(ctx, space, radius):
    all_rows = []
    near_rows = []
    for _, cs in sorted(ctx["by_traj"].items()):
        for a_i in range(len(cs)):
            for b_i in range(a_i + 1, len(cs)):
                a, b = cs[a_i], cs[b_i]
                dist = source_space.distance(space, int(a["source_idx"]), int(b["source_idx"]))
                m = source_space.pair_metrics(a, b, space, metric_distance=dist)
                all_rows.append(m)
                if dist <= radius:
                    near_rows.append(m)
    return {
        "all_pair_target_divergent_rate": al.finite_mean([r["target_divergent"] for r in all_rows]),
        "all_pair_joint_disagreement_rate": al.finite_mean([r["joint_good_disagreement"] for r in all_rows]),
        "all_pair_mean_target_utility_gap": al.finite_mean([r["target_utility_gap"] for r in all_rows]),
        "source_distance_matched_pair_target_divergent_rate": al.finite_mean(
            [r["target_divergent"] for r in near_rows]),
        "source_distance_matched_pair_joint_disagreement_rate": al.finite_mean(
            [r["joint_good_disagreement"] for r in near_rows]),
        "source_distance_matched_pair_mean_target_utility_gap": al.finite_mean(
            [r["target_utility_gap"] for r in near_rows]),
        "source_distance_matched_n_pairs": len(near_rows),
        "all_pair_n_pairs": len(all_rows),
    }


def audit(ctx):
    space = source_space.build_space(ctx)
    radii = source_space.epsilon_radii(ctx, space)
    rows = []
    baselines = []
    for row in ctx["registry"]:
        for scope in schema.NEAREST_SCOPES:
            cand = _eligible(ctx, row, scope)
            if not cand:
                continue
            nn, dist = _nearest(row, cand, space)
            rows.append(_pair_row(scope, row, nn, dist, space, radii, "nearest_source"))
            b = _baseline_neighbor(row, cand)
            if b is not None:
                bdist = source_space.distance(space, int(row["source_idx"]), int(b["source_idx"]))
                baselines.append(_pair_row(scope, row, b, bdist, space, radii, "scope_conditioned_baseline"))
    radius_q10 = radii[schema.SOURCE_EQUIVALENT_Q]
    pairs = source_equivalent_pairs(ctx, space, radius_q10)
    summary = _summarize(rows, baselines)
    summary["within_trajectory_pair_baseline"] = within_trajectory_pair_baselines(ctx, space, radius_q10)
    summary["epsilon_radii"] = {f"q{int(q * 100):02d}": v for q, v in radii.items()}
    summary["source_equivalent_target_divergent_pair_count"] = len(pairs)
    summary["trajectories_with_source_equivalent_divergent_pair_fraction"] = (
        len({(r["seed"], r["target"], r["level"], r["regime"]) for r in pairs}) / len(ctx["by_traj"])
        if ctx["by_traj"] else None)
    return {
        "rows": rows,
        "baseline_rows": baselines,
        "source_equivalent_pairs": pairs,
        "summary": summary,
        "space": space,
    }
