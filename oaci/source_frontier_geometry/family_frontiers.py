"""C44 fixed family-reduced Pareto frontiers."""
from __future__ import annotations

from . import artifact_loader as al
from . import objective_registry, pareto_nulls, schema


def _auc_depth_target(rows, layers):
    total = correct = ties = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            du = float(rows[i]["target_utility_score"]) - float(rows[j]["target_utility_score"])
            if abs(du) <= 1e-12:
                continue
            ds = -layers[i] - (-layers[j])
            total += 1
            prod = du * ds
            if prod > 0:
                correct += 1
            elif abs(prod) <= 1e-12:
                ties += 1
    return (correct + 0.5 * ties) / total if total else None


def _subset_frontier(ctx, subset_name, families):
    specs = objective_registry.family_specs(ctx, families)
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        mat = pareto_nulls.oriented_matrix(rs, specs)
        front, _, _ = pareto_nulls.dominance_stats(mat)
        layers = pareto_nulls.pareto_layers(mat)
        fidx = [i for i, flag in enumerate(front) if flag]
        not_front = [i for i, flag in enumerate(front) if not flag]
        out = {
            "subset": subset_name,
            "trajectory_id": tid,
            "n_objectives": len(specs),
            "n_candidates": len(rs),
            "front_fraction": float(front.mean()),
            "joint_good_front_fraction": (
                sum(1 for i, r in enumerate(rs) if int(r["primary_joint_good"]) and front[i]) /
                max(1, sum(int(r["primary_joint_good"]) for r in rs))),
            "pareto_good_front_fraction": (
                sum(1 for i, r in enumerate(rs) if int(r["pareto_good"]) and front[i]) /
                max(1, sum(int(r["pareto_good"]) for r in rs))),
            "target_bad_front_fraction": (
                sum(1 for i, r in enumerate(rs) if not int(r["primary_joint_good"]) and front[i]) /
                max(1, sum(1 for r in rs if not int(r["primary_joint_good"])))),
            "p_joint_good_given_front": al.finite_mean([rs[i]["primary_joint_good"] for i in fidx]),
            "p_joint_good_given_not_front": al.finite_mean([rs[i]["primary_joint_good"] for i in not_front]),
            "dominance_depth_auc_vs_target_utility": _auc_depth_target(rs, layers),
            "target_labels_diagnostic_only": 1,
        }
        base = sum(int(r["primary_joint_good"]) for r in rs) / len(rs)
        out["front_joint_good_enrichment"] = out["p_joint_good_given_front"] / base if base > 0 else ""
        rows.append(out)
    return rows


def audit(ctx):
    rows = []
    for subset_name, families in schema.FAMILY_SUBSETS:
        rows.extend(_subset_frontier(ctx, subset_name, families))
    summary_rows = []
    for subset_name, _ in schema.FAMILY_SUBSETS:
        rs = [r for r in rows if r["subset"] == subset_name]
        summary_rows.append({
            "subset": subset_name,
            "n_trajectories": len(rs),
            "mean_front_fraction": al.finite_mean([r["front_fraction"] for r in rs]),
            "mean_joint_good_front_fraction": al.finite_mean([r["joint_good_front_fraction"] for r in rs]),
            "mean_pareto_good_front_fraction": al.finite_mean([r["pareto_good_front_fraction"] for r in rs]),
            "mean_target_bad_front_fraction": al.finite_mean([r["target_bad_front_fraction"] for r in rs]),
            "mean_p_joint_good_given_front": al.finite_mean([r["p_joint_good_given_front"] for r in rs]),
            "mean_p_joint_good_given_not_front": al.finite_mean([r["p_joint_good_given_not_front"] for r in rs]),
            "mean_front_joint_good_enrichment": al.finite_mean([r["front_joint_good_enrichment"] for r in rs]),
            "mean_depth_auc_vs_target_utility": al.finite_mean(
                [r["dominance_depth_auc_vs_target_utility"] for r in rs]),
        })
    summary = {r["subset"]: r for r in summary_rows}
    return {"rows": summary_rows, "trajectory_rows": rows, "summary": summary}
