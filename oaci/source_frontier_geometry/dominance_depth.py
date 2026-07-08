"""C44 dominance depth target-alignment audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import objective_registry, pareto_nulls


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
    return ranks


def _spearman(x, y):
    import numpy as np
    rx, ry = _rankdata(x), _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _pairwise_auc(rows, score_values):
    total = correct = ties = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            du = float(rows[i]["target_utility_score"]) - float(rows[j]["target_utility_score"])
            if abs(du) <= 1e-12:
                continue
            ds = score_values[i] - score_values[j]
            total += 1
            prod = du * ds
            if prod > 0:
                correct += 1
            elif abs(prod) <= 1e-12:
                ties += 1
    return (correct + 0.5 * ties) / total if total else None


def audit(ctx):
    specs = objective_registry.source_pareto_specs(ctx)
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        mat = pareto_nulls.oriented_matrix(rs, specs)
        front, dominators, dominated = pareto_nulls.dominance_stats(mat)
        layers = pareto_nulls.pareto_layers(mat)
        layer_score = [-int(v) for v in layers]
        dom_score = [-int(v) for v in dominators]
        dominated_score = [int(v) for v in dominated]
        utility = [float(r["target_utility_score"]) for r in rs]
        front_idx = [i for i, v in enumerate(front) if v]
        rows.append({
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "n_candidates": len(rs),
            "front_fraction": float(front.mean()),
            "max_pareto_layer": int(max(layers)),
            "mean_n_dominators": al.finite_mean(dominators),
            "mean_n_dominated": al.finite_mean(dominated),
            "layer_auc_vs_target_utility": _pairwise_auc(rs, layer_score),
            "n_dominators_auc_vs_target_utility": _pairwise_auc(rs, dom_score),
            "n_dominated_auc_vs_target_utility": _pairwise_auc(rs, dominated_score),
            "layer_spearman_vs_target_utility": _spearman(layer_score, utility),
            "front_joint_good_rate": al.finite_mean([rs[i]["primary_joint_good"] for i in front_idx]),
            "front_pareto_good_rate": al.finite_mean([rs[i]["pareto_good"] for i in front_idx]),
            "target_labels_diagnostic_only": 1,
        })
    summary = {
        "mean_layer_auc_vs_target_utility": al.finite_mean([r["layer_auc_vs_target_utility"] for r in rows]),
        "mean_n_dominators_auc_vs_target_utility": al.finite_mean(
            [r["n_dominators_auc_vs_target_utility"] for r in rows]),
        "mean_n_dominated_auc_vs_target_utility": al.finite_mean(
            [r["n_dominated_auc_vs_target_utility"] for r in rows]),
        "mean_layer_spearman_vs_target_utility": al.finite_mean(
            [r["layer_spearman_vs_target_utility"] for r in rows]),
        "mean_front_joint_good_rate": al.finite_mean([r["front_joint_good_rate"] for r in rows]),
        "mean_front_pareto_good_rate": al.finite_mean([r["front_pareto_good_rate"] for r in rows]),
        "mean_max_pareto_layer": al.finite_mean([r["max_pareto_layer"] for r in rows]),
    }
    return {"rows": rows, "summary": summary}
