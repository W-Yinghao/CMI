"""C43 leakage-vs-rank frontier conflict diagnostics."""
from __future__ import annotations

import numpy as np

from . import actionability_metrics, artifact_loader as al, scalarization_grid, schema


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
    return np.asarray(ranks, dtype=float)


def _spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _percentile(rows, candidate, key, *, higher_better):
    ordered = sorted(rows, key=lambda r: (float(r[key]), -int(r["candidate_order"])), reverse=higher_better)
    for i, r in enumerate(ordered):
        if r is candidate:
            return i / (len(ordered) - 1) if len(ordered) > 1 else 0.0
    return None


def audit(ctx, grid, mult):
    best_id = mult["summary"]["best_scalarization_id"]
    best_scalar = next(r for r in grid["rows"] if r["scalarization_id"] == best_id)
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        selected = next(r for r in rs if int(r["selected_oaci"]) == 1)
        rank_top = max(rs, key=lambda r: (float(r["source_rank_score"]), -int(r["candidate_order"])))
        scores = actionability_metrics.scalar_scores(rs, best_scalar)
        compromise = max(rs, key=lambda r: (scores[r["candidate_order"]], -int(r["candidate_order"])))
        leakage_rank_corr = _spearman(
            [-float(r["selection_leakage_point"]) for r in rs],
            [float(r["source_rank_score"]) for r in rs],
        )
        rank_delta = float(rank_top["target_utility_score"]) - float(selected["target_utility_score"])
        comp_delta = float(compromise["target_utility_score"]) - float(selected["target_utility_score"])
        leak_delta_rank = float(rank_top["selection_leakage_point"]) - float(selected["selection_leakage_point"])
        rows.append({
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "best_scalarization_id": best_id,
            "leakage_rank_spearman": leakage_rank_corr,
            "oaci_leakage_rank_percentile": _percentile(rs, selected, "selection_leakage_point", higher_better=False),
            "oaci_source_rank_percentile": _percentile(rs, selected, "source_rank_score", higher_better=True),
            "rank_top_target_utility_delta_vs_oaci": rank_delta,
            "rank_top_selection_leakage_delta_vs_oaci": leak_delta_rank,
            "rank_top_target_better_than_oaci": int(rank_delta > 1e-12),
            "leakage_blocks_rank_better_candidate": int(rank_delta > 1e-12 and leak_delta_rank > 1e-12),
            "best_scalarization_target_utility_delta_vs_oaci": comp_delta,
            "best_scalarization_target_better_than_oaci": int(comp_delta > 1e-12),
            "best_scalarization_joint_good": int(compromise["primary_joint_good"]),
            "best_scalarization_pareto_good": int(compromise["pareto_good"]),
            "no_candidate_id_emitted": 1,
        })
    summary = {
        "best_scalarization_id": best_id,
        "mean_leakage_rank_spearman": al.finite_mean([r["leakage_rank_spearman"] for r in rows]),
        "negative_leakage_rank_corr_fraction": al.finite_mean(
            [int((r["leakage_rank_spearman"] or 0) < 0) for r in rows]),
        "mean_oaci_leakage_rank_percentile": al.finite_mean([r["oaci_leakage_rank_percentile"] for r in rows]),
        "mean_oaci_source_rank_percentile": al.finite_mean([r["oaci_source_rank_percentile"] for r in rows]),
        "rank_top_target_better_fraction": al.finite_mean([r["rank_top_target_better_than_oaci"] for r in rows]),
        "leakage_blocks_rank_better_fraction": al.finite_mean(
            [r["leakage_blocks_rank_better_candidate"] for r in rows]),
        "best_scalarization_target_better_fraction": al.finite_mean(
            [r["best_scalarization_target_better_than_oaci"] for r in rows]),
        "best_scalarization_joint_good_rate": al.finite_mean([r["best_scalarization_joint_good"] for r in rows]),
    }
    summary["source_rank_leakage_tradeoff_real"] = bool(
        (summary["mean_leakage_rank_spearman"] or 0) <= schema.TRADEOFF_NEGATIVE_CORR_GATE)
    summary["leakage_extreme_blocks_rank_frontier"] = bool(
        (summary["leakage_blocks_rank_better_fraction"] or 0) >= schema.LEAKAGE_BLOCKS_GATE)
    return {"rows": rows, "summary": summary}
