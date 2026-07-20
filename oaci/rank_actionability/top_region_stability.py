"""C42 source-rank top-region stability audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import auc_to_topk_gap, schema, score_registry


def audit(ctx):
    spec = next(s for s in score_registry.SCORES if s["score"] == "C30_source_rank_score")
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        ordered = auc_to_topk_gap.order_rows(rs, spec)
        if len(ordered) < 3:
            continue
        top = ordered[0]
        top_score = float(top["source_rank_score"])
        plateau = [r for r in ordered if top_score - float(r["source_rank_score"]) <= schema.PLATEAU_EPS + 1e-12]
        rows.append({
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "top1_top2_margin": top_score - float(ordered[1]["source_rank_score"]),
            "top1_top3_margin": top_score - float(ordered[2]["source_rank_score"]),
            "plateau_epsilon": schema.PLATEAU_EPS,
            "plateau_size": len(plateau),
            "plateau_fraction": len(plateau) / len(ordered),
            "plateau_joint_good_rate": al.finite_mean([r["primary_joint_good"] for r in plateau]),
            "plateau_pareto_good_rate": al.finite_mean([r["pareto_good"] for r in plateau]),
            "top1_joint_good": int(top["primary_joint_good"]),
            "top1_pareto_good": int(top["pareto_good"]),
            "top_region_low_margin": int(top_score - float(ordered[1]["source_rank_score"]) <= schema.PLATEAU_EPS),
            "target_labels_diagnostic_only": 1,
        })
    summary = {
        "n_trajectories": len(rows),
        "median_top1_top2_margin": al.finite_median([r["top1_top2_margin"] for r in rows]),
        "mean_top1_top2_margin": al.finite_mean([r["top1_top2_margin"] for r in rows]),
        "mean_plateau_size": al.finite_mean([r["plateau_size"] for r in rows]),
        "median_plateau_size": al.finite_median([r["plateau_size"] for r in rows]),
        "low_margin_fraction": al.finite_mean([r["top_region_low_margin"] for r in rows]),
        "mean_plateau_joint_good_rate": al.finite_mean([r["plateau_joint_good_rate"] for r in rows]),
        "top_region_plateau_or_instability": False,
    }
    summary["top_region_plateau_or_instability"] = bool(
        (summary["mean_plateau_size"] or 0) >= schema.PLATEAU_MEAN_SIZE_GATE or
        (summary["low_margin_fraction"] or 0) >= schema.PLATEAU_LOW_MARGIN_FRACTION_GATE)
    return {"rows": rows, "summary": summary}
