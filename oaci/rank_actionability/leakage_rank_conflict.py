"""C42 leakage-vs-source-rank conflict audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import auc_to_topk_gap, score_registry, schema


def audit(ctx):
    spec = next(s for s in score_registry.SCORES if s["score"] == "C30_source_rank_score")
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        selected = next(r for r in rs if int(r["selected_oaci"]) == 1)
        rank_top = auc_to_topk_gap.order_rows(rs, spec)[0]
        utility_delta = float(rank_top["target_utility_score"]) - float(selected["target_utility_score"])
        leakage_delta = float(rank_top["selection_leakage_point"]) - float(selected["selection_leakage_point"])
        target_better = utility_delta > 1e-12
        rank_higher_leakage = leakage_delta > 1e-12
        rows.append({
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "target_utility_delta_rank_top_minus_oaci": utility_delta,
            "selection_leakage_delta_rank_top_minus_oaci": leakage_delta,
            "audit_leakage_delta_rank_top_minus_oaci": (
                float(rank_top["audit_leakage_point"]) - float(selected["audit_leakage_point"])),
            "R_src_delta_rank_top_minus_oaci": float(rank_top["R_src"]) - float(selected["R_src"]),
            "rank_top_target_better_than_oaci": int(target_better),
            "rank_top_higher_selection_leakage_than_oaci": int(rank_higher_leakage),
            "leakage_blocks_rank_better_candidate": int(target_better and rank_higher_leakage),
            "oaci_joint_good": int(selected["primary_joint_good"]),
            "rank_top_joint_good": int(rank_top["primary_joint_good"]),
            "oaci_pareto_good": int(selected["pareto_good"]),
            "rank_top_pareto_good": int(rank_top["pareto_good"]),
            "target_gauge_delta_available": 0,
            "target_gauge_delta_rank_top_minus_oaci": "",
            "no_candidate_id_emitted": 1,
        })
    summary = {
        "n_trajectories": len(rows),
        "rank_top_target_better_than_oaci_count": sum(r["rank_top_target_better_than_oaci"] for r in rows),
        "rank_top_target_better_than_oaci_fraction": al.finite_mean(
            [r["rank_top_target_better_than_oaci"] for r in rows]),
        "leakage_blocks_rank_better_count": sum(r["leakage_blocks_rank_better_candidate"] for r in rows),
        "leakage_blocks_rank_better_fraction": al.finite_mean(
            [r["leakage_blocks_rank_better_candidate"] for r in rows]),
        "mean_target_utility_delta_rank_top_minus_oaci": al.finite_mean(
            [r["target_utility_delta_rank_top_minus_oaci"] for r in rows]),
        "mean_selection_leakage_delta_rank_top_minus_oaci": al.finite_mean(
            [r["selection_leakage_delta_rank_top_minus_oaci"] for r in rows]),
        "target_gauge_delta_available": False,
    }
    summary["leakage_blocks_rank_better_candidates"] = bool(
        (summary["leakage_blocks_rank_better_fraction"] or 0) >= schema.LEAKAGE_BLOCKS_FRACTION_GATE)
    return {"rows": rows, "summary": summary}
