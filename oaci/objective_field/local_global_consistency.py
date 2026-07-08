"""C41 local-vs-global leakage conflict consistency."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def _rank_percentile(rs, order, field, *, lower_better):
    vals = sorted([(float(r[field]), int(r["candidate_order"])) for r in rs if al.finite(r[field])],
                  key=lambda x: (x[0], x[1]), reverse=not lower_better)
    for i, (_, o) in enumerate(vals):
        if o == int(order):
            return i / (len(vals) - 1) if len(vals) > 1 else 0.0
    return ""


def audit(ctx, alignment):
    align_by_traj_field = {(r["trajectory_id"], r["field"]): r for r in alignment["rows"]}
    rows = []
    for pair in ctx["tables"]["c37"]["exact"]:
        tid = "|".join([pair["seed"], pair["target"], pair["level"], pair["regime"]])
        rs = ctx["by_traj"][tid]
        selected = ctx["by_candidate"][(pair["seed"], pair["target"], pair["level"], pair["regime"],
                                         pair["selected_order"])]
        better = ctx["by_candidate"][(pair["seed"], pair["target"], pair["level"], pair["regime"],
                                      pair["better_order"])]
        global_row = align_by_traj_field.get((tid, "selection_leakage_point"), {})
        auc = al.as_float(global_row.get("pairwise_auc_oriented_field_ranks_target_utility"))
        local_selected_low_leakage = float(selected["selection_leakage_point"]) < float(better["selection_leakage_point"])
        local_better_target = float(better["target_utility_score"]) > float(selected["target_utility_score"])
        selected_leak_rank = _rank_percentile(rs, pair["selected_order"], "selection_leakage_point", lower_better=True)
        selected_target_rank = _rank_percentile(rs, pair["selected_order"], "target_utility_score", lower_better=False)
        rows.append({
            "pair_id": pair["pair_id"],
            "pair_key": pair["pair_key"],
            "seed": pair["seed"],
            "target": pair["target"],
            "level": pair["level"],
            "regime": pair["regime"],
            "selected_order": pair["selected_order"],
            "better_order": pair["better_order"],
            "global_selection_leakage_auc": auc,
            "global_selection_leakage_class": global_row.get("sign_class", "unavailable"),
            "local_selected_lower_leakage": int(local_selected_low_leakage),
            "local_better_higher_target_utility": int(local_better_target),
            "local_conflict": int(local_selected_low_leakage and local_better_target),
            "selected_low_leakage_rank_percentile": selected_leak_rank,
            "selected_target_utility_rank_percentile": selected_target_rank,
            "selected_near_global_leakage_optimum": int(al.finite(selected_leak_rank) and float(selected_leak_rank) <= 0.10),
            "selected_away_from_target_optimum": int(al.finite(selected_target_rank) and float(selected_target_rank) > 0.25),
            "local_conflict_representative_of_global_field": int(al.finite(auc) and auc <= schema.ALIGNMENT_AUC_HIGH),
            "local_tail_only_flag": int(al.finite(selected_leak_rank) and float(selected_leak_rank) > 0.50),
        })
    summary = {
        "n_pairs": len(rows),
        "local_conflict_count": sum(r["local_conflict"] for r in rows),
        "representative_count": sum(r["local_conflict_representative_of_global_field"] for r in rows),
        "representative_fraction": al.finite_mean([r["local_conflict_representative_of_global_field"] for r in rows]),
        "tail_only_count": sum(r["local_tail_only_flag"] for r in rows),
        "tail_only_fraction": al.finite_mean([r["local_tail_only_flag"] for r in rows]),
        "mean_selected_low_leakage_rank_percentile": al.finite_mean([r["selected_low_leakage_rank_percentile"] for r in rows]),
        "mean_selected_target_utility_rank_percentile": al.finite_mean([r["selected_target_utility_rank_percentile"] for r in rows]),
    }
    return {"rows": rows, "summary": summary}


def target_gauge_vs_leakage(ctx):
    rows = []
    for r in ctx["tables"]["c38"]["gauge"]:
        exact = ctx["by_pair"]["c37_exact"][r["pair_id"]]
        rows.append({
            "pair_id": r["pair_id"],
            "pair_key": r["pair_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selection_point_delta_better_minus_selected": exact["point_delta_better_minus_selected"],
            "selection_point_prefers": exact["point_prefers"],
            "target_gauge_delta_better_minus_selected": r["target_gauge_delta_better_minus_selected"],
            "target_gauge_prefers": r["target_gauge_prefers"],
            "leakage_target_gauge_conflict": r["leakage_target_gauge_conflict"],
            "candidate_level_target_gauge_available": 0,
            "target_gauge_non_source_only": 1,
        })
    summary = {
        "n_pairs": len(rows),
        "leakage_target_gauge_conflict_count": sum(int(r["leakage_target_gauge_conflict"]) for r in rows),
        "leakage_target_gauge_conflict_fraction": al.finite_mean([r["leakage_target_gauge_conflict"] for r in rows]),
        "candidate_level_target_gauge_available": False,
    }
    return {"rows": rows, "summary": summary}
