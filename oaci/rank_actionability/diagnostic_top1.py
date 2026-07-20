"""C42 aggregate diagnostic top-1 audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import auc_to_topk_gap, schema, score_registry


def _oracle(rows):
    return max(float(r["target_utility_score"]) for r in rows)


def _selected(rows):
    ss = [r for r in rows if int(r["selected_oaci"]) == 1]
    return ss[0] if len(ss) == 1 else None


def _top1(rows, spec):
    ordered = auc_to_topk_gap.order_rows(rows, spec)
    return ordered[0] if ordered else None


def audit(ctx):
    rows = []
    specs = {s["score"]: s for s in score_registry.SCORES}
    score_names = ("actual_oaci_selector", "selection_leakage_point", "source_audit_leakage", "R_src",
                   "C19_robust_core_score", "C30_source_rank_score", "target_grouped_diagnostic_ceiling")
    random_joint = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        random_joint.append(sum(int(r["primary_joint_good"]) for r in rs) / len(rs))
    random_top1_joint = al.finite_mean(random_joint)
    for name in score_names:
        spec = specs[name]
        chosen = []
        deltas = []
        better_than_oaci = []
        for tid, rs in sorted(ctx["by_traj"].items()):
            if name == "actual_oaci_selector":
                c = _selected(rs)
            else:
                c = _top1(rs, spec)
            o = _selected(rs)
            if c is None or o is None:
                continue
            oracle = _oracle(rs)
            chosen.append((c, oracle))
            deltas.append(float(c["target_utility_score"]) - float(o["target_utility_score"]))
            better_than_oaci.append(float(c["target_utility_score"]) > float(o["target_utility_score"]) + 1e-12)
        rows.append({
            "score": name,
            "n_trajectories": len(chosen),
            "top1_joint_good_rate": al.finite_mean([c["primary_joint_good"] for c, _ in chosen]),
            "top1_pareto_good_rate": al.finite_mean([c["pareto_good"] for c, _ in chosen]),
            "top1_preference_robust_utility_rate": al.finite_mean(
                [c["preference_robust_better_candidate"] for c, _ in chosen]),
            "top1_regret_vs_target_oracle": al.finite_mean(
                [oracle - float(c["target_utility_score"]) for c, oracle in chosen]),
            "mean_target_utility_delta_vs_actual_oaci": al.finite_mean(deltas),
            "fraction_top1_target_better_than_actual_oaci": al.finite_mean(better_than_oaci),
            "top1_joint_good_gain_vs_random": (
                al.finite_mean([c["primary_joint_good"] for c, _ in chosen]) - random_top1_joint),
            "non_source_only": spec["non_source_only"],
            "diagnostic_ceiling": spec["diagnostic_ceiling"],
        })
    summary = {r["score"]: r for r in rows}
    summary["random_trajectory_conditioned"] = {"top1_joint_good_rate": random_top1_joint}
    return {"rows": rows, "summary": summary}
