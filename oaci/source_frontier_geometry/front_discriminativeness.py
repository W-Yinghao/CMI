"""C44 front membership discriminativeness audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import objective_registry, pareto_nulls


LABELS = (
    ("primary_joint_good", "joint_good"),
    ("pareto_good", "pareto_good"),
    ("preference_robust_better_candidate", "preference_robust"),
)


def audit(ctx):
    specs = objective_registry.source_pareto_specs(ctx)
    rows = []
    co_rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        mat = pareto_nulls.oriented_matrix(rs, specs)
        front, _, _ = pareto_nulls.dominance_stats(mat)
        front_idx = [i for i, v in enumerate(front) if v]
        not_idx = [i for i, v in enumerate(front) if not v]
        target_bad_front = [
            i for i in front_idx if not int(rs[i]["primary_joint_good"])
        ]
        target_good_front = [
            i for i in front_idx if int(rs[i]["primary_joint_good"])
        ]
        co_rows.append({
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "n_candidates": len(rs),
            "n_front": len(front_idx),
            "n_not_front": len(not_idx),
            "n_joint_good_front": len(target_good_front),
            "n_joint_bad_front": len(target_bad_front),
            "front_contains_joint_good": int(len(target_good_front) > 0),
            "front_contains_joint_bad": int(len(target_bad_front) > 0),
            "front_contains_both_good_and_bad": int(len(target_good_front) > 0 and len(target_bad_front) > 0),
            "target_labels_diagnostic_only": 1,
        })
        for field, label in LABELS:
            p_front = al.finite_mean([rs[i][field] for i in front_idx])
            p_not = al.finite_mean([rs[i][field] for i in not_idx])
            base = sum(int(r[field]) for r in rs) / len(rs)
            rows.append({
                "trajectory_id": tid,
                "label": label,
                "n_front": len(front_idx),
                "n_not_front": len(not_idx),
                "p_label_given_front": p_front,
                "p_label_given_not_front": p_not,
                "trajectory_baseline": base,
                "front_enrichment_over_trajectory": p_front / base if base > 0 else "",
                "front_minus_not_front": "" if p_not is None else p_front - p_not,
                "target_labels_diagnostic_only": 1,
            })
    summary_rows = []
    for _, label in LABELS:
        rs = [r for r in rows if r["label"] == label]
        summary_rows.append({
            "label": label,
            "n_trajectories": len(rs),
            "mean_p_label_given_front": al.finite_mean([r["p_label_given_front"] for r in rs]),
            "mean_p_label_given_not_front": al.finite_mean([r["p_label_given_not_front"] for r in rs]),
            "mean_trajectory_baseline": al.finite_mean([r["trajectory_baseline"] for r in rs]),
            "mean_front_enrichment_over_trajectory": al.finite_mean(
                [r["front_enrichment_over_trajectory"] for r in rs]),
            "mean_front_minus_not_front": al.finite_mean([r["front_minus_not_front"] for r in rs]),
        })
    co_summary = {
        "front_contains_both_good_and_bad_fraction": al.finite_mean(
            [r["front_contains_both_good_and_bad"] for r in co_rows]),
        "mean_front_fraction": al.finite_mean([r["n_front"] / r["n_candidates"] for r in co_rows]),
        "mean_not_front_count": al.finite_mean([r["n_not_front"] for r in co_rows]),
    }
    return {"rows": summary_rows, "trajectory_rows": rows, "cooccupancy_rows": co_rows,
            "summary": {r["label"]: r for r in summary_rows}, "cooccupancy_summary": co_summary}
