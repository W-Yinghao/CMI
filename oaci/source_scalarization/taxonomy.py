"""C43 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def classify(frontier, actionability, mult, conflict):
    best_id = mult["summary"]["best_scalarization_id"]
    best_key = (best_id, "top1", "primary_joint_good")
    best_metric = actionability["summary"][best_key]
    best_top1 = mult["summary"]["best_top1_joint_good_rate"]
    best_gain = mult["summary"]["best_top1_gain_vs_random"]
    reliable = bool(mult["summary"]["any_positive_scalarization_claim_allowed"])
    front_good = max(
        frontier["summary"]["joint_good_front_fraction"] or 0,
        frontier["summary"]["pareto_good_front_fraction"] or 0,
        frontier["summary"]["preference_robust_front_fraction"] or 0,
    )
    rejected_good = max(
        frontier["summary"]["joint_good_rejected_fraction"] or 0,
        frontier["summary"]["pareto_good_rejected_fraction"] or 0,
        frontier["summary"]["preference_robust_rejected_fraction"] or 0,
    )
    heterogeneous = (
        mult["summary"]["best_per_target_sign_consistency"] < schema.TARGET_SIGN_CONSISTENCY_GATE or
        mult["summary"]["best_bh_q_value"] >= 0.05)
    established = {
        schema.F1: front_good >= schema.FRONT_CONTAINS_GOOD_GATE,
        schema.F2: rejected_good >= schema.FRONT_REJECTS_GOOD_GATE and front_good < schema.FRONT_CONTAINS_GOOD_GATE,
        schema.F3: bool(conflict["summary"]["leakage_extreme_blocks_rank_frontier"]),
        schema.F4: not reliable,
        schema.F5: best_gain >= schema.BEST_GAIN_MIN and best_top1 < schema.WEAK_CEILING_TOP1_GATE,
        schema.F6: bool(heterogeneous),
        schema.F7: bool(conflict["summary"]["source_rank_leakage_tradeoff_real"]),
        schema.F8: not reliable,
        schema.F9: reliable,
        schema.F10: False,
    }
    evidence = {
        schema.F1: (
            f"front_good max joint/pareto/robust={frontier['summary']['joint_good_front_fraction']}/"
            f"{frontier['summary']['pareto_good_front_fraction']}/"
            f"{frontier['summary']['preference_robust_front_fraction']}"),
        schema.F2: f"max_rejected_good_fraction={rejected_good}",
        schema.F3: f"leakage_blocks_rank_better_fraction={conflict['summary']['leakage_blocks_rank_better_fraction']}",
        schema.F4: f"best_top1={best_top1}, reliable_positive_claim_allowed={reliable}",
        schema.F5: f"best={best_id}, top1={best_top1}, gain_vs_random={best_gain}",
        schema.F6: (
            f"best_sign_consistency={mult['summary']['best_per_target_sign_consistency']}, "
            f"best_bh_q={mult['summary']['best_bh_q_value']}"),
        schema.F7: f"mean_leakage_rank_spearman={conflict['summary']['mean_leakage_rank_spearman']}",
        schema.F8: "no fixed source-only scalarization passes reliability and multiplicity gates",
        schema.F9: f"best={best_id}, metric={best_metric}",
        schema.F10: "candidate registry and scalarization grid complete",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {"cases": [c for c in schema.ALL_CASES if established[c]], "case_rows": rows,
            "established": established, "evidence": evidence}
