"""Gauge-locality diagnostics for continuous local regret."""
from __future__ import annotations

import numpy as np

from . import endpoint_utility, schema


def gauge_locality(selected_pairs, direction) -> dict:
    rows = []
    for p in selected_pairs["pairs"]:
        if p["comparison"] not in ("nearest_continuous_better", "nearest_pareto_better", "target_oracle"):
            continue
        meaningful = bool(p.get("meaningful_continuous_regret"))
        gauge_jump = bool(abs(float(p.get("target_gauge_delta") or 0.0)) >= schema.GAUGE_JUMP_EPS)
        source_insensitive = bool(abs(float(p.get("source_score_delta") or 0.0)) <= schema.SOURCE_FLAT_EPS)
        rows.append({
            "seed": p["seed"], "target": p["target"], "level": p["level"], "regime": p["regime"],
            "comparison": p["comparison"],
            "meaningful_continuous_regret": int(meaningful),
            "joint_min_margin_delta": p.get("joint_min_margin_delta"),
            "endpoint_norm_regret_reduction": p.get("endpoint_vector_norm_regret_reduction"),
            "target_gauge_delta": p.get("target_gauge_delta"),
            "target_margin_mean_delta": p.get("target_margin_mean_delta"),
            "source_score_delta": p.get("source_score_delta"),
            "target_unlabeled_R3_delta": p.get("target_unlabeled_R3_delta"),
            "target_grouped_centered_delta": p.get("target_grouped_centered_delta"),
            "gauge_jump": int(gauge_jump),
            "source_insensitive_to_gauge": int(gauge_jump and source_insensitive),
            "gauge_jump_unseen_by_source": int(meaningful and gauge_jump and source_insensitive),
        })
    meaningful = [r for r in rows if r["meaningful_continuous_regret"]]
    gauge = [r for r in meaningful if r["gauge_jump"]]
    nongauge = [r for r in meaningful if not r["gauge_jump"]]

    def mean(rs, key):
        vals = [float(r[key]) for r in rs if endpoint_utility.finite(r.get(key))]
        return float(np.mean(vals)) if vals else None

    tu_rows = direction["random_aggregate"]
    source_pm = {r["neighborhood"]: r for r in tu_rows if r["strategy"] == "source_score"}
    tu_summary_rows = []
    for r in tu_rows:
        if r["strategy"] not in ("source_score", "target_unlabeled_r3_score", "target_grouped_centered_score",
                                 "target_label_oracle_score"):
            continue
        src = source_pm.get(r["neighborhood"])
        tu_summary_rows.append({
            "strategy": r["strategy"], "info_class": r["info_class"], "neighborhood": r["neighborhood"],
            "mean_strategy_top1_regret": r["mean_strategy_top1_regret"],
            "mean_local_random_regret": r["mean_local_random_regret"],
            "regret_delta_vs_source": (r["mean_strategy_top1_regret"] - src["mean_strategy_top1_regret"]
                                       if src is not None else None),
            "non_source_only": int(r["strategy"] in ("target_unlabeled_r3_score", "target_grouped_centered_score",
                                                     "target_label_oracle_score")),
        })

    tu_pm1 = next((r for r in tu_summary_rows
                   if r["strategy"] == "target_unlabeled_r3_score" and r["neighborhood"] == "pm1"), None)
    source_pm1 = next((r for r in tu_summary_rows
                       if r["strategy"] == "source_score" and r["neighborhood"] == "pm1"), None)
    summary = {
        "n_gauge_rows": len(rows),
        "meaningful_regret_gauge_jump_fraction": (
            float(np.mean([r["gauge_jump"] for r in meaningful])) if meaningful else None),
        "meaningful_regret_gauge_unseen_fraction": (
            float(np.mean([r["gauge_jump_unseen_by_source"] for r in meaningful])) if meaningful else None),
        "mean_joint_min_gain_with_gauge_jump": mean(gauge, "joint_min_margin_delta"),
        "mean_joint_min_gain_without_gauge_jump": mean(nongauge, "joint_min_margin_delta"),
        "target_unlabeled_pm1_regret_delta_vs_source": None if tu_pm1 is None else tu_pm1.get("regret_delta_vs_source"),
        "source_pm1_mean_regret": None if source_pm1 is None else source_pm1.get("mean_strategy_top1_regret"),
        "target_unlabeled_pm1_mean_regret": None if tu_pm1 is None else tu_pm1.get("mean_strategy_top1_regret"),
    }
    return {"summary": summary, "gauge_rows": rows, "target_unlabeled_rows": tu_summary_rows}
