"""C32 deterministic taxonomy, tightened by the C32R remote-audit reinterpretation."""
from __future__ import annotations

from . import schema


def _topk(table, strategy, k):
    for r in table:
        if r.get("strategy") == strategy and r.get("k") == k:
            return r
    return None


def classify(landscape, random_base, source_topk, selected_regret, ladder) -> dict:
    cases, evidence = [], {}
    random_top1 = next(r for r in random_base["topk"] if r["k"] == 1)
    source_k1 = next(r for r in source_topk["topk"] if r["k"] == 1)
    source_k3 = next(r for r in source_topk["topk"] if r["k"] == 3)
    source_k5 = next(r for r in source_topk["topk"] if r["k"] == 5)
    selected = selected_regret["summary"]
    tu_gain = ladder["meta"]["target_unlabeled_pooled_auc_gain_over_source"]
    tu_top1_gain = ladder["meta"]["target_unlabeled_top1_gain_over_source"]
    grouped_gain = ladder["meta"]["target_grouped_pooled_auc_gain_over_source"]
    scarcity = selected["category_fractions"].get("scarcity_no_joint_good", 0.0)
    selected_random_gap = ((selected["selected_joint_hit_rate"] or 0) - (random_top1["hit_rate"] or 0))

    # C32R: the primary headline is dense/high-base-rate localization. This is NOT selector skill:
    # random trajectory-conditioned top-1 is already high, selected OACI sits near random, and nearest joint-good is
    # usually one order step away at the frozen primary margin.
    j8 = bool((random_top1["hit_rate"] or 0) >= schema.RANDOM_TOP1_NONTRIVIAL and
              abs(selected_random_gap) <= schema.SELECTED_RANDOM_TOL and
              selected["median_nearest_order_distance"] is not None and
              selected["median_nearest_order_distance"] <= schema.NEAR_ORDER_DISTANCE)
    evidence["J8"] = {"joint_good_rate": landscape["joint_good_rate"],
                      "random_top1_hit_rate": random_top1["hit_rate"],
                      "selected_joint_hit_rate": selected["selected_joint_hit_rate"],
                      "selected_minus_random_top1": selected_random_gap,
                      "median_nearest_order_distance": selected["median_nearest_order_distance"]}
    if j8:
        cases.append(schema.J8)

    j1 = bool((landscape["joint_good_rate"] or 0) >= schema.JOINT_COMMON_RATE and
              (landscape["trajectory_any_joint_fraction"] or 0) >= schema.TRAJECTORY_WITH_JOINT_FRACTION)
    evidence["J1"] = {"joint_good_rate": landscape["joint_good_rate"],
                      "trajectory_any_joint_fraction": landscape["trajectory_any_joint_fraction"]}
    if j1:
        cases.append(schema.J1)

    # Weak enrichment exists, but it is not reliable localization.
    j2 = bool((source_k1["hit_enrichment"] or 0) > 1.0 and
              (source_k1["hit_enrichment"] or 0) < schema.WEAK_TOPK_ENRICHMENT_MAX)
    evidence["J2"] = {"source_top1_hit_rate": source_k1["hit_rate"],
                      "random_top1_hit_rate": random_top1["hit_rate"],
                      "source_top1_enrichment": source_k1["hit_enrichment"],
                      "source_top3_enrichment": source_k3["hit_enrichment"],
                      "source_top5_enrichment": source_k5["hit_enrichment"]}
    if j2:
        cases.append(schema.J2)

    j3 = bool((source_k1["hit_enrichment"] or 0) < schema.WEAK_TOPK_ENRICHMENT_MAX and
              (source_k5["hit_enrichment"] or 0) <= schema.TOP5_NOT_BETTER_THAN_RANDOM)
    evidence["J3"] = {"source_top1_hit_rate": source_k1["hit_rate"],
                      "source_top1_enrichment": source_k1["hit_enrichment"],
                      "source_top5_hit_rate": source_k5["hit_rate"],
                      "source_top5_enrichment": source_k5["hit_enrichment"]}
    if j3:
        cases.append(schema.J3)

    # C32R: do NOT promote "far from joint-good" as a primary case. Median nearest distance=1 contradicts it at
    # the frozen primary margin; retain J4 only as an explicit future tail/margin-specific slot.
    j4 = False
    evidence["J4"] = {"selected_joint_hit_rate": selected["selected_joint_hit_rate"],
                      "random_top1_hit_rate": random_top1["hit_rate"],
                      "selected_minus_random_top1": selected_random_gap,
                      "scarcity_fraction": scarcity,
                      "median_nearest_order_distance": selected["median_nearest_order_distance"],
                      "not_primary_reason": "median nearest joint-good order distance is small; far-region claim is not established globally"}
    if j4:
        cases.append(schema.J4)

    j5 = bool(selected["median_nearest_order_distance"] is not None and
              selected["median_nearest_order_distance"] <= schema.NEAR_ORDER_DISTANCE)
    evidence["J5"] = {"median_nearest_order_distance": selected["median_nearest_order_distance"],
                      "mean_nearest_order_distance": selected["mean_nearest_order_distance"],
                      "median_nearest_epoch_distance": selected["median_nearest_epoch_distance"]}
    if j5:
        cases.append(schema.J5)

    j6 = bool(tu_gain is not None and tu_gain >= schema.TARGET_UNLABELED_POOLED_AUC_GAIN and
              tu_top1_gain is not None and tu_top1_gain <= 0.0)
    evidence["J6"] = {"target_unlabeled_pooled_auc_gain_over_source": tu_gain,
                      "target_unlabeled_top1_gain_over_source": tu_top1_gain}
    if j6:
        cases.append(schema.J6)

    j7 = bool(grouped_gain is not None and grouped_gain >= schema.GROUPED_POOLED_AUC_GAIN)
    evidence["J7"] = {"target_grouped_pooled_auc_gain_over_source": grouped_gain,
                      "non_deployable": True}
    if j7:
        cases.append(schema.J7)

    if not cases:
        cases.append(schema.J9)
    return {"cases": cases, "evidence": evidence}
