"""Selected-OACI vs continuous-better local regret anatomy."""
from __future__ import annotations

import numpy as np

from ..local_boundary.artifact_loader import units
from . import endpoint_utility, schema


def _selected_index(cs):
    idx = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
    return idx[0] if len(idx) == 1 else None


def _order(c):
    return c.get("order") if c.get("order") is not None else c.get("epoch", 0)


def _nearest(cs, si, candidates):
    if not candidates:
        return None
    return min(candidates, key=lambda i: (abs(i - si), abs(float(_order(cs[i])) - float(_order(cs[si])))))


def _raw_dominates(a, b):
    av = endpoint_utility.endpoint_raw_vector(a)
    bv = endpoint_utility.endpoint_raw_vector(b)
    return bool(np.all(av >= bv) and np.any(av > bv))


def _continuous_better_indices(cs, si):
    s = cs[si]
    out = []
    for i, c in enumerate(cs):
        if i == si:
            continue
        min_gain = float(c["continuous_joint_min_margin"] - s["continuous_joint_min_margin"])
        norm_gain = float(s["endpoint_vector_norm_regret"] - c["endpoint_vector_norm_regret"])
        if min_gain > schema.STANDARDIZED_TINY_REGRET or (
                norm_gain > schema.STANDARDIZED_TINY_REGRET and np.any(endpoint_utility.endpoint_z_vector(c) >
                                                                        endpoint_utility.endpoint_z_vector(s))):
            out.append(i)
    return out


def _pareto_better_indices(cs, si):
    s = cs[si]
    out = []
    for i, c in enumerate(cs):
        if i == si:
            continue
        if _raw_dominates(c, s):
            out.append(i)
        elif c.get("pareto_distance") == 0.0 and (
                float(s["endpoint_vector_norm_regret"] - c["endpoint_vector_norm_regret"]) >
                schema.STANDARDIZED_TINY_REGRET):
            out.append(i)
    return out


def _target_oracle_index(cs):
    return max(range(len(cs)), key=lambda i: (float(cs[i]["continuous_joint_min_margin"]),
                                             -float(cs[i]["endpoint_vector_norm_regret"]),
                                             float(cs[i]["target_bacc_delta"]),
                                             float(cs[i]["target_nll_delta"]),
                                             float(cs[i]["target_ece_delta"])))


def selected_pair_targets(rows):
    """Internal row-reference targets for component/gauge audits."""
    out = []
    for key, cs in units(rows).items():
        si = _selected_index(cs)
        if si is None:
            continue
        comparisons = {
            "nearest_binary_joint_good": _nearest(cs, si, [i for i, c in enumerate(cs) if c.get("joint_good")]),
            "nearest_continuous_better": _nearest(cs, si, _continuous_better_indices(cs, si)),
            "nearest_pareto_better": _nearest(cs, si, _pareto_better_indices(cs, si)),
            "target_oracle": _target_oracle_index(cs),
        }
        for name, ci in comparisons.items():
            if ci is not None and ci != si:
                out.append({"unit": key, "comparison": name, "selected": cs[si], "candidate": cs[ci],
                            "selected_index": si, "candidate_index": ci})
            elif name == "target_oracle":
                out.append({"unit": key, "comparison": name, "selected": cs[si], "candidate": cs[si],
                            "selected_index": si, "candidate_index": si})
    return out


def _candidate_delta(selected, candidate):
    d = endpoint_utility.endpoint_delta(selected, candidate)
    d.update({
        "joint_min_margin_delta": float(candidate["continuous_joint_min_margin"] -
                                        selected["continuous_joint_min_margin"]),
        "pareto_distance_delta": float(candidate["pareto_distance"] - selected["pareto_distance"]),
        "pareto_distance_reduction": float(selected["pareto_distance"] - candidate["pareto_distance"]),
        "endpoint_vector_norm_regret_delta": float(candidate["endpoint_vector_norm_regret"] -
                                                   selected["endpoint_vector_norm_regret"]),
        "endpoint_vector_norm_regret_reduction": float(selected["endpoint_vector_norm_regret"] -
                                                       candidate["endpoint_vector_norm_regret"]),
        "dominated_hypervolume_regret_delta": float(candidate["dominated_hypervolume_regret"] -
                                                    selected["dominated_hypervolume_regret"]),
        "source_score_delta": float(candidate["score"] - selected["score"]),
        "R_src_delta": float(candidate["R_src"] - selected["R_src"]),
        "c30_source_rank_delta": float(candidate.get("c30_source_rank", np.nan) -
                                       selected.get("c30_source_rank", np.nan)),
        "target_gauge_delta": float((candidate.get("joint_margin") or 0.0) -
                                    (selected.get("joint_margin") or 0.0)),
        "target_margin_mean_delta": float(candidate.get("target_margin_mean", np.nan) -
                                          selected.get("target_margin_mean", np.nan)),
        "target_unlabeled_R3_delta": float(candidate.get("target_unlabeled_r3_score", np.nan) -
                                           selected.get("target_unlabeled_r3_score", np.nan)),
        "target_grouped_centered_delta": float(candidate.get("target_grouped_centered_score", np.nan) -
                                               selected.get("target_grouped_centered_score", np.nan)),
    })
    return d


def _case(comparison, selected, d):
    tiny = (
        abs(d["joint_min_margin_delta"]) <= schema.STANDARDIZED_TINY_REGRET and
        abs(d["endpoint_vector_norm_regret_reduction"]) <= schema.STANDARDIZED_TINY_REGRET
    )
    meaningful = (
        d["joint_min_margin_delta"] >= schema.STANDARDIZED_MEANINGFUL_REGRET or
        d["endpoint_vector_norm_regret_reduction"] >= schema.STANDARDIZED_MEANINGFUL_REGRET
    )
    if comparison == "nearest_binary_joint_good" and not selected.get("joint_good") and tiny:
        return "threshold_only"
    if meaningful and d["source_score_delta"] < -schema.SOURCE_FLAT_EPS:
        return "source_active_misranking"
    if meaningful and d["target_gauge_delta"] >= schema.GAUGE_JUMP_EPS and d["source_score_delta"] <= schema.SOURCE_FLAT_EPS:
        return "gauge_jump"
    if d["endpoint_tradeoff"]:
        return "endpoint_tradeoff"
    if meaningful:
        return "real_endpoint_regret"
    return "continuous_tiny_or_no_regret"


def selected_pair_regret(rows) -> dict:
    pairs, per_unit = [], []
    targets = selected_pair_targets(rows)
    for t in targets:
        key = t["unit"]
        s, c = t["selected"], t["candidate"]
        d = _candidate_delta(s, c)
        meaningful = int(d["joint_min_margin_delta"] >= schema.STANDARDIZED_MEANINGFUL_REGRET or
                         d["endpoint_vector_norm_regret_reduction"] >= schema.STANDARDIZED_MEANINGFUL_REGRET)
        tiny = int(abs(d["joint_min_margin_delta"]) <= schema.STANDARDIZED_TINY_REGRET and
                   abs(d["endpoint_vector_norm_regret_reduction"]) <= schema.STANDARDIZED_TINY_REGRET)
        row = {
            "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
            "comparison": t["comparison"], "selected_joint_good": int(s.get("joint_good", 0)),
            "candidate_joint_good": int(c.get("joint_good", 0)),
            "selected_order": s.get("order"), "candidate_order": c.get("order"),
            "order_distance": abs(t["candidate_index"] - t["selected_index"]),
            "epoch_distance": abs(float(c.get("epoch", 0)) - float(s.get("epoch", 0))),
            "selected_target_bacc_delta": s.get("target_bacc_delta"),
            "selected_target_nll_delta": s.get("target_nll_delta"),
            "selected_target_ece_delta": s.get("target_ece_delta"),
            "candidate_target_bacc_delta": c.get("target_bacc_delta"),
            "candidate_target_nll_delta": c.get("target_nll_delta"),
            "candidate_target_ece_delta": c.get("target_ece_delta"),
            **d,
            "meaningful_continuous_regret": meaningful,
            "tiny_continuous_difference": tiny,
            "threshold_artifact": int(t["comparison"] == "nearest_binary_joint_good" and
                                      not s.get("joint_good") and tiny),
        }
        row["pair_case"] = _case(t["comparison"], s, d)
        pairs.append(row)
    by_unit = {}
    for p in pairs:
        key = (p["seed"], p["target"], p["level"], p["regime"])
        by_unit.setdefault(key, {})[p["comparison"]] = p
    for key, comps in sorted(by_unit.items()):
        cont = comps.get("nearest_continuous_better") or comps.get("target_oracle")
        oracle = comps.get("target_oracle")
        binary = comps.get("nearest_binary_joint_good")
        per_unit.append({
            "seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
            "selected_joint_good": (binary or cont or oracle or {}).get("selected_joint_good"),
            "has_binary_joint_neighbor": int(binary is not None),
            "has_continuous_better": int(comps.get("nearest_continuous_better") is not None),
            "selected_to_continuous_better_joint_min_delta": None if cont is None else cont.get("joint_min_margin_delta"),
            "selected_to_continuous_better_norm_regret_reduction": None if cont is None else cont.get(
                "endpoint_vector_norm_regret_reduction"),
            "selected_to_oracle_joint_min_delta": None if oracle is None else oracle.get("joint_min_margin_delta"),
            "selected_to_oracle_norm_regret_reduction": None if oracle is None else oracle.get(
                "endpoint_vector_norm_regret_reduction"),
            "threshold_artifact": None if binary is None else binary.get("threshold_artifact"),
            "continuous_pair_case": None if cont is None else cont.get("pair_case"),
        })

    def mean(rows_, key):
        vals = [float(r[key]) for r in rows_ if r.get(key) is not None and endpoint_utility.finite(r.get(key))]
        return float(np.mean(vals)) if vals else None

    def median(rows_, key):
        vals = [float(r[key]) for r in rows_ if r.get(key) is not None and endpoint_utility.finite(r.get(key))]
        return float(np.median(vals)) if vals else None

    cont_pairs = [p for p in pairs if p["comparison"] == "nearest_continuous_better"]
    binary_misses = [p for p in pairs if p["comparison"] == "nearest_binary_joint_good" and
                     not p["selected_joint_good"]]
    meaningful = [p for p in cont_pairs if p["meaningful_continuous_regret"]]
    case_counts = {}
    for p in pairs:
        case_counts[p["pair_case"]] = case_counts.get(p["pair_case"], 0) + 1
    summary = {
        "n_units": len(per_unit),
        "n_selected_continuous_better_pairs": len(cont_pairs),
        "n_meaningful_continuous_regret_pairs": len(meaningful),
        "continuous_raw_pareto_nonworse_count": sum(
            int(p["target_bacc_delta"] >= 0 and p["target_nll_delta"] >= 0 and p["target_ece_delta"] >= 0)
            for p in cont_pairs),
        "continuous_raw_endpoint_backward_count": sum(
            int(p["target_bacc_delta"] < 0 or p["target_nll_delta"] < 0 or p["target_ece_delta"] < 0)
            for p in cont_pairs),
        "continuous_joint_min_negative_count": sum(int(p["joint_min_margin_delta"] < 0) for p in cont_pairs),
        "binary_miss_count": len(binary_misses),
        "binary_threshold_tiny_count": sum(int(p["threshold_artifact"]) for p in binary_misses),
        "binary_endpoint_tradeoff_count": sum(int(p["endpoint_tradeoff"]) for p in binary_misses),
        "binary_worse_by_scalar_or_norm_count": sum(
            int(p["joint_min_margin_delta"] < -schema.STANDARDIZED_TINY_REGRET or
                p["endpoint_vector_norm_regret_reduction"] < -schema.STANDARDIZED_TINY_REGRET)
            for p in binary_misses),
        "selected_joint_good_rate": mean(per_unit, "selected_joint_good"),
        "mean_continuous_joint_min_regret": mean(cont_pairs, "joint_min_margin_delta"),
        "median_continuous_joint_min_regret": median(cont_pairs, "joint_min_margin_delta"),
        "mean_endpoint_norm_regret_reduction": mean(cont_pairs, "endpoint_vector_norm_regret_reduction"),
        "median_endpoint_norm_regret_reduction": median(cont_pairs, "endpoint_vector_norm_regret_reduction"),
        "threshold_only_fraction_among_binary_misses": (
            float(np.mean([p["threshold_artifact"] for p in binary_misses])) if binary_misses else None),
        "real_endpoint_regret_fraction": (
            float(np.mean([p["meaningful_continuous_regret"] for p in cont_pairs])) if cont_pairs else None),
        "source_active_misranking_fraction": (
            float(np.mean([p["pair_case"] == "source_active_misranking" for p in meaningful])) if meaningful else None),
        "gauge_jump_case_fraction": (
            float(np.mean([p["pair_case"] == "gauge_jump" for p in meaningful])) if meaningful else None),
        "endpoint_tradeoff_fraction": (
            float(np.mean([p["endpoint_tradeoff"] for p in cont_pairs])) if cont_pairs else None),
        "median_continuous_order_distance": median(cont_pairs, "order_distance"),
        "median_continuous_epoch_distance": median(cont_pairs, "epoch_distance"),
        "pair_case_counts": case_counts,
    }
    return {"summary": summary, "pairs": pairs, "per_unit": per_unit}
