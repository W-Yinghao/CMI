"""C32 selected-OACI regret and nearest joint-good anatomy."""
from __future__ import annotations

import numpy as np

from .landscape import by_trajectory


def _score_rank(cs, candidate):
    ordered = sorted(cs, key=lambda c: c["score"], reverse=True)
    for i, c in enumerate(ordered, start=1):
        if c["model_hash"] == candidate["model_hash"]:
            return i
    return None


def _nearest_joint(selected, joint):
    def order_value(c):
        return c.get("order") if c.get("order") is not None else c.get("epoch")
    return min(joint, key=lambda c: abs(float(order_value(c)) - float(order_value(selected))))


def selected_oaci_regret(rows) -> dict:
    by = by_trajectory(rows)
    per = []
    for traj, cs in sorted(by.items()):
        selected = [c for c in cs if c.get("selected_oaci")]
        joint = [c for c in cs if c["joint_good"]]
        if len(selected) != 1:
            per.append({"seed": traj[0], "target": traj[1], "level": traj[2],
                        "regime": traj[3], "category": "selected_missing_or_duplicate",
                        "has_joint_good": int(bool(joint))})
            continue
        s = selected[0]
        top = sorted(cs, key=lambda c: c["score"], reverse=True)
        row = {"seed": traj[0], "target": traj[1], "level": traj[2],
               "regime": traj[3],
               "has_joint_good": int(bool(joint)), "selected_joint_good": int(s["joint_good"]),
               "selected_score_rank": _score_rank(cs, s),
               "source_top1_joint": int(top[0]["joint_good"]),
               "source_top3_has_joint": int(any(c["joint_good"] for c in top[:3])),
               "source_top5_has_joint": int(any(c["joint_good"] for c in top[:5]))}
        if not joint:
            row.update({"category": "scarcity_no_joint_good"})
        else:
            nearest = _nearest_joint(s, joint)
            order_span = max(c.get("order", 0) for c in cs) - min(c.get("order", 0) for c in cs)
            order_distance = abs(float(nearest.get("order", nearest["epoch"])) - float(s.get("order", s["epoch"])))
            epoch_distance = abs(float(nearest["epoch"]) - float(s["epoch"]))
            row.update({
                "nearest_order_distance": order_distance,
                "nearest_order_distance_norm": (order_distance / order_span if order_span else 0.0),
                "nearest_epoch_distance": epoch_distance,
                "best_joint_score_rank": min(_score_rank(cs, c) for c in joint),
                "bacc_regret_to_best_joint": max(0.0, max(c["bacc"] for c in joint) - s["bacc"]),
                "nll_regret_to_best_joint": max(0.0, s["nll"] - min(c["nll"] for c in joint)),
                "ece_regret_to_best_joint": max(0.0, s["ece"] - min(c["ece"] for c in joint)),
            })
            if s["joint_good"]:
                row["category"] = "selected_joint_good"
            elif order_distance <= 1.0:
                row["category"] = "adjacent_near_miss"
            elif row["source_top5_has_joint"]:
                row["category"] = "source_top5_available_selection_missed"
            else:
                row["category"] = "source_rank_not_enriched_enough"
        per.append(row)
    n = len(per)
    cats = {}
    for r in per:
        cats[r["category"]] = cats.get(r["category"], 0) + 1
    with_joint = [r for r in per if r.get("has_joint_good")]
    misses_with_joint = [r for r in with_joint if not r.get("selected_joint_good")]

    def mean_key(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.mean(vals)) if vals else None

    def median_key(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.median(vals)) if vals else None

    def max_key(rows_, key):
        vals = [r[key] for r in rows_ if r.get(key) is not None]
        return float(np.max(vals)) if vals else None

    summary = {
        "n_trajectories": n,
        "selected_joint_hit_rate": mean_key(per, "selected_joint_good"),
        "selected_has_joint_trajectory_fraction": mean_key(per, "has_joint_good"),
        "source_top1_joint_hit_rate": mean_key(per, "source_top1_joint"),
        "source_top3_has_joint_rate": mean_key(per, "source_top3_has_joint"),
        "source_top5_has_joint_rate": mean_key(per, "source_top5_has_joint"),
        "mean_nearest_order_distance": mean_key(with_joint, "nearest_order_distance"),
        "median_nearest_order_distance": median_key(with_joint, "nearest_order_distance"),
        "mean_nearest_epoch_distance": mean_key(with_joint, "nearest_epoch_distance"),
        "median_nearest_epoch_distance": median_key(with_joint, "nearest_epoch_distance"),
        "miss_mean_nearest_order_distance": mean_key(misses_with_joint, "nearest_order_distance"),
        "miss_median_nearest_order_distance": median_key(misses_with_joint, "nearest_order_distance"),
        "mean_bacc_regret_to_best_joint": mean_key(with_joint, "bacc_regret_to_best_joint"),
        "median_bacc_regret_to_best_joint": median_key(with_joint, "bacc_regret_to_best_joint"),
        "max_bacc_regret_to_best_joint": max_key(with_joint, "bacc_regret_to_best_joint"),
        "mean_nll_regret_to_best_joint": mean_key(with_joint, "nll_regret_to_best_joint"),
        "mean_ece_regret_to_best_joint": mean_key(with_joint, "ece_regret_to_best_joint"),
        "category_counts": cats,
        "category_fractions": {k: v / n for k, v in cats.items()} if n else {},
    }
    return {"summary": summary, "per_trajectory": per}


def strategy_top1_regret(rows, score_maps) -> list:
    """Aggregate top-1 diagnostic regret for each scoring strategy. No per-candidate hashes are emitted."""
    by = by_trajectory(rows)
    out = []
    for name, score_getter in score_maps.items():
        hits, bregs, nregs, eregs = [], [], [], []
        for cs in by.values():
            chosen = max(cs, key=score_getter)
            joint = [c for c in cs if c["joint_good"]]
            hits.append(int(chosen["joint_good"]))
            if joint:
                bregs.append(max(0.0, max(c["bacc"] for c in joint) - chosen["bacc"]))
                nregs.append(max(0.0, chosen["nll"] - min(c["nll"] for c in joint)))
                eregs.append(max(0.0, chosen["ece"] - min(c["ece"] for c in joint)))
        out.append({"strategy": name, "top1_joint_hit_rate": float(np.mean(hits)) if hits else None,
                    "mean_bacc_regret_to_best_joint": float(np.mean(bregs)) if bregs else None,
                    "mean_nll_regret_to_best_joint": float(np.mean(nregs)) if nregs else None,
                    "mean_ece_regret_to_best_joint": float(np.mean(eregs)) if eregs else None})
    return out
