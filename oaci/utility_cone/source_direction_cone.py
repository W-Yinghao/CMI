"""Source-score direction under endpoint utility cone categories."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def _index_simplex(rows):
    return {r["pair_id"]: r for r in rows if r["comparison"] == "nearest_continuous_better"}


def _pair_id_from_c34(r):
    return "|".join(map(str, (r["seed"], r["target"], r["level"], r.get("regime", ""), r["comparison"],
                             r.get("selected_order"), r.get("candidate_order"))))


def source_direction_by_cone(c34_pairs, simplex_rows):
    sim = _index_simplex(simplex_rows)
    pair_rows = []
    for p in c34_pairs:
        if p["comparison"] != "nearest_continuous_better":
            continue
        s = sim.get(_pair_id_from_c34(p))
        if s is None:
            continue
        source_delta = artifact_loader._as_float(p["source_score_delta"])
        alt_wins = s["fraction_weights_alt_beats_selected"] > 0
        if not alt_wins:
            direction = "no_utility_cone_regret"
        elif abs(source_delta) <= schema.SOURCE_FLAT_EPS:
            direction = "source_flat"
        elif source_delta < 0:
            direction = "source_favors_selected_misrank"
        else:
            direction = "source_favors_alternative"
        pair_rows.append({
            "pair_id": s["pair_id"], "seed": p["seed"], "target": p["target"], "level": p["level"],
            "regime": p.get("regime", ""), "utility_cone_category": s["utility_cone_category"],
            "fraction_weights_alt_beats_selected": s["fraction_weights_alt_beats_selected"],
            "source_score_delta": source_delta,
            "source_direction_case": direction,
            "source_misranking": int(direction == "source_favors_selected_misrank"),
            "source_agreement": int(direction == "source_favors_alternative"),
            "source_flat": int(direction == "source_flat"),
        })
    aggregate = []
    for cat in ("preference_robust_regret", "preference_dependent_regret", "narrow_scalarization_regret", "no_regret"):
        rs = [r for r in pair_rows if r["utility_cone_category"] == cat]
        if not rs:
            aggregate.append({"utility_cone_category": cat, "n_pairs": 0, "source_misranking_rate": None,
                              "source_agreement_rate": None, "source_flat_rate": None,
                              "random_baseline": 0.5, "mean_source_score_delta": None,
                              "per_target_sign_consistency": None})
            continue
        per_target = []
        for t in sorted({r["target"] for r in rs}):
            trs = [r for r in rs if r["target"] == t]
            per_target.append(np.mean([r["source_agreement"] for r in trs]))
        aggregate.append({
            "utility_cone_category": cat,
            "n_pairs": len(rs),
            "source_misranking_rate": float(np.mean([r["source_misranking"] for r in rs])),
            "source_agreement_rate": float(np.mean([r["source_agreement"] for r in rs])),
            "source_flat_rate": float(np.mean([r["source_flat"] for r in rs])),
            "random_baseline": 0.5,
            "mean_source_score_delta": float(np.mean([r["source_score_delta"] for r in rs])),
            "per_target_sign_consistency": float(np.mean([int(v > 0.5) for v in per_target])) if per_target else None,
        })
    summary = {r["utility_cone_category"]: r for r in aggregate}
    return {"summary": summary, "pair_rows": pair_rows, "aggregate": aggregate}
