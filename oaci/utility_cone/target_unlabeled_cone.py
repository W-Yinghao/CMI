"""Target-unlabeled local behavior under endpoint utility-cone categories."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema
from .source_direction_cone import _index_simplex, _pair_id_from_c34


def target_unlabeled_by_cone(c34_pairs, simplex_rows):
    sim = _index_simplex(simplex_rows)
    pair_rows = []
    for p in c34_pairs:
        if p["comparison"] != "nearest_continuous_better":
            continue
        s = sim.get(_pair_id_from_c34(p))
        if s is None:
            continue
        delta = artifact_loader._as_float(p["target_unlabeled_R3_delta"])
        alt_wins = s["fraction_weights_alt_beats_selected"] > 0
        if not alt_wins:
            direction = "no_utility_cone_regret"
        elif abs(delta) <= schema.TU_FLAT_EPS:
            direction = "target_unlabeled_flat"
        elif delta < 0:
            direction = "target_unlabeled_favors_selected_misrank"
        else:
            direction = "target_unlabeled_favors_alternative"
        pair_rows.append({
            "pair_id": s["pair_id"], "seed": p["seed"], "target": p["target"], "level": p["level"],
            "regime": p.get("regime", ""), "utility_cone_category": s["utility_cone_category"],
            "fraction_weights_alt_beats_selected": s["fraction_weights_alt_beats_selected"],
            "target_unlabeled_R3_delta": delta,
            "target_unlabeled_direction_case": direction,
            "target_unlabeled_misranking": int(direction == "target_unlabeled_favors_selected_misrank"),
            "target_unlabeled_agreement": int(direction == "target_unlabeled_favors_alternative"),
            "target_unlabeled_flat": int(direction == "target_unlabeled_flat"),
            "non_source_only": 1,
        })
    aggregate = []
    for cat in ("preference_robust_regret", "preference_dependent_regret", "narrow_scalarization_regret", "no_regret"):
        rs = [r for r in pair_rows if r["utility_cone_category"] == cat]
        if not rs:
            aggregate.append({"utility_cone_category": cat, "n_pairs": 0,
                              "target_unlabeled_misranking_rate": None,
                              "target_unlabeled_agreement_rate": None,
                              "target_unlabeled_flat_rate": None, "random_baseline": 0.5,
                              "mean_target_unlabeled_R3_delta": None, "non_source_only": 1})
            continue
        aggregate.append({
            "utility_cone_category": cat,
            "n_pairs": len(rs),
            "target_unlabeled_misranking_rate": float(np.mean([r["target_unlabeled_misranking"] for r in rs])),
            "target_unlabeled_agreement_rate": float(np.mean([r["target_unlabeled_agreement"] for r in rs])),
            "target_unlabeled_flat_rate": float(np.mean([r["target_unlabeled_flat"] for r in rs])),
            "random_baseline": 0.5,
            "mean_target_unlabeled_R3_delta": float(np.mean([r["target_unlabeled_R3_delta"] for r in rs])),
            "non_source_only": 1,
        })
    return {"summary": {r["utility_cone_category"]: r for r in aggregate},
            "pair_rows": pair_rows, "aggregate": aggregate}
