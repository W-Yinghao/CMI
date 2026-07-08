"""Diagnostic target-gauge residual witnesses for C45."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema


def _corr(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if al.finite(x) and al.finite(y)]
    if not pairs:
        return None
    xs = np.asarray([p[0] for p in pairs], dtype=float)
    ys = np.asarray([p[1] for p in pairs], dtype=float)
    n = len(pairs)
    if n < 2 or float(np.std(xs)) <= 1e-12 or float(np.std(ys)) <= 1e-12:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])


def audit(witness):
    rows = []
    for r in witness["source_equivalent_pairs"]:
        large_gauge = int(float(r["target_gauge_gap"]) >= schema.GAUGE_JUMP_EPS)
        rows.append({
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "candidate_order": r["candidate_order"],
            "neighbor_order": r["neighbor_order"],
            "source_distance_primary": r["source_distance_primary"],
            "target_utility_gap": r["target_utility_gap"],
            "target_gauge_gap": r["target_gauge_gap"],
            "large_target_gauge_gap": large_gauge,
            "joint_good_disagreement": r["joint_good_disagreement"],
            "target_divergent": r["target_divergent"],
            "target_gauge_non_source_only": 1,
            "c27_class_conditioned_confidence_global_available": 1,
            "c29_representation_projection_global_available": 1,
            "candidate_level_representation_projection_atom_available": 0,
        })
    xs = [r["target_gauge_gap"] for r in rows]
    ys = [r["target_utility_gap"] for r in rows]
    n_eq = len(witness["source_equivalent_pairs"])
    large = [r for r in rows if int(r["large_target_gauge_gap"])]
    summary = {
        "n_source_equivalent_divergent_pairs": n_eq,
        "n_source_equivalent_gauge_witnesses": len(large),
        "source_equivalent_gauge_witness_fraction": len(large) / n_eq if n_eq else None,
        "mean_target_gauge_gap": al.finite_mean(xs),
        "mean_target_utility_gap": al.finite_mean(ys),
        "gauge_gap_target_utility_gap_corr": _corr(xs, ys),
        "large_gauge_mean_target_utility_gap": al.finite_mean(
            [r["target_utility_gap"] for r in large]),
        "target_gauge_candidate_level_source_available": False,
        "target_gauge_non_source_only": True,
        "c27_c29_global_origin_available": True,
        "pair_local_representation_projection_atom_available": False,
    }
    return {"rows": rows, "summary": summary}
