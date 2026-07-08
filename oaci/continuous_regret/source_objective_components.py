"""Diagnostic decomposition of source-objective component conflicts."""
from __future__ import annotations

import numpy as np

from . import endpoint_utility, local_direction, schema, selected_pair_regret


def _component_delta(selected, candidate, spec):
    if not spec.get("available", True):
        return np.nan
    return local_direction.component_value(candidate, spec) - local_direction.component_value(selected, spec)


def _target_gain(selected, candidate):
    return float(candidate["continuous_joint_min_margin"] - selected["continuous_joint_min_margin"])


def source_objective_conflict(rows) -> dict:
    targets = [t for t in selected_pair_regret.selected_pair_targets(rows)
               if t["comparison"] in ("nearest_continuous_better", "target_oracle")]
    conflict_rows = []
    for t in targets:
        selected, candidate = t["selected"], t["candidate"]
        gain = _target_gain(selected, candidate)
        if gain < schema.STANDARDIZED_MEANINGFUL_REGRET:
            continue
        for spec in schema.SOURCE_COMPONENTS:
            delta = _component_delta(selected, candidate, spec)
            available = bool(spec.get("available", True) and endpoint_utility.finite(delta))
            if not available:
                direction = "unavailable"
            elif abs(delta) <= schema.COMPONENT_FLAT_EPS:
                direction = "flat_or_too_weak"
            elif delta < 0:
                direction = "wrong_direction"
            else:
                direction = "correct_direction"
            conflict_rows.append({
                "seed": t["unit"][0], "target": t["unit"][1], "level": t["unit"][2], "regime": t["unit"][3],
                "comparison": t["comparison"], "component": spec["component"], "component_family": spec["family"],
                "available": available, "target_joint_min_gain": gain,
                "target_norm_regret_reduction": float(selected["endpoint_vector_norm_regret"] -
                                                       candidate["endpoint_vector_norm_regret"]),
                "component_delta": delta if available else None,
                "direction_case": direction,
            })

    aggregate = []
    for spec in schema.SOURCE_COMPONENTS:
        rs = [r for r in conflict_rows if r["component"] == spec["component"]]
        avail = [r for r in rs if r["available"]]
        if not avail:
            aggregate.append({"component": spec["component"], "component_family": spec["family"],
                              "available": False, "n_pairs": 0, "wrong_direction_fraction": None,
                              "flat_fraction": None, "correct_fraction": None, "mean_component_delta": None,
                              "mean_target_joint_min_gain": None})
            continue
        aggregate.append({
            "component": spec["component"], "component_family": spec["family"], "available": True,
            "n_pairs": len(avail),
            "wrong_direction_fraction": float(np.mean([r["direction_case"] == "wrong_direction" for r in avail])),
            "flat_fraction": float(np.mean([r["direction_case"] == "flat_or_too_weak" for r in avail])),
            "correct_fraction": float(np.mean([r["direction_case"] == "correct_direction" for r in avail])),
            "mean_component_delta": float(np.mean([r["component_delta"] for r in avail])),
            "mean_target_joint_min_gain": float(np.mean([r["target_joint_min_gain"] for r in avail])),
        })

    by_component = {r["component"]: r for r in aggregate}
    leakage = [by_component.get("selection_leakage"), by_component.get("audit_leakage")]
    risk = [by_component.get("R_src"), by_component.get("source_guard_nll"), by_component.get("source_audit_nll")]
    leakage_wrong = [r["wrong_direction_fraction"] for r in leakage if r and r["wrong_direction_fraction"] is not None]
    risk_wrong = [r["wrong_direction_fraction"] for r in risk if r and r["wrong_direction_fraction"] is not None]
    source = by_component.get("source_score", {})
    summary = {
        "n_meaningful_component_pairs": len({(r["seed"], r["target"], r["level"], r["regime"], r["comparison"])
                                             for r in conflict_rows if r["available"]}),
        "source_score_wrong_direction_fraction": source.get("wrong_direction_fraction"),
        "source_score_flat_fraction": source.get("flat_fraction"),
        "source_score_correct_fraction": source.get("correct_fraction"),
        "leakage_mean_wrong_direction_fraction": float(np.mean(leakage_wrong)) if leakage_wrong else None,
        "risk_mean_wrong_direction_fraction": float(np.mean(risk_wrong)) if risk_wrong else None,
    }
    return {"summary": summary, "rows": conflict_rows, "aggregate": aggregate}
