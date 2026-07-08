"""C43 source-side Pareto frontier audit."""
from __future__ import annotations

from . import artifact_loader as al
from . import objective_registry, schema


def _objective_specs():
    reg = {field: (field, orientation) for field, _, orientation, used_pareto, _, _ in objective_registry.OBJECTIVES
           if used_pareto and field in schema.PARETO_OBJECTIVES}
    return [(field, reg[field][1]) for field in schema.PARETO_OBJECTIVES if field in reg]


def _oriented_vector(row, specs):
    return [objective_registry.oriented_value(row, field, orient) for field, orient in specs]


def _dominates(a, b):
    return all(x >= y - 1e-12 for x, y in zip(a, b)) and any(x > y + 1e-12 for x, y in zip(a, b))


def pareto_layers(rows):
    specs = _objective_specs()
    remaining = list(rows)
    layers = {}
    depth = 0
    while remaining:
        vecs = [(r, _oriented_vector(r, specs)) for r in remaining]
        front = []
        for r, v in vecs:
            if not any(q is not r and _dominates(u, v) for q, u in vecs):
                front.append(r)
        if not front:
            break
        for r in front:
            layers[r["candidate_order"]] = depth
        fset = set(id(r) for r in front)
        remaining = [r for r in remaining if id(r) not in fset]
        depth += 1
    return layers


def _source_rank_top(rows):
    return max(rows, key=lambda r: (float(r["source_rank_score"]), -int(r["candidate_order"])))


def _mean_layer(rows, layers):
    vals = [layers[r["candidate_order"]] for r in rows if r["candidate_order"] in layers]
    return al.finite_mean(vals)


def audit(ctx):
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        layers = pareto_layers(rs)
        front = [r for r in rs if layers.get(r["candidate_order"]) == 0]
        oaci = next(r for r in rs if int(r["selected_oaci"]) == 1)
        rank_top = _source_rank_top(rs)
        label_groups = {
            "joint_good": [r for r in rs if int(r["primary_joint_good"]) == 1],
            "pareto_good": [r for r in rs if int(r["pareto_good"]) == 1],
            "preference_robust_target_better": [
                r for r in rs if int(r["preference_robust_better_candidate"]) == 1],
        }
        out = {
            "trajectory_id": tid,
            "seed": seed,
            "target": target,
            "level": level,
            "regime": regime,
            "n_candidates": len(rs),
            "n_source_pareto_front": len(front),
            "source_pareto_front_fraction": len(front) / len(rs),
            "oaci_selected_on_source_front": int(layers.get(oaci["candidate_order"]) == 0),
            "oaci_selected_source_pareto_layer": layers.get(oaci["candidate_order"]),
            "source_rank_top_on_source_front": int(layers.get(rank_top["candidate_order"]) == 0),
            "source_rank_top_source_pareto_layer": layers.get(rank_top["candidate_order"]),
            "no_candidate_id_emitted": 1,
        }
        for label, group in label_groups.items():
            if group:
                out[f"{label}_count"] = len(group)
                out[f"{label}_front_fraction"] = (
                    sum(1 for r in group if layers.get(r["candidate_order"]) == 0) / len(group))
                out[f"{label}_mean_source_pareto_layer"] = _mean_layer(group, layers)
                out[f"{label}_source_pareto_rejected_fraction"] = 1.0 - out[f"{label}_front_fraction"]
            else:
                out[f"{label}_count"] = 0
                out[f"{label}_front_fraction"] = ""
                out[f"{label}_mean_source_pareto_layer"] = ""
                out[f"{label}_source_pareto_rejected_fraction"] = ""
        rows.append(out)
    summary = {
        "mean_front_fraction": al.finite_mean([r["source_pareto_front_fraction"] for r in rows]),
        "oaci_selected_front_rate": al.finite_mean([r["oaci_selected_on_source_front"] for r in rows]),
        "source_rank_top_front_rate": al.finite_mean([r["source_rank_top_on_source_front"] for r in rows]),
        "joint_good_front_fraction": al.finite_mean([r["joint_good_front_fraction"] for r in rows]),
        "pareto_good_front_fraction": al.finite_mean([r["pareto_good_front_fraction"] for r in rows]),
        "preference_robust_front_fraction": al.finite_mean(
            [r["preference_robust_target_better_front_fraction"] for r in rows]),
        "joint_good_rejected_fraction": al.finite_mean([r["joint_good_source_pareto_rejected_fraction"] for r in rows]),
        "pareto_good_rejected_fraction": al.finite_mean([r["pareto_good_source_pareto_rejected_fraction"] for r in rows]),
        "preference_robust_rejected_fraction": al.finite_mean(
            [r["preference_robust_target_better_source_pareto_rejected_fraction"] for r in rows]),
    }
    return {"rows": rows, "summary": summary}
