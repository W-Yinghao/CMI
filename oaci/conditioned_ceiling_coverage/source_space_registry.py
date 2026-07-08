"""C49 fixed source-space registry and radii."""
from __future__ import annotations

from ..source_nonidentifiability import source_space
from . import artifact_loader as al
from . import schema


def registry(ctx):
    rows = []
    spaces = {}
    for source_space_name, families in schema.SOURCE_SPACES:
        fam = tuple(families)
        sp = source_space.build_space(ctx, families=fam or None)
        spaces[source_space_name] = sp
        rows.append({
            "source_space": source_space_name,
            "families": ",".join(fam) if fam else "all",
            "n_source_objectives": len(sp["specs"]),
            "objectives": ",".join(s["objective"] for s in sp["specs"]),
            "distance_metric": schema.PRIMARY_DISTANCE,
            "target_label_used": 0,
            "source_only": 1,
        })
    return {"rows": rows, "spaces": spaces}


def epsilon_radii(ctx, spaces):
    rows = []
    summary = {}
    for name, sp in spaces.items():
        vals = source_space.within_trajectory_pair_distances(ctx, sp)
        summary[name] = {}
        for q in schema.EPSILON_QUANTILES:
            radius = al.finite_quantile(vals, q)
            summary[name][q] = radius
            rows.append({
                "source_space": name,
                "epsilon_quantile": q,
                "epsilon_radius": radius,
                "distance_metric": schema.PRIMARY_DISTANCE,
                "computed_from": "within_trajectory_pair_distances",
            })
    return {"rows": rows, "summary": summary}
