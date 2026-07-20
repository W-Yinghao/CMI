"""C44 objective registry inherited from C43."""
from __future__ import annotations

from ..source_scalarization import objective_registry as c43_objectives
from . import artifact_loader as al


def inherited_registry(ctx):
    rows = c43_objectives.registry(ctx)["rows"]
    return {"rows": rows, "summary": {r["objective"]: r for r in rows}}


def source_pareto_specs(ctx):
    rows = inherited_registry(ctx)["rows"]
    specs = []
    for r in rows:
        if int(r["used_for_source_pareto"]) and int(r["n_available"]) == int(r["n_candidate_rows"]):
            specs.append({
                "objective": r["objective"],
                "family": r["family"],
                "orientation": r["orientation"],
            })
    return specs


def oriented(row, spec):
    value = float(row[spec["objective"]])
    return -value if spec["orientation"] == "lower" else value


def family_specs(ctx, families):
    famset = set(families)
    return [s for s in source_pareto_specs(ctx) if s["family"] in famset]
