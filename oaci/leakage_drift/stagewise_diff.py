"""Stagewise localization of the first observed drift boundary."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


STAGES = (
    "feature_population",
    "support_graph",
    "fold_plan",
    "cell_mass_accounting",
    "atom_additive_aggregation",
    "persisted_aggregate_point_identity",
)


def _stage_status(row):
    statuses = []
    statuses.append(("feature_population", int(row["feature_population_hash_matches"]) == 1,
                     "feature population hash matches fold population"))
    statuses.append(("support_graph", bool(row["support_graph_hash"]), "support graph hash available"))
    statuses.append(("fold_plan", bool(row["fold_plan_hash"]), "fold plan hash available"))
    statuses.append(("cell_mass_accounting", al.as_float(row["max_class_mass_abs_diff"]) == 0.0,
                     "class overlap mass equals OOF mass"))
    statuses.append(("atom_additive_aggregation", al.as_float(row["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL,
                     "sum atoms equals recomputed point"))
    if row["split"] == "selection":
        statuses.append(("persisted_aggregate_point_identity",
                         al.as_float(row["point_abs_diff"]) <= schema.POINT_IDENTITY_TOL,
                         "recomputed point equals persisted C37 point at 1e-9"))
    else:
        statuses.append(("persisted_aggregate_point_identity", False,
                         "source-audit persisted aggregate point is unavailable"))
    return statuses


def localize(ctx):
    rows = []
    for r in ctx["tables"]["c39"]["identity"]:
        statuses = _stage_status(r)
        first = next((stage for stage, ok, _ in statuses if not ok), "none")
        rows.append({
            "job_key": r["job_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "candidate_role": r["candidate_role"],
            "candidate_order": r["candidate_order"],
            "candidate_id": r["candidate_id"],
            "split": r["split"],
            "feature_population_match": int(statuses[0][1]),
            "support_graph_available": int(statuses[1][1]),
            "fold_plan_available": int(statuses[2][1]),
            "cell_mass_accounting_pass": int(statuses[3][1]),
            "atom_additive_pass": int(statuses[4][1]),
            "persisted_aggregate_identity_pass": int(statuses[5][1]),
            "first_divergent_stage": first,
            "stagewise_interpretation": statuses[5][2] if first == "persisted_aggregate_point_identity" else
            next((note for stage, ok, note in statuses if stage == first), "all observed stages pass"),
        })
    selection = [r for r in rows if r["split"] == "selection"]
    summary = {
        "n_rows": len(rows),
        "n_selection_rows": len(selection),
        "selection_first_divergent_stage_counts": _counts([r["first_divergent_stage"] for r in selection]),
        "observed_semantic_mismatch_count": sum(
            1 for r in selection if r["first_divergent_stage"] not in ("persisted_aggregate_point_identity", "none")),
        "aggregate_vs_atom_path_divergence_count": sum(
            1 for r in selection if r["first_divergent_stage"] == "persisted_aggregate_point_identity"),
    }
    return {"rows": rows, "summary": summary}


def aggregate_path_diff(ctx):
    rows = []
    for r in al.selection_identity_rows(ctx):
        persisted = al.as_float(r["expected_point"])
        recomputed = al.as_float(r["recomputed_point"])
        atom_sum = al.as_float(r["atom_sum"])
        rows.append({
            "job_key": r["job_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "candidate_role": r["candidate_role"],
            "candidate_order": r["candidate_order"],
            "candidate_id": r["candidate_id"],
            "persisted_point": persisted,
            "recomputed_point": recomputed,
            "atom_sum": atom_sum,
            "recomputed_minus_persisted": recomputed - persisted,
            "atom_sum_minus_recomputed": atom_sum - recomputed,
            "first_path_difference": "persisted_aggregate_vs_recomputed_point"
            if abs(recomputed - persisted) > schema.POINT_IDENTITY_TOL else "none",
        })
    return {"rows": rows, "summary": {"n_rows": len(rows),
            "path_difference_count": sum(1 for r in rows if r["first_path_difference"] != "none")}}


def _counts(vals):
    out = {}
    for v in vals:
        out[v] = out.get(v, 0) + 1
    return out
