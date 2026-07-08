"""C40 drift manifest from C39 identity rows."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def build(ctx):
    rows = []
    for r in ctx["tables"]["c39"]["identity"]:
        persisted = al.as_float(r.get("expected_point"))
        recomputed = al.as_float(r.get("recomputed_point"))
        signed = recomputed - persisted if al.finite(persisted) and al.finite(recomputed) else None
        abs_drift = abs(signed) if signed is not None else None
        rows.append({
            "job_key": r["job_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "candidate_role": r["candidate_role"],
            "candidate_order": r["candidate_order"],
            "candidate_id": r["candidate_id"],
            "split": r["split"],
            "persisted_point_available": int(al.finite(persisted)),
            "persisted_point": persisted,
            "recomputed_point": recomputed,
            "signed_drift_recomputed_minus_persisted": signed,
            "abs_drift": abs_drift,
            "pass_1e_9": int(abs_drift is not None and abs_drift <= schema.POINT_IDENTITY_TOL),
            "selected_capacity": r["selected_capacity"],
            "atom_sum": al.as_float(r["atom_sum"]),
            "additive_abs_diff": al.as_float(r["additive_abs_diff"]),
            "atom_additive_identity_pass": int(al.as_float(r["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL),
            "support_graph_hash": r["support_graph_hash"],
            "fold_plan_hash": r["fold_plan_hash"],
            "bootstrap_plan_hash": r["bootstrap_plan_hash"],
            "population_hash": r["population_hash"],
            "feature_population_hash_matches": r["feature_population_hash_matches"],
            "target_labels_loaded_for_replay": r["target_labels_loaded_for_replay"],
        })
    selection = [r for r in rows if r["split"] == "selection"]
    persisted = [r for r in selection if r["persisted_point_available"]]
    summary = {
        "n_rows": len(rows),
        "n_selection_candidates": len(selection),
        "n_selection_persisted_available": len(persisted),
        "n_selection_pass_1e_9": sum(r["pass_1e_9"] for r in persisted),
        "selection_identity_pass": bool(persisted and all(r["pass_1e_9"] for r in persisted)),
        "max_abs_drift": max([float(r["abs_drift"]) for r in persisted], default=None),
        "mean_abs_drift": al.finite_mean([r["abs_drift"] for r in persisted]),
        "n_additive_pass": sum(r["atom_additive_identity_pass"] for r in rows),
        "all_additive_pass": bool(rows and all(r["atom_additive_identity_pass"] for r in rows)),
    }
    return {"rows": rows, "summary": summary}

