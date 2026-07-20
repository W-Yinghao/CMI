"""C39 additive identity gate summaries."""
from __future__ import annotations

from . import schema


def audit(identity_rows):
    rows = []
    for r in identity_rows:
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
            "expected_point": r["expected_point"],
            "recomputed_point": r["recomputed_point"],
            "point_abs_diff": r["point_abs_diff"],
            "selected_capacity": r["selected_capacity"],
            "atom_sum": r["atom_sum"],
            "additive_abs_diff": r["additive_abs_diff"],
            "max_class_mass_abs_diff": r["max_class_mass_abs_diff"],
            "point_identity_pass": int(
                r["split"] != "selection" or float(r["point_abs_diff"]) <= schema.POINT_IDENTITY_TOL),
            "atom_additive_identity_pass": int(float(r["additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL),
            "identity_pass": r["identity_pass"],
            "support_graph_hash": r["support_graph_hash"],
            "fold_plan_hash": r["fold_plan_hash"],
            "bootstrap_plan_hash": r["bootstrap_plan_hash"],
            "population_hash": r["population_hash"],
            "feature_population_hash_matches": r["feature_population_hash_matches"],
            "target_labels_loaded_for_replay": r["target_labels_loaded_for_replay"],
            "n_atoms": r["n_atoms"],
        })
    selection = [r for r in rows if r["split"] == "selection"]
    audit_rows = [r for r in rows if r["split"] == "source_audit"]
    summary = {
        "n_rows": len(rows),
        "n_selection_candidates": len(selection),
        "n_selection_identity_pass": sum(int(r["identity_pass"]) for r in selection),
        "selection_identity_pass": bool(selection and all(int(r["identity_pass"]) for r in selection)),
        "n_source_audit_candidates": len(audit_rows),
        "n_source_audit_additive_pass": sum(int(r["atom_additive_identity_pass"]) for r in audit_rows),
        "source_audit_additive_pass": bool(audit_rows and
                                           all(int(r["atom_additive_identity_pass"]) for r in audit_rows)),
        "max_selection_point_abs_diff": max([float(r["point_abs_diff"]) for r in selection], default=None),
        "max_selection_additive_abs_diff": max([float(r["additive_abs_diff"]) for r in selection], default=None),
        "max_audit_additive_abs_diff": max([float(r["additive_abs_diff"]) for r in audit_rows], default=None),
    }
    return {"rows": rows, "summary": summary}
