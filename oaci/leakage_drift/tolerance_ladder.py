"""Diagnostic-only tolerance ladder for C40."""
from __future__ import annotations

from collections import defaultdict

from . import artifact_loader as al
from . import schema


def compute(manifest_rows):
    selection = [r for r in manifest_rows if r["split"] == "selection" and r["persisted_point_available"]]
    rows = []
    for tol in schema.TOLERANCE_LADDER:
        passed = [r for r in selection if float(r["abs_drift"]) <= tol]
        by_role = defaultdict(int)
        by_role_total = defaultdict(int)
        for r in selection:
            by_role_total[r["candidate_role"]] += 1
            if float(r["abs_drift"]) <= tol:
                by_role[r["candidate_role"]] += 1
        rows.append({
            "tolerance": tol,
            "n_selection_candidates": len(selection),
            "n_pass": len(passed),
            "pass_fraction": len(passed) / len(selection) if selection else None,
            "selected_role_pass": by_role.get("selected", 0),
            "selected_role_total": by_role_total.get("selected", 0),
            "better_role_pass": by_role.get("better", 0),
            "better_role_total": by_role_total.get("better", 0),
            "diagnostic_only": 1,
            "elevates_atom_claims": 0,
        })
    summary = {
        "pass_counts": {str(r["tolerance"]): r["n_pass"] for r in rows},
        "all_pass_at_1e_3": bool(rows and rows[-1]["n_pass"] == rows[-1]["n_selection_candidates"]),
        "all_pass_at_frozen_1e_9": bool(rows and rows[0]["n_pass"] == rows[0]["n_selection_candidates"]),
    }
    return {"rows": rows, "summary": summary}

