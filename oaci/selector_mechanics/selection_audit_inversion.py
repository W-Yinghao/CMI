"""C36 selection-vs-audit inversion audit."""
from __future__ import annotations

import numpy as np


def audit(pair_rows):
    rows = []
    for r in pair_rows:
        selection_pref = r["selection_leakage_point_prefers"]
        audit_pref = r["audit_leakage_point_prefers"]
        audit_endpoint_pref = r["source_audit_endpoint_prefers"]
        rows.append({
            "pair_id": r["pair_id"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selection_leakage_prefers": selection_pref,
            "audit_leakage_prefers": audit_pref,
            "source_audit_endpoint_prefers": audit_endpoint_pref,
            "target_endpoint_prefers": "better",
            "selection_to_audit_inversion": int(selection_pref == "selected" and audit_pref in ("better", "flat")),
            "audit_to_target_inversion": int(audit_pref == "selected"),
            "local_leakage_target_conflict": int(selection_pref == "selected"),
        })
    n = len(rows)
    mean = lambda k: float(np.mean([r[k] for r in rows])) if rows else None
    return {
        "rows": rows,
        "summary": {
            "n_pairs": n,
            "selection_to_audit_inversion_rate": mean("selection_to_audit_inversion"),
            "audit_to_target_inversion_rate": mean("audit_to_target_inversion"),
            "local_leakage_target_conflict_rate": mean("local_leakage_target_conflict"),
        },
    }

