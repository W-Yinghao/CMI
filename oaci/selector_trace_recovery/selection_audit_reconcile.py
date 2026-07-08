"""C37 selection-UCL vs source-audit reconciliation."""
from __future__ import annotations

import numpy as np


def audit(comparisons, c36_selection_audit_rows):
    c36 = {r["pair_id"]: r for r in c36_selection_audit_rows}
    rows = []
    for r in comparisons:
        a = c36.get(r["pair_id"], {})
        audit_pref = a.get("audit_leakage_prefers", "unavailable")
        ucl_pref = r["ucl_prefers"]
        rows.append({
            "pair_id": r["pair_id"],
            "pair_key": r["pair_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selection_ucl_prefers": ucl_pref,
            "audit_leakage_prefers": audit_pref,
            "target_endpoint_prefers": "better",
            "selection_audit_inversion_exact": int(ucl_pref == "selected" and audit_pref in ("better", "flat")),
            "selection_target_conflict_exact": int(ucl_pref == "selected"),
        })
    return {"rows": rows, "summary": {
        "n_pairs": len(rows),
        "selection_audit_inversion_exact_rate": (
            float(np.mean([r["selection_audit_inversion_exact"] for r in rows])) if rows else None),
        "selection_target_conflict_exact_rate": (
            float(np.mean([r["selection_target_conflict_exact"] for r in rows])) if rows else None),
    }}

