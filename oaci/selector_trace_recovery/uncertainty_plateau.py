"""C37 UCL plateau and uncertainty classification."""
from __future__ import annotations

import numpy as np

from . import artifact_loader as al
from . import schema


def audit(comparisons):
    rows = []
    for r in comparisons:
        d = al.as_float(r.get("ucl_delta_better_minus_selected"))
        plateau = int(al.finite(d) and abs(d) <= schema.UCL_PLATEAU_EPS)
        rows.append({
            "pair_id": r["pair_id"],
            "pair_key": r["pair_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selected_order": r["selected_order"],
            "better_order": r["better_order"],
            "ucl_delta_better_minus_selected": r.get("ucl_delta_better_minus_selected"),
            "ucl_plateau_eps": schema.UCL_PLATEAU_EPS,
            "ucl_plateau": plateau,
            "ucl_prefers": r["ucl_prefers"],
            "uncertainty_class": "ucl_plateau" if plateau else f"ucl_prefers_{r['ucl_prefers']}",
        })
    n = len(rows)
    return {"rows": rows, "summary": {
        "n_pairs": n,
        "ucl_plateau_fraction": float(np.mean([r["ucl_plateau"] for r in rows])) if rows else None,
    }}

