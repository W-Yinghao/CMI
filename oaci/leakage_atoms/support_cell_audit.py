"""Support-cell and estimability artifact audit for C39."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import schema


def audit(point_atom_rows):
    by_pair = defaultdict(list)
    for r in point_atom_rows:
        by_pair[r["pair_id"]].append(r)
    rows = []
    for pair_id, rs in sorted(by_pair.items()):
        masses = np.asarray([float(r["cell_mass"]) for r in rs], dtype=float)
        low_cut = float(np.quantile(masses, schema.LOW_MASS_QUANTILE)) if masses.size else 0.0
        pos_sum = float(sum(float(r["positive_selected_advantage"]) for r in rs))
        low_mass_pos = 0.0
        support_edge_pos = 0.0
        top = max(rs, key=lambda r: float(r["positive_selected_advantage"])) if rs else None
        for r in rs:
            low = float(r["cell_mass"]) < low_cut
            edge = int(r["support_edge"]) == 1
            adv = float(r["positive_selected_advantage"])
            if low:
                low_mass_pos += adv
            if edge:
                support_edge_pos += adv
        first = rs[0]
        rows.append({
            "pair_id": pair_id,
            "pair_key": first["pair_key"],
            "seed": first["seed"],
            "target": first["target"],
            "level": first["level"],
            "regime": first["regime"],
            "selected_order": first["selected_order"],
            "better_order": first["better_order"],
            "n_atoms": len(rs),
            "low_mass_cut": low_cut,
            "positive_advantage_sum": pos_sum,
            "low_mass_positive_share": low_mass_pos / pos_sum if pos_sum > 0 else 0.0,
            "support_edge_positive_share": support_edge_pos / pos_sum if pos_sum > 0 else 0.0,
            "dominant_atom_id": top["atom_id"] if top else "",
            "dominant_atom_positive_share": top["positive_advantage_share"] if top else 0.0,
            "dominant_atom_support_count": top["support_count"] if top else "",
            "dominant_atom_support_m": top["support_m"] if top else "",
            "dominant_atom_cell_mass": top["cell_mass"] if top else "",
            "dominant_atom_low_mass": int(float(top["cell_mass"]) < low_cut) if top else 0,
            "dominant_atom_support_edge": int(top["support_edge"]) if top else 0,
            "bootstrap_atom_variance_available": 0,
            "support_artifact_flag": int(
                (low_mass_pos / pos_sum if pos_sum > 0 else 0.0) >= schema.SUPPORT_ARTIFACT_SHARE_GATE or
                (support_edge_pos / pos_sum if pos_sum > 0 else 0.0) >= schema.SUPPORT_ARTIFACT_SHARE_GATE),
        })
    summary = {
        "n_pairs": len(rows),
        "mean_low_mass_positive_share": al.finite_mean([r["low_mass_positive_share"] for r in rows]),
        "mean_support_edge_positive_share": al.finite_mean([r["support_edge_positive_share"] for r in rows]),
        "dominant_atom_low_mass_fraction": al.finite_mean([r["dominant_atom_low_mass"] for r in rows]),
        "dominant_atom_support_edge_fraction": al.finite_mean([r["dominant_atom_support_edge"] for r in rows]),
        "support_artifact_pair_count": sum(int(r["support_artifact_flag"]) for r in rows),
        "support_artifact_pair_fraction": (
            sum(int(r["support_artifact_flag"]) for r in rows) / len(rows) if rows else None),
    }
    return {"rows": rows, "summary": summary}
