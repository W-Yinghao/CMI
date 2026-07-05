"""C18 — support-graph metrics + estimability loss per regime (feeds H5 and the severity axis). All read
from the (masked) SupportGraph; no target information. Unsupported cells are counted, never imputed."""
from __future__ import annotations

import numpy as np


def support_graph_metrics(sg) -> dict:
    present = np.asarray(sg.present, dtype=bool); eligible = np.asarray(sg.eligible, dtype=bool)
    mass = np.asarray(sg.cell_mass, dtype=np.float64); pref = np.asarray(sg.reference_prior, dtype=np.float64)
    pm = float(mass[present].sum())
    return {"n_domains": int(sg.n_domains), "n_classes": int(sg.n_classes),
            "n_present_cells": int(present.sum()), "n_eligible_cells": int(eligible.sum()),
            "n_comparable_classes": int(len(sg.comparable_classes)),
            "n_singleton_classes": int(len(sg.singleton_classes)),
            "n_empty_classes": int(len(sg.empty_classes)),
            "n_coupling_components": int(len(sg.coupling_components)),
            "comparable_prior_mass": float(sum(pref[c] for c in sg.comparable_classes)),
            "eligible_cell_fraction": (float(eligible.sum()) / float(present.sum()) if present.any() else 0.0),
            "eligible_mass_fraction": (float(mass[eligible].sum()) / pm if pm > 0 else 0.0),
            "total_cell_mass": float(mass.sum())}


def estimability_loss(base_sg, masked_sg) -> dict:
    b, m = support_graph_metrics(base_sg), support_graph_metrics(masked_sg)
    base_elig = set(map(tuple, np.argwhere(np.asarray(base_sg.eligible, dtype=bool)).tolist()))
    mask_elig = set(map(tuple, np.argwhere(np.asarray(masked_sg.eligible, dtype=bool)).tolist()))
    removed = base_elig - mask_elig
    return {"d_eligible_cells": m["n_eligible_cells"] - b["n_eligible_cells"],
            "eligible_cells_removed": int(len(removed)),
            "d_comparable_classes": m["n_comparable_classes"] - b["n_comparable_classes"],
            "comparable_classes_lost": int(b["n_comparable_classes"] - m["n_comparable_classes"]),
            "d_coupling_components": m["n_coupling_components"] - b["n_coupling_components"],
            "became_singleton": sorted(set(masked_sg.singleton_classes) - set(base_sg.singleton_classes)),
            "became_empty": sorted(set(masked_sg.empty_classes) - set(base_sg.empty_classes)),
            "mass_removed_fraction": (1.0 - m["total_cell_mass"] / b["total_cell_mass"]) if b["total_cell_mass"] > 0 else 0.0,
            "eligible_mass_fraction_after": m["eligible_mass_fraction"],
            "comparable_prior_mass_after": m["comparable_prior_mass"],
            "any_comparable_remaining": m["n_comparable_classes"] >= 1}
