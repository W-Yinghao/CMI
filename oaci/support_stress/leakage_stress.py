"""C18-R (H5) — support-aware leakage estimability / abstention under support degradation. For each regime,
apply the (dual-side) masks to every fold-level's committed support graphs and measure whether the support-
aware leakage machinery would ESTIMATE or ABSTAIN (non-estimable): a masked cell family that loses its last
comparable class is reported as non-estimable, NOT smoothed. H5 holds if, as cells become rare/non-estimable/
missing, the fraction of estimable fold-levels drops (the framework abstains rather than hallucinating
unsupported conditional invariance). No probe fitting here — pure support-graph structure."""
from __future__ import annotations

import numpy as np

from . import masks, schema, support_metrics
from . import source_signal_recompute as ssr
from . import stress_plan as sp


def _mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def leakage_estimability_stress(extract_dir, *, boundary_classes, n_perturb=2, folds=None) -> dict:
    fold_dirs = folds if folds is not None else ssr._list_folds(extract_dir)
    acc = {r: {"n": 0, "src_elig_removed": [], "src_comp_lost": [], "audit_comp_lost": [],
               "src_estimable": [], "audit_estimable": [], "mass_removed": []} for r in schema.REGIME_ORDER}
    for (seed, target) in fold_dirs:
        for level in ssr._levels(extract_dir, seed, target):
            fld = ssr.load_fold_level(extract_dir, seed, target, level)
            for r in schema.REGIME_ORDER:
                a = acc[r]; a["n"] += 1
                for side, sg in (("src", fld["support_source"]), ("audit", fld["support_audit"])):
                    if sg is None:
                        continue
                    na, _ = ssr._regime_name_actions(r, sg, boundary_classes=boundary_classes, seed=seed,
                                                     target=target, level=level, n_perturb=n_perturb)
                    masked = masks.apply_to_support_graph(na, sg)
                    loss = support_metrics.estimability_loss(sg, masked)
                    estimable = loss["any_comparable_remaining"]
                    a[f"{side}_estimable"].append(1.0 if estimable else 0.0)
                    if side == "src":
                        a["src_elig_removed"].append(loss["eligible_cells_removed"])
                        a["src_comp_lost"].append(loss["comparable_classes_lost"])
                        a["mass_removed"].append(loss["mass_removed_fraction"])
                    else:
                        a["audit_comp_lost"].append(loss["comparable_classes_lost"])
    per_regime = {}
    for r, a in acc.items():
        per_regime[r] = {"n_fold_levels": a["n"], "severity": schema.REGIME_SEVERITY[r],
                         "mean_src_eligible_cells_removed": _mean(a["src_elig_removed"]),
                         "mean_src_comparable_classes_lost": _mean(a["src_comp_lost"]),
                         "mean_audit_comparable_classes_lost": _mean(a["audit_comp_lost"]),
                         "mean_mass_removed_fraction": _mean(a["mass_removed"]),
                         "source_estimable_fraction": _mean(a["src_estimable"]),
                         "audit_estimable_fraction": _mean(a["audit_estimable"]),
                         "any_estimable": (_mean(a["src_estimable"]) or 0.0) > 0.0}
    # abstention grows with severity iff estimable-fraction is monotone-decreasing on the deleting regimes
    deleting = ("S3_nonestimable_cells", "S4_missing_cells", "S5_block_class_by_domain",
                "S6_boundary_aligned_mask", "S7_random_matched_mask")
    est_del = [per_regime[r]["source_estimable_fraction"] for r in deleting
               if per_regime[r]["source_estimable_fraction"] is not None]
    s0 = per_regime["S0_full_support"]["source_estimable_fraction"]
    abstention = bool(est_del and s0 is not None and min(est_del) < s0)
    return {"per_regime": per_regime,
            "s0_source_estimable_fraction": s0,
            "min_deleting_estimable_fraction": (min(est_del) if est_del else None),
            "abstains_under_degradation": abstention,
            "note": ("support-aware leakage reports non-estimability (abstains) as masked cell families lose "
                     "their last comparable class; unsupported cells are never smoothed or imputed.")}
