"""C22 Q4 — feature-level explanation of non-transport. For each frozen robust-core feature: how much of its
variance is between-target OFFSET (breaks pooling) vs within-target RANK contribution (carries signal), and how
stable its level is across regimes. NOT feature selection -- an interpretive decomposition of the FROZEN set."""
from __future__ import annotations

import numpy as np

from . import schema
from .epoch_confound import _spearman


def _between_fraction(vals, groups):
    means = {g: np.mean([vals[i] for i in idx]) for g, idx in groups.items()}
    grand = float(np.mean(vals)); n = len(vals)
    between = sum(len(idx) * (means[g] - grand) ** 2 for g, idx in groups.items()) / n
    within = sum(sum((vals[i] - means[g]) ** 2 for i in idx) for g, idx in groups.items()) / n
    total = between + within
    return (between / total) if total > 0 else None, {str(g): float(means[g]) for g in means}


def feature_shift(rows) -> dict:
    inr = [r for r in rows if r["mode"] == "in_regime"]
    per_feature = {}
    tgroups = {}; rgroups = {}
    for i, r in enumerate(inr):
        tgroups.setdefault(r["target"], []).append(i); rgroups.setdefault(r["regime"], []).append(i)
    for f in schema.ROBUST_CORE:
        col = "feat__" + f
        vals = [r.get(col) for r in inr]
        if any(v is None or (isinstance(v, float) and v != v) for v in vals):
            vals = None
        if vals is None:
            per_feature[f] = {"target_between_fraction": None, "regime_between_fraction": None,
                              "within_target_label_spearman": None, "usable_ranking": None, "offset_dominated": None}
            continue
        vals = np.array(vals, float)
        tbf, _ = _between_fraction(vals, tgroups)
        rbf, regime_means = _between_fraction(vals, rgroups)
        # within-target Spearman(feature, label)
        wt = []
        for t, idx in tgroups.items():
            if len(idx) >= 5:
                sp = _spearman(vals[idx], [inr[i]["label"] for i in idx])
                if sp is not None:
                    wt.append(sp)
        wt_mean = float(np.mean([abs(x) for x in wt])) if wt else None
        per_feature[f] = {"target_between_fraction": tbf, "regime_between_fraction": rbf,
                          "within_target_label_spearman": wt_mean,
                          "usable_ranking": bool(wt_mean is not None and wt_mean >= 0.10),
                          "offset_dominated": bool(tbf is not None and tbf >= schema.OFFSET_DOMINATED_FRACTION)}
    n = len(per_feature)
    n_offset = sum(1 for v in per_feature.values() if v["offset_dominated"])
    n_rank = sum(1 for v in per_feature.values() if v["usable_ranking"])
    return {"per_feature": per_feature, "n_features": n, "n_offset_dominated": n_offset, "n_usable_ranking": n_rank,
            "offset_dominated_fraction": (n_offset / n if n else None),
            "note": "interpretive decomposition of the FROZEN robust-core set; NOT feature selection."}
