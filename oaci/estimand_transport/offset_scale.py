"""C22 Q1 — target/regime score offset + scale variance components. Decomposes total score variance into
within-target (usable ranking) vs between-target (offset that breaks pooling), and checks whether the target
offset is confounded with the target base-rate."""
from __future__ import annotations

import numpy as np


def _var_components(rows, key):
    """Between-group vs within-group variance of the score, grouped by `key` (target or regime)."""
    groups = {}
    for r in rows:
        groups.setdefault(r[key], []).append(r["score"])
    means = {g: float(np.mean(v)) for g, v in groups.items()}
    grand = float(np.mean([r["score"] for r in rows]))
    n = len(rows)
    between = sum(len(v) * (means[g] - grand) ** 2 for g, v in groups.items()) / n
    within = sum(sum((x - means[g]) ** 2 for x in v) for g, v in groups.items()) / n
    total = between + within
    return {"between": between, "within": within, "between_fraction": (between / total if total > 0 else None),
            "group_means": {str(g): means[g] for g in means}, "group_scales":
            {str(g): float(np.std(groups[g])) for g in groups}}


def _corr(x, y):
    if len(x) < 3:
        return None
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.std() < 1e-9 or y.std() < 1e-9:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def offset_scale(rows) -> dict:
    """Per (mode) -> target and regime variance components + offset<->base-rate confound."""
    out = {}
    for mode in sorted({r["mode"] for r in rows}):
        mr = [r for r in rows if r["mode"] == mode]
        tvc = _var_components(mr, "target")
        rvc = _var_components(mr, "regime")
        # is the per-target score offset correlated with the per-target base rate? (a confound that inflates pooled offset)
        by_t = {}
        for r in mr:
            by_t.setdefault(r["target"], []).append(r["label"])
        offsets = []; brates = []
        for t, labs in by_t.items():
            if str(t) in tvc["group_means"]:
                offsets.append(tvc["group_means"][str(t)]); brates.append(float(np.mean(labs)))
        out[mode] = {"target_between_fraction": tvc["between_fraction"], "target_within": tvc["within"],
                     "target_between": tvc["between"], "regime_between_fraction": rvc["between_fraction"],
                     "target_offset_vs_baserate_corr": _corr(offsets, brates),
                     "target_scales": tvc["group_scales"], "regime_offsets": rvc["group_means"]}
    return out
