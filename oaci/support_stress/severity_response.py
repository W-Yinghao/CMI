"""C18 — severity response. Assembles the per-regime H2 (probe), H3 (calibration-vs-accuracy visibility),
H4 (boundary), H5 (leakage estimability) summaries into a severity-ordered table and a monotonicity readout.
Order-invariant by construction (each regime is computed independently; results are keyed by regime, never by
computation order)."""
from __future__ import annotations

import numpy as np

from . import schema


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        for k in range(i, j + 1):
            r[order[k]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return r


def _spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    return (sxy / (sxx * syy) ** 0.5) if sxx > 1e-12 and syy > 1e-12 else None


def severity_response(probe_by_regime, boundary_by_regime, leakage_by_regime, axis_by_regime=None) -> dict:
    rows = []
    for r in schema.REGIME_ORDER:
        p = probe_by_regime.get(r, {}); b = boundary_by_regime.get(r, {}); lk = leakage_by_regime.get(r, {})
        ax = (axis_by_regime or {}).get(r, {})
        rows.append({"regime": r, "severity": schema.REGIME_SEVERITY[r],
                     "loto_auc": p.get("loto_auc"), "beats_permutation": p.get("beats_permutation"),
                     "permutation_p": p.get("permutation_p"), "n_used": p.get("n_used"),
                     "boundary_corr": b.get("source_target_recall_delta_corr"),
                     "leakage_source_estimable_fraction": lk.get("source_estimable_fraction"),
                     "accuracy_visibility": ax.get("accuracy_visibility"),
                     "calibration_visibility": ax.get("calibration_visibility")})
    # monotone trend of LOTO AUC and boundary corr vs severity (Spearman over regimes with finite values)
    sev = [x["severity"] for x in rows]

    def trend(key):
        pairs = [(x["severity"], x[key]) for x in rows if x[key] is not None]
        if len(pairs) < 3:
            return None
        return _spearman([a for a, _ in pairs], [b for _, b in pairs])

    return {"severity_rows": rows, "loto_auc_vs_severity_spearman": trend("loto_auc"),
            "boundary_corr_vs_severity_spearman": trend("boundary_corr"),
            "leakage_estimable_vs_severity_spearman": trend("leakage_source_estimable_fraction"),
            "order_invariant": True,
            "note": "keyed by regime; independent of computation order (test asserts permutation invariance)."}
