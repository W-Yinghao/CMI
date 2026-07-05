"""C14b — source->target instability diagnostics. Descriptive (not inferential): transfer correlations, an
anti-transfer index (ATI), a source-target instability score (STI), and harm localization. Operates on the
C12 SRC cells (per target/temp/level source-vs-target deltas) and the C10a audit->target correlations.

Margins: a delta counts as an improvement only if it beats 0 by `MARGIN` (NLL lower better, bAcc higher).
"""
from __future__ import annotations

import math

MARGIN = 1e-9
_EPS = 1e-12


def _clean(xs, ys):
    x, y = [], []
    for a, b in zip(xs, ys):
        if a is None or b is None:
            continue
        a, b = float(a), float(b)
        if a != a or b != b:
            continue
        x.append(a); y.append(b)
    return x, y


def _rank(v):
    order = sorted(range(len(v)), key=lambda i: v[i]); r = [0.0] * len(v); i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def pearson(xs, ys):
    x, y = _clean(xs, ys); n = len(x)
    if n < 3:
        return {"r": None, "n": n}
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    if sxx < _EPS or syy < _EPS:
        return {"r": None, "n": n, "note": "constant series"}
    return {"r": sxy / math.sqrt(sxx * syy), "n": n}


def spearman(xs, ys):
    x, y = _clean(xs, ys)
    if len(x) < 3:
        return {"rho": None, "n": len(x)}
    return {"rho": pearson(_rank(x), _rank(y)).get("r"), "n": len(x)}


def _sign_agreement(xs, ys):
    x, y = _clean(xs, ys)
    if not x:
        return None
    return sum(1 for a, b in zip(x, y) if (a > 0) == (b > 0)) / len(x)


def transfer_correlations(c12_cells, c10_part1) -> dict:
    """Source->target transfer (from C12 active cells) + audit->target transfer (from C10a)."""
    active = [c for c in c12_cells if not c["src_fallback_erm"]]
    d_src_nll = [(c["src_source_guard_nll"] - c["erm_source_guard_nll"])
                 if (c["src_source_guard_nll"] is not None and c["erm_source_guard_nll"] is not None) else None
                 for c in active]
    d_tgt_nll = [c["d_nll_vs_erm"] for c in active]
    d_tgt_bacc = [c["d_bacc_vs_erm"] for c in active]
    a2t = c10_part1.get("audit_to_target_transfer", {})
    return {
        "source_nll_to_target_nll": {"pearson": pearson(d_src_nll, d_tgt_nll), "spearman": spearman(d_src_nll, d_tgt_nll),
                                     "sign_agreement": _sign_agreement(d_src_nll, d_tgt_nll)},
        "source_nll_to_target_bacc": {"pearson": pearson(d_src_nll, d_tgt_bacc)},
        "audit_leakage_to_target_worst_bacc": a2t.get("corr_audit_vs_target_worst_bacc"),
        "audit_leakage_to_target_worst_nll": a2t.get("corr_audit_vs_target_worst_nll"),
        "note": ("positive source_nll->target_nll corr = source improvement REDUCES target loss (transfer); "
                 "here it is expected near zero / negative (anti-transfer)"),
    }


def instability_metrics(c12_cells) -> dict:
    """ATI (anti-transfer index), ATI severity, and STI (instability score) over the SRC active cells."""
    active = [c for c in c12_cells if not c["src_fallback_erm"]]
    src_improved = []
    anti = []
    for c in active:
        si = (c["src_source_guard_nll"] is not None and c["erm_source_guard_nll"] is not None
              and c["src_source_guard_nll"] < c["erm_source_guard_nll"] - MARGIN)
        tw = c["d_nll_vs_erm"] is not None and c["d_nll_vs_erm"] > MARGIN         # target NLL worsened
        src_improved.append(si)
        anti.append(si and tw)
    n_active = len(active)
    n_src_improved = sum(src_improved)
    n_anti = sum(anti)
    ati_nll = (n_anti / n_active) if n_active else None
    ati_severity = None
    harms = [c["d_nll_vs_erm"] for c, a in zip(active, anti) if a and c["d_nll_vs_erm"] is not None]
    if harms:
        ati_severity = sum(harms) / len(harms)
    # STI: fraction of source-improving units whose target metric worsens
    sti = (n_anti / n_src_improved) if n_src_improved else None
    return {"n_active": n_active, "n_source_improved": n_src_improved, "n_anti_transfer": n_anti,
            "ATI_NLL": ati_nll, "ATI_severity_mean_target_nll_harm": ati_severity,
            "source_target_instability_score": sti,
            "interpretation": ("STI≈1 means EVERY source-improving checkpoint harms target NLL "
                               "(source-side optimization anti-transfers)")}


def harm_localization(c12_cells) -> dict:
    """Group harm flags by target / level / temperature / method (SRC)."""
    def flags(c):
        return {"target_nll_blowup": bool(c["target_nll_blowup"]),
                "target_bacc_harmed": c["d_bacc_vs_erm"] is not None and c["d_bacc_vs_erm"] < -MARGIN,
                "target_nll_harmed": c["d_nll_vs_erm"] is not None and c["d_nll_vs_erm"] > MARGIN,
                "fallback_erm": bool(c["src_fallback_erm"])}
    by_target, by_level, by_temp = {}, {}, {}
    for c in c12_cells:
        f = flags(c)
        for k, v in (("target", c["target"]), ("level", c["level"]), ("temp", c["temp"])):
            d = {"target": by_target, "level": by_level, "temp": by_temp}[k]
            b = d.setdefault(str(v), {"cells": 0, "blowup": 0, "bacc_harmed": 0, "nll_harmed": 0, "fallback": 0})
            b["cells"] += 1; b["blowup"] += f["target_nll_blowup"]; b["bacc_harmed"] += f["target_bacc_harmed"]
            b["nll_harmed"] += f["target_nll_harmed"]; b["fallback"] += f["fallback_erm"]
    return {"by_target": by_target, "by_level": by_level, "by_temperature": by_temp,
            "per_cell": [{"target": c["target"], "temp": c["temp"], "level": c["level"], **flags(c)} for c in c12_cells]}
