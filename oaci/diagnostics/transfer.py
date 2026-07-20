"""C10 Part 1 — transfer / failure-mode analysis on the SELECTED checkpoints (artifact-only). All quantities
are paired within a (seed, target, level) fold-level; deltas are OACI − baseline. Order-invariant (results
depend only on the fold-level SET). Correlations are Pearson + Spearman with None/NaN pairs dropped.

Leakage: lower is better (Δ<0 ⇒ OACI reduces leakage). bAcc: higher better. NLL/ECE: lower better.
"""
from __future__ import annotations

import math

_EPS = 1e-12


def _clean_pairs(xs, ys):
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
    order = sorted(range(len(v)), key=lambda i: v[i])
    r = [0.0] * len(v)
    i = 0
    while i < len(v):
        j = i
        while j + 1 < len(v) and v[order[j + 1]] == v[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            r[order[k]] = avg
        i = j + 1
    return r


def _pearson(xs, ys):
    x, y = _clean_pairs(xs, ys)
    n = len(x)
    if n < 3:
        return {"r": None, "n": n}
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x); syy = sum((b - my) ** 2 for b in y)
    if sxx < _EPS or syy < _EPS:
        return {"r": None, "n": n, "note": "zero variance"}
    return {"r": sxy / math.sqrt(sxx * syy), "n": n}


def _spearman(xs, ys):
    x, y = _clean_pairs(xs, ys)
    if len(x) < 3:
        return {"rho": None, "n": len(x)}
    p = _pearson(_rank(x), _rank(y))
    return {"rho": p.get("r"), "n": len(x)}


def _corr(xs, ys):
    return {"pearson": _pearson(xs, ys), "spearman": _spearman(xs, ys)}


def _stats(vals):
    v = [float(x) for x in vals if x is not None and float(x) == float(x)]
    if not v:
        return {"n": 0, "mean": None, "median": None, "min": None, "max": None}
    s = sorted(v); n = len(s); m = n // 2
    return {"n": n, "mean": sum(v) / n, "median": s[m] if n % 2 else 0.5 * (s[m - 1] + s[m]),
            "min": min(v), "max": max(v)}


# ---- per-fold-level extraction (OACI vs a baseline) ----
def _fl(folds):
    """Iterate (seed, target, level, methods-dict, k1)."""
    for f in folds:
        for L, lv in f["levels"].items():
            yield f["seed"], f["target"], L, lv["methods"], lv["k1"]


def _leak(md, method, kind, field="bootstrap_ucl"):
    return md[method][kind][field]


def _tgt(md, method, key):
    return md[method]["roles"]["target_audit"][key]


def _guard(md, method, key):
    return md[method]["roles"]["source_guard"][key]


def _delta(md, method, base, getter):
    a, b = getter(md, method), getter(md, base)
    return None if (a is None or b is None) else float(a) - float(b)


def selection_to_audit_optimism(folds, method="OACI", base="ERM") -> dict:
    """Q1: does the selection-time leakage improvement transfer to the held-out audit split?"""
    d_sel, d_audit, keys = [], [], []
    for s, t, L, md, _ in _fl(folds):
        d_sel.append(_delta(md, method, base, lambda m, x: _leak(m, x, "sel_leakage")))
        d_audit.append(_delta(md, method, base, lambda m, x: _leak(m, x, "audit_leakage")))
        keys.append((s, t, L))
    ds = _stats(d_sel); da = _stats(d_audit)
    n_sel = sum(1 for x in d_sel if x is not None and x < 0)
    n_audit = sum(1 for x in d_audit if x is not None and x < 0)
    return {"delta_selection_leakage": ds, "delta_audit_leakage": da,
            "n_selection_reduced": n_sel, "n_audit_reduced": n_audit, "n_fold_levels": len(keys),
            "corr_selection_vs_audit_delta": _corr(d_sel, d_audit),
            "interpretation_hint": "selection ≪ 0 but audit ≈ 0 ⇒ selection-optimism / criterion overfit"}


def audit_to_target_transfer(folds, method="OACI", base="ERM") -> dict:
    """Q2: does the held-out audit leakage change predict target worst-domain metric change?"""
    d_audit, tgt = [], {"worst_bacc": [], "worst_nll": [], "worst_ece": []}
    for s, t, L, md, _ in _fl(folds):
        d_audit.append(_delta(md, method, base, lambda m, x: _leak(m, x, "audit_leakage")))
        for k in tgt:
            tgt[k].append(_delta(md, method, base, lambda m, x, kk=k: _tgt(m, x, kk)))
    out = {"delta_audit_leakage": _stats(d_audit)}
    for k in tgt:
        # expected-if-leakage-helps: Δaudit<0 ↔ Δbacc>0 (neg corr) / Δnll<0 (pos corr) / Δece<0 (pos corr)
        out[f"delta_target_{k}"] = _stats(tgt[k])
        out[f"corr_audit_vs_target_{k}"] = _corr(d_audit, tgt[k])
    out["interpretation_hint"] = ("if |r| small / wrong sign ⇒ audit leakage reduction is orthogonal to "
                                  "downstream worst-domain DG (points toward case C)")
    return out


def risk_tradeoff(folds, method="OACI", base="ERM") -> dict:
    """Q: did OACI trade source risk / larger lambda / later epoch for target degradation?"""
    lam, epoch, r_src_gap, d_bacc, d_nll = [], [], [], [], []
    for s, t, L, md, _ in _fl(folds):
        lam.append(md[method]["selected_lambda"]); epoch.append(md[method]["selected_epoch"])
        rs_a, rs_b = md[method]["R_src"], md[base]["R_src"]
        r_src_gap.append(None if (rs_a is None or rs_b is None) else float(rs_a) - float(rs_b))
        d_bacc.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_bacc")))
        d_nll.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_nll")))
    return {"lambda": _stats(lam), "selected_epoch": _stats(epoch), "R_src_gap_OACI_minus_ERM": _stats(r_src_gap),
            "corr_lambda_vs_delta_target_worst_bacc": _corr(lam, d_bacc),
            "corr_lambda_vs_delta_target_worst_nll": _corr(lam, d_nll),
            "corr_R_src_gap_vs_delta_target_worst_bacc": _corr(r_src_gap, d_bacc),
            "corr_epoch_vs_delta_target_worst_bacc": _corr(epoch, d_bacc)}


def level_effect(folds, method="OACI", base="ERM") -> dict:
    """Q4: level-0 vs missing-cell level-1."""
    out = {}
    for L in sorted({L for _, _, L, _, _ in _fl(folds)}):
        db, dn, da = [], [], []
        for s, t, LL, md, _ in _fl(folds):
            if LL != L:
                continue
            db.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_bacc")))
            dn.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_nll")))
            da.append(_delta(md, method, base, lambda m, x: _leak(m, x, "audit_leakage")))
        out[str(L)] = {"level": L, "delta_target_worst_bacc": _stats(db),
                       "delta_target_worst_nll": _stats(dn), "delta_audit_leakage": _stats(da)}
    return out


def method_comparison(folds, method="OACI", baselines=("ERM", "global_lpc", "uniform")) -> dict:
    """Q5: OACI vs each baseline on target worst-domain bAcc/NLL/ECE + audit leakage."""
    out = {}
    for base in baselines:
        db, dn, de, dl = [], [], [], []
        for s, t, L, md, _ in _fl(folds):
            db.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_bacc")))
            dn.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_nll")))
            de.append(_delta(md, method, base, lambda m, x: _tgt(m, x, "worst_ece")))
            dl.append(_delta(md, method, base, lambda m, x: _leak(m, x, "audit_leakage")))
        out[base] = {"delta_target_worst_bacc": _stats(db), "delta_target_worst_nll": _stats(dn),
                     "delta_target_worst_ece": _stats(de), "delta_audit_leakage": _stats(dl),
                     "n_bacc_improved": sum(1 for x in db if x is not None and x > 0),
                     "n_bacc_harmed": sum(1 for x in db if x is not None and x < 0)}
    return out


def harm_localization(folds, method="OACI", base="ERM") -> dict:
    """Q6: which (seed, target) target subjects lose bAcc/NLL under OACI, and how concentrated is the harm."""
    per = []
    for s, t, L, md, _ in _fl(folds):
        db = _delta(md, method, base, lambda m, x: _tgt(m, x, "worst_bacc"))
        dn = _delta(md, method, base, lambda m, x: _tgt(m, x, "worst_nll"))
        per.append({"seed": s, "target": t, "level": L, "delta_worst_bacc": db, "delta_worst_nll": dn})
    harmed_bacc = [p for p in per if p["delta_worst_bacc"] is not None and p["delta_worst_bacc"] < 0]
    harmed_nll = [p for p in per if p["delta_worst_nll"] is not None and p["delta_worst_nll"] < 0]
    tot_bacc_loss = -sum(p["delta_worst_bacc"] for p in harmed_bacc) if harmed_bacc else 0.0
    worst5 = sorted(harmed_bacc, key=lambda p: p["delta_worst_bacc"])[:5]
    top5_share = ((-sum(p["delta_worst_bacc"] for p in worst5) / tot_bacc_loss)
                  if tot_bacc_loss > _EPS else None)
    # per target-subject aggregation across seeds/levels
    by_t = {}
    for p in per:
        if p["delta_worst_bacc"] is not None:
            by_t.setdefault(p["target"], []).append(p["delta_worst_bacc"])
    subj = {str(t): _stats(v) for t, v in sorted(by_t.items())}
    return {"n_fold_levels": len(per), "n_harmed_bacc": len(harmed_bacc), "n_harmed_nll": len(harmed_nll),
            "total_bacc_loss": tot_bacc_loss, "top5_harm_share_of_total_loss": top5_share,
            "worst5_fold_levels": worst5, "per_target_subject_delta_worst_bacc": subj, "per_fold_level": per}


def run_all_transfer(folds) -> dict:
    return {"selection_to_audit_optimism": selection_to_audit_optimism(folds),
            "audit_to_target_transfer": audit_to_target_transfer(folds),
            "risk_tradeoff": risk_tradeoff(folds), "level_effect": level_effect(folds),
            "method_comparison": method_comparison(folds), "harm_localization": harm_localization(folds)}
