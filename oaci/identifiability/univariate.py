"""C17-B — univariate source-signal identifiability. For each source-only signal, does ranking candidates by
it recover the target-accuracy-good checkpoints WITHIN each fold-level (the granularity at which the oracle
picks)? We report the mean within-fold-level Spearman with the target bAcc and NLL deltas, top-k enrichment,
sign agreement, and a within-(target,seed,level) permutation p-value (shuffle target labels inside each
fold-level; deterministic). The verdict per signal: does it identify ACCURACY, only CALIBRATION, or neither."""
from __future__ import annotations

import math

import numpy as np

from .signal_atlas import SIGNAL_AXIS, SOURCE_SIGNALS

_N_PERM = 300
_SIG = 0.05
_WEAK = 0.15                 # |mean within-fold rho| for a WEAK (perm-significant) ranking signal
_STRONG = 0.25              # |mean within-fold rho| for a usable/strong ranking signal


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


def _spearman(x, y):
    n = len(x)
    if n < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx, my = sum(rx) / n, sum(ry) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    sxx = sum((a - mx) ** 2 for a in rx); syy = sum((b - my) ** 2 for b in ry)
    if sxx < 1e-12 or syy < 1e-12:
        return None
    return sxy / (sxx * syy) ** 0.5


def _finite(v):
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))   # drop None, NaN, +/-inf


def _by_fold(rows):
    fl = {}
    for r in rows:
        fl.setdefault((r["seed"], r["target"], r["level"]), []).append(r)
    return fl


def _mean_within_fold_spearman(fold_levels, sig, tgt):
    vals = []
    for cs0 in fold_levels.values():
        cs = [c for c in cs0 if _finite(c[sig]) and _finite(c[tgt])]
        if len(cs) < 3:
            continue
        xs = [c[sig] for c in cs]; ys = [c[tgt] for c in cs]
        rho = _spearman(xs, ys)
        if rho is not None:
            vals.append(rho)
    return (sum(vals) / len(vals)) if vals else None, len(vals)


def _perm_p(fold_levels, sig, tgt, observed, seed=0):
    if observed is None:
        return None
    rng = np.random.RandomState(seed)
    ge = 0
    for _ in range(_N_PERM):
        vals = []
        for cs0 in fold_levels.values():
            cs = [c for c in cs0 if _finite(c[sig]) and _finite(c[tgt])]
            if len(cs) < 3:
                continue
            xs = [c[sig] for c in cs]; ys = [c[tgt] for c in cs]
            yp = list(ys); rng.shuffle(yp)
            rho = _spearman(xs, yp)
            if rho is not None:
                vals.append(rho)
        m = (sum(vals) / len(vals)) if vals else 0.0
        if abs(m) >= abs(observed):
            ge += 1
    return (ge + 1) / (_N_PERM + 1)


def _topk_enrichment(fold_levels, sig, k=3):
    """Fraction of the top-k candidates by |signal| (in the target-improving direction) that are bAcc-good,
    vs the base rate. Direction: bAcc-like signals ranked high; nll/ece/leakage/risk ranked low."""
    high_good = SIGNAL_AXIS.get(sig.replace("src__", "")) == "accuracy"   # higher accuracy signal -> pick high
    hit, tot, base_num, base_den = 0, 0, 0, 0
    for cs in fold_levels.values():
        good = [c for c in cs if c["tgt__target_bacc_good"]]
        base_num += len(good); base_den += len(cs)
        valid = [c for c in cs if _finite(c[sig])]
        if len(valid) < k:
            continue
        valid.sort(key=lambda c: (-c[sig] if high_good else c[sig]))
        top = valid[:k]
        hit += sum(1 for c in top if c["tgt__target_bacc_good"]); tot += len(top)
    return {"topk_good_rate": (hit / tot if tot else None), "base_rate": (base_num / base_den if base_den else None), "k": k}


def univariate_identifiability(rows, *, perm_seed=0) -> dict:
    fl = _by_fold(rows)
    per_signal = {}
    for s in SOURCE_SIGNALS:
        sig = "src__" + s
        rho_b, n = _mean_within_fold_spearman(fl, sig, "tgt__target_bacc_delta")
        rho_n, _ = _mean_within_fold_spearman(fl, sig, "tgt__target_nll_delta")
        p_b = _perm_p(fl, sig, "tgt__target_bacc_delta", rho_b, seed=perm_seed)
        enr = _topk_enrichment(fl, sig)
        sig_p = p_b is not None and p_b < _SIG
        strong_acc = rho_b is not None and abs(rho_b) >= _STRONG and sig_p
        weak_acc = rho_b is not None and _WEAK <= abs(rho_b) < _STRONG and sig_p
        ident_nll = rho_n is not None and abs(rho_n) >= _WEAK
        per_signal[s] = {"axis": SIGNAL_AXIS.get(s), "mean_within_fold_spearman_bacc": rho_b,
                         "mean_within_fold_spearman_nll": rho_n, "perm_p_bacc": p_b, "n_fold_levels": n,
                         "topk_enrichment": enr, "strong_accuracy_signal": bool(strong_acc),
                         "weak_accuracy_signal": bool(weak_acc), "identifies_target_nll": bool(ident_nll)}
    n_strong = sum(1 for v in per_signal.values() if v["strong_accuracy_signal"])
    n_weak = sum(1 for v in per_signal.values() if v["weak_accuracy_signal"])
    n_nll = sum(1 for v in per_signal.values() if v["identifies_target_nll"])
    # which signal family carries the (weak) accuracy signal, and how strong is the ORACLE's signal?
    oracle_sig = per_signal["source_audit_worst_bacc"]
    best_acc = max((abs(v["mean_within_fold_spearman_bacc"]) for v in per_signal.values()
                    if v["mean_within_fold_spearman_bacc"] is not None), default=None)
    acc_families = sorted({v["axis"] for k, v in per_signal.items() if v["strong_accuracy_signal"] or v["weak_accuracy_signal"]})
    if n_strong >= 1:
        verdict = "accuracy_identifiable_univariate"
    elif n_weak >= 1:
        verdict = "weak_accuracy_needs_multivariate"          # Case III candidate (weak, may combine)
    elif n_nll >= 1:
        verdict = "calibration_identifiable_only"             # Case II
    else:
        verdict = "source_unidentifiable_univariate"          # Case IV candidate
    return {"per_signal": per_signal, "n_strong_accuracy_signals": n_strong, "n_weak_accuracy_signals": n_weak,
            "n_signals_identify_nll": n_nll, "max_abs_accuracy_spearman": best_acc,
            "oracle_signal_spearman_bacc": oracle_sig["mean_within_fold_spearman_bacc"],
            "accuracy_signal_families": acc_families, "univariate_verdict": verdict,
            "n_fold_levels": len(fl), "n_candidates": len(rows),
            "note": ("C10's oracle used source_audit_worst_bacc (a WEAK target-accuracy signal); the strongest "
                     "univariate accuracy signals are source RISK/training signals, not the leakage-endpoint "
                     "family — but all are weak (|rho|<0.25). The multivariate probe decides Case I vs III vs IV.")}
