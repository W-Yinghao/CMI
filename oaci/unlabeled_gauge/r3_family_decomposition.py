"""C25 Q1 — decompose the weak R3 recovery across the FROZEN target-unlabeled families. family-only + leave-one-
family-out + EXACT Shapley (3 families -> 8 subset evaluations, permutation-average marginal gap-closure) +
per-regime stability. No feature selection: family membership is fixed; only pre-declared unions are scored."""
from __future__ import annotations

import itertools
import math

import numpy as np

from ..information_ladder import target_unlabeled_features as tuf
from ..score_gauge import offset_model
from ..score_gauge.ceiling_ladder import _pooled_auc
from . import schema


def _subset_gauge(gauge_table, feats):
    return {t: {"gauge": {f: gauge_table[t]["gauge"][f] for f in feats}, "offset": gauge_table[t]["offset"]}
            for t in gauge_table}


def _fit_gap(mr, gauge_table, feats, raw, oracle):
    """LOTO offset fit on a feature subset -> pooled-AUC gap closed toward the target-centered oracle."""
    if not feats:
        return 0.0, None
    sg = _subset_gauge(gauge_table, feats)
    fit = offset_model.fit_offsets(sg, names=list(feats))
    oh = fit["offset_hat_loto"]
    auc = _pooled_auc(mr, subtract=lambda r: oh.get(r["target"], 0.0))
    gap = ((auc - raw) / (oracle - raw)) if (auc is not None and (oracle - raw) > 1e-6) else None
    return gap, fit["loto_r2"]


def family_only(rows, gauge_table, mode, raw, oracle) -> list:
    mr = [r for r in rows if r["mode"] == mode]
    out = []
    for fam, feats in schema.FAMILIES.items():
        perm = tuf.r3_loto_permutation(rows, _subset_gauge(gauge_table, feats), list(feats), mode, raw, oracle)
        out.append({"family": fam, "n_features": len(feats), "gap_closed": perm["gap_closed"],
                    "auc_improve": perm["auc_improve"], "perm_p": perm["auc_improve_perm_p"],
                    "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]})
    return out


def leave_one_family_out(rows, gauge_table, mode, raw, oracle) -> list:
    mr = [r for r in rows if r["mode"] == mode]
    full = [f for feats in schema.FAMILIES.values() for f in feats]
    out = []
    for fam, feats in schema.FAMILIES.items():
        rest = [f for f in full if f not in feats]
        perm = tuf.r3_loto_permutation(rows, _subset_gauge(gauge_table, rest), list(rest), mode, raw, oracle)
        out.append({"dropped_family": fam, "remaining_features": len(rest), "gap_closed": perm["gap_closed"],
                    "perm_p": perm["auc_improve_perm_p"], "survives_permutation": perm["survives_permutation"]})
    return out


def shapley(rows, gauge_table, mode, raw, oracle) -> dict:
    """Exact Shapley gap-closure per family (value function = pooled-AUC gap closed of a family-union's gauge)."""
    mr = [r for r in rows if r["mode"] == mode]
    fams = list(schema.FAMILIES)
    n = len(fams)
    # precompute v(S) for all 2^n subsets
    v = {}
    for k in range(n + 1):
        for S in itertools.combinations(fams, k):
            feats = [f for fam in S for f in schema.FAMILIES[fam]]
            g, _ = _fit_gap(mr, gauge_table, feats, raw, oracle)
            v[frozenset(S)] = (g if g is not None else 0.0)
    phi = {}
    for fam in fams:
        s = 0.0
        others = [f for f in fams if f != fam]
        for k in range(len(others) + 1):
            w = math.factorial(k) * math.factorial(n - k - 1) / math.factorial(n)
            for S in itertools.combinations(others, k):
                s += w * (v[frozenset(S + (fam,))] - v[frozenset(S)])
        phi[fam] = s
    total_pos = sum(max(x, 0.0) for x in phi.values())
    share = {f: (max(phi[f], 0.0) / total_pos if total_pos > 1e-9 else None) for f in fams}
    dominant = max(fams, key=lambda f: phi[f])
    dominant_share = share[dominant]
    return {"shapley": phi, "positive_share": share, "full_gap": v[frozenset(fams)],
            "dominant_family": dominant, "dominant_share": dominant_share,
            "single_family_dominates": bool(dominant_share is not None and dominant_share >= schema.DOMINANT_FAMILY_SHARE)}


def per_regime_stability(rows, gauge_table, mode, feature_names) -> list:
    """Does the full-R3 offset_hat help WITHIN each in-regime regime? (fit once, evaluate per regime)."""
    mr = [r for r in rows if r["mode"] == mode]
    fit = offset_model.fit_offsets(gauge_table, names=list(feature_names))
    oh = fit["offset_hat_loto"]
    out = []
    for regime in sorted({r["regime"] for r in mr}):
        rr = [r for r in mr if r["regime"] == regime]
        tmean = {t: float(np.mean([c["score"] for c in rr if c["target"] == t])) for t in {r["target"] for r in rr}}
        raw = _pooled_auc(rr); oracle = _pooled_auc(rr, subtract=lambda r: tmean[r["target"]])
        gauge = _pooled_auc(rr, subtract=lambda r: oh.get(r["target"], 0.0))
        gap = ((gauge - raw) / (oracle - raw)) if (gauge is not None and oracle is not None and (oracle - raw) > 1e-6) else None
        out.append({"regime": regime, "raw_pooled": raw, "gauge_pooled": gauge, "oracle": oracle, "gap_closed": gap})
    return out
