"""CMI-Trace — DG-erasure ORACLE (P1 go/no-go). Separates SAFE cleaning from DG intervention.

The exact-head-null oracle removes functionally-UNUSED leakage (can't help DG). Here the objective is INVERTED:
minimize SOURCE-HELD-OUT risk (the DG-relevant quantity), with subject-CMI reduction only as a CONSTRAINT
confirming the deletion acts on subject leakage — NOT as the objective.

    argmin_S  R_source-heldout(delete S)   s.t.  Ihat(kept;D|Y) <= Ihat(Z;D|Y) - gamma,  R_source drop <= delta

Three oracles / selectors over a candidate basis B (axis subsets S):
  * target_dg_oracle       : uses TARGET labels — the non-deployable UPPER BOUND (does any deletion help target?).
  * source_meta_subset_oracle : SOURCE-ONLY — source-LOSO meta-validation picks S*; the go/no-go (is a
                                target-beneficial subset identifiable from source subjects alone?).
  * cmi_only_selector      : picks the subset that most reduces subject-CMI (the OLD objective) — for contrast.
Reports best-prefix vs best-subset (ordering test) + matched-rank random + magnitude baselines. Pure numpy+sklearn.
"""
from __future__ import annotations
from itertools import combinations, chain

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score


def _head_bacc(Ztr, ytr, Zte, yte, seed=0):
    mu, sd = Ztr.mean(0), Ztr.std(0) + 1e-8
    if len(np.unique(ytr)) < 2:
        return float("nan")
    clf = LogisticRegression(max_iter=500, C=1.0).fit((Ztr - mu) / sd, ytr)
    return float(balanced_accuracy_score(yte, clf.predict((Zte - mu) / sd)))


def delete(Z, B, S):
    """Remove span of the candidate directions B[S] (B [r,D] orthonormal rows): Z (I - B_S^T B_S)."""
    if len(S) == 0:
        return Z
    Bs = B[list(S)]
    return Z - (Z @ Bs.T) @ Bs


def cmi_proxy(Z, y, d, seed=0):
    """Label-conditional linear subject decode (cheap CMI proxy): within each Y, decode subject, average."""
    accs = []; rng = np.random.default_rng(seed)
    for c in np.unique(y):
        m = y == c; zz, dd = Z[m], d[m]
        if len(np.unique(dd)) < 2:
            continue
        idx = rng.permutation(len(zz)); cut = int(0.7 * len(idx))
        if len(np.unique(dd[idx[:cut]])) < 2 or cut >= len(idx):
            continue
        accs.append(balanced_accuracy_score(dd[idx[cut:]],
                    LogisticRegression(max_iter=300).fit(zz[idx[:cut]], dd[idx[:cut]]).predict(zz[idx[cut:]])))
    return float(np.mean(accs)) if accs else float("nan")


def _subsets(r, max_exhaustive=12, k_cap=None):
    kc = r if k_cap is None else min(k_cap, r)
    if 2 ** r <= 2 ** max_exhaustive:
        return list(chain.from_iterable(combinations(range(r), m) for m in range(kc + 1)))
    return [tuple(range(m)) for m in range(kc + 1)] + [(i,) for i in range(r)]   # prefix + singletons fallback


def target_dg_oracle(Z, y, d, target_dom, B, seed=0, max_exhaustive=12):
    """UPPER BOUND (uses target labels). For each subset S: fresh head on SOURCE-erased, score TARGET-erased.
    Returns per-subset target utility (vs identity) + the best subset/prefix + matched-rank random."""
    src = d != target_dom; tgt = d == target_dom
    Zs, ys, Zt, yt = Z[src], y[src], Z[tgt], y[tgt]
    r = B.shape[0]
    ident = _head_bacc(delete(Zs, B, ()), ys, delete(Zt, B, ()), yt, seed)
    rows = []
    for S in _subsets(r, max_exhaustive):
        u = _head_bacc(delete(Zs, B, S), ys, delete(Zt, B, S), yt, seed)
        rows.append({"S": list(S), "k": len(S), "target_bacc": u, "d_target": u - ident})
    best = max(rows, key=lambda x: x["d_target"])
    best_prefix = max([x for x in rows if x["S"] == list(range(x["k"]))], key=lambda x: x["d_target"])
    krand = best["k"]; rng = np.random.default_rng(7 + seed)
    rand = np.mean([_head_bacc(delete(Zs, B, tuple(rng.choice(r, min(krand, r), replace=False))), ys,
                               delete(Zt, B, tuple(rng.choice(r, min(krand, r), replace=False))), yt, seed)
                    for _ in range(15)]) - ident if krand else 0.0
    return {"identity_target_bacc": ident, "best": best, "best_prefix": best_prefix,
            "d_target_best": best["d_target"], "d_target_best_prefix": best_prefix["d_target"],
            "d_target_random": float(rand), "rows": rows}


def source_meta_subset_oracle(Z, y, d, target_dom, B, seed=0, max_exhaustive=12, gamma_cmi=0.0):
    """SOURCE-ONLY selector (the go/no-go). For each subset S, source-LOSO over source subjects: train a fresh
    head on source-except-one and score the held-out SOURCE subject; average = source-meta bAcc. Optionally
    require the subject-CMI to drop by >= gamma_cmi (constraint, source-only). Select S* = max source-meta bAcc
    among constraint-satisfying subsets. Returns S* + its source-meta score (target NEVER used here)."""
    src = d != target_dom
    Zs, ys, ds = Z[src], y[src], d[src]
    source_doms = np.unique(ds)
    r = B.shape[0]
    full_cmi = cmi_proxy(Zs, ys, ds, seed)
    rows = []
    for S in _subsets(r, max_exhaustive):
        Zsd = delete(Zs, B, S)
        # source-LOSO meta-validation
        accs = []
        for mv in source_doms:
            tr = ds != mv; te = ds == mv
            if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
                continue
            accs.append(_head_bacc(Zsd[tr], ys[tr], Zsd[te], ys[te], seed))
        meta = float(np.mean(accs)) if accs else float("nan")
        cmi = cmi_proxy(Zsd, ys, ds, seed)
        rows.append({"S": list(S), "k": len(S), "source_meta_bacc": meta, "cmi": cmi,
                     "cmi_reduction": full_cmi - cmi})
    feasible = [x for x in rows if np.isfinite(x["source_meta_bacc"]) and x["cmi_reduction"] >= gamma_cmi and x["k"] > 0]
    star = max(feasible, key=lambda x: x["source_meta_bacc"]) if feasible else {"S": [], "k": 0}
    return {"full_source_cmi": full_cmi, "S_star": star.get("S", []), "star": star, "rows": rows}


def cmi_only_selector(Z, y, d, target_dom, B, seed=0, max_exhaustive=12):
    """OLD objective, for contrast: pick the subset (source-only) that most reduces subject-CMI (any rank)."""
    src = d != target_dom; Zs, ys, ds = Z[src], y[src], d[src]
    r = B.shape[0]
    rows = [{"S": list(S), "k": len(S), "cmi": cmi_proxy(delete(Zs, B, S), ys, ds, seed)}
            for S in _subsets(r, max_exhaustive) if len(S) > 0]
    star = min(rows, key=lambda x: x["cmi"])
    return {"S_cmi": star["S"], "star": star}


def evaluate_on_target(Z, y, d, target_dom, B, S, seed=0):
    """Final target evaluation of a chosen subset S: fresh head on all source-erased, score target-erased."""
    src = d != target_dom; tgt = d == target_dom
    ident = _head_bacc(delete(Z[src], B, ()), y[src], delete(Z[tgt], B, ()), y[tgt], seed)
    got = _head_bacc(delete(Z[src], B, tuple(S)), y[src], delete(Z[tgt], B, tuple(S)), y[tgt], seed)
    return {"identity_target_bacc": ident, "chosen_target_bacc": got, "d_target": got - ident, "chosen_S": list(S)}


# --------------------------------------------------------------- greedy variants (O(r^2), for real EEG where 2^r is too slow)
def _source_meta_bacc(Zs, ys, ds, B, S, source_doms, seed):
    Zsd = delete(Zs, B, S); accs = []
    for mv in source_doms:
        tr = ds != mv; te = ds == mv
        if len(np.unique(ys[tr])) < 2 or te.sum() == 0:
            continue
        accs.append(_head_bacc(Zsd[tr], ys[tr], Zsd[te], ys[te], seed))
    return float(np.mean(accs)) if accs else float("nan")


def source_meta_greedy(Z, y, d, target_dom, B, seed=0, max_k=None, gamma_cmi=0.0):
    """Greedy forward selection of the DG deletion subset by SOURCE-LOSO meta-validation (O(r^2)). Adds the
    direction that most improves source-held-out bAcc until no gain (>1e-4) or max_k. Source-only."""
    src = d != target_dom; Zs, ys, ds = Z[src], y[src], d[src]
    source_doms = np.unique(ds); r = B.shape[0]; max_k = r if max_k is None else min(max_k, r)
    full_cmi = cmi_proxy(Zs, ys, ds, seed)
    S, cur = [], _source_meta_bacc(Zs, ys, ds, B, [], source_doms, seed)
    base = cur
    for _ in range(max_k):
        cand = [(_source_meta_bacc(Zs, ys, ds, B, S + [j], source_doms, seed), j) for j in range(r) if j not in S]
        if not cand:
            break
        bm, bj = max(cand, key=lambda x: (x[0] if np.isfinite(x[0]) else -1))
        if not np.isfinite(bm) or bm <= cur + 1e-4:
            break
        S.append(bj); cur = bm
    Zsd = delete(Zs, B, S)
    return {"full_source_cmi": full_cmi, "S_star": S, "star": {"source_meta_bacc": cur,
            "source_meta_gain": cur - base, "cmi_reduction": full_cmi - cmi_proxy(Zsd, ys, ds, seed)}}


def target_dg_greedy(Z, y, d, target_dom, B, seed=0, max_k=None):
    """Greedy target-label UPPER BOUND: add the direction that most improves TARGET bAcc (O(r^2))."""
    src = d != target_dom; tgt = d == target_dom
    Zs, ys, Zt, yt = Z[src], y[src], Z[tgt], y[tgt]; r = B.shape[0]; max_k = r if max_k is None else min(max_k, r)
    ident = _head_bacc(Zs, ys, Zt, yt, seed)
    tgt_b = lambda S: _head_bacc(delete(Zs, B, S), ys, delete(Zt, B, S), yt, seed)
    S, cur = [], ident
    for _ in range(max_k):
        cand = [(tgt_b(S + [j]), j) for j in range(r) if j not in S]
        if not cand:
            break
        bm, bj = max(cand, key=lambda x: (x[0] if np.isfinite(x[0]) else -1))
        if not np.isfinite(bm) or bm <= cur + 1e-4:
            break
        S.append(bj); cur = bm
    return {"identity_target_bacc": ident, "best": {"S": S, "k": len(S)}, "d_target_best": cur - ident}
