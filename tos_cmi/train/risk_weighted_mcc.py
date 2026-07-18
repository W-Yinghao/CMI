"""Risk-Weighted MCC (RW-MCC). Weight the direction-normalized MCC consistency term per (source subject, class pair)
by SOURCE-LOSO excess pairwise predictive risk: how much more a classifier trained on the OTHER source subjects
fails on subject d for pair (a,b) than on its own training subjects. Weights are computed once at the ERM warm-up
(source-only; never touches target) and frozen for continuation. PRIMARY control = per-pair weight permutation
across subjects. Manuscript FROZEN; only the project owner stops a scientific line."""
from __future__ import annotations
import hashlib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tos_cmi.train.mechanism_consistency import class_pairs

EPS = 1e-9


def source_loso_excess_risk_weights(Z, y, d, winsor_q=0.90, clip=4.0, eps=EPS):
    """r_{d,p} = [ l_hold_{d,p} - l_ref_{-d,p} ]_+ over source-only features; winsorize(p90) -> mean-normalize ->
    clip(4), no re-mean. Returns weights {(d,(a,b)): w} + diagnostics. all-zero -> NO_POSITIVE_SOURCE_TRANSFER_GAP
    (weights 0, RW-MCC a no-op; NEVER fall back to uniform)."""
    Z = np.asarray(Z, float); y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    subs = sorted(np.unique(d).tolist()); classes = sorted(np.unique(y).tolist()); pairs = class_pairs(classes)
    for s in subs:
        for c in classes:
            if int(((d == s) & (y == c)).sum()) == 0:
                raise ValueError(f"RW weights: source subject {s} missing class {c} (fail loud, no reweight)")
    r, hold, ref = {}, {}, {}
    for dd in subs:
        oth = d != dd
        sc = StandardScaler().fit(Z[oth])
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500).fit(sc.transform(Z[oth]), y[oth])
        P = clf.predict_proba(sc.transform(Z)); cidx = {int(c): i for i, c in enumerate(clf.classes_)}
        for (a, b) in pairs:
            pa = P[:, cidx[a]]; pb = P[:, cidx[b]]; ss = pa + pb + eps; pta = pa / ss; ptb = pb / ss
            ma = (d == dd) & (y == a); mb = (d == dd) & (y == b)
            l_hold = -0.5 * (float(np.mean(np.log(pta[ma] + eps))) + float(np.mean(np.log(ptb[mb] + eps))))
            refs = []
            for e in subs:
                if e == dd:
                    continue
                ea = (d == e) & (y == a); eb = (d == e) & (y == b)
                if ea.sum() and eb.sum():
                    refs.append(-0.5 * (float(np.mean(np.log(pta[ea] + eps))) + float(np.mean(np.log(ptb[eb] + eps)))))
            l_ref = float(np.mean(refs))
            hold[(dd, (a, b))] = l_hold; ref[(dd, (a, b))] = l_ref; r[(dd, (a, b))] = max(0.0, l_hold - l_ref)
    keys = list(r); rv = np.array([r[k] for k in keys]); pos = rv[rv > 0]
    if pos.size == 0:
        return dict(status="NO_POSITIVE_SOURCE_TRANSFER_GAP", weights={k: 0.0 for k in keys}, r=r, hold=hold, ref=ref,
                    subs=subs, pairs=pairs, positive_weight_fraction=0.0, effective_weight_support=0.0,
                    max_weight=0.0, weight_entropy=0.0, winsor_threshold=0.0)
    wt = float(np.quantile(pos, winsor_q)); rwin = {k: min(r[k], wt) for k in keys}
    mean_rwin = float(np.mean([rwin[k] for k in keys])) + eps
    w = {k: float(min(rwin[k] / mean_rwin, clip)) for k in keys}
    wv = np.array([w[k] for k in keys]); pw = wv[wv > 0]
    ent = float(-np.sum(pw / pw.sum() * np.log(pw / pw.sum() + eps))) if pw.size else 0.0
    return dict(status="ok", weights=w, r=r, hold=hold, ref=ref, subs=subs, pairs=pairs,
                positive_weight_fraction=float(np.mean(wv > 0)), effective_weight_support=float(wv.sum() ** 2 / (np.sum(wv ** 2) + eps)),
                max_weight=float(wv.max()), weight_entropy=ent, winsor_threshold=wt)


def permute_weights(weights, subs, pairs, seed=0):
    """Per-pair fixed permutation of {w_{d,p}} across source subjects (PRIMARY control): same multiset + per-pair
    total, only WHICH subject is more constrained changes."""
    rng = np.random.default_rng(seed); out = dict(weights)
    for p in pairs:
        vals = [weights[(dd, p)] for dd in subs]; perm = rng.permutation(len(subs))
        for i, dd in enumerate(subs):
            out[(dd, p)] = vals[perm[i]]
    return out


def weight_hash(weights, subs, pairs):
    v = np.array([weights[(dd, p)] for dd in subs for p in pairs], float)
    return hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest()[:16]


def rw_mcc_loss(Z, y, d, weights, pairs=None, eps=1e-6):
    """L_RW-MCC = (1/(|S||P|)) sum_{d,p} w_{d,p} [1 - <u_{d,p}, sg(ubar_{-d,p})>]. weights keyed by (subject,(a,b))."""
    if not torch.is_tensor(y):
        y = torch.as_tensor(np.asarray(y), device=Z.device)
    if not torch.is_tensor(d):
        d = torch.as_tensor(np.asarray(d), device=Z.device)
    y = y.long(); d = d.long()
    subs = sorted(torch.unique(d).tolist()); classes = sorted(torch.unique(y).tolist())
    if pairs is None:
        pairs = class_pairs(classes)
    mean = {}
    for s in subs:
        for c in classes:
            m = (d == s) & (y == c)
            if int(m.sum()) == 0:
                raise ValueError(f"RW-MCC: subject {s} missing class {c} in batch")
            mean[(s, c)] = Z[m].mean(0)
    terms = []; wsum = 0.0
    for (a, b) in pairs:
        U = torch.stack([F.normalize(mean[(s, a)] - mean[(s, b)], dim=0, eps=eps) for s in subs])
        Ud = U.detach(); tot = Ud.sum(0)
        for i, s in enumerate(subs):
            ubar = F.normalize(tot - Ud[i], dim=0, eps=eps)
            w = float(weights.get((s, (a, b)), 0.0)); wsum += w
            terms.append(w * (1.0 - torch.dot(U[i], ubar)))
    return torch.stack(terms).mean(), dict(weight_sum=wsum, n_terms=len(terms))
