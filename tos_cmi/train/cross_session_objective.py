"""Cross-Session objective audit primitives. Cross-session risk weight: for each source subject d and class pair
(a,b), fit a readout on the EARLIEST session of the OTHER source subjects, then r^sess = [ l_late(d) - l_early(d) ]_+
(same classifier for early/late; early controls the subject's own difficulty; the difference isolates session drift).
Two candidate objectives share these weights: CS-RW-MCC (weighted direction-consistency) and CS-Risk (weighted
later-session task loss). All source-only; target labels never enter any weight or objective. Manuscript FROZEN."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

from tos_cmi.train.mechanism_consistency import class_pairs

EPS = 1e-9


def _normalize_weights(r, keys, winsor_q=0.90, clip=4.0, eps=EPS):
    """winsor(p90 of positives) -> mean-normalize -> clip(4), NO re-mean; all-zero -> status NO_POSITIVE."""
    rv = np.array([r[k] for k in keys]); pos = rv[rv > 0]
    if pos.size == 0:
        return {k: 0.0 for k in keys}, "NO_POSITIVE_SOURCE_TRANSFER_GAP", 0.0
    wt = float(np.quantile(pos, winsor_q)); rwin = {k: min(r[k], wt) for k in keys}
    mean_rwin = float(np.mean([rwin[k] for k in keys])) + eps
    w = {k: float(min(rwin[k] / mean_rwin, clip)) for k in keys}
    return w, "ok", wt


def _early_late(sess):
    """Ordered unique session labels -> (earliest, set-of-later). Handles '0train'/'1test' and '0A'/'1B'/'2C'."""
    u = sorted(set(str(s) for s in sess))
    return u[0], set(u[1:])


def _pairwise_balanced_logloss(P, cidx, a, b, mask_a, mask_b, eps=EPS):
    pa = P[:, cidx[a]]; pb = P[:, cidx[b]]; ss = pa + pb + eps; pta = pa / ss; ptb = pb / ss
    if mask_a.sum() == 0 or mask_b.sum() == 0:
        return float("nan")
    return -0.5 * (float(np.mean(np.log(pta[mask_a] + eps))) + float(np.mean(np.log(ptb[mask_b] + eps))))


def cross_session_risk_weights(Z, y, d, sess, winsor_q=0.90, clip=4.0, eps=EPS):
    """r^sess_{d,p} = [ l_late_{d,p} - l_early_{d,p} ]_+ using h^early_{-d} (fit on the earliest session of the OTHER
    source subjects). Returns {(d,(a,b)): w} + diagnostics + the raw r/early/late. Fails loud if a subject lacks an
    early or late session for a class."""
    Z = np.asarray(Z, float); y = np.asarray(y).astype(int); d = np.asarray(d).astype(int); sess = np.asarray(sess).astype(str)
    subs = sorted(np.unique(d).tolist()); classes = sorted(np.unique(y).tolist()); pairs = class_pairs(classes)
    early, later = _early_late(sess)
    is_early = sess == early; is_late = np.isin(sess, list(later))
    r, l_e, l_l = {}, {}, {}
    for dd in subs:
        oth_early = (d != dd) & is_early
        if len(np.unique(y[oth_early])) < len(classes) or oth_early.sum() < len(classes) * 2:
            raise ValueError(f"cross-session: subject {dd} others' early session missing a class (fail loud)")
        sc = StandardScaler().fit(Z[oth_early])
        clf = LogisticRegression(penalty="l2", C=1.0, class_weight="balanced", solver="lbfgs", max_iter=500).fit(sc.transform(Z[oth_early]), y[oth_early])
        P = clf.predict_proba(sc.transform(Z)); cidx = {int(c): i for i, c in enumerate(clf.classes_)}
        for (a, b) in pairs:
            le = _pairwise_balanced_logloss(P, cidx, a, b, (d == dd) & is_early & (y == a), (d == dd) & is_early & (y == b))
            ll = _pairwise_balanced_logloss(P, cidx, a, b, (d == dd) & is_late & (y == a), (d == dd) & is_late & (y == b))
            if not (np.isfinite(le) and np.isfinite(ll)):
                raise ValueError(f"cross-session: subject {dd} pair {(a,b)} missing early/late class trials")
            l_e[(dd, (a, b))] = le; l_l[(dd, (a, b))] = ll; r[(dd, (a, b))] = max(0.0, ll - le)
    keys = list(r); w, status, wt = _normalize_weights(r, keys, winsor_q, clip)
    wv = np.array([w[k] for k in keys]); pw = wv[wv > 0]
    ent = float(-np.sum(pw / pw.sum() * np.log(pw / pw.sum() + eps))) if pw.size else 0.0
    return dict(status=status, weights=w, r=r, l_early=l_e, l_late=l_l, subs=subs, pairs=pairs, classes=classes,
                is_late=is_late, positive_weight_fraction=float(np.mean(wv > 0)),
                effective_weight_support=float(wv.sum() ** 2 / (np.sum(wv ** 2) + eps)), max_weight=float(wv.max()), weight_entropy=ent)


# ---- gradients (all w.r.t. encoder params; BN frozen/eval so per-sample micro-batching is exact) ----
def _flat_grad(loss, params):
    g = torch.autograd.grad(loss, params, allow_unused=True, retain_graph=False)
    return torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten() for gi, p in zip(g, params)]).detach().cpu().numpy()


def weighted_late_task_gradient(bb, X, y, d, sess_is_late, weights, device, bs=512):
    """CS-Risk: g_theta of Sum_{d,p} w_{d,p} * pairwise-balanced-CE on subject d's LATER-session trials, through the
    warm-up model's task head. Per-sample task loss (normal backprop). BN frozen."""
    bb.eval(); params = list(bb.parameters())
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int); late = np.asarray(sess_is_late)
    subs = sorted(np.unique(d).tolist()); classes = sorted(np.unique(y).tolist()); pairs = class_pairs(classes)
    for p in params:
        p.grad = None
    total = torch.zeros((), device=device)
    # accumulate the weighted loss in micro-batches over LATE-session source trials, grouped by (subject, pair)
    for (a, b) in pairs:
        for s in subs:
            w = float(weights.get((s, (a, b)), 0.0))
            if w == 0.0:
                continue
            idx = np.where(late & (d == s) & np.isin(y, [a, b]))[0]
            if idx.size == 0:
                continue
            xb = torch.tensor(X[idx], dtype=torch.float32).to(device)
            logits = bb(xb)[0]
            yy = torch.tensor(np.where(y[idx] == a, a, b), dtype=torch.long, device=device)
            # class-balanced pairwise CE (restricted to the pair columns)
            lg = logits[:, [a, b]]; tgt = (y[idx] == b).astype(int)
            ce = F.cross_entropy(lg, torch.tensor(tgt, dtype=torch.long, device=device))
            total = total + w * ce
    total.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).detach().cpu().numpy()


def task_gradient(bb, X, y, device, mask=None, bs=512):
    """Ordinary CE gradient w.r.t. encoder params over X[mask] (used for g_task = source, g_target = target future
    session). Accumulated over micro-batches, mean CE. BN frozen."""
    bb.eval(); params = list(bb.parameters())
    idx = np.arange(len(X)) if mask is None else np.where(mask)[0]
    for p in params:
        p.grad = None
    n = len(idx)
    for i0 in range(0, n, bs):
        j = idx[i0:i0 + bs]
        logits = bb(torch.tensor(X[j], dtype=torch.float32).to(device))[0]
        ce = F.cross_entropy(logits, torch.tensor(np.asarray(y)[j], dtype=torch.long, device=device), reduction="sum") / n
        ce.backward()
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in params]).detach().cpu().numpy()


def _weighted_mcc_from_means(means, subs, classes, pairs, weights, eps=1e-6):
    terms = []
    for (a, b) in pairs:
        U = torch.stack([F.normalize(means[(s, a)] - means[(s, b)], dim=0, eps=eps) for s in subs])
        Ud = U.detach(); tot = Ud.sum(0)
        for i, s in enumerate(subs):
            ubar = F.normalize(tot - Ud[i], dim=0, eps=eps); w = float(weights.get((s, (a, b)), 0.0))
            terms.append(w * (1.0 - torch.dot(U[i], ubar)))
    return torch.stack(terms).mean()


def exact_weighted_mcc_gradient(bb, X, y, d, weights, device, bs=256):
    """Two-pass EXACT g_theta of the WEIGHTED full-source MCC loss (BN frozen so micro-batching is exact). Mirrors
    the verified mcc_estimator_audit two-pass with a per-(d,pair)-weighted prototype loss."""
    bb.eval(); assert not bb.training
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    classes = sorted(np.unique(y).tolist()); subs = sorted(np.unique(d).tolist()); pairs = class_pairs(classes)
    cell = {(s, c): np.where((d == s) & (y == c))[0] for s in subs for c in classes}
    for k, idx in cell.items():
        if len(idx) == 0:
            raise ValueError(f"weighted-MCC grad: empty cell {k}")
    with torch.no_grad():
        z_all = torch.cat([bb(torch.tensor(X[i:i + bs], dtype=torch.float32).to(device))[1].detach().cpu() for i in range(0, len(X), bs)])
    means = {k: z_all[idx].mean(0).clone().to(device).requires_grad_(True) for k, idx in cell.items()}
    counts = {k: len(idx) for k, idx in cell.items()}
    L = _weighted_mcc_from_means(means, subs, classes, pairs, weights)
    keys = list(cell); gmu_l = torch.autograd.grad(L, [means[k] for k in keys]); gmu = {k: gmu_l[i].detach() for i, k in enumerate(keys)}
    cell_of = np.empty(len(X), dtype=object)
    for k, idx in cell.items():
        for i in idx:
            cell_of[i] = k
    for p in bb.parameters():
        p.grad = None
    for i0 in range(0, len(X), bs):
        idx = np.arange(i0, min(i0 + bs, len(X)))
        z = bb(torch.tensor(X[idx], dtype=torch.float32).to(device))[1]
        up = torch.stack([gmu[cell_of[i]] / counts[cell_of[i]] for i in idx]).to(device)
        z.backward(gradient=up)
    return torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten() for p in bb.parameters()]).detach().cpu().numpy()


def per_trial_cs_weights(weights, subs, pairs, classes):
    """Aggregate per-(subject,pair) weights to per-(subject,class): v_{s,c} = mean over pairs containing c."""
    v = {}
    for s in subs:
        for c in classes:
            ws = [weights[(s, (a, b))] for (a, b) in pairs if c in (a, b)]
            v[(s, c)] = float(np.mean(ws)) if ws else 0.0
    return v


def direct_risk_loss(logits, y, d, is_late, v, device, eps=1e-6):
    """CS-Risk TRAINING term: L = (Sum_{late i} v_{d(i),y(i)} CE_i) / (Sum v + eps) -- weighted later-session
    predictive risk (upweights drift-prone late-session cells). NO cosine geometry. is_late = per-batch-trial bool."""
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int); late = np.asarray(is_late)
    assert late.dtype == np.bool_, f"is_late must be a bool mask, got {late.dtype} (pass np.isin(sess, later), NOT raw session strings)"
    ce = F.cross_entropy(logits, torch.tensor(y, dtype=torch.long, device=device), reduction="none")
    w = torch.tensor([v.get((int(d[i]), int(y[i])), 0.0) if late[i] else 0.0 for i in range(len(y))],
                     dtype=logits.dtype, device=device)
    return (w * ce).sum() / (w.sum() + eps)


def cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12))
