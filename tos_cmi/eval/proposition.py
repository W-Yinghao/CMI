"""Empirical check of the idealized proposition (THEORY.md, Prop. 1).

Proposition (informal). If Z = (Z_Y, Z_N) with I(Y; Z_N | Z_Y) = 0 and all conditional
domain leakage in Z_N, then removing Z_N leaves the Bayes risk unchanged while removing
the leakage. When task and domain subspaces overlap, removing the domain-rich subspace
*raises* the risk, so the selective method must refuse (identity).

This module instruments that with frozen linear probes on a held-out split:

  acc_full   : label probe on Z                 (the achievable accuracy)
  acc_task   : label probe on (I - P_N) Z        (after removing the nuisance subspace)
  leak_full  : conditional-domain advantage on Z
  leak_nuis  : conditional-domain advantage on P_N Z
  leak_task  : conditional-domain advantage on (I - P_N) Z

The proposition predicts, for a *correctly* selected subspace:
  acc_task ~ acc_full   (risk preserved)  AND  leak_task ~ 0 << leak_full ~ leak_nuis
and, under overlap, the selector returns identity so acc_task == acc_full by construction.

Probes are plain multinomial logistic regressions trained with full-batch GD (no sklearn
dependency, deterministic given the seed).
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F


def _fit_logreg(X, t, n_out, epochs=400, lr=0.5, wd=1e-3, seed=0):
    torch.manual_seed(seed)
    Xt = torch.tensor(X, dtype=torch.float32)
    tt = torch.tensor(t, dtype=torch.long)
    mu, sd = Xt.mean(0, keepdim=True), Xt.std(0, keepdim=True).clamp_min(1e-6)
    Xt = (Xt - mu) / sd
    W = torch.zeros(Xt.shape[1], n_out, requires_grad=True)
    b = torch.zeros(n_out, requires_grad=True)
    opt = torch.optim.Adam([W, b], lr=lr, weight_decay=wd)
    for _ in range(epochs):
        opt.zero_grad()
        loss = F.cross_entropy(Xt @ W + b, tt)
        loss.backward()
        opt.step()
    return (W.detach(), b.detach(), mu, sd)


def _predict(model, X):
    W, b, mu, sd = model
    Xt = (torch.tensor(X, dtype=torch.float32) - mu) / sd
    return (Xt @ W + b).argmax(1).numpy()


def _balanced_acc(pred, true, n):
    accs = []
    for k in range(n):
        m = true == k
        if m.sum() > 0:
            accs.append((pred[m] == k).mean())
    return float(np.mean(accs)) if accs else 0.0


def _label_acc(Ztr, ytr, Zte, yte, n_cls, seed=0):
    model = _fit_logreg(Ztr, ytr, n_cls, seed=seed)
    return _balanced_acc(_predict(model, Zte), yte, n_cls)


def _cond_domain_adv(Ztr, ytr, dtr, Zte, yte, dte, n_cls, n_dom, seed=0):
    """Balanced accuracy of predicting D from [Z, onehot(Y)], minus the per-class
    majority-domain baseline (the label-conditional leakage advantage)."""
    def feat(Z, y):
        oh = np.eye(n_cls)[y]
        return np.concatenate([Z, oh], 1)
    model = _fit_logreg(feat(Ztr, ytr), dtr, n_dom, seed=seed)
    pred = _predict(model, feat(Zte, yte))
    probe = _balanced_acc(pred, dte, n_dom)
    # baseline: predict, within each class, that class's most frequent domain
    base_pred = np.zeros_like(dte)
    for c in range(n_cls):
        mtr = ytr == c
        if mtr.sum() == 0:
            continue
        maj = np.bincount(dtr[mtr], minlength=n_dom).argmax()
        base_pred[yte == c] = maj
    base = _balanced_acc(base_pred, dte, n_dom)
    return float(probe - base)


def bayes_risk_check(data, selector, train_frac=0.6, seed=0) -> dict:
    """data: dict from synthetic.make (or any {Z,y,d,n_cls,n_dom}); selector: a *refreshed*
    SubspaceSelector. Returns the probe panel above plus the selector decision."""
    Z, y, d = data["Z"], data["y"], data["d"]
    spec = data.get("spec")
    n_cls = spec.n_cls if spec is not None else int(y.max() + 1)
    n_dom = spec.n_dom if spec is not None else int(d.max() + 1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    cut = int(train_frac * len(Z))
    tr, te = idx[:cut], idx[cut:]

    P = selector.P.cpu().numpy()                      # nuisance projector [d,d]
    Zn = Z @ P.T                                      # P_N Z
    Zt = Z - Zn                                       # (I - P_N) Z

    out = {
        "is_identity": bool(selector.is_identity),
        "k": 0 if selector.report is None else selector.report.k,
        "acc_full": _label_acc(Z[tr], y[tr], Z[te], y[te], n_cls, seed),
        "acc_task": _label_acc(Zt[tr], y[tr], Zt[te], y[te], n_cls, seed),
        "leak_full": _cond_domain_adv(Z[tr], y[tr], d[tr], Z[te], y[te], d[te], n_cls, n_dom, seed),
        "leak_nuis": _cond_domain_adv(Zn[tr], y[tr], d[tr], Zn[te], y[te], d[te], n_cls, n_dom, seed),
        "leak_task": _cond_domain_adv(Zt[tr], y[tr], d[tr], Zt[te], y[te], d[te], n_cls, n_dom, seed),
    }
    out["acc_drop"] = out["acc_full"] - out["acc_task"]
    out["leak_removed"] = out["leak_full"] - out["leak_task"]
    return out
