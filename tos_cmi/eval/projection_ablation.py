"""Leakage-free linear-probe projection ablation (was `bayes_risk_check`).

What this measures (honestly): on data held out from the subspace estimation, how a *linear*
probe's label balanced-accuracy and a *linear* conditional-domain probe's balanced-accuracy
*advantage* change when we project Z onto / away from the selected nuisance subspace P_N.

Three disjoint splits (this is the fix for the selection-leakage bug — previously P_N was
estimated on ALL data, including the probe-test rows, so test labels/domains entered the
construction of P_N):

  selector-train : the ONLY data used to estimate F_Y / F_{D|Y} and hence P_N
  risk-val       : held out from selection; used to report the risk delta of removing P_N
                   (a diagnostic; the formal source-only risk *gate* is a separate step)
  probe-test     : held out from both; the ONLY data used for the reported numbers

Naming caveats (the reviewer's point):
  * `domadv_*` is a conditional-domain linear-probe **balanced-accuracy advantage**, NOT a
    mutual information. It lower-bounds leakage detectable by a linear probe only.
  * "leakage removed" is descriptive of the projection arithmetic, not a causal claim — the
    simulator *plants* the leakage in this subspace; the method identifies and projects it.

Probes are plain multinomial logistic regressions (full-batch GD, deterministic given seed),
trained on selector-train, evaluated on probe-test.
"""
from __future__ import annotations
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F

from ..config import FisherConfig, SubspaceConfig
from ..subspace import SubspaceSelector


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
        F.cross_entropy(Xt @ W + b, tt).backward()
        opt.step()
    return (W.detach(), b.detach(), mu, sd)


def _predict(model, X):
    W, b, mu, sd = model
    Xt = (torch.tensor(X, dtype=torch.float32) - mu) / sd
    return (Xt @ W + b).argmax(1).numpy()


def _balanced_acc(pred, true, n):
    accs = [(pred[true == k] == k).mean() for k in range(n) if (true == k).sum() > 0]
    return float(np.mean(accs)) if accs else 0.0


def _label_acc(Ztr, ytr, Zte, yte, n_cls, seed=0):
    return _balanced_acc(_predict(_fit_logreg(Ztr, ytr, n_cls, seed=seed), Zte), yte, n_cls)


def _cond_domain_adv(Ztr, ytr, dtr, Zte, yte, dte, n_cls, n_dom, seed=0):
    """Balanced-accuracy advantage of predicting D from [Z, onehot(Y)] over the per-class
    majority-domain baseline. A LINEAR conditional-domain leakage proxy, not CMI."""
    feat = lambda Z, y: np.concatenate([Z, np.eye(n_cls)[y]], 1)
    pred = _predict(_fit_logreg(feat(Ztr, ytr), dtr, n_dom, seed=seed), feat(Zte, yte))
    probe = _balanced_acc(pred, dte, n_dom)
    base_pred = np.zeros_like(dte)
    for c in range(n_cls):
        if (ytr == c).sum():
            base_pred[yte == c] = np.bincount(dtr[ytr == c], minlength=n_dom).argmax()
    return float(probe - _balanced_acc(base_pred, dte, n_dom))


def linear_probe_projection_ablation(data, fcfg: Optional[FisherConfig] = None,
                                     scfg: Optional[SubspaceConfig] = None,
                                     seed=0, fracs=(0.45, 0.15, 0.40)):
    """Estimate P_N on selector-train only, report probe metrics on a disjoint probe-test.
    Returns the metric panel plus the fitted selector (for summary / stability reuse)."""
    Z, y, d = data["Z"], data["y"], data["d"]
    spec = data.get("spec")
    n_cls = spec.n_cls if spec is not None else int(y.max() + 1)
    n_dom = spec.n_dom if spec is not None else int(d.max() + 1)

    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    n = len(Z)
    a = int(fracs[0] * n)
    b = a + int(fracs[1] * n)
    sel, val, test = idx[:a], idx[a:b], idx[b:]

    # P_N from selector-train ONLY -- no probe-test labels/domains enter the construction
    selector = SubspaceSelector(Z.shape[1], n_cls, n_dom, fcfg, scfg)
    selector.refresh(torch.tensor(Z[sel]), torch.tensor(y[sel]), torch.tensor(d[sel]), seed=seed)
    P = selector.P.cpu().numpy()
    Zn, Zt = Z @ P.T, Z - Z @ P.T                # nuisance / task components

    out = {
        "is_identity": bool(selector.is_identity),
        "k": 0 if selector.report is None else selector.report.k,
        # label balanced-accuracy: probe trained on selector-train, evaluated on probe-test
        "acc_full": _label_acc(Z[sel], y[sel], Z[test], y[test], n_cls, seed),
        "acc_task": _label_acc(Zt[sel], y[sel], Zt[test], y[test], n_cls, seed),
        # conditional-domain linear-probe balanced-accuracy ADVANTAGE (not CMI)
        "domadv_full": _cond_domain_adv(Z[sel], y[sel], d[sel], Z[test], y[test], d[test], n_cls, n_dom, seed),
        "domadv_nuis": _cond_domain_adv(Zn[sel], y[sel], d[sel], Zn[test], y[test], d[test], n_cls, n_dom, seed),
        "domadv_task": _cond_domain_adv(Zt[sel], y[sel], d[sel], Zt[test], y[test], d[test], n_cls, n_dom, seed),
    }
    out["acc_drop"] = out["acc_full"] - out["acc_task"]
    # diagnostic risk delta on the held-out risk-val split (not the gate; see THEORY/risk gate)
    if len(val):
        out["risk_val_acc_drop"] = (_label_acc(Z[sel], y[sel], Z[val], y[val], n_cls, seed)
                                    - _label_acc(Zt[sel], y[sel], Zt[val], y[val], n_cls, seed))
    return out, selector
