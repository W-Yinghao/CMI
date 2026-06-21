"""Action registry. Each action: (state, z_batch) -> (p_a [n,K], z_tilde [n,d] or None). Source-free, label-free.

`identity` is f_0 (the no-adaptation reference). `matched_coral` is the deployed transductive lever. `spdim` and
`t3a` are the source-free TTA baselines. All consume only the serialized source state from fit_source_state — never
raw source, never target labels. z_tilde (post-features) is returned where the action is geometric, else None.
"""
from __future__ import annotations
import numpy as np

from cmi.eval.source_state import pmct_predict_serialized
from cmi.eval.tta_baselines import t3a_predict, spdim_predict


def _full(state, p):
    out = np.zeros((len(p), state["n_cls"]))
    out[:, state["clf"].classes_] = p
    return out


def act_identity(state, z):
    return np.asarray(_full(state, state["clf"].predict_proba(z)), float), np.asarray(z, float)


def act_matched_coral(state, z):
    prob, _, ztil = pmct_predict_serialized(state, z, ref="pooled", tmap="wc", em_iters=3, return_ztilde=True)
    return np.asarray(prob, float), np.asarray(ztil, float)


def act_spdim(state, z, steps=100, lr=0.05, div_w=1.0):
    """IM recentering. Reproduces tta_baselines.spdim_predict and additionally exposes z_tilde = z + b (a pure
    mean shift), so its transport-size (Bures) feature is well-defined. prob path is asserted-equal to the
    validated baseline in tests/test_leakage_guard.py."""
    import torch
    z = np.asarray(z, float)
    clf, n_cls = state["clf"], state["n_cls"]
    cls = clf.classes_
    W, c = np.atleast_2d(clf.coef_), np.atleast_1d(clf.intercept_)
    if W.shape[0] == 1 and n_cls == 2:
        W = np.vstack([-W[0], W[0]]); c = np.array([-c[0], c[0]])
    Z = torch.tensor(z, dtype=torch.float64)
    Wt = torch.tensor(W, dtype=torch.float64); ct = torch.tensor(c, dtype=torch.float64)
    b = torch.zeros(z.shape[1], dtype=torch.float64, requires_grad=True)
    opt = torch.optim.Adam([b], lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        p = torch.softmax((Z + b) @ Wt.T + ct, 1)
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(1).mean()
        pm = p.mean(0)
        div = -(pm * torch.log(pm.clamp_min(1e-9))).sum()
        (ent - div_w * div).backward()
        opt.step()
    with torch.no_grad():
        bias = b.detach().numpy()
        p = torch.softmax((Z + b) @ Wt.T + ct, 1).numpy()
    out = np.zeros((len(z), n_cls)); out[:, cls] = p[:, :len(cls)]
    return np.asarray(out, float), (z + bias[None]).astype(float)


def act_t3a(state, z):
    return np.asarray(t3a_predict(state, z), float), None     # adjusts the classifier, not the embedding


REGISTRY = {
    "identity": act_identity,
    "matched_coral": act_matched_coral,
    "spdim": act_spdim,
    "t3a": act_t3a,
}


def apply_action(name, state, z):
    """Returns (p_a, z_tilde_or_None). On <MIN_BATCH or degenerate batch the caller forces identity upstream."""
    return REGISTRY[name](state, np.asarray(z, float))
