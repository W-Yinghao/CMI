"""CMI-Trace — FFW-EEG: Finding Fantastic Weights for subject-leakage debiasing (ECCV'24 adaptation).

First version of Nahon et al. 2024 (arXiv:2403.14200) for the EEG cross-subject setting. On a FROZEN
vanilla model (encoder E + linear task head C), learn a structured GATING MASK on the bottleneck z (=graph_z)
such that subject identity D becomes NON-EXTRACTABLE by C, WITHOUT retraining E or C (only the mask is
learned). Complements the exact-head-null oracle: the oracle removes an arbitrary FEATURE SUBSPACE with an
algebraic softmax guarantee; FFW removes/masks NEURONS (axis-aligned, a learned sub-network) with the task
retained through the frozen head.

Bias = subject, made LABEL-CONDITIONAL to match our estimand I(Z;D|Y): the bias head P is q(D|z⊙m, Y) and the
MI term reuses the posterior-KL leakage ruler (cmi.eval.graph_leakage). FFW objective (paper Eq. 10):

    J(m) = CE(y, C(z⊙gate(m,τ))) + gamma * Ihat(D ; P(z⊙gate(m,τ)) | Y)

Gate (paper Eq. 8/9, structured on z): gate(m,τ)=Theta(-m)·2σ(m/τ)+Theta(m); init m=1 (=vanilla, gate 1);
straight-through estimator for Theta; anneal τ ×0.5 until the hard mask matches. Only m is trainable; E,C,P
weights are frozen w.r.t. the mask update (P is trained alternately to EXTRACT the bias). Pure torch.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def gate(m, tau):
    """Smooth structured neuron gate g_j = σ(m_j/τ) in [0,1]. Differentiable everywhere (no forward-flat /
    backward-nonzero STE mismatch, which caused phantom-gradient collapse). As τ→0 it → {0,1} with threshold
    m=0; the deployed hard mask is 1[m>0]. m is initialized large-positive (gate≈1 = vanilla) — see M_INIT."""
    return torch.sigmoid(m / tau)


M_INIT = 1.0            # sigmoid(1)=0.73: keeps a HEALTHY gradient dσ/dm (init 4 saturates -> mask can't learn).
                        # The deployed mask uses the learned ORDERING (1[m>0]) + a task-safe frontier, so the
                        # soft gate need not be exactly 1 at init.


def _bias_head(z_dim, n_cls, n_dom, hidden=64):
    """Privacy/bias extraction head P: q(D | z, Y). Same style as the CMI critic (label-conditional)."""
    return nn.Sequential(nn.Linear(z_dim + n_cls, hidden), nn.ReLU(), nn.Linear(hidden, n_dom))


def find_fantastic_weights(Z, y, d, W, b, n_cls, n_dom, *, gamma=10.0, hidden=64, tau0=1.0,
                           n_temps=5, inner_epochs=60, p_epochs=40, lr_m=0.05, lr_p=1e-3,
                           device="cpu", seed=0, smoothing=1e-3):
    """Learn a structured graph_z neuron mask on FROZEN (E via z + head W,b). Returns (mask [d] in {0,1},
    m params, diagnostics). Z,y,d numpy (source). W [n_cls,d], b [n_cls] the frozen task head."""
    from cmi.eval.graph_leakage import compute_label_domain_prior, conditional_kl_to_prior
    torch.manual_seed(int(seed)); np.random.seed(int(seed))
    Zt = torch.tensor(np.asarray(Z, float), dtype=torch.float32, device=device)
    yt = torch.tensor(np.asarray(y), dtype=torch.long, device=device)
    dt = torch.tensor(np.asarray(d), dtype=torch.long, device=device)
    Wt = torch.tensor(np.asarray(W, float), dtype=torch.float32, device=device)
    bt = torch.tensor(np.asarray(b, float), dtype=torch.float32, device=device)
    y_oh = F.one_hot(yt, n_cls).float()
    dd = Z.shape[1]
    pi_y = compute_label_domain_prior(y, d, n_cls, n_dom, smoothing).to(device)

    m = nn.Parameter(torch.full((dd,), M_INIT, device=device))   # init large-positive -> gate ~1 -> vanilla
    P = _bias_head(dd, n_cls, n_dom, hidden).to(device)
    opt_m = torch.optim.Adam([m], lr=lr_m)
    opt_p = torch.optim.Adam(P.parameters(), lr=lr_p)
    tau = float(tau0)
    hist = []
    for _t in range(int(n_temps)):
        # (A) train the bias head P to EXTRACT subject from the current masked z (mask frozen; no grad to m)
        with torch.no_grad():
            zm = Zt * gate(m.detach(), tau)
        for _ in range(int(p_epochs)):
            opt_p.zero_grad()
            F.cross_entropy(P(torch.cat([zm, y_oh], 1)), dt).backward(); opt_p.step()
        # (B) optimize the mask m: task CE (frozen head) + gamma * label-conditional leakage of P (minimize)
        for _ in range(int(inner_epochs)):
            zmask = Zt * gate(m, tau)
            logits_task = zmask @ Wt.t() + bt                # frozen task head C
            ce = F.cross_entropy(logits_task, yt)
            probs_d = F.softmax(P(torch.cat([zmask, y_oh], 1)), 1)
            leak = conditional_kl_to_prior(probs_d, yt, pi_y).mean()   # Ihat(D;P|Y) proxy
            opt_m.zero_grad(); (ce + gamma * leak).backward(); opt_m.step()
        hist.append(dict(tau=tau, kept=int((m.detach() > 0).sum()), ce=float(ce), leak=float(leak)))
        tau *= 0.5
    scores = m.detach().cpu().numpy()                        # per-neuron importance: LOW m => prune first (subject-carrying)
    mask = (scores > 0).astype(float)                        # FFW default hard mask (1=kept)
    return scores, mask, {"history": hist, "n_kept": int(mask.sum()),
                          "n_masked": int((mask == 0).sum()), "gamma": gamma}


def apply_mask(Z, mask):
    return np.asarray(Z, float) * np.asarray(mask, float)[None, :]


def mask_from_scores(scores, n_masked):
    """Keep all but the `n_masked` LOWEST-score neurons (FFW learns low score = subject-carrying)."""
    d = len(scores); mask = np.ones(d)
    if n_masked > 0:
        mask[np.argsort(scores)[:min(n_masked, d)]] = 0.0
    return mask


def random_mask(d, n_masked, seed):
    rng = np.random.default_rng(seed)
    mask = np.ones(d)
    if n_masked > 0:
        mask[rng.choice(d, size=min(n_masked, d), replace=False)] = 0.0
    return mask


def prune_frontier(Z, y, d, W, b, scores, cmi_fn, task_fn, ks, seed=0):
    """FFW neuron-importance frontier: for each prune count k in `ks`, prune the k lowest-score neurons and
    report (source task bAcc, subject CMI) vs a same-count RANDOM prune. Returns the frontier + the task-safe
    best (max CMI reduction s.t. source-task drop <= 0.02); identity if none. task_fn/cmi_fn take masked Z."""
    d_dim = Z.shape[1]
    full_task = task_fn(Z); full_cmi = cmi_fn(Z)
    rows = []
    for k in ks:
        m_ffw = mask_from_scores(scores, k); m_rnd = random_mask(d_dim, k, seed)
        zf, zr = apply_mask(Z, m_ffw), apply_mask(Z, m_rnd)
        rows.append({"k": int(k), "ffw_task": task_fn(zf), "ffw_cmi": cmi_fn(zf),
                     "rand_task": task_fn(zr), "rand_cmi": cmi_fn(zr)})
    safe = [r for r in rows if (full_task - r["ffw_task"]) <= 0.02 and r["k"] > 0]
    best = max(safe, key=lambda r: full_cmi - r["ffw_cmi"]) if safe else {"k": 0, "ffw_task": full_task,
                                                                          "ffw_cmi": full_cmi, "rand_cmi": full_cmi}
    return {"full_task": full_task, "full_cmi": full_cmi, "frontier": rows, "task_safe_best": best,
            "task_safe_exists": bool(best["k"] > 0 and (full_cmi - best["ffw_cmi"]) > 0)}
