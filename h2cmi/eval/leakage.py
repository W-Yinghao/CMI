"""Cross-fitted hierarchical conditional-leakage estimate on a FROZEN encoder.

Reports, per penalised factor j, the SIGNED held-out estimate

    I_hat_j = H(D_j | Y, Pa) - H_psi(D_j | Z_c, Y, Pa)

with critics fit on one fold and evaluated on the held-out fold (cross-fitted, averaged
both ways), plus a within-(Y,Pa) permutation null so a positive estimate can be tested
against estimator noise (review: keep signed estimates + a permutation/bootstrap null;
do NOT truncate negatives, which biases upward).
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from h2cmi.cmi.hierarchical import ConditionalDomainCritic, reference_conditional_entropy
from h2cmi.domains import DomainDAG, DomainLabels


def _perm_within(d: np.ndarray, ctx: np.ndarray, rng) -> np.ndarray:
    out = d.copy()
    for c in np.unique(ctx):
        idx = np.where(ctx == c)[0]
        if len(idx) > 1:
            out[idx] = rng.permutation(out[idx])
    return out


def _fit_critic(Zfit, yfit, pkfit, dfit, n_levels, n_cls, n_pa, device, epochs, lr, seed):
    torch.manual_seed(seed)
    crit = ConditionalDomainCritic(Zfit.shape[1], n_levels, n_cls, n_pa).to(device)
    opt = torch.optim.Adam(crit.parameters(), lr=lr)
    Z = torch.as_tensor(Zfit, dtype=torch.float32, device=device)
    y = torch.as_tensor(yfit, dtype=torch.long, device=device)
    pk = torch.as_tensor(pkfit, dtype=torch.long, device=device)
    d = torch.as_tensor(dfit, dtype=torch.long, device=device)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(crit(Z, y, pk), d).backward()
        opt.step()
    return crit


@torch.no_grad()
def _eval_signed(crit, Zev, yev, pkev, dev_, n_levels, n_cls, n_pa, device):
    Z = torch.as_tensor(Zev, dtype=torch.float32, device=device)
    y = torch.as_tensor(yev, dtype=torch.long, device=device)
    pk = torch.as_tensor(pkev, dtype=torch.long, device=device)
    logits = crit(Z, y, pk)
    ce = float(F.cross_entropy(logits, torch.as_tensor(dev_, dtype=torch.long, device=device)))
    h_ref = reference_conditional_entropy(dev_, yev, pkev, n_levels, n_cls, n_pa)
    dom_acc = float((logits.argmax(1).cpu().numpy() == dev_).mean())
    return h_ref - ce, ce, h_ref, dom_acc


def crossfit_conditional_leakage(Z: np.ndarray, y: np.ndarray, domains: DomainLabels,
                                 dag: DomainDAG, n_classes: int, device: str = "cpu",
                                 epochs: int = 120, lr: float = 2e-3, n_perm: int = 0,
                                 seed: int = 0) -> dict:
    """Per-factor signed cross-fitted leakage with optional permutation null."""
    rng = np.random.default_rng(seed)
    N = len(y)
    perm = rng.permutation(N)
    fold = np.zeros(N, dtype=int); fold[perm[: N // 2]] = 0; fold[perm[N // 2:]] = 1
    out = {}
    for f in dag.penalised_factors():
        n_levels = f.n_levels
        pidx = dag.parent_indices(f.name)
        n_pa = int(np.prod([dag.factors[j].n_levels for j in pidx])) if pidx else 1
        d_all = domains.factor(f.name)
        pk_all = domains.parent_key(f.name)
        signed, ce_, href_, acc_ = [], [], [], []
        for fit_fold in (0, 1):
            fit = fold == fit_fold
            ev = fold != fit_fold
            if fit.sum() < 4 or ev.sum() < 4:
                continue
            crit = _fit_critic(Z[fit], y[fit], pk_all[fit], d_all[fit],
                               n_levels, n_classes, n_pa, device, epochs, lr, seed + fit_fold)
            s, ce, href, acc = _eval_signed(crit, Z[ev], y[ev], pk_all[ev], d_all[ev],
                                            n_levels, n_classes, n_pa, device)
            signed.append(s); ce_.append(ce); href_.append(href); acc_.append(acc)
            if n_perm:
                ctx_ev = y[ev].astype(np.int64) * n_pa + pk_all[ev].astype(np.int64)
                nulls = []
                for k in range(n_perm):
                    d_null = _perm_within(d_all[ev], ctx_ev, rng)
                    sn, *_ = _eval_signed(crit, Z[ev], y[ev], pk_all[ev], d_null,
                                          n_levels, n_classes, n_pa, device)
                    nulls.append(sn)
                out.setdefault(f"{f.name}__nulls", []).extend(nulls)
        rec = dict(I_hat=float(np.mean(signed)) if signed else float("nan"),
                   ce=float(np.mean(ce_)) if ce_ else float("nan"),
                   h_ref=float(np.mean(href_)) if href_ else float("nan"),
                   cond_dom_acc=float(np.mean(acc_)) if acc_ else float("nan"),
                   budget=f.budget)
        if n_perm and f"{f.name}__nulls" in out:
            nulls = np.asarray(out.pop(f"{f.name}__nulls"))
            q = float(np.quantile(nulls, 0.95))
            rec.update(null_q95=q, null_mean=float(nulls.mean()),
                       excess=float(max(rec["I_hat"] - q, 0.0)))
        out[f.name] = rec
    return out
