"""MCC estimator audit — EXACT full-source MCC gradient (two-pass, BN-frozen) vs the K=4 / K=16 episodic estimators.
Frozen diagnostic (NO training): tests whether the episodic estimator is a poor estimate of the exact population
MCC gradient. Cannot and does not test geometry->DG. EEGNet BatchNorm is kept in eval() (frozen running stats), so
z_i is an independent function of x_i and per-sample micro-batching is EXACT. Manuscript FROZEN; only the project
owner stops a scientific line."""
from __future__ import annotations
import numpy as np
import torch
import torch.nn.functional as F

from tos_cmi.train.mechanism_consistency import class_pairs, mcc_loss, BalancedSubjectClassSampler

EPS = 1e-6


def _shuffle_within_class_np(y, d, rng):
    d = np.asarray(d).copy()
    for c in np.unique(y):
        idx = np.where(y == c)[0]; d[idx] = d[idx][rng.permutation(len(idx))]
    return d


def mcc_loss_from_means(means, subs, classes, pairs, eps=EPS):
    """L_MCC computed directly from per-(subject,class) mean tensors (leaf); same LOSO-consensus + stop-grad form as
    mechanism_consistency.mcc_loss, but starting from prototypes (for the exact population gradient Pass 1)."""
    terms = []
    for (a, b) in pairs:
        U = torch.stack([F.normalize(means[(s, a)] - means[(s, b)], dim=0, eps=eps) for s in subs])
        Udet = U.detach(); tot = Udet.sum(0)
        for i in range(len(subs)):
            ubar = F.normalize(tot - Udet[i], dim=0, eps=eps)
            terms.append(1.0 - torch.dot(U[i], ubar))
    return torch.stack(terms).mean()


def _forward_z(bb, X, device, bs=256, grad=False):
    ctx = torch.enable_grad() if grad else torch.no_grad()
    with ctx:
        return [bb(torch.tensor(X[i:i + bs], dtype=torch.float32).to(device))[1] for i in range(0, len(X), bs)]


def exact_population_gradient(bb, X, y, d, device, shuffle=False, rng=None, bs=256):
    """Two-pass EXACT full-source MCC gradient w.r.t. the encoder params. Returns
    (g_theta [flat np], means {(d,c): np}, g_mu {(d,c): np}, loss). BN MUST be frozen (eval)."""
    bb.eval()
    assert not bb.training, "BatchNorm must be frozen (eval) for the exact two-pass gradient"
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    if shuffle:
        d = _shuffle_within_class_np(y, d, rng)
    classes = sorted(np.unique(y).tolist()); subs = sorted(np.unique(d).tolist()); pairs = class_pairs(classes)
    cell = {(s, c): np.where((d == s) & (y == c))[0] for s in subs for c in classes}
    for k, idx in cell.items():
        if len(idx) == 0:
            raise ValueError(f"MCC audit: empty subject-class cell {k} (fail loud, do not reweight)")
    # Pass 1: exact prototypes (leaf) -> population MCC -> prototype gradient g_mu
    z_all = torch.cat([z.detach().cpu() for z in _forward_z(bb, X, device, bs, grad=False)])
    means = {k: z_all[idx].mean(0).clone().to(device).requires_grad_(True) for k, idx in cell.items()}
    counts = {k: len(idx) for k, idx in cell.items()}
    L = mcc_loss_from_means(means, subs, classes, pairs)
    keys = list(cell); gmu_list = torch.autograd.grad(L, [means[k] for k in keys])
    gmu = {k: gmu_list[i].detach() for i, k in enumerate(keys)}
    # Pass 2: exact backprop to theta via z.backward(upstream = g_mu[cell(i)]/n_cell(i)), accumulated over micro-batches
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
    # params AFTER z (the classifier head) are not in z's graph -> MCC gradient is legitimately zero there.
    g_theta = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).flatten()
                         for p in bb.parameters()]).detach().cpu().numpy()
    return (g_theta, {k: means[k].detach().cpu().numpy() for k in means},
            {k: gmu[k].detach().cpu().numpy() for k in gmu}, float(L))


def episodic_theta_gradients(bb, X, y, d, device, K, R, seed, shuffle=False, bs=512):
    """R independent balanced-K episodes -> R encoder-param gradients (flat np). BN frozen (eval)."""
    bb.eval()
    y = np.asarray(y).astype(int); d = np.asarray(d).astype(int)
    samp = BalancedSubjectClassSampler(d, y, K=K, n_batches=R, seed=seed)
    gen = torch.Generator(device="cpu").manual_seed(seed + 991)
    out = []
    params = list(bb.parameters())
    for local in samp:
        z = bb(torch.tensor(X[local], dtype=torch.float32).to(device))[1]
        L, _ = mcc_loss(z, y[local], d[local], shuffle_subjects=shuffle, generator=gen)
        g = torch.autograd.grad(L, params, allow_unused=True)   # head params (post-z) legitimately get None -> 0
        out.append(torch.cat([(gi if gi is not None else torch.zeros_like(p)).flatten()
                              for gi, p in zip(g, params)]).detach().cpu().numpy())
    return np.asarray(out)


def _cos(a, b):
    na = np.linalg.norm(a); nb = np.linalg.norm(b)
    return float(a @ b / (na * nb + 1e-12))


def gradient_diagnostics(g_full, g_K):
    """A_K (alignment), B_K (relative bias), SNR_K from R episodic gradients g_K [R, P] vs exact g_full [P]."""
    mean = g_K.mean(0)
    var = float(np.mean(np.sum((g_K - mean) ** 2, axis=1)))
    return dict(A_K=_cos(mean, g_full), B_K=float(np.linalg.norm(mean - g_full) / (np.linalg.norm(g_full) + 1e-12)),
                SNR_K=float(np.sum(mean ** 2) / (var + 1e-12)), mean_grad_norm=float(np.linalg.norm(mean)),
                full_grad_norm=float(np.linalg.norm(g_full)))


# ---- prototype-space normalized one-step WSCI (scale-vs-direction) ----
def wsci_from_means(means, subs, classes, pairs):
    """Source direction-consistency (WSCI) on prototype means = mean LOSO-consensus cosine of unit class contrasts."""
    cons = []
    for (a, b) in pairs:
        U = np.array([means[(s, a)] - means[(s, b)] for s in subs])
        U = U / (np.linalg.norm(U, axis=1, keepdims=True) + 1e-12); tot = U.sum(0)
        for i in range(len(subs)):
            ub = tot - U[i]; ub = ub / (np.linalg.norm(ub) + 1e-12); cons.append(float(U[i] @ ub))
    return float(np.mean(cons)) if cons else float("nan")


def one_step_prototype_wsci(means, g_mu, subs, classes, pairs, alpha=0.1):
    """Normalize the prototype gradient to unit norm, take a step M' = M - alpha * g/||g|| in prototype space,
    return WSCI(M') - WSCI(M). Answers: does this estimator's DIRECTION move the population consistency?"""
    keys = [(s, c) for s in subs for c in classes]
    gvec = np.concatenate([g_mu[k].ravel() for k in keys]); gn = gvec / (np.linalg.norm(gvec) + 1e-12)
    base = wsci_from_means(means, subs, classes, pairs)
    stepped = {}; off = 0
    for k in keys:
        p = means[k].size; stepped[k] = means[k] - alpha * gn[off:off + p].reshape(means[k].shape); off += p
    return wsci_from_means(stepped, subs, classes, pairs) - base


def prototype_gradient_from_means(means, subs, classes, pairs):
    """g_mu for an arbitrary set of prototype means (cheap; no encoder). Used for the K-episode one-step direction."""
    mt = {k: torch.tensor(means[k], dtype=torch.float64, requires_grad=True) for k in means}
    L = mcc_loss_from_means(mt, subs, classes, pairs)
    g = torch.autograd.grad(L, [mt[k] for k in mt])
    return {k: g[i].detach().numpy() for i, k in enumerate(mt)}


def episodic_prototype_one_step_wsci(bb, X, y, d, device, K, R, seed, full_means, subs, classes, pairs, alpha=0.1, bs=512):
    """Mean over R K-episodes of the one-step WSCI movement of the FULL prototypes stepped in each episode's (noisy)
    prototype-gradient direction. Compares the K-estimator DIRECTION's effect on population WSCI vs the exact one."""
    bb.eval()
    ya = np.asarray(y).astype(int); da = np.asarray(d).astype(int)
    samp = BalancedSubjectClassSampler(da, ya, K=K, n_batches=R, seed=seed)
    dws = []
    for local in samp:
        with torch.no_grad():
            z = torch.cat([bb(torch.tensor(X[local[i:i + bs]], dtype=torch.float32).to(device))[1].cpu()
                           for i in range(0, len(local), bs)]).numpy()
        yl, dl = ya[local], da[local]
        em = {(s, c): z[(dl == s) & (yl == c)].mean(0) for s in subs for c in classes}
        if any(np.isnan(em[k]).any() for k in em):
            continue
        gmu_e = prototype_gradient_from_means(em, subs, classes, pairs)
        dws.append(one_step_prototype_wsci(full_means, gmu_e, subs, classes, pairs, alpha))
    return float(np.mean(dws)) if dws else float("nan")
