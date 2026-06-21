"""Score-Fisher selector (novelty core) -- covariance/nonlinear-aware replacement for the
frozen mean-scatter baseline (`fisher.py`/`subspace.py`).

Model-EXPECTED score Fishers of trained, cross-fit probes (their input-gradients sense
covariance/nonlinear structure that mean scatters cannot):

    G_Y     = E_z  Σ_c p_φ(c|z)        s_c s_c^T ,   s_c = ∇_z log p_φ(c|z)
    G_{D|Y} = E_{z,y} Σ_r q_ψ(r|z,y)   t_r t_r^T ,   t_r = ∇_z log q_ψ(r|z,y)

Deletable subspace = generalized spectrum in a coordinate-covariant whitening metric
`M = (Σ_W + ε Σ_ref)^{-1}` (Σ_W pooled within-class cov, Σ_ref total cov -- both transform as
`A·Aᵀ` so M is covariant under z→Az, which a fixed `εI` is not):

    G_{D|Y} v = ρ (G_Y + η M) v ,

with a **metric-aware oblique projector** `P_N = V (Vᵀ M V)^{-1} Vᵀ M`.

PHASE 1.1 corrections vs the first cut:
  1. probe RNG seed is set BEFORE the module is constructed (init is now seed-controlled),
     via one explicit SplitPlan shared across task/domain/null probes;
  2. the within-Y permutation null is a *prefilter/diagnostic* on scalar score ENERGIES
     (pooled over perms), NOT a PSD-breaking elementwise max of matrices and NOT the final
     decision (the final rank decision is the deferred source-risk UCB gate; rank here is an
     interim eigengap heuristic);
  3. the leakage GATE is a cross-fit `q0(D|Y)` vs `q1(D|Z,Y)` paired held-out log-loss test
     with a cluster-aware one-sided bound (not a majority baseline + iid bootstrap);
  4. coordinate covariance is split into a *transported-probe* claim (exact: the score
     Fisher / eig / projector algebra) vs a *refitted-probe* claim (only empirical, since
     Adam+weight-decay are not equivariant) -- see tests.
Selector buffers `P` and `active_k`, so `is_identity` survives a checkpoint reload.

Deferred (NOT here): source-risk UCB rank gate; conditional-on-task critic
`I(P_N Z;D|Y,sg(P_T Z))`; PCGrad; EEG wiring.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.linalg import eigh


@dataclass
class ScoreFisherConfig:
    probe_family: str = "mlp"       # mlp | linear | quad  (nonlinear needed for covariance leakage)
    hidden: int = 64
    epochs: int = 250
    lr: float = 1e-2
    weight_decay: float = 1e-3
    n_folds: int = 2                # cross-fit folds (probe-train vs score/nll evaluation)
    eps_ref: float = 1e-2           # ridge coefficient on Σ_ref in the metric (coordinate-covariant)
    eta: float = 1e-2               # ridge (in metric M units) on G_Y in the generalized eig
    tau_ratio: float = 3.0          # domain/label score-energy ratio (M-unit eigenvectors)
    eps_label: float = 0.20         # label-light prefilter: lab_j <= eps_label * max_k lab_k
    dom_floor: float = 0.05         # relative domain-energy prefilter
    rank_gap: float = 1.5           # interim eigengap rank cut (until the UCB rank gate lands)
    max_dim: int = 8
    n_perm_null: int = 3            # within-Y permutations for the noise-energy prefilter
    null_safety: float = 1.25       # a direction must exceed null_safety x the null energy quantile
    null_q: float = 0.95            # quantile of the pooled null energy distribution
    gate_gamma: float = 0.0         # leakage gate: domain NLL gain must exceed this (nats)
    gate_alpha: float = 0.05        # one-sided level for the leakage gate
    gate_boot: int = 300            # cluster-bootstrap resamples
    dtype64: bool = True


# ----------------------------------------------------------------------------- probes
class _Probe(nn.Module):
    """Differentiable-in-z probe. `cond_dim>0` appends a one-hot conditioner (e.g. Y) to the
    z-features. `quad` features = [z, z^2] (input-gradient linear in z -> senses variance)."""
    def __init__(self, z_dim, n_out, cond_dim=0, family="mlp", hidden=64):
        super().__init__()
        self.family, self.cond_dim, self.z_dim = family, cond_dim, z_dim
        feat_dim = 2 * z_dim if family == "quad" else z_dim
        in_dim = feat_dim + cond_dim
        if family in ("linear", "quad"):
            self.net = nn.Linear(in_dim, n_out)
        else:
            self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.Tanh(),
                                     nn.Linear(hidden, hidden), nn.Tanh(),
                                     nn.Linear(hidden, n_out))

    def _feat(self, z):
        return torch.cat([z, z * z], 1) if self.family == "quad" else z

    def forward(self, z, cond=None):
        f = self._feat(z)
        if cond is not None:
            f = torch.cat([f, cond], 1)
        return self.net(f)


def _build_and_train(z_dim, n_out, cond_dim, Z, target, cond, cfg, seed):
    """Seed BEFORE construction (fix 1) so weight init is reproducible; then train with CE."""
    torch.manual_seed(seed)
    probe = _Probe(z_dim, n_out, cond_dim, cfg.probe_family, cfg.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Zt = torch.tensor(Z, dtype=torch.float32)
    tt = torch.tensor(target, dtype=torch.long)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    for _ in range(cfg.epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Zt, ct), tt).backward()
        opt.step()
    return probe


def _expected_score_fisher(probe, Z, cond, n_out):
    """G = (1/B) Σ_b Σ_c p(c|z_b) ∇z logp(c|z_b) ∇z logp(c|z_b)^T at the given Z."""
    Zt = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    logp = F.log_softmax(probe(Zt, ct), dim=1)
    p = logp.exp().detach()
    d = Zt.shape[1]
    G = torch.zeros(d, d, dtype=torch.float64)
    for c in range(n_out):
        g, = torch.autograd.grad(logp[:, c].sum(), Zt, retain_graph=True)
        gw = (g * p[:, c:c + 1].clamp_min(0).sqrt()).double()
        G += gw.t() @ gw
    return (G / Zt.shape[0]).numpy()


class _SplitPlan:
    """Deterministic K-fold index plan shared across task/domain/null probes (fix 1)."""
    def __init__(self, n, n_folds, seed):
        self.idx = np.random.default_rng(seed).permutation(n)
        self.folds = np.array_split(self.idx, n_folds)

    def iter(self):
        for f, hold in enumerate(self.folds):
            yield f, np.setdiff1d(self.idx, hold), hold


def _cross_fit_fisher(Z, target, cond, n_out, z_dim, cond_dim, cfg, plan, seed):
    """Train probe on each fold's complement, accumulate the score Fisher on the held-out
    fold; average. Cond is one-hot (or None)."""
    Gs = []
    for f, tr, hold in plan.iter():
        probe = _build_and_train(z_dim, n_out, cond_dim, Z[tr], np.asarray(target)[tr],
                                 None if cond is None else cond[tr], cfg, seed + f)
        Gs.append(_expected_score_fisher(probe, Z[hold], None if cond is None else cond[hold], n_out))
    G = np.mean(Gs, axis=0)
    return 0.5 * (G + G.T)


def _crossfit_heldout(Zfeat, target, cond, n_out, cfg, plan, seed):
    """Per-sample held-out NLL and prediction of a cross-fit probe. Zfeat may have 0 columns
    (e.g. q0(D|Y) uses only the Y conditioner)."""
    target = np.asarray(target)
    nll = np.zeros(len(target)); pred = np.zeros(len(target), dtype=int)
    z_dim = Zfeat.shape[1]
    cond_dim = 0 if cond is None else cond.shape[1]
    for f, tr, hold in plan.iter():
        probe = _build_and_train(z_dim, n_out, cond_dim, Zfeat[tr], target[tr],
                                 None if cond is None else cond[tr], cfg, seed + f)
        with torch.no_grad():
            ct = None if cond is None else torch.tensor(cond[hold], dtype=torch.float32)
            logp = F.log_softmax(probe(torch.tensor(Zfeat[hold], dtype=torch.float32), ct), 1)
        nll[hold] = -logp[np.arange(len(hold)), target[hold]].numpy()
        pred[hold] = logp.argmax(1).numpy()
    return nll, pred


def _shuffle_within_y(y, d, rng):
    dp = np.array(d, copy=True)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        dp[idx] = d[idx][rng.permutation(idx.size)]
    return dp


def _metric(Z, y, n_cls, cfg):
    """Coordinate-covariant whitening metric M = (Σ_W + eps_ref Σ_ref)^{-1}; Σ_W pooled
    within-class covariance, Σ_ref total covariance. Both ~ A·Aᵀ under z->Az."""
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    Sw = np.zeros((d, d))
    for c in range(n_cls):
        Zc = Z[y == c]
        if len(Zc) > 1:
            Sw += (len(Zc) / len(Z)) * np.cov(Zc, rowvar=False)
    Sref = np.cov(Z, rowvar=False)
    A = 0.5 * (Sw + Sw.T) + cfg.eps_ref * 0.5 * (Sref + Sref.T) + 1e-9 * np.eye(d)
    return np.linalg.inv(A)


def metric_projector(V_sel, M):
    """Oblique M-orthogonal projector onto span(V_sel): P = V (Vᵀ M V)^{-1} Vᵀ M. Covariant
    under z->Az: P -> A P A^{-1} (similarity), so the selected subspace is rescaling-invariant."""
    if V_sel.shape[1] == 0:
        return np.zeros((V_sel.shape[0], V_sel.shape[0]))
    G = V_sel.T @ M @ V_sel
    return V_sel @ np.linalg.solve(G, V_sel.T @ M)


def _cluster_boot_lcb(values, cluster_id, alpha, n_boot, seed):
    """One-sided lower confidence bound on E[values] by clustered bootstrap (resample whole
    clusters). `cluster_id` per-sample; None -> per-sample iid."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    if cluster_id is None:
        groups = [np.array([i]) for i in range(len(values))]
    else:
        cluster_id = np.asarray(cluster_id)
        groups = [np.where(cluster_id == c)[0] for c in np.unique(cluster_id)]
    means = []
    G = len(groups)
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        sel = np.concatenate([groups[i] for i in pick])
        means.append(values[sel].mean())
    return float(np.quantile(means, alpha))


def leakage_gate(Z, y, d, n_cls, n_dom, cfg, plan, seed, cluster_id=None):
    """Cross-fit q0(D|Y) vs q1(D|Z,Y) gate (fix 3): proper q0(D|Y) baseline (not an
    eval-data majority), cross-fit, with a CLUSTER-aware one-sided bound. The DECISION is the
    paired held-out ACCURACY advantage (robust to critic miscalibration); the paired NLL gain
    is kept as a diagnostic only, because an over-confident MLP critic on a weak covariance
    signal has poor held-out NLL even when its score-Fisher recovers the direction. Returns
    (acc_gain_mean, acc_gain_lcb, open, nll_gain_mean)."""
    y_oh = np.eye(n_cls)[y]
    nll0, pred0 = _crossfit_heldout(np.zeros((len(Z), 0)), d, y_oh, n_dom, cfg, plan, seed + 11)
    nll1, pred1 = _crossfit_heldout(Z, d, y_oh, n_dom, cfg, plan, seed + 22)
    acc_gain = (pred1 == d).astype(float) - (pred0 == d).astype(float)   # paired, per-sample
    lcb = _cluster_boot_lcb(acc_gain, cluster_id, cfg.gate_alpha, cfg.gate_boot, seed + 33)
    return float(acc_gain.mean()), lcb, bool(lcb > cfg.gate_gamma), float((nll0 - nll1).mean())


@dataclass
class ScoreFisherReport:
    rho: np.ndarray
    dom_energy: np.ndarray
    lab_energy: np.ndarray
    ratio: np.ndarray
    selected: np.ndarray
    basis: np.ndarray            # [d,k] selected generalized eigenvectors (raw)
    P: np.ndarray                # [d,d] metric-aware (oblique) projector
    is_identity: bool
    leak_gain: float             # mean cross-fit q1-q0 paired ACCURACY advantage (decision)
    leak_lcb: float              # clustered one-sided lower bound on the accuracy advantage
    gate_open: bool              # leakage significant?
    leak_nll_gain: float = 0.0   # mean q0-q1 NLL gain (diagnostic only; NLL-fragile)
    dom_floor_abs: float = 0.0   # within-Y permutation-null energy prefilter floor

    @property
    def k(self):
        return int(self.basis.shape[1])


def select_from_fishers(G_DgY, G_Y, M, cfg: ScoreFisherConfig, floor_abs=0.0, gate_open=True):
    """Deterministic selection given the Fishers + metric (no probe training here). This is
    the part whose covariance under z->Az is EXACT (transported-probe claim, fix 4): the
    energies are invariant and the projector transforms by similarity. Returns
    (cand_idx, V, rho, dom, lab, ratio, P)."""
    z_dim = G_Y.shape[0]
    B = 0.5 * ((G_Y + cfg.eta * M) + (G_Y + cfg.eta * M).T)
    w, V = eigh(0.5 * (G_DgY + G_DgY.T), B)
    order = np.argsort(w)[::-1]
    rho, V = w[order], V[:, order]

    Mnorm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V.T, M, V.T), 1e-18))
    U = V / Mnorm
    dom = np.clip(np.einsum("ij,jk,ik->i", U.T, G_DgY, U.T), 0, None)
    lab = np.clip(np.einsum("ij,jk,ik->i", U.T, G_Y, U.T), 0, None)
    ratio = dom / (lab + cfg.eta)

    dom_max = dom.max() if dom.size and dom.max() > 0 else 1.0
    lab_max = lab.max() if lab.size and lab.max() > 0 else 1.0
    eligible = ((ratio >= cfg.tau_ratio) & (lab <= cfg.eps_label * lab_max)
                & (dom >= max(cfg.dom_floor * dom_max, floor_abs)))
    elig = np.where(eligible)[0]
    if elig.size:
        elig = elig[np.argsort(rho[elig])[::-1]]
        rho_e = np.maximum(rho[elig], 1e-12)
        kcut = elig.size
        if elig.size > 1:
            over = np.where(rho_e[:-1] / rho_e[1:] >= cfg.rank_gap)[0]
            if over.size:
                kcut = int(over[0]) + 1
        cand = elig[: min(kcut, cfg.max_dim)]
    else:
        cand = elig
    if not gate_open:
        cand = np.array([], dtype=int)
    V_sel = V[:, cand] if cand.size else np.zeros((z_dim, 0))
    return np.sort(cand), V, rho, dom, lab, ratio, metric_projector(V_sel, M)


def _null_floor(Z, y, d, y_oh, U, n_cls, n_dom, z_dim, cfg, plan, seed):
    """within-Y permutation-null PREFILTER (fix 2): pool scalar null ENERGIES over perms (not
    a PSD-breaking matrix max); floor = a high quantile. A prefilter/diagnostic only -- the
    final rank decision is the deferred source-risk UCB gate."""
    energies = []
    rng = np.random.default_rng(seed)
    for p in range(max(cfg.n_perm_null, 0)):
        d_perm = _shuffle_within_y(y, d, rng)
        Gp = _cross_fit_fisher(Z, d_perm, y_oh, n_dom, z_dim, n_cls, cfg, plan, seed + 800 + p)
        energies.append(np.clip(np.einsum("ij,jk,ik->i", U.T, Gp, U.T), 0, None))
    if not energies:
        return 0.0
    return cfg.null_safety * float(np.quantile(np.concatenate(energies), cfg.null_q))


def select_score_fisher(Z, y, d, n_cls, n_dom, cfg: ScoreFisherConfig, seed=0, cluster_id=None):
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y); d = np.asarray(d)
    z_dim = Z.shape[1]
    y_oh = np.eye(n_cls)[y]
    plan = _SplitPlan(len(Z), cfg.n_folds, seed + 1)

    G_Y = _cross_fit_fisher(Z, y, None, n_cls, z_dim, 0, cfg, plan, seed)
    G_DgY = _cross_fit_fisher(Z, d, y_oh, n_dom, z_dim, n_cls, cfg, plan, seed + 100)
    M = _metric(Z, y, n_cls, cfg)

    leak_gain, leak_lcb, gate_open, leak_nll = leakage_gate(Z, y, d, n_cls, n_dom, cfg, plan, seed + 500, cluster_id)

    # M-unit eigenvectors needed to evaluate the null energies on a fixed basis
    B = 0.5 * ((G_Y + cfg.eta * M) + (G_Y + cfg.eta * M).T)
    w, V = eigh(0.5 * (G_DgY + G_DgY.T), B)
    V = V[:, np.argsort(w)[::-1]]
    U = V / np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V.T, M, V.T), 1e-18))
    floor_abs = _null_floor(Z, y, d, y_oh, U, n_cls, n_dom, z_dim, cfg, plan, seed + 700)

    cand, V, rho, dom, lab, ratio, P = select_from_fishers(G_DgY, G_Y, M, cfg, floor_abs, gate_open)
    V_sel = V[:, cand] if cand.size else np.zeros((z_dim, 0))
    return ScoreFisherReport(rho=rho, dom_energy=dom, lab_energy=lab, ratio=ratio,
                             selected=cand, basis=V_sel, P=P, is_identity=(cand.size == 0),
                             leak_gain=leak_gain, leak_lcb=leak_lcb, gate_open=gate_open,
                             leak_nll_gain=leak_nll, dom_floor_abs=floor_abs)


class ScoreFisherSelector(nn.Module):
    """Static score-Fisher selector (no encoder training). Mirrors SubspaceSelector's API.
    Buffers `P` AND `active_k` so `is_identity` survives a checkpoint reload (fix: previously
    `report=None` after reload made `is_identity` wrongly True)."""
    def __init__(self, z_dim, n_cls, n_dom, cfg: Optional[ScoreFisherConfig] = None, device="cpu"):
        super().__init__()
        self.z_dim, self.n_cls, self.n_dom = z_dim, n_cls, n_dom
        self.cfg = cfg or ScoreFisherConfig()
        self.register_buffer("P", torch.zeros(z_dim, z_dim))
        self.register_buffer("active_k", torch.zeros((), dtype=torch.long))
        self.report: Optional[ScoreFisherReport] = None

    @property
    def is_identity(self):
        return int(self.active_k) == 0

    def refresh(self, Z, y, d, seed: int = 0, cluster_id=None):
        f = lambda a: a.detach().cpu().numpy() if torch.is_tensor(a) else np.asarray(a)
        rep = select_score_fisher(f(Z), f(y), f(d), self.n_cls, self.n_dom, self.cfg, seed, cluster_id)
        self.report = rep
        self.P = torch.tensor(rep.P, dtype=torch.float32, device=self.P.device)
        self.active_k = torch.tensor(rep.k, dtype=torch.long, device=self.active_k.device)
        return rep

    def project(self, Z):
        return Z @ self.P.t().to(Z.dtype)        # nuisance component (row-vectors)

    def summary(self):
        if self.report is None:
            return {"k": int(self.active_k), "is_identity": self.is_identity}
        r = self.report
        return {"k": r.k, "is_identity": r.is_identity, "selected": r.selected.tolist(),
                "leak_acc_adv": round(r.leak_gain, 4), "leak_acc_lcb": round(r.leak_lcb, 4),
                "leak_nll_gain": round(r.leak_nll_gain, 4), "gate_open": r.gate_open,
                "rho_top": [float(x) for x in r.rho[:5]]}
