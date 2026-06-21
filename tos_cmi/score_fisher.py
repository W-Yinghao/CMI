"""Score-Fisher selector (Phase 1 of the novelty core) -- the covariance/nonlinear-aware
replacement for the frozen mean-scatter baseline (`fisher.py`/`subspace.py`).

The mean-scatter Fishers see only group MEAN shifts, so they are blind to task/domain
information carried in covariance or nonlinear structure (`tests/test_limits.py`). Here we
instead use **model-expected score Fishers** of *trained probes*, whose input-gradients pick
up exactly that structure:

    G_Y     = E_z  Σ_c p_φ(c|z)        s_c s_c^T ,   s_c   = ∇_z log p_φ(c|z)
    G_{D|Y} = E_{z,y} Σ_r q_ψ(r|z,y)   t_r t_r^T ,   t_r   = ∇_z log q_ψ(r|z,y)

with `p_φ` the task posterior probe and `q_ψ` the conditional-domain critic. A *nonlinear*
probe (MLP / quadratic) makes `∇_z log q` large along directions where D is decodable from
Z|Y even through variance -- which is why this recovers covariance leakage a mean-scatter
selector cannot. Both probes are **cross-fit**: trained on one split, scores accumulated on a
disjoint split, so the Fishers are not overfit-inflated.

The deletable subspace is the generalized spectrum in the **whitening metric**
`M = (Σ_W + εI)^{-1}` (Σ_W = pooled within-class covariance):

    G_{D|Y} v = ρ (G_Y + η M) v ,

and the projector is **metric-aware / oblique** (M-orthogonal), `P_N = V (Vᵀ M V)^{-1} Vᵀ M`,
so the selection is covariant under invertible coordinate rescalings of Z (a Euclidean QR
projector is not). A global **cross-fit critic-advantage gate** (bootstrap one-sided lower
bound on the domain critic's held-out balanced-accuracy advantage) replaces the mean-scatter
permutation-null floor -- shuffling D *after* training the critic is not a valid null when
the critic was trained on the real D.

Deferred to later phases (NOT here): source-risk UCB rank gate, conditional-on-task critic
`I(P_N Z;D|Y,sg(P_T Z))`, parameter-level PCGrad, EEG wiring.
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
    hidden: int = 128
    epochs: int = 300
    lr: float = 1e-2
    weight_decay: float = 1e-3
    n_folds: int = 2                # cross-fit folds for probe-train vs score-accumulation
    eps_white: float = 1e-2         # ridge on Σ_W before inversion
    eta: float = 1e-2               # ridge (in metric M units) on G_Y in the generalized eig
    tau_ratio: float = 3.0          # domain/label score-energy ratio (M-unit eigenvectors)
    eps_label: float = 0.20         # label-light: lab_j <= eps_label * max_k lab_k
    dom_floor: float = 0.05         # reject near-null directions (relative domain energy)
    rank_gap: float = 1.5           # cut rank at the first multiplicative rho-gap >= this
    max_dim: int = 8
    gate_alpha: float = 0.05        # one-sided level for the critic-advantage gate
    gate_boot: int = 300            # bootstrap resamples for the gate
    n_perm_null: int = 1            # within-Y permutations to calibrate the domain-energy floor
    null_safety: float = 1.25       # a direction must exceed null_safety x the null floor
    dtype64: bool = True


# ----------------------------------------------------------------------------- probes
class _Probe(nn.Module):
    """Differentiable-in-z probe. `cond_dim>0` appends a one-hot conditioner (e.g. Y) to the
    z-features. `family` controls the z-feature map; `quad` = [z, z^2] (its input-gradient is
    linear in z, so it already senses variance), `mlp` = a 1-hidden-layer net."""
    def __init__(self, z_dim, n_out, cond_dim=0, family="mlp", hidden=128):
        super().__init__()
        self.family, self.cond_dim = family, cond_dim
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


def _train_probe(probe, Z, target, cond, epochs, lr, wd, seed):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(probe.parameters(), lr=lr, weight_decay=wd)
    Zt = torch.tensor(Z, dtype=torch.float32)
    tt = torch.tensor(target, dtype=torch.long)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    for _ in range(epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Zt, ct), tt).backward()
        opt.step()
    return probe


def _expected_score_fisher(probe, Z, cond, n_out):
    """G = (1/B) Σ_b Σ_c p(c|z_b) ∇z logp(c|z_b) ∇z logp(c|z_b)^T, evaluated at the given Z."""
    Zt = torch.tensor(Z, dtype=torch.float32, requires_grad=True)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    logp = F.log_softmax(probe(Zt, ct), dim=1)        # [B, n_out]
    p = logp.exp().detach()
    d = Zt.shape[1]
    G = torch.zeros(d, d, dtype=torch.float64)
    for c in range(n_out):
        g, = torch.autograd.grad(logp[:, c].sum(), Zt, retain_graph=True)   # [B,d] = s_c per sample
        gw = (g * p[:, c:c + 1].clamp_min(0).sqrt()).double()              # weight by sqrt p(c|z)
        G += gw.t() @ gw
    return (G / Zt.shape[0]).numpy()


def _cross_fit_fisher(Z, target, cond, n_out, z_dim, cond_dim, cfg: ScoreFisherConfig, seed):
    """Train probe on each fold's complement, accumulate the score Fisher on the held-out
    fold; average. Also return cross-fit held-out predictions for the advantage gate."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(Z))
    folds = np.array_split(idx, cfg.n_folds)
    Gs, preds, trues = [], np.zeros(len(Z), dtype=int) - 1, np.asarray(target)
    for f, hold in enumerate(folds):
        tr = np.setdiff1d(idx, hold)
        probe = _Probe(z_dim, n_out, cond_dim, cfg.probe_family, cfg.hidden)
        _train_probe(probe, Z[tr], np.asarray(target)[tr],
                     None if cond is None else cond[tr], cfg.epochs, cfg.lr, cfg.weight_decay, seed + f)
        Gs.append(_expected_score_fisher(probe, Z[hold], None if cond is None else cond[hold], n_out))
        with torch.no_grad():
            ct = None if cond is None else torch.tensor(cond[hold], dtype=torch.float32)
            preds[hold] = probe(torch.tensor(Z[hold], dtype=torch.float32), ct).argmax(1).numpy()
    G = np.mean(Gs, axis=0)
    return 0.5 * (G + G.T), preds, trues


def _shuffle_within_y(y, d, rng):
    """Permute D independently within each class -- breaks D|Y while keeping p(d|y)."""
    dp = np.array(d, copy=True)
    for c in np.unique(y):
        idx = np.where(y == c)[0]
        dp[idx] = d[idx][rng.permutation(idx.size)]
    return dp


def _within_class_cov(Z, y, n_cls, eps):
    """Σ_W = Σ_y p(y) Cov(Z | Y=y); M = (Σ_W + eps I)^{-1}."""
    Z = np.asarray(Z, dtype=np.float64)
    d = Z.shape[1]
    S = np.zeros((d, d))
    for c in range(n_cls):
        Zc = Z[y == c]
        if len(Zc) > 1:
            S += (len(Zc) / len(Z)) * np.cov(Zc, rowvar=False)
    S = 0.5 * (S + S.T) + eps * np.eye(d)
    return np.linalg.inv(S)


def _balanced_adv(pred, true, group, n_group, n_cls):
    """Balanced-acc advantage of `pred` over the per-class majority baseline for `group`."""
    def bacc(pr):
        accs = [(pr[true == k] == k).mean() for k in range(n_group) if (true == k).sum() > 0]
        return float(np.mean(accs)) if accs else 0.0
    base = np.zeros_like(true)
    for c in range(n_cls):
        m = group == c
        if m.sum():
            base[m] = np.bincount(true[m], minlength=n_group).argmax()
    return bacc(pred) - bacc(base)


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
    critic_adv: float            # cross-fit domain-critic balanced-acc advantage
    critic_adv_lb: float         # one-sided bootstrap lower bound
    gate_open: bool              # leakage significant?
    dom_floor_abs: float = 0.0   # within-Y permutation-null domain-energy floor

    @property
    def k(self):
        return int(self.basis.shape[1])


def metric_projector(V_sel, M):
    """Oblique M-orthogonal projector onto span(V_sel): P = V (Vᵀ M V)^{-1} Vᵀ M.
    Covariant under z->Az: P transforms by similarity A P A^{-1}, so the selected subspace is
    invariant to invertible rescalings (a Euclidean Q Qᵀ projector is not)."""
    if V_sel.shape[1] == 0:
        d = V_sel.shape[0]
        return np.zeros((d, d))
    G = V_sel.T @ M @ V_sel
    return V_sel @ np.linalg.solve(G, V_sel.T @ M)


def select_score_fisher(Z, y, d, n_cls, n_dom, cfg: ScoreFisherConfig, seed=0) -> ScoreFisherReport:
    Z = np.asarray(Z, dtype=np.float64)
    y = np.asarray(y); d = np.asarray(d)
    z_dim = Z.shape[1]
    y_oh = np.eye(n_cls)[y]

    G_Y, _, _ = _cross_fit_fisher(Z, y, None, n_cls, z_dim, 0, cfg, seed)
    G_DgY, dpred, _ = _cross_fit_fisher(Z, d, y_oh, n_dom, z_dim, n_cls, cfg, seed + 100)
    M = _within_class_cov(Z, y, n_cls, cfg.eps_white)

    # within-Y permutation null of the DOMAIN score Fisher: retrain the critic on shuffled D
    # (shuffling D *after* training is not a valid null since the critic saw the real D). The
    # null G captures the score energy attainable with NO real D|Y dependence -- the floor a
    # direction must clear, which kills the label-null/noise directions that otherwise blow up
    # rho where (G_Y + eta M) -> eta M.
    rng_null = np.random.default_rng(seed + 300)
    G_null = np.zeros_like(G_DgY)
    for p in range(max(cfg.n_perm_null, 0)):
        d_perm = _shuffle_within_y(y, d, rng_null)
        Gp, _, _ = _cross_fit_fisher(Z, d_perm, y_oh, n_dom, z_dim, n_cls, cfg, seed + 400 + p)
        G_null = np.maximum(G_null, Gp) if p else Gp

    # global leakage gate: is the cross-fit domain critic better than the per-class baseline?
    adv = _balanced_adv(dpred, d, y, n_dom, n_cls)
    rng = np.random.default_rng(seed + 7)
    boot = []
    for _ in range(cfg.gate_boot):
        b = rng.integers(0, len(Z), len(Z))
        boot.append(_balanced_adv(dpred[b], d[b], y[b], n_dom, n_cls))
    adv_lb = float(np.quantile(boot, cfg.gate_alpha))
    gate_open = adv_lb > 0.0

    # generalized eig in the whitening metric: G_DgY v = rho (G_Y + eta M) v
    B = G_Y + cfg.eta * M
    B = 0.5 * (B + B.T)
    w, V = eigh(0.5 * (G_DgY + G_DgY.T), B)
    order = np.argsort(w)[::-1]
    rho, V = w[order], V[:, order]

    # scale-consistent energies on M-unit eigenvectors (u^T M u = 1)
    Mnorm = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V.T, M, V.T), 1e-18))
    U = V / Mnorm
    dom = np.clip(np.einsum("ij,jk,ik->i", U.T, G_DgY, U.T), 0, None)
    lab = np.clip(np.einsum("ij,jk,ik->i", U.T, G_Y, U.T), 0, None)
    dom_null = np.clip(np.einsum("ij,jk,ik->i", U.T, G_null, U.T), 0, None)  # null floor per dir
    ratio = dom / (lab + cfg.eta)

    dom_max = dom.max() if dom.size and dom.max() > 0 else 1.0
    lab_max = lab.max() if lab.size and lab.max() > 0 else 1.0
    floor_abs = cfg.null_safety * (dom_null.max() if dom_null.size else 0.0)
    eligible = ((ratio >= cfg.tau_ratio) & (lab <= cfg.eps_label * lab_max)
                & (dom >= max(cfg.dom_floor * dom_max, floor_abs)))    # above noise/null floor
    # rank by generalized eigenvalue, then cut at the first large multiplicative rho-gap
    # (an eigengap heuristic; the principled rank choice is the deferred source-risk UCB gate).
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
        cand = np.array([], dtype=int)                 # no significant leakage -> identity

    V_sel = V[:, cand] if cand.size else np.zeros((z_dim, 0))
    P = metric_projector(V_sel, M)
    return ScoreFisherReport(rho=rho, dom_energy=dom, lab_energy=lab, ratio=ratio,
                             selected=np.sort(cand), basis=V_sel, P=P,
                             is_identity=(cand.size == 0), critic_adv=adv,
                             critic_adv_lb=adv_lb, gate_open=gate_open, dom_floor_abs=floor_abs)


class ScoreFisherSelector(nn.Module):
    """Static score-Fisher selector (no encoder training). Mirrors SubspaceSelector's API
    (`refresh`, `project`, `is_identity`, `report`, `summary`) so it can later drop into the
    same trainer hook, but uses score Fishers + the metric-aware oblique projector."""
    def __init__(self, z_dim, n_cls, n_dom, cfg: Optional[ScoreFisherConfig] = None, device="cpu"):
        super().__init__()
        self.z_dim, self.n_cls, self.n_dom = z_dim, n_cls, n_dom
        self.cfg = cfg or ScoreFisherConfig()
        self.register_buffer("P", torch.zeros(z_dim, z_dim))
        self.report: Optional[ScoreFisherReport] = None

    @property
    def is_identity(self):
        return self.report is None or self.report.is_identity

    def refresh(self, Z, y, d, seed: int = 0):
        # NOT under no_grad: probe training and the autograd score-Fisher need grad enabled.
        # The projector/eig are numpy and the P buffer is set from numpy, so nothing here
        # leaks autograd state into the selector.
        Znp = Z.detach().cpu().numpy() if torch.is_tensor(Z) else np.asarray(Z)
        ynp = y.detach().cpu().numpy() if torch.is_tensor(y) else np.asarray(y)
        dnp = d.detach().cpu().numpy() if torch.is_tensor(d) else np.asarray(d)
        rep = select_score_fisher(Znp, ynp, dnp, self.n_cls, self.n_dom, self.cfg, seed)
        self.report = rep
        self.P = torch.tensor(rep.P, dtype=torch.float32, device=self.P.device)
        return rep

    def project(self, Z):
        # nuisance component: row-vector Z [B,d] times P^T (P maps column vectors)
        return Z @ self.P.t().to(Z.dtype)

    def summary(self):
        if self.report is None:
            return {"k": 0, "is_identity": True}
        r = self.report
        return {"k": r.k, "is_identity": r.is_identity, "selected": r.selected.tolist(),
                "critic_adv": r.critic_adv, "critic_adv_lb": r.critic_adv_lb,
                "gate_open": r.gate_open, "rho_top": [float(x) for x in r.rho[:5]]}
