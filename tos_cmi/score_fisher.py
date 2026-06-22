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

PHASE 1.2 adds the SOURCE-RISK UCB RANK GATE (the rank decision; the eigengap is now only a
diagnostic). Four-way split: S_sel ranks candidate directions; S_gate runs a cross-fitted
dual gate over nested prefixes P_N^(k) -- task NLL cost UCB(Delta_Y) <= delta_Y AND domain
BRIER gain LCB(Delta_D) > gamma_D, with a simultaneous (Bonferroni) cluster-bootstrap band;
k* = argmax_{feasible} LCB(Delta_D) (ties -> smaller k); empty -> identity. The domain critic
in the gate is q_k = sg(logit q0(D|Y)) + zero-init residual r(P_k Z, Y). The leakage prefilter
gate compares q0(D|Y) vs q1(D|Z,Y) on the BRIER score (bounded; NLL kept diagnostic only).
GROUP-aware cross-fitting (whole clusters per fold) + fold-coverage abstention.

Deferred (NOT here): conditional-on-task TRAINING critic `I(P_N Z;D|Y,sg(P_T Z))` as an
encoder penalty; PCGrad; EEG wiring. (The gate's residual q_k is eval-only, not the trainer.)
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
    gate_gamma: float = 0.0         # leakage gate: domain BRIER gain LCB must exceed this
    gate_alpha: float = 0.05        # one-sided level for the leakage gate
    gate_boot: int = 300            # cluster-bootstrap resamples
    boot_estimand: str = "sample"   # sample (trial-weighted) | cluster_equal (subject-equal)
    # --- source-risk UCB rank gate (Phase 1.2) ---
    gate_split: float = 0.45        # fraction of refresh data held out as S_gate
    delta_Y: float = 0.03           # max tolerated task NLL/Brier increase from removing P_N (UCB)
    gamma_D: float = 0.0            # min per-prefix domain Brier gain (LCB) to count as domain-rich
    task_protect: bool = False      # use the task-protected direct-sum projector (protect T=est.
                                    # task carrier). OFF by default pending the task-head-fidelity
                                    # decision (the nonlinear task-risk head charges a residual
                                    # cost on rank-reduced (I-P)Z even when (I-P)T=T exactly).
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


import copy as _copy


def _fit_temperature(logits, target, iters=100):
    """Scalar temperature minimizing held-out NLL (calibration); returns T>0."""
    logT = torch.zeros(1, requires_grad=True)
    opt = torch.optim.LBFGS([logT], lr=0.1, max_iter=iters)
    L = logits.detach(); t = target

    def closure():
        opt.zero_grad(); loss = F.cross_entropy(L / logT.exp(), t); loss.backward(); return loss
    try:
        opt.step(closure)
    except Exception:
        return 1.0
    return float(logT.exp().clamp(0.2, 5.0))


def _build_and_train(z_dim, n_out, cond_dim, Z, target, cond, cfg, seed, want_temp=False):
    """Seed BEFORE construction (fix 1); train with CE + EARLY STOPPING on a val sub-split
    (prevents the confident-wrong overfit that wrecks held-out Brier/NLL on weak signals). If
    want_temp, also returns a temperature fit on the val split for calibrated probabilities."""
    torch.manual_seed(seed)
    probe = _Probe(z_dim, n_out, cond_dim, cfg.probe_family, cfg.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Zt = torch.tensor(Z, dtype=torch.float32)
    tt = torch.tensor(target, dtype=torch.long)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    n = len(target)
    nval = int(0.15 * n) if n >= 40 else 0
    g = torch.Generator().manual_seed(seed + 1)
    perm = torch.randperm(n, generator=g)
    val, trn = (perm[:nval], perm[nval:]) if nval else (perm[:0], perm)
    cv = None if ct is None else ct[val]
    best, best_state, bad, patience = float("inf"), None, 0, max(8, cfg.epochs // 5)
    for ep in range(cfg.epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Zt[trn], None if ct is None else ct[trn]), tt[trn]).backward()
        opt.step()
        if nval and ep % 3 == 0:
            with torch.no_grad():
                v = F.cross_entropy(probe(Zt[val], cv), tt[val]).item()
            if v < best - 1e-4:
                best, best_state, bad = v, _copy.deepcopy(probe.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        probe.load_state_dict(best_state)
    if not want_temp:
        return probe
    with torch.no_grad():
        T = _fit_temperature(probe(Zt[val], cv), tt[val]) if nval else 1.0
    return probe, T


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
    """Deterministic GROUP-aware K-fold plan shared across task/domain/null probes. Whole
    `group_id` units are assigned to a single fold (round-robin over shuffled groups), so a
    recording/session/block never straddles train and held-out -- cluster bootstrap alone
    cannot fix that train/eval leakage (Phase 1.2 preflight 1). group_id=None => per-sample."""
    def __init__(self, n, n_folds, seed, group_id=None):
        rng = np.random.default_rng(seed)
        if group_id is None:
            units = [np.array([i]) for i in range(n)]
        else:
            group_id = np.asarray(group_id)
            units = [np.where(group_id == g)[0] for g in np.unique(group_id)]
        order = rng.permutation(len(units))
        buckets = [[] for _ in range(n_folds)]
        for j, u in enumerate(order):
            buckets[j % n_folds].append(units[u])
        self.folds = [np.concatenate(b) if b else np.array([], dtype=int) for b in buckets]
        self.idx = np.arange(n)

    def iter(self):
        for f, hold in enumerate(self.folds):
            yield f, np.setdiff1d(self.idx, hold), hold

    def coverage_ok(self, target, n_classes):
        """True iff every TRAIN fold sees all classes (else the cross-fit probe can't learn
        them -> the selector should abstain with FOLD_COVERAGE_FAILURE)."""
        target = np.asarray(target)
        for _, tr, _ in self.iter():
            if len(np.unique(target[tr])) < n_classes:
                return False
        return True


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
    """Per-sample held-out NLL, prediction and multiclass BRIER of a cross-fit probe. Zfeat
    may have 0 columns (e.g. q0(D|Y) uses only the Y conditioner)."""
    target = np.asarray(target)
    nll = np.zeros(len(target)); pred = np.zeros(len(target), dtype=int); brier = np.zeros(len(target))
    z_dim = Zfeat.shape[1]
    cond_dim = 0 if cond is None else cond.shape[1]
    for f, tr, hold in plan.iter():
        probe = _build_and_train(z_dim, n_out, cond_dim, Zfeat[tr], target[tr],
                                 None if cond is None else cond[tr], cfg, seed + f)
        with torch.no_grad():
            ct = None if cond is None else torch.tensor(cond[hold], dtype=torch.float32)
            logp = F.log_softmax(probe(torch.tensor(Zfeat[hold], dtype=torch.float32), ct), 1)
        p = logp.exp().numpy()
        t = target[hold]
        nll[hold] = -logp[np.arange(len(hold)), t].numpy()
        pred[hold] = p.argmax(1)
        brier[hold] = (p ** 2).sum(1) - 2 * p[np.arange(len(hold)), t] + 1.0   # Σ_c (p_c-1[c=t])^2
    return nll, pred, brier


def _groups_of(cluster_id, n):
    if cluster_id is None:
        return [np.array([i]) for i in range(n)]
    cluster_id = np.asarray(cluster_id)
    return [np.where(cluster_id == c)[0] for c in np.unique(cluster_id)]


def _boot_estimates(values, groups, n_boot, rng, estimand):
    """Cluster bootstrap estimates of E[values]. values [N] or [N,K]. estimand:
      'sample'        -> mean over all resampled trials (trial-weighted);
      'cluster_equal' -> mean of per-cluster means (cluster/subject-equal weighting)."""
    G = len(groups)
    cmeans = np.stack([values[g].mean(0) for g in groups])   # [G] or [G,K]
    csizes = np.array([len(g) for g in groups], dtype=float)
    out = []
    for _ in range(n_boot):
        pick = rng.integers(0, G, G)
        if estimand == "cluster_equal":
            out.append(cmeans[pick].mean(0))
        else:
            w = csizes[pick]; w = w / w.sum()
            out.append((cmeans[pick] * (w[:, None] if cmeans.ndim == 2 else w)).sum(0))
    return np.stack(out)


def _one_sided_bound(values, cluster_id, alpha, side, n_boot, seed, estimand):
    """One-sided bound on E[values]. If values is 2D [N,K], returns a per-column SIMULTANEOUS
    band via Bonferroni (level alpha/K) -- a valid (conservative) simultaneous reference."""
    values = np.asarray(values, dtype=float)
    boot = _boot_estimates(values, _groups_of(cluster_id, len(values)), n_boot,
                           np.random.default_rng(seed), estimand)
    K = boot.shape[1] if boot.ndim == 2 else 1
    lvl = alpha / K
    q = lvl if side == "lower" else 1 - lvl
    return np.quantile(boot, q, axis=0)


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
    # NO fixed `eps*I` jitter: an isotropic ridge does NOT transform as A·Aᵀ under z->Az, so it
    # would break the exact coordinate-covariance the transported-probe test certifies. Both
    # terms here are data covariances (covariant). For n >> d this is well-conditioned; for
    # d >= n one must restrict to the empirical support of Σ_ref and claim covariance only
    # there (not yet needed on synthetic) -- do NOT reintroduce a fixed I-ridge.
    A = 0.5 * (Sw + Sw.T) + cfg.eps_ref * 0.5 * (Sref + Sref.T)
    cond = np.linalg.cond(A)
    if not np.isfinite(cond) or cond > 1e10:
        raise np.linalg.LinAlgError(f"metric ill-conditioned (cond={cond:.2e}); restrict to "
                                    "Σ_ref support (d>=n regime) rather than adding an I-ridge")
    return np.linalg.inv(A)


def metric_projector(V_sel, M):
    """Oblique M-orthogonal projector onto span(V_sel): P = V (Vᵀ M V)^{-1} Vᵀ M. Covariant
    under z->Az: P -> A P A^{-1}. NOTE: range=span(V), kernel = M-orthogonal complement -- this
    is NOT task-preserving in general (it distorts a Euclidean task-orthogonal subspace). Kept
    as an ablation; the deployed projector is `task_protected_projector` (Phase 1.2.2)."""
    if V_sel.shape[1] == 0:
        return np.zeros((V_sel.shape[0], V_sel.shape[0]))
    G = V_sel.T @ M @ V_sel
    return V_sel @ np.linalg.solve(G, V_sel.T @ M)


def _m_proj(B, M):
    """M-orthogonal projector onto span(B): B (Bᵀ M B)^{-1} Bᵀ M (0 if empty)."""
    if B is None or B.shape[1] == 0:
        return np.zeros((M.shape[0], M.shape[0]))
    return B @ np.linalg.solve(B.T @ M @ B, B.T @ M)


def _direct_sum_min_sin(V, T, M):
    """Per-nuisance-direction sin-angle to span(T) in the M-metric = ||(I-Pi_T^M) v||_M/||v||_M.
    Min over columns ~ sin of the smallest principal angle between span(V) and span(T); ->0 iff
    a nuisance direction lies in T (the direct-sum condition N∩T={0} fails)."""
    if V.shape[1] == 0:
        return 1.0
    PiT = _m_proj(T, M)
    Vp = V - PiT @ V
    vn = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V.T, M, V.T), 1e-18))
    vpn = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", Vp.T, M, Vp.T), 0.0))
    return float((vpn / vn).min())


def task_protected_projector(V, T, M, min_sin=0.05):
    """Coordinate-covariant DIRECT-SUM projector with range = span(V) and kernel ⊇ span(T):
        Π_T^M = T(TᵀMT)^{-1}TᵀM ;  V⊥ = (I-Π_T^M)V ;  P = V (V⊥ᵀ M V⊥)^{-1} V⊥ᵀ M .
    Satisfies P V = V, P T = 0, P² = P, and P(AV,AT,A^{-T}MA^{-1}) = A P A^{-1} (so deleting the
    nuisance preserves the task carrier T EXACTLY while staying rescaling-covariant). Unifies
    the earlier choices: T=∅ -> metric_projector; V ⟂_M T -> metric_projector; M=I & V⟂T -> QQᵀ.
    If span(V) ∩ span(T) ≠ {0} (a direction is both nuisance and task), NO such projector exists
    -> returns (None, info) and the caller abstains (TASK_SUBSPACE_INTERSECTION).
    Returns (P [d,d] or None, info dict with min_sin / op_norm / cond / intersects)."""
    d = M.shape[0]
    if V.shape[1] == 0:
        return np.zeros((d, d)), {"intersects": False, "min_sin": 1.0, "op_norm": 0.0, "cond": 1.0}
    ms = _direct_sum_min_sin(V, T, M)
    if ms < min_sin:
        return None, {"intersects": True, "min_sin": ms, "op_norm": float("inf"), "cond": float("inf")}
    Vp = V - _m_proj(T, M) @ V
    G = Vp.T @ M @ Vp
    P = V @ np.linalg.solve(G, Vp.T @ M)
    return P, {"intersects": False, "min_sin": ms,
               "op_norm": float(np.linalg.norm(P, 2)), "cond": float(np.linalg.cond(G))}


def estimate_task_basis(G_Y, M, cfg, energy_frac=0.9):
    """Protected task carrier T = top eigenvectors of the LABEL score-Fisher in the M-metric
    (G_Y v = lambda M v), keeping directions up to `energy_frac` of total label energy (capped
    at max_dim). These are the directions the task posterior is most sensitive to -- what the
    deletion must preserve. Estimated on S_sel only."""
    w, V = eigh(0.5 * (G_Y + G_Y.T), 0.5 * (M + M.T))
    order = np.argsort(w)[::-1]
    w, V = np.clip(w[order], 0, None), V[:, order]
    if w.sum() <= 0:
        return np.zeros((G_Y.shape[0], 0))
    keep = int(np.searchsorted(np.cumsum(w) / w.sum(), energy_frac) + 1)
    return V[:, : min(keep, cfg.max_dim)]


def leakage_gate(Z, y, d, n_cls, n_dom, cfg, plan, seed, cluster_id=None):
    """Leakage prefilter (Phase 1.2): does Z carry conditional-domain info beyond Y? Compares
    q0(D|Y) to the RESIDUAL critic q1 = sg(logit q0) + zero-init r(Z,Y) -- the full-Z case of
    the same residual+Brier construction used per-prefix. Starting q1 AT q0 is what makes the
    held-out BRIER gain robust: a plain q1(D|Z,Y) MLP overfits a weak covariance signal and
    gets WORSE held-out Brier than q0 (observed -0.5), even though the signal is real; the
    residual can only help if the signal generalizes. GROUP-aware cross-fit, CLUSTER one-sided
    bound. Returns dict(brier_gain, brier_lcb, open)."""
    y_oh = np.eye(n_cls)[y]
    gplan = _GatePlan(plan, seed + 5)
    br0, cache = _domain_q0(d, y_oh, n_dom, n_cls, cfg, gplan, seed + 11)
    br1 = _domain_residual_brier(Z, d, y_oh, cache, n_dom, n_cls, cfg, seed + 22)
    gain = br0 - br1                                       # >0 iff Z helps predict D beyond Y
    lcb = float(_one_sided_bound(gain, cluster_id, cfg.gate_alpha, "lower",
                                 cfg.gate_boot, seed + 33, cfg.boot_estimand))
    return {"brier_gain": float(gain.mean()), "brier_lcb": lcb, "open": bool(lcb > cfg.gate_gamma)}


@dataclass
class ScoreFisherReport:
    rho: np.ndarray              # generalized eigenvalues on S_sel (descending)
    cand_order: np.ndarray       # eligible candidate eigen-indices, ranked (the prefix order)
    basis: np.ndarray            # [d,k*] SELECTED generalized eigenvectors (= cand_order[:k*])
    P: np.ndarray                # [d,d] metric-aware (oblique) projector
    is_identity: bool
    k_star: int                  # rank chosen by the UCB gate
    gate: dict                   # leakage-prefilter gate dict (brier_gain/lcb/open/nll/acc)
    rank_records: list           # per-prefix structured records (task UCB / domain LCB / reason)
    decision_reason: str         # ACCEPTED | NO_CANDIDATE | DOMAIN_GATE_CLOSED | ...
    dom_floor_abs: float = 0.0   # within-Y permutation-null energy prefilter floor
    eigengap_k: int = 0          # interim eigengap rank (DIAGNOSTIC only now)

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


def candidate_order(G_DgY, G_Y, M, cfg, floor_abs=0.0):
    """Ordered candidate eigenvectors for the UCB gate: eligible (domain-rich + label-light +
    above the null-energy prefilter) directions ranked by rho. NO eigengap cut here -- the UCB
    gate decides the rank over nested prefixes. Returns (V_cand[d,K], rho, dom, lab, ratio,
    eigengap_k(diagnostic))."""
    z_dim = G_Y.shape[0]
    B = 0.5 * ((G_Y + cfg.eta * M) + (G_Y + cfg.eta * M).T)
    w, V = eigh(0.5 * (G_DgY + G_DgY.T), B)
    order = np.argsort(w)[::-1]
    rho, V = w[order], V[:, order]
    U = V / np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V.T, M, V.T), 1e-18))
    dom = np.clip(np.einsum("ij,jk,ik->i", U.T, G_DgY, U.T), 0, None)
    lab = np.clip(np.einsum("ij,jk,ik->i", U.T, G_Y, U.T), 0, None)
    ratio = dom / (lab + cfg.eta)
    dom_max = dom.max() if dom.size and dom.max() > 0 else 1.0
    lab_max = lab.max() if lab.size and lab.max() > 0 else 1.0
    eligible = ((ratio >= cfg.tau_ratio) & (lab <= cfg.eps_label * lab_max)
                & (dom >= max(cfg.dom_floor * dom_max, floor_abs)))
    elig = np.where(eligible)[0]
    elig = elig[np.argsort(rho[elig])[::-1]][: cfg.max_dim]
    # interim eigengap rank (diagnostic only): first big multiplicative rho-drop among eligible
    egk = elig.size
    if elig.size > 1:
        re = np.maximum(rho[elig], 1e-12)
        over = np.where(re[:-1] / re[1:] >= cfg.rank_gap)[0]
        if over.size:
            egk = int(over[0]) + 1
    return V[:, elig], rho, dom, lab, ratio, int(egk)


def _zero_last_layer(probe):
    last = probe.net[-1] if isinstance(probe.net, nn.Sequential) else probe.net
    nn.init.zeros_(last.weight); nn.init.zeros_(last.bias)
    return probe


def _brier(p, t):
    return (p ** 2).sum(1) - 2 * p[np.arange(len(t)), t] + 1.0


def _fit_nonneg_scale(base_logits, resid, target, grid=(0.0, 0.25, 0.5, 1.0, 1.5, 2.0)):
    """alpha >= 0 minimizing NLL(softmax(base + alpha*resid)) on a small grid; alpha=0 in the
    grid means a zero-information residual EXACTLY recovers the baseline (strict nesting)."""
    best_a, best_l = 0.0, float("inf")
    for a in grid:
        l = F.cross_entropy(base_logits + a * resid, target).item()
        if l < best_l:
            best_l, best_a = l, a
    return best_a


class _GatePlan:
    """Per outer fold (from a GROUP-aware plan): a fixed (tr, hold), a fixed inner train/val
    split of tr, and a fixed init seed -- ALL SHARED across the task heads h_0..h_K and the
    domain q0/residual critics, so every paired comparison differs only in its INPUT, not in
    init / splits / optimisation budget (Phase 1.2.1 fix 2)."""
    def __init__(self, plan, seed):
        self.folds = []
        for f, tr, hold in plan.iter():
            perm = np.random.default_rng(seed + f).permutation(len(tr))
            nval = int(0.15 * len(tr)) if len(tr) >= 40 else 0
            self.folds.append((tr, hold, tr[perm[nval:]], tr[perm[:nval]], int(seed + 1000 + f)))


def _train_head(z_dim, n_out, cond_dim, Zf, target, cond, cfg, init_seed, itr, iva):
    """Train a probe from a FIXED init (init_seed) on global indices `itr`, early-stop on
    `iva`; restore best-val weights. Zf/target/cond are full arrays indexed by global ids."""
    torch.manual_seed(init_seed)
    probe = _Probe(z_dim, n_out, cond_dim, cfg.probe_family, cfg.hidden)
    opt = torch.optim.Adam(probe.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    Zt = torch.tensor(Zf, dtype=torch.float32); tt = torch.tensor(target, dtype=torch.long)
    ct = None if cond is None else torch.tensor(cond, dtype=torch.float32)
    itr_t = torch.as_tensor(itr); iva_t = torch.as_tensor(iva)
    best, best_state, bad, patience = float("inf"), None, 0, max(8, cfg.epochs // 5)
    for ep in range(cfg.epochs):
        opt.zero_grad()
        F.cross_entropy(probe(Zt[itr_t], None if ct is None else ct[itr_t]), tt[itr_t]).backward()
        opt.step()
        if len(iva_t) and ep % 3 == 0:
            with torch.no_grad():
                v = F.cross_entropy(probe(Zt[iva_t], None if ct is None else ct[iva_t]), tt[iva_t]).item()
            if v < best - 1e-4:
                best, best_state, bad = v, _copy.deepcopy(probe.state_dict()), 0
            else:
                bad += 1
                if bad >= patience:
                    break
    if best_state is not None:
        probe.load_state_dict(best_state)
    return probe


def _paired_task_nll(Zg, yg, transforms, n_cls, cfg, gplan):
    """Per-sample held-out NLL of task heads h_t(Y | transform_t), all sharing per-fold init +
    inner splits so Delta_Y is a clean PAIRED difference (only the input differs)."""
    nll = [np.zeros(len(yg)) for _ in transforms]
    z_dim = Zg.shape[1]
    for (tr, hold, itr, iva, iseed) in gplan.folds:
        for t, Zf in enumerate(transforms):
            probe = _train_head(z_dim, n_cls, 0, Zf, yg, None, cfg, iseed, itr, iva)
            with torch.no_grad():
                logp = F.log_softmax(probe(torch.tensor(Zf[hold], dtype=torch.float32)), 1)
            nll[t][hold] = -logp[np.arange(len(hold)), yg[hold]].numpy()
    return nll


def _domain_q0(dg, yg_oh, n_dom, n_cls, cfg, gplan, seed):
    """Train the calibrated baseline q0(D|Y) ONCE per fold; cache calibrated logits
    L~0 = L0/T0 on tr and hold. Returns (br0[N], cache) with cache[f]=(tr,hold,itr,iva,L~0_tr,L~0_ho)."""
    br0 = np.zeros(len(dg)); cache = []
    for (tr, hold, itr, iva, iseed) in gplan.folds:
        q0 = _train_head(0, n_dom, n_cls, np.zeros((len(dg), 0)), dg, yg_oh, cfg, iseed + 5, itr, iva)
        with torch.no_grad():
            Lva = q0(torch.zeros(len(iva), 0), torch.tensor(yg_oh[iva], dtype=torch.float32))
            T0 = _fit_temperature(Lva, torch.tensor(dg[iva], dtype=torch.long)) if len(iva) else 1.0
            Ltr = q0(torch.zeros(len(tr), 0), torch.tensor(yg_oh[tr], dtype=torch.float32)) / T0
            Lho = q0(torch.zeros(len(hold), 0), torch.tensor(yg_oh[hold], dtype=torch.float32)) / T0
        br0[hold] = _brier(torch.softmax(Lho, 1).numpy(), dg[hold])
        cache.append((tr, hold, itr, iva, Ltr.detach(), Lho.detach()))
    return br0, cache


def _domain_residual_brier(Zfeat, dg, yg_oh, cache, n_dom, n_cls, cfg, seed):
    """Strictly-nested domain critic q_k = softmax(L~0 + alpha*r(Zfeat,Y)), r zero-init,
    alpha>=0 fit on inner-val (alpha=0 recovers q0 EXACTLY -- so any Brier gain is from
    P_k Z's increment, not recalibration). Shares the cached calibrated q0. Returns brk[N]."""
    brk = np.zeros(len(dg)); z_dim = Zfeat.shape[1]
    for (tr, hold, itr, iva, Ltr_cal, Lho_cal) in cache:
        pos = {int(g): i for i, g in enumerate(tr)}
        itr_l = np.array([pos[int(g)] for g in itr]); iva_l = np.array([pos[int(g)] for g in iva])
        torch.manual_seed(seed)
        r = _zero_last_layer(_Probe(z_dim, n_dom, n_cls, cfg.probe_family, cfg.hidden))
        opt = torch.optim.Adam(r.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        Ztr = torch.tensor(Zfeat[tr], dtype=torch.float32); ytr = torch.tensor(yg_oh[tr], dtype=torch.float32)
        dtr = torch.tensor(dg[tr], dtype=torch.long)
        itl = torch.as_tensor(itr_l); ivl = torch.as_tensor(iva_l)
        best, best_state, bad, patience = float("inf"), None, 0, max(8, cfg.epochs // 5)
        for ep in range(cfg.epochs):
            opt.zero_grad()
            F.cross_entropy((Ltr_cal + r(Ztr, ytr))[itl], dtr[itl]).backward(); opt.step()
            if len(ivl) and ep % 3 == 0:
                with torch.no_grad():
                    v = F.cross_entropy((Ltr_cal + r(Ztr, ytr))[ivl], dtr[ivl]).item()
                if v < best - 1e-4:
                    best, best_state, bad = v, _copy.deepcopy(r.state_dict()), 0
                else:
                    bad += 1
                    if bad >= patience:
                        break
        if best_state is not None:
            r.load_state_dict(best_state)
        with torch.no_grad():
            r_tr = r(Ztr, ytr)
            alpha = (_fit_nonneg_scale(Ltr_cal[ivl], r_tr[ivl], dtr[ivl]) if len(ivl) else 1.0)
            Lk_ho = Lho_cal + alpha * r(torch.tensor(Zfeat[hold], dtype=torch.float32),
                                        torch.tensor(yg_oh[hold], dtype=torch.float32))
        brk[hold] = _brier(torch.softmax(Lk_ho, 1).numpy(), dg[hold])
    return brk


_REASONS = ("ACCEPTED", "NO_CANDIDATE", "DOMAIN_GATE_CLOSED", "DOMAIN_GAIN_TOO_SMALL",
            "TASK_RISK_UCB", "FOLD_COVERAGE_FAILURE", "INSUFFICIENT_GROUPS",
            "TASK_SUBSPACE_INTERSECTION", "NUMERICAL_FAILURE")


def ucb_rank_gate(Zg, yg, dg, V_cand, M, n_cls, n_dom, cfg, seed, cluster_id=None, T_task=None):
    """Source-risk dual-gate rank selection on the gate split, with the TASK-PROTECTED
    direct-sum projector P_N^(k) (range=span(V[:, :k]), kernel ⊇ span(T_task)). For each nested
    prefix: task cost Delta_Y(k)=NLL(h_k((I-P_k)Z)) - NLL(h_0(Z)) (PAIRED heads) and domain gain
    Delta_D(k)=Brier(q0(D|Y)) - Brier(q_k(D|Y,P_k Z)). Feasible iff simultaneous UCB Delta_Y(k)
    <= delta_Y AND simultaneous LCB Delta_D(k) > gamma_D; k* = argmax_{feasible} LCB Delta_D(k)
    (ties -> smaller k); empty -> 0. If a prefix's span intersects span(T_task) (no direct sum)
    that prefix and all larger ones are infeasible (TASK_SUBSPACE_INTERSECTION)."""
    K = V_cand.shape[1]
    if K == 0:
        return 0, [], "NO_CANDIDATE"
    plan_g = _SplitPlan(len(Zg), cfg.n_folds, seed + 3, group_id=cluster_id)
    if not (plan_g.coverage_ok(yg, n_cls) and plan_g.coverage_ok(dg, n_dom)):
        return 0, [], "FOLD_COVERAGE_FAILURE"
    gplan = _GatePlan(plan_g, seed + 5)
    yg_oh = np.eye(n_cls)[yg]
    # task-protected projectors for each nested prefix; stop at the first direct-sum failure
    Ps, intersect_at = [], None
    for k in range(1, K + 1):
        Pk, info = task_protected_projector(V_cand[:, :k], T_task, M)
        if Pk is None:
            intersect_at = k
            break
        Ps.append(Pk)
    Kv = len(Ps)
    if Kv == 0:
        return 0, [{"k": 1, "risk_feasible": False, "decision_reason": "TASK_SUBSPACE_INTERSECTION"}], \
               "TASK_SUBSPACE_INTERSECTION"
    try:
        transforms = [Zg] + [Zg - Zg @ Pk.T for Pk in Ps]
        nll = _paired_task_nll(Zg, yg, transforms, n_cls, cfg, gplan)
        dY = np.stack([nll[k] - nll[0] for k in range(1, Kv + 1)], 1)       # [N,Kv]
        br0, q0cache = _domain_q0(dg, yg_oh, n_dom, n_cls, cfg, gplan, seed + 30)
        dD = np.stack([br0 - _domain_residual_brier(Zg @ Ps[k - 1].T, dg, yg_oh, q0cache,
                       n_dom, n_cls, cfg, seed + 40 + k) for k in range(1, Kv + 1)], 1)
    except (np.linalg.LinAlgError, KeyError):
        return 0, [], "NUMERICAL_FAILURE"
    ucb_Y = _one_sided_bound(dY, cluster_id, cfg.gate_alpha, "upper", cfg.gate_boot, seed + 60, cfg.boot_estimand)
    lcb_D = _one_sided_bound(dD, cluster_id, cfg.gate_alpha, "lower", cfg.gate_boot, seed + 70, cfg.boot_estimand)
    feasible = (ucb_Y <= cfg.delta_Y) & (lcb_D > cfg.gamma_D)
    records = []
    for k in range(1, Kv + 1):
        i = k - 1
        rsn = ("ACCEPTED" if feasible[i] else
               "TASK_RISK_UCB" if ucb_Y[i] > cfg.delta_Y else "DOMAIN_GAIN_TOO_SMALL")
        records.append({"k": k, "task_delta_mean": float(dY[:, i].mean()), "task_ucb": float(ucb_Y[i]),
                        "domain_gain_mean": float(dD[:, i].mean()), "domain_lcb": float(lcb_D[i]),
                        "risk_feasible": bool(feasible[i]), "decision_reason": rsn})
    if intersect_at is not None:
        records.append({"k": intersect_at, "risk_feasible": False,
                        "decision_reason": "TASK_SUBSPACE_INTERSECTION"})
    feas = np.where(feasible)[0]
    if feas.size == 0:
        reason = "TASK_RISK_UCB" if np.any(ucb_Y > cfg.delta_Y) else "DOMAIN_GAIN_TOO_SMALL"
        return 0, records, reason
    best = feas[np.argmax(lcb_D[feas])]              # ties: argmax returns the first => smaller k
    return int(best + 1), records, "ACCEPTED"


def select_score_fisher(Z, y, d, n_cls, n_dom, cfg: ScoreFisherConfig, seed=0, cluster_id=None):
    """Four-way isolation: S_sel (candidate ordering) | S_gate (UCB rank gate). S_test is the
    caller's (eval only). The projector is built from S_sel candidates + S_sel metric."""
    Z = np.asarray(Z, dtype=np.float64); y = np.asarray(y); d = np.asarray(d)
    z_dim = Z.shape[1]
    rng = np.random.default_rng(seed + 9)
    # group-aware top split into S_sel / S_gate (whole clusters per side)
    groups = _groups_of(cluster_id, len(Z))
    gperm = rng.permutation(len(groups))
    n_sel = max(1, int(round((1 - cfg.gate_split) * len(groups))))
    sel = np.concatenate([groups[i] for i in gperm[:n_sel]])
    blank = lambda reason: ScoreFisherReport(
        rho=np.zeros(z_dim), cand_order=np.array([], int), basis=np.zeros((z_dim, 0)),
        P=np.zeros((z_dim, z_dim)), is_identity=True, k_star=0, gate={},
        rank_records=[], decision_reason=reason)
    # need a DISJOINT, non-trivial gate split (and enough groups per side for cross-fit)
    if n_sel >= len(groups) or (len(groups) - n_sel) < cfg.n_folds or n_sel < cfg.n_folds:
        return blank("INSUFFICIENT_GROUPS")
    gat = np.concatenate([groups[i] for i in gperm[n_sel:]])
    cid_sel = None if cluster_id is None else np.asarray(cluster_id)[sel]
    cid_g = None if cluster_id is None else np.asarray(cluster_id)[gat]

    y_oh = np.eye(n_cls)[y]
    # S_sel cross-fit is GROUP-aware too (Phase 1.2.1 fix 1): pass the sel-subset cluster ids
    plan = _SplitPlan(len(sel), cfg.n_folds, seed + 1, group_id=cid_sel)
    if not (plan.coverage_ok(y[sel], n_cls) and plan.coverage_ok(d[sel], n_dom)):
        return blank("FOLD_COVERAGE_FAILURE")

    try:
        G_Y = _cross_fit_fisher(Z[sel], y[sel], None, n_cls, z_dim, 0, cfg, plan, seed)
        G_DgY = _cross_fit_fisher(Z[sel], d[sel], y_oh[sel], n_dom, z_dim, n_cls, cfg, plan, seed + 100)
        M = _metric(Z[sel], y[sel], n_cls, cfg)
    except np.linalg.LinAlgError:
        return blank("NUMERICAL_FAILURE")

    gate = leakage_gate(Z[sel], y[sel], d[sel], n_cls, n_dom, cfg, plan, seed + 500,
                        None if cluster_id is None else np.asarray(cluster_id)[sel])

    # null-energy prefilter floor (diagnostic/prefilter; not the decision)
    B = 0.5 * ((G_Y + cfg.eta * M) + (G_Y + cfg.eta * M).T)
    w, V0 = eigh(0.5 * (G_DgY + G_DgY.T), B)
    V0 = V0[:, np.argsort(w)[::-1]]
    U = V0 / np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", V0.T, M, V0.T), 1e-18))
    floor_abs = _null_floor(Z[sel], y[sel], d[sel], y_oh[sel], U, n_cls, n_dom, z_dim, cfg, plan, seed + 700)

    V_cand, rho, dom, lab, ratio, egk = candidate_order(G_DgY, G_Y, M, cfg, floor_abs)
    # protected task carrier T (top label score-Fisher eigenspace), estimated on S_sel only.
    # OFF by default (cfg.task_protect) -> T_task=None -> the projector reduces to the oblique
    # metric_projector (Phase 1.2.1 behaviour) pending the task-head-fidelity decision.
    T_task = estimate_task_basis(G_Y, M, cfg) if cfg.task_protect else None
    if not gate["open"]:
        return ScoreFisherReport(
            rho=rho, cand_order=np.array([], int), basis=np.zeros((z_dim, 0)),
            P=np.zeros((z_dim, z_dim)), is_identity=True, k_star=0, gate=gate,
            rank_records=[], decision_reason="DOMAIN_GATE_CLOSED",
            dom_floor_abs=floor_abs, eigengap_k=egk)

    # UCB rank gate on S_gate with the TASK-PROTECTED projector (S_sel metric M + task basis T)
    kstar, records, reason = ucb_rank_gate(Z[gat], y[gat], d[gat], V_cand, M,
                                           n_cls, n_dom, cfg, seed + 2000, cid_g, T_task=T_task)
    V_sel = V_cand[:, :kstar] if kstar else np.zeros((z_dim, 0))
    P, _ = task_protected_projector(V_sel, T_task, M)
    if P is None:
        P = np.zeros((z_dim, z_dim)); kstar = 0; reason = "TASK_SUBSPACE_INTERSECTION"
        V_sel = np.zeros((z_dim, 0))
    return ScoreFisherReport(rho=rho, cand_order=np.arange(V_cand.shape[1]), basis=V_sel, P=P,
                             is_identity=(kstar == 0), k_star=kstar, gate=gate,
                             rank_records=records, decision_reason=reason,
                             dom_floor_abs=floor_abs, eigengap_k=egk)


class ScoreFisherSelector(nn.Module):
    """Static score-Fisher selector (no encoder training). Buffers `P` AND `active_k` so
    `is_identity` survives a checkpoint reload."""
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
        self.active_k = torch.tensor(rep.k_star, dtype=torch.long, device=self.active_k.device)
        return rep

    def project(self, Z):
        return Z @ self.P.t().to(Z.dtype)        # nuisance component (row-vectors)

    def summary(self):
        if self.report is None:
            return {"k": int(self.active_k), "is_identity": self.is_identity}
        r = self.report
        return {"k": r.k_star, "is_identity": r.is_identity, "decision_reason": r.decision_reason,
                "gate": {kk: (round(vv, 4) if isinstance(vv, float) else vv) for kk, vv in r.gate.items()},
                "eigengap_k_diag": r.eigengap_k,
                "rank_records": r.rank_records}
