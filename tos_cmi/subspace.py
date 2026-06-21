"""The task-orthogonal nuisance subspace: the generalized eigenproblem and the
risk-feasible selection that decides *what* (if anything) is safe to make invariant.

Given the two Fishers from `fisher.py` we solve

    F_{D|Y} v_j = rho_j (F_Y + eta I) v_j .

A large rho_j means direction v_j carries a lot of class-conditional *domain* energy
per unit of *label* energy -- i.e. domain-rich and label-light. We then evaluate each
direction's energies on its Euclidean-unit version (scale-consistent) and keep the
directions that are simultaneously:

  * domain-rich   : ratio_j = dom_j / (lab_j + eta) >= tau_ratio
  * label-light   : lab_j <= eps_label * max_k lab_k       (the *risk-feasibility* gate)
  * non-null      : dom_j  >= dom_floor * max_k dom_k

The label-light gate is the risk-feasible constraint in spectral form: a direction is
only deletable if removing it costs almost no label information. When task and domain
subspaces overlap, every domain-rich direction is also label-rich, NO direction
qualifies, and the projector becomes the zero map -- the method *refuses to delete* and
degrades to identity. That refusal is the whole point: it is what global LPC (which
deletes unconditionally and collapses TSMNet) cannot do.

The Euclidean orthogonal projector onto the selected span is returned as a fixed buffer;
the penalty in `selective_cmi.py` applies leakage pressure only to `Z @ P_N`.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import eigh

from .config import FisherConfig, SubspaceConfig
from .fisher import fisher_pair, null_domain_energy_floor


@dataclass
class SubspaceReport:
    """Everything needed to judge the selection (and to diff it across seeds/folds)."""
    rho: np.ndarray            # generalized eigenvalues, descending
    dom_energy: np.ndarray     # F_{D|Y} energy of each unit eigenvector (descending by rho)
    lab_energy: np.ndarray     # F_Y energy of each unit eigenvector
    ratio: np.ndarray          # dom/(lab+eta)
    selected: np.ndarray       # indices (into the descending order) chosen as nuisance
    basis: np.ndarray          # [d, k] orthonormal basis of the selected span (k may be 0)
    is_identity: bool          # True iff nothing was selected (projector is zero)
    eta: float
    dom_floor_abs: float = 0.0 # absolute domain-energy floor from the within-Y null
    eigengap: float = 0.0      # ratio gap at the selection boundary (separation quality)

    @property
    def k(self) -> int:
        return int(self.basis.shape[1])


def solve_generalized(F_DgY: torch.Tensor, F_Y: torch.Tensor, eta: float):
    """Solve F_DgY v = rho (F_Y + eta I) v. Returns (rho desc [d], V desc [d,d]).
    B = F_Y + eta I is SPD for eta>0, so this is a stable symmetric generalized problem.
    Eigenvectors are B-orthonormal (V^T B V = I); we re-normalize per-direction for gating."""
    A = F_DgY.detach().cpu().double().numpy()
    B = F_Y.detach().cpu().double().numpy()
    A = 0.5 * (A + A.T)
    B = 0.5 * (B + B.T)
    d = B.shape[0]
    B = B + eta * np.eye(d)
    w, V = eigh(A, B)                       # ascending
    order = np.argsort(w)[::-1]            # descending rho
    return w[order], V[:, order]


def _orthonormal_projector(basis_cols: np.ndarray):
    """Euclidean orthogonal projector P = Q Q^T onto span(columns). basis_cols: [d,k]."""
    if basis_cols.shape[1] == 0:
        d = basis_cols.shape[0]
        return np.zeros((d, d)), np.zeros((d, 0))
    Q, _ = np.linalg.qr(basis_cols)        # [d,k] orthonormal
    return Q @ Q.T, Q


def select_nuisance(F_DgY, F_Y, fcfg: FisherConfig, scfg: SubspaceConfig,
                    dom_floor_abs: float = 0.0) -> SubspaceReport:
    """Run the generalized eig and apply the risk-feasible selection rule.
    `dom_floor_abs` is the absolute domain-energy floor from the within-Y permutation null
    (see fisher.null_domain_energy_floor); directions below it are sampling noise and are
    rejected even if their (label-null) ratio looks large."""
    rho, V = solve_generalized(F_DgY, F_Y, fcfg.eta)
    A = 0.5 * (F_DgY.detach().cpu().double().numpy() + F_DgY.detach().cpu().double().numpy().T)
    B = 0.5 * (F_Y.detach().cpu().double().numpy() + F_Y.detach().cpu().double().numpy().T)
    d = V.shape[0]

    # scale-consistent energies on Euclidean-unit eigenvectors
    Vn = V / (np.linalg.norm(V, axis=0, keepdims=True) + 1e-12)
    dom = np.einsum("ij,jk,ik->i", Vn.T, A, Vn.T)       # dom_j = u_j^T A u_j
    lab = np.einsum("ij,jk,ik->i", Vn.T, B, Vn.T)       # lab_j = u_j^T B u_j
    dom = np.clip(dom, 0.0, None)
    lab = np.clip(lab, 0.0, None)
    ratio = dom / (lab + fcfg.eta)

    dom_max = dom.max() if dom.size and dom.max() > 0 else 1.0
    lab_max = lab.max() if lab.size and lab.max() > 0 else 1.0
    floor = max(scfg.dom_floor * dom_max, scfg.null_safety * dom_floor_abs)
    eligible = (
        (ratio >= scfg.tau_ratio)        # domain-rich relative to its own label content
        & (lab <= scfg.eps_label * lab_max)  # label-light (the risk-feasibility gate)
        & (dom >= floor)                 # above the sampling-noise / null floor
    )
    cand = np.where(eligible)[0]
    cand = cand[np.argsort(ratio[cand])[::-1]]          # highest ratio first
    selected = cand[: scfg.max_dim]

    if scfg.min_dim > 0 and selected.size < scfg.min_dim:
        selected = np.array([], dtype=int)              # not enough safe room -> identity

    # eigengap: ratio separation between the lowest selected and the highest rejected
    # direction -- a clean selection has a large gap; a borderline one a small gap.
    ratio_sorted = np.sort(ratio)[::-1]
    k = selected.size
    eigengap = float(ratio_sorted[k - 1] - ratio_sorted[k]) if 0 < k < ratio_sorted.size else 0.0

    basis_cols = Vn[:, selected] if selected.size else np.zeros((d, 0))
    _, Q = _orthonormal_projector(basis_cols)
    return SubspaceReport(
        rho=rho, dom_energy=dom, lab_energy=lab, ratio=ratio,
        selected=np.sort(selected), basis=Q, is_identity=(Q.shape[1] == 0), eta=fcfg.eta,
        dom_floor_abs=float(dom_floor_abs), eigengap=eigengap,
    )


class SubspaceSelector(nn.Module):
    """Stateful holder of the current nuisance projector P_N, refreshed from data.

    An nn.Module so the projector is a registered BUFFER: it enters `state_dict`, moves with
    `.to(device)`, and survives checkpoint/DDP (the previous plain-attribute version did
    not). `Q` (the [d,k] orthonormal basis) is kept as a python attribute since k varies
    across refreshes; the fixed [d,d] projector `P` is the buffer.

    Usage (mirrors the project's Step-A / Step-B discipline):
        sel = SubspaceSelector(z_dim, n_cls, n_dom)
        sel.refresh(Z, y, d)                 # recompute P_N (no grad)
        Zn = sel.project(Z)                  # [B,d] nuisance component, grad flows through P_N
        penalty = lam * I(Zn; D | Y)         # applied only on Zn (selective_cmi.py)
    """

    def __init__(self, z_dim, n_cls, n_dom,
                 fcfg: Optional[FisherConfig] = None,
                 scfg: Optional[SubspaceConfig] = None,
                 device="cpu"):
        super().__init__()
        self.z_dim, self.n_cls, self.n_dom = z_dim, n_cls, n_dom
        self.fcfg = fcfg or FisherConfig()
        self.scfg = scfg or SubspaceConfig()
        self.register_buffer("P", torch.zeros(z_dim, z_dim))   # nuisance projector (in state_dict)
        self.Q = None                                          # [d,k] basis (k varies)
        self.report: Optional[SubspaceReport] = None
        self.to(device)

    @property
    def is_identity(self) -> bool:
        """True when no safe nuisance subspace exists -> the penalty is off."""
        return self.report is None or self.report.is_identity

    @torch.no_grad()
    def refresh(self, Z, y, d, seed: int = 0) -> SubspaceReport:
        F_DgY, F_Y = fisher_pair(Z, y, d, self.n_cls, self.n_dom, self.fcfg)
        floor = null_domain_energy_floor(Z, y, d, self.n_cls, self.n_dom, self.fcfg,
                                         n_perm=self.scfg.n_perm, seed=seed)
        rep = select_nuisance(F_DgY, F_Y, self.fcfg, self.scfg, dom_floor_abs=floor)
        self.report = rep
        dev = self.P.device
        self.Q = torch.tensor(rep.basis, dtype=torch.float32, device=dev)   # [d,k]
        P = rep.basis @ rep.basis.T                                          # [d,d], zeros if identity
        self.P = torch.tensor(P, dtype=torch.float32, device=dev)           # updates the buffer
        return rep

    def project(self, Z: torch.Tensor) -> torch.Tensor:
        """Z @ P_N : the nuisance component of Z (grad flows; P_N is a constant buffer)."""
        return Z @ self.P.to(Z.dtype)

    def project_k(self, Z: torch.Tensor) -> torch.Tensor:
        """Z @ Q_N : the nuisance coordinates in R^k (k = current selected dim). Cleaner
        critic input than the ambient-dim P_N Z; returns shape [B,0] under identity."""
        if self.Q is None or self.Q.shape[1] == 0:
            return Z.new_zeros((Z.shape[0], 0))
        return Z @ self.Q.to(Z.dtype)

    def summary(self) -> dict:
        if self.report is None:
            return {"k": 0, "is_identity": True}
        r = self.report
        return {
            "k": r.k,
            "is_identity": r.is_identity,
            "selected": r.selected.tolist(),
            "dom_floor_abs": float(r.dom_floor_abs),
            "eigengap": float(r.eigengap),
            "rho_top": [float(x) for x in r.rho[:5]],
            "ratio_selected": [float(r.ratio[i]) for i in r.selected],
            "lab_share_selected": [float(r.lab_energy[i] / (r.lab_energy.max() + 1e-12))
                                   for i in r.selected],
        }
