"""ACAR v3 candidate predictors (HSCR). DESIGN/DEV stage — SYNTHETIC only until DEV_DESIGN_LOCK.

Disease-specific DeepSets over a WindowActionSet -> raw-ΔR-unit outputs. Three frozen candidates:
  C1 mean-only (Huber)        : point=μ̂,  upper_center=μ̂,   scale=None        ; U=μ̂+q          ; score=ΔR−μ̂
  C2 mean+scale (β-NLL)       : point=μ̂,  upper_center=μ̂,   scale=max(σ̂,σmin) ; U=μ̂+max(q,0)σ̃  ; score=(ΔR−μ̂)/σ̃
  C3 additive CQR (pinball)   : point=q̂₅₀, upper_center=q̂₉₀, scale=None        ; U=q̂₉₀+q        ; score=ΔR−q̂₉₀
Only C2 clamps q (q⁺=max(q,0)) — it is the only candidate with the negative-q × scale uncertainty inversion;
C1/C3 use raw additive q. q_raw and q_used are both recorded by the caller. All hyperparameters are PINNED here
(DEV_DESIGN_LOCK content); nothing is left to be chosen after seeing DEV numbers.

This module computes NO ΔR target from labels itself; `fit()` is GIVEN the offline ΔR targets (Phase-2, DEV only).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn as nn

from .set_features import (WindowActionSet, PER_WINDOW_FEATURES, CONTEXT_FEATURES, action_index)

CANDIDATES = ("C1", "C2", "C3")
EPS = 1e-6

# ---- PINNED hyperparameters (DEV_DESIGN_LOCK; do not tune post-DEV) ----
HP = dict(
    psi_hidden=64, psi_layers=2, rho_hidden=64, rho_layers=2, action_emb_dim=8, activation="relu",
    pooling=("mean", "std"), dropout=0.0,
    optimizer="adam", lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
    max_epochs=200, patience=20, min_delta=1e-4, batch_sets=64,
    target_sd_floor=1e-3, huber_delta=1.0, beta_nll=0.5, eps=EPS,
    seed_outer=0, seed_fitcal=1, seed_es=2, k_folds=5, fit_frac=0.70, train_frac=0.80,
)
_F = len(PER_WINDOW_FEATURES)
_C = len(CONTEXT_FEATURES)
_WIN_IN = 2 * _F            # values + mask
_CTX_IN = 2 * _C


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate: str
    disease: str
    action: str
    point: float            # G1 point predictor / width center (raw ΔR units): μ̂ (C1/C2) or q̂₅₀ (C3)
    upper_center: float     # μ̂ (C1/C2) or q̂₉₀ (C3)
    scale: float | None     # max(σ̂,σmin) for C2; None for C1/C3


def _mlp(din, dh, dout, layers):
    mods, d = [], din
    for _ in range(layers - 1):
        mods += [nn.Linear(d, dh), nn.ReLU()]; d = dh
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class DeepSetsNet(nn.Module):
    """Permutation-invariant set encoder + action-specific output head. Pinned architecture (HP)."""

    def __init__(self, candidate: str, seed: int):
        super().__init__()
        if candidate not in CANDIDATES:
            raise ValueError(f"unknown candidate {candidate!r}")
        self.candidate = candidate
        torch.manual_seed(seed)
        self.psi = _mlp(_WIN_IN, HP["psi_hidden"], HP["psi_hidden"], HP["psi_layers"])
        self.act_emb = nn.Embedding(len(("identity",) + CANDIDATES) - 1 + 1, HP["action_emb_dim"])  # >= n actions
        pooled = HP["psi_hidden"] * len(HP["pooling"])
        out = 1 if candidate == "C1" else 2
        self.rho = _mlp(pooled + _CTX_IN + HP["action_emb_dim"], HP["rho_hidden"], out, HP["rho_layers"])
        self.eval()

    def forward(self, win, ctx, a_idx):
        # win [n, 2F], ctx [2C], a_idx int
        h = self.psi(win)                                     # [n, H]
        pooled = torch.cat([h.mean(0), h.std(0, unbiased=False)], 0)   # mean+std (permutation-invariant)
        a = self.act_emb(torch.tensor(int(a_idx)))
        z = torch.cat([pooled, ctx, a], 0)
        o = self.rho(z)
        if self.candidate == "C1":
            return o[0], None
        if self.candidate == "C2":
            mu = o[0]; v = torch.nn.functional.softplus(o[1]) + EPS
            return mu, torch.sqrt(v)                          # (mu, sigma) standardized units
        q50 = o[0]; gap = torch.nn.functional.softplus(o[1]) + EPS
        return q50, gap                                       # (q50, q90-q50) standardized units


class Candidate:
    """Wraps a disease-specific net + frozen target standardization (raw<->std) + σ_min per action."""

    def __init__(self, candidate, disease, seed, target_mean=0.0, target_sd=1.0, sigma_min=None):
        self.candidate = candidate; self.disease = disease
        self.net = DeepSetsNet(candidate, seed)
        self.target_mean = float(target_mean)
        self.target_sd = float(max(target_sd, HP["target_sd_floor"]))
        self.sigma_min = dict(sigma_min or {})               # {action: raw-unit floor}

    def _inputs(self, was: WindowActionSet):
        win = torch.tensor(np.concatenate([was.values, was.availability_mask.astype(np.float64)], axis=1),
                           dtype=torch.float32)
        ctx = torch.tensor(np.concatenate([was.context_values, was.context_mask.astype(np.float64)]),
                           dtype=torch.float32)
        return win, ctx

    def predict(self, was: WindowActionSet) -> CandidatePrediction:
        win, ctx = self._inputs(was)
        with torch.no_grad():
            a, b = self.net(win, ctx, action_index(was.action_name))
        sd, mu0 = self.target_sd, self.target_mean
        if self.candidate == "C1":
            mu = float(a) * sd + mu0
            return CandidatePrediction("C1", self.disease, was.action_name, mu, mu, None)
        if self.candidate == "C2":
            mu = float(a) * sd + mu0
            sig = float(b) * sd                               # de-standardize scale
            sig = max(sig, self.sigma_min.get(was.action_name, 0.0))
            return CandidatePrediction("C2", self.disease, was.action_name, mu, mu, sig)
        q50 = float(a) * sd + mu0
        q90 = q50 + float(b) * sd                             # gap is positive (softplus) -> no crossing
        return CandidatePrediction("C3", self.disease, was.action_name, q50, q90, None)


# ---- per-candidate nonconformity score + upper bound (exact, frozen) ----
def score(pred: CandidatePrediction, delta_r: float) -> float:
    """Nonconformity: C1/C3 additive (ΔR − upper_center); C2 standardized (ΔR − μ̂)/σ̃."""
    s = float(delta_r) - pred.upper_center
    return s / pred.scale if pred.scale is not None else s


def upper_bound(pred: CandidatePrediction, q: float) -> float:
    """U: C2 = μ̂ + max(q,0)·σ̃ (q⁺ clamp — uncertainty-inversion fix); C1/C3 = upper_center + q (NO clamp)."""
    if pred.scale is not None:                                # C2
        return pred.upper_center + max(float(q), 0.0) * pred.scale
    return pred.upper_center + float(q)                       # C1/C3
