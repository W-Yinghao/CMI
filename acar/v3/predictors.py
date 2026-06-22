"""ACAR v3 candidate predictors (HSCR). DESIGN/DEV stage — SYNTHETIC only until DEV_DESIGN_LOCK.

Disease-specific DeepSets (shared ψ trunk + shared ρ trunk + per-action ModuleDict heads) over a WindowActionSet ->
raw-ΔR-unit outputs. Frozen formulas (notes/ACAR_V3_FREEZE_SKELETON.md S1):
  C1 mean (Huber)      : point=μ̂,  upper_center=μ̂,  scale=None         ; U=μ̂+q          ; s=ΔR−μ̂
  C2 mean+scale (β-NLL): point=μ̂,  upper_center=μ̂,  scale_used=max(σ̂,σmin) ; U=μ̂+max(q,0)σ̃ ; s=(ΔR−μ̂)/σ̃
  C3 additive CQR      : point=q̂₅₀, upper_center=q̂₉₀, scale=None         ; U=q̂₉₀+q        ; s=ΔR−q̂₉₀
score()/upper_bound() dispatch on the validated `candidate` field. Only C2 clamps q (q⁺); C1/C3 raw additive q.
Training lives in training.py; `FittedCandidate` is the immutable, hashed artifact used at deploy.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import math
import sys
import numpy as np
import torch
import torch.nn as nn

from .set_features import WindowActionSet, PER_WINDOW_FEATURES, CONTEXT_FEATURES, NON_IDENTITY, ACTION_VOCAB
from .normalizers import InputNormalizer, TargetNormalizer

SCHEMA_VERSION = "acar-v3-pred/1"
CANDIDATES = ("C1", "C2", "C3")
EPS = 1e-6
_F = len(PER_WINDOW_FEATURES)
_C = len(CONTEXT_FEATURES)
_WIN_IN = 2 * _F
_CTX_IN = 2 * _C

HP = dict(
    psi_hidden=64, psi_layers=2, rho_hidden=64, rho_layers=2, activation="relu",
    pooling=("mean", "std"), dropout=0.0,
    optimizer="adam", lr=1e-3, weight_decay=1e-4, grad_clip=1.0,
    max_epochs=200, patience=20, min_delta=1e-4, batch_subjects=32,
    target_sd_floor=1e-6, huber_delta=1.0, beta_nll=0.5, eps=EPS,
    sigma_min_quantile=0.05, seed_outer=0, seed_fitcal=1, seed_es=2,
    k_folds=5, fit_frac=0.70, train_frac=0.80,
)


def env_versions():
    return {"python": sys.version.split()[0], "torch": torch.__version__,
            "numpy": np.__version__, "scipy": __import__("scipy").__version__}


def _mlp(din, dh, dout, layers):
    mods, d = [], din
    for _ in range(layers - 1):
        mods += [nn.Linear(d, dh), nn.ReLU()]; d = dh
    mods += [nn.Linear(d, dout)]
    return nn.Sequential(*mods)


class DeepSetsNet(nn.Module):
    """Shared ψ trunk + shared ρ trunk + per-action heads (ModuleDict keyed by NON_IDENTITY)."""

    def __init__(self, candidate: str, seed: int):
        super().__init__()
        if candidate not in CANDIDATES:
            raise ValueError(f"unknown candidate {candidate!r}")
        self.candidate = candidate
        torch.manual_seed(seed)
        self.psi = _mlp(_WIN_IN, HP["psi_hidden"], HP["psi_hidden"], HP["psi_layers"])
        pooled = HP["psi_hidden"] * len(HP["pooling"])
        self.rho = _mlp(pooled + _CTX_IN, HP["rho_hidden"], HP["rho_hidden"], HP["rho_layers"])
        out = 1 if candidate == "C1" else 2
        self.heads = nn.ModuleDict({a: nn.Linear(HP["rho_hidden"], out) for a in NON_IDENTITY})

    def trunk(self, win, ctx):
        h = self.psi(win)
        pooled = torch.cat([h.mean(0), h.std(0, unbiased=False)], 0)
        return torch.relu(self.rho(torch.cat([pooled, ctx], 0)))

    def forward(self, win, ctx, action):
        r = self.trunk(win, ctx)
        o = self.heads[action](r)
        if self.candidate == "C1":
            return o[0], None
        if self.candidate == "C2":
            return o[0], torch.sqrt(torch.nn.functional.softplus(o[1]) + EPS)   # (μ, σ) std units
        return o[0], torch.nn.functional.softplus(o[1]) + EPS                   # (q50, gap) std units


@dataclass(frozen=True, slots=True)
class CandidatePrediction:
    candidate: str
    disease: str
    action: str
    point: float
    upper_center: float
    scale_used: float | None = None
    scale_raw: float | None = None
    scale_floor: float | None = None

    def __post_init__(self):
        if self.candidate not in CANDIDATES:
            raise ValueError("bad candidate")
        if self.action not in NON_IDENTITY:
            raise ValueError("bad action")
        if not (math.isfinite(self.point) and math.isfinite(self.upper_center)):
            raise ValueError("non-finite point/upper_center")
        if self.candidate in ("C1", "C3"):
            if self.scale_used is not None or self.scale_raw is not None or self.scale_floor is not None:
                raise ValueError(f"{self.candidate} must have no scale")
            if self.candidate == "C1" and self.upper_center != self.point:
                raise ValueError("C1 upper_center must equal point")
            if self.candidate == "C3" and not (self.upper_center > self.point):
                raise ValueError("C3 requires q̂₉₀ > q̂₅₀")
        else:  # C2
            if self.upper_center != self.point:
                raise ValueError("C2 upper_center must equal point (μ̂)")
            for nm, val in (("scale_raw", self.scale_raw), ("scale_floor", self.scale_floor),
                            ("scale_used", self.scale_used)):
                if val is None or not math.isfinite(val):
                    raise ValueError(f"C2 {nm} must be finite")
            if not (self.scale_raw > 0 and self.scale_used > 0):
                raise ValueError("C2 scale_raw/scale_used must be > 0")
            if abs(self.scale_used - max(self.scale_raw, self.scale_floor)) > 1e-12:
                raise ValueError("C2 scale_used must equal max(scale_raw, scale_floor)")


def score(pred: CandidatePrediction, delta_r: float) -> float:
    if not math.isfinite(float(delta_r)):
        raise ValueError("ΔR must be finite")
    s = float(delta_r) - pred.upper_center
    if pred.candidate == "C2":
        return s / pred.scale_used
    return s                                                 # C1/C3


def upper_bound(pred: CandidatePrediction, q: float) -> float:
    q = float(q)
    if math.isnan(q):
        raise ValueError("q must not be NaN")
    if pred.candidate == "C2":
        return pred.upper_center + max(q, 0.0) * pred.scale_used    # q⁺ clamp
    return pred.upper_center + q                              # C1/C3 (q may be +inf)


@dataclass(frozen=True, slots=True)
class FittedCandidate:
    candidate: str
    disease: str
    net: nn.Module
    input_norm: InputNormalizer
    target_norm: TargetNormalizer
    sigma_min: tuple        # (action, floor) pairs, raw units — must cover all NON_IDENTITY for C2
    training_epoch: int
    env: tuple              # (key,val) pairs
    artifact_sha256: str = ""

    def __post_init__(self):
        if self.candidate not in CANDIDATES or self.disease not in ("PD", "SCZ"):
            raise ValueError("bad candidate/disease")
        sm = dict(self.sigma_min)
        if self.candidate == "C2" and set(sm) != set(NON_IDENTITY):
            raise ValueError("C2 sigma_min must cover EXACTLY the non-identity actions")
        self.net.eval()
        for p in self.net.parameters():
            p.requires_grad_(False)
        object.__setattr__(self, "artifact_sha256", self._compute_hash())

    def _floor(self, action):
        sm = dict(self.sigma_min)
        if action not in sm:
            raise KeyError(f"sigma_min missing action {action!r}")
        return float(sm[action])

    def predict(self, was: WindowActionSet) -> CandidatePrediction:
        nw = self.input_norm.transform(was)
        win = torch.tensor(np.concatenate([nw.values, nw.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
        ctx = torch.tensor(np.concatenate([nw.context_values, nw.context_mask.astype(np.float64)]), dtype=torch.float32)
        with torch.no_grad():
            a, b = self.net(win, ctx, was.action_name)
        if self.candidate == "C1":
            mu = float(self.target_norm.destandardize(float(a)))
            return CandidatePrediction("C1", self.disease, was.action_name, mu, mu)
        if self.candidate == "C2":
            mu = float(self.target_norm.destandardize(float(a)))
            sig_raw = float(b) * self.target_norm.sd          # scale de-standardization (×sd only)
            floor = self._floor(was.action_name)
            used = max(sig_raw, floor)
            return CandidatePrediction("C2", self.disease, was.action_name, mu, mu, used, sig_raw, floor)
        q50 = float(self.target_norm.destandardize(float(a)))
        q90 = q50 + float(b) * self.target_norm.sd
        return CandidatePrediction("C3", self.disease, was.action_name, q50, q90)

    def _compute_hash(self):
        h = hashlib.sha256()
        meta = json.dumps({"schema": SCHEMA_VERSION, "candidate": self.candidate, "disease": self.disease,
                           "action_vocab": list(ACTION_VOCAB),
                           "feature_schema": {"per_window": list(PER_WINDOW_FEATURES), "context": list(CONTEXT_FEATURES)},
                           "hp": {k: HP[k] for k in sorted(HP)}, "env": dict(self.env),
                           "training_epoch": int(self.training_epoch),
                           "sigma_min": {a: round(float(v), 12) for a, v in dict(self.sigma_min).items()}},
                          sort_keys=True).encode()
        h.update(b"META\x00"); h.update(meta)
        self.input_norm.digest_update(h); self.target_norm.digest_update(h)
        for name, t in self.net.state_dict().items():
            h.update(b"T\x00"); h.update(name.encode()); h.update(str(t.dtype).encode())
            h.update(str(tuple(t.shape)).encode())
            h.update(np.ascontiguousarray(t.detach().cpu().numpy()).tobytes())
        return h.hexdigest()
