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
from dataclasses import dataclass, field
import hashlib
import json
import math
import struct
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
    target_sd_floor=1e-3, input_sd_floor=1e-6, huber_delta=1.0, beta_nll=0.5, eps=EPS,
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
        if self.disease not in ("PD", "SCZ"):
            raise ValueError("CandidatePrediction.disease must be PD or SCZ")
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
            if not (self.scale_raw > 0 and self.scale_used > 0 and self.scale_floor > 0):
                raise ValueError("C2 scale_raw/scale_used/scale_floor must be > 0")
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


def state_items(net) -> tuple:
    """Canonical, immutable snapshot: sorted-by-name (name, canonical little-endian dtype, shape, raw LE bytes).
    Preserves native precision but normalizes byte order so the hash is platform-canonical."""
    sd = net.state_dict()
    items = []
    for name in sorted(sd):
        a = sd[name].detach().cpu().numpy()
        le = np.ascontiguousarray(a.astype(a.dtype.newbyteorder("<")))   # explicit little-endian, same precision
        items.append((name, le.dtype.str, tuple(int(s) for s in le.shape), le.tobytes()))
    return tuple(items)


def _arch_items(candidate):
    """Reference (name -> (dtype, shape)) for a candidate's FROZEN architecture (params are float32 -> '<f4')."""
    ref = DeepSetsNet(candidate, 0).state_dict()
    return {name: ("<f4", tuple(int(s) for s in ref[name].shape)) for name in sorted(ref)}


def _validate_items(items, candidate=None):
    names = [n for n, _, _, _ in items]
    if len(set(names)) != len(names) or names != sorted(names):
        raise ValueError("state_items must have unique, sorted names")
    for name, dt, shape, buf in items:
        if dt != "<f4":
            raise ValueError(f"state_items dtype must be '<f4' (got {dt} for {name})")
        n = int(np.prod(shape)) if shape else 1
        if len(buf) != n * np.dtype(dt).itemsize:
            raise ValueError(f"state_items byte length mismatch for {name}")
        if not np.all(np.isfinite(np.frombuffer(buf, dtype=np.dtype(dt)))):
            raise ValueError(f"non-finite parameter bytes in {name}")
    if candidate is not None:
        spec = _arch_items(candidate)
        got = {name: (dt, tuple(shape)) for name, dt, shape, _ in items}
        if got != spec:
            raise ValueError("state_items names/dtype/shape do not exactly match the frozen architecture")


def build_net(candidate, items, seed=0):
    """Rebuild a deterministic net from canonical state bytes. Raises on architecture/candidate mismatch."""
    _validate_items(items, candidate)
    net = DeepSetsNet(candidate, seed)
    sd = net.state_dict()
    if {n for n, _, _, _ in items} != set(sd):
        raise ValueError("state_items keys do not match the candidate architecture")
    new = {}
    for name, dt, shape, buf in items:
        arr = np.frombuffer(buf, dtype=np.dtype(dt)).reshape(shape).astype(np.float32).copy()
        new[name] = torch.tensor(arr)
    try:
        net.load_state_dict(new)
    except RuntimeError as e:
        raise ValueError(f"state_items incompatible with {candidate} architecture: {e}")
    net.eval()
    for p in net.parameters():
        p.requires_grad_(False)
    return net


def _pos_int(x, name):
    if isinstance(x, bool) or not isinstance(x, int) or x < 1:
        raise ValueError(f"{name} must be a positive non-bool int")
    return x


def _canon_pairs(pairs, name):
    """Validate a sequence of (str, value) pairs has unique keys; return a sorted, immutable tuple."""
    items = tuple((str(k), v) for k, v in pairs)
    keys = [k for k, _ in items]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} has duplicate keys")
    return tuple(sorted(items))


@dataclass(frozen=True, slots=True)
class FittedCandidateArtifact:
    """Immutable deployment artifact. Stores canonical parameter BYTES (NOT a live mutable module). The hash covers a
    unique canonical representation of every stored field; verify_integrity() recomputes it and builds an EPHEMERAL
    net (no shared mutable state). Disease-bound."""
    candidate: str
    disease: str
    state_items: tuple
    input_norm: InputNormalizer
    target_norm: TargetNormalizer
    sigma_min: tuple          # canonical sorted (action, floor) raw units; C2 covers NON_IDENTITY, floors finite>0
    best_epoch_zero_based: int   # 0-based best checkpoint epoch (= n_epochs-1 for refit)
    checkpoint_epoch_count: int  # = best_epoch_zero_based + 1 (epochs up to the restored checkpoint)
    n_epochs_executed: int       # optimizer epochs actually run (>= checkpoint_epoch_count; == for refit)
    env: tuple
    arch_schema: str = SCHEMA_VERSION
    hp_snapshot: str = field(default="", init=False)   # immutable HP snapshot (hash never reads global HP)
    artifact_sha256: str = field(default="", init=False)

    def __post_init__(self):
        if self.candidate not in CANDIDATES or self.disease not in ("PD", "SCZ"):
            raise ValueError("bad candidate/disease")
        if self.arch_schema != SCHEMA_VERSION:
            raise ValueError("arch_schema must equal SCHEMA_VERSION")
        _pos_int(self.checkpoint_epoch_count, "checkpoint_epoch_count")
        _pos_int(self.n_epochs_executed, "n_epochs_executed")
        if isinstance(self.best_epoch_zero_based, bool) or not isinstance(self.best_epoch_zero_based, int) \
                or self.best_epoch_zero_based < 0 or self.checkpoint_epoch_count != self.best_epoch_zero_based + 1 \
                or self.n_epochs_executed < self.checkpoint_epoch_count:
            raise ValueError("inconsistent epoch provenance fields")
        object.__setattr__(self, "sigma_min", _canon_pairs(self.sigma_min, "sigma_min"))
        object.__setattr__(self, "env", _canon_pairs(self.env, "env"))
        for a, v in self.env:
            if not (isinstance(a, str) and isinstance(v, str)):
                raise ValueError("env entries must be (str, str)")
        sm = dict(self.sigma_min)
        if self.candidate == "C2":
            if set(sm) != set(NON_IDENTITY):
                raise ValueError("C2 sigma_min must cover EXACTLY the non-identity actions")
            for a, v in sm.items():
                if not isinstance(v, float):
                    raise ValueError("C2 sigma_min floors must be Python float")
                if not (math.isfinite(v) and v > 0):
                    raise ValueError(f"C2 sigma_min[{a}] must be finite and > 0 (got {v})")
        elif sm:
            raise ValueError(f"{self.candidate} must have empty sigma_min")
        _validate_items(self.state_items, self.candidate)     # exact name/dtype(<f4)/shape vs frozen arch
        object.__setattr__(self, "hp_snapshot", json.dumps({k: HP[k] for k in sorted(HP)}, sort_keys=True))
        object.__setattr__(self, "artifact_sha256", self._hash())
        build_net(self.candidate, self.state_items)           # loadability cross-check (discarded)

    def _hash(self):
        def lp(b):                                            # length-prefixed (injective) field encoding
            h.update(struct.pack(">Q", len(b))); h.update(b)
        h = hashlib.sha256()
        meta = json.dumps({"schema": self.arch_schema, "candidate": self.candidate, "disease": self.disease,
                           "action_vocab": list(ACTION_VOCAB),
                           "feature_schema": {"per_window": list(PER_WINDOW_FEATURES), "context": list(CONTEXT_FEATURES)},
                           "hp_snapshot": self.hp_snapshot,    # the artifact's OWN snapshot, not global HP
                           "best_epoch_zero_based": int(self.best_epoch_zero_based),
                           "checkpoint_epoch_count": int(self.checkpoint_epoch_count),
                           "n_epochs_executed": int(self.n_epochs_executed)}, sort_keys=True).encode()
        h.update(b"META"); lp(meta)
        self.input_norm.digest_update(h); self.target_norm.digest_update(h)
        for a, v in self.env:                                 # length-prefixed -> injective (no NUL collision)
            h.update(b"E"); lp(a.encode()); lp(v.encode())
        for a, v in self.sigma_min:                           # raw float64 bytes — NO rounding
            h.update(b"S"); lp(a.encode()); lp(np.array([v], dtype="<f8").tobytes())
        for name, dt, shape, buf in self.state_items:         # already sorted, dtype '<f4'
            h.update(b"T"); lp(name.encode()); lp(dt.encode()); lp(str(shape).encode()); lp(buf)
        return h.hexdigest()

    def verify_integrity(self):
        """Recompute the hash from the IMMUTABLE bytes; raise on mismatch. Returns None (no module is exposed)."""
        if self._hash() != self.artifact_sha256:
            raise ValueError("artifact integrity failure: recomputed hash != stored artifact_sha256")

    def _fresh_net(self):
        """Private: verify then build an EPHEMERAL net that shares no state with the artifact."""
        self.verify_integrity()
        net = build_net(self.candidate, self.state_items)
        if net.candidate != self.candidate or set(net.heads.keys()) != set(NON_IDENTITY):
            raise ValueError("artifact integrity failure: net architecture mismatch")
        return net

    def assert_disease(self, disease):
        if disease != self.disease:
            raise ValueError(f"disease mismatch: artifact is {self.disease}, batch is {disease}")

    def predict(self, was: WindowActionSet) -> CandidatePrediction:
        net = self._fresh_net()
        nw = self.input_norm.transform(was)
        win = torch.tensor(np.concatenate([nw.values, nw.availability_mask.astype(np.float64)], 1), dtype=torch.float32)
        ctx = torch.tensor(np.concatenate([nw.context_values, nw.context_mask.astype(np.float64)]), dtype=torch.float32)
        with torch.no_grad():
            a, b = net(win, ctx, was.action_name)
        if self.candidate == "C1":
            mu = float(self.target_norm.destandardize(float(a)))
            return CandidatePrediction("C1", self.disease, was.action_name, mu, mu)
        if self.candidate == "C2":
            mu = float(self.target_norm.destandardize(float(a)))
            sig_raw = float(b) * self.target_norm.sd
            floor = float(dict(self.sigma_min)[was.action_name])
            return CandidatePrediction("C2", self.disease, was.action_name, mu, mu, max(sig_raw, floor), sig_raw, floor)
        q50 = float(self.target_norm.destandardize(float(a)))
        q90 = q50 + float(b) * self.target_norm.sd
        return CandidatePrediction("C3", self.disease, was.action_name, q50, q90)


def make_artifact(candidate, disease, net, input_norm, target_norm, sigma_min,
                  best_epoch_zero_based, n_epochs_executed, env) -> FittedCandidateArtifact:
    if candidate not in CANDIDATES:
        raise ValueError("bad candidate")
    if getattr(net, "candidate", None) != candidate or set(getattr(net, "heads", {}).keys()) != set(NON_IDENTITY):
        raise ValueError("net.candidate / heads do not match the requested candidate")
    for nm, v in (("best_epoch_zero_based", best_epoch_zero_based), ("n_epochs_executed", n_epochs_executed)):
        if isinstance(v, bool) or not isinstance(v, int):
            raise ValueError(f"{nm} must be a non-bool int (no float/bool coercion)")
    # dict input has unique keys; a sequence-of-pairs is passed through UNCHANGED so duplicates are caught downstream.
    sm = tuple(sigma_min.items()) if isinstance(sigma_min, dict) else tuple(sigma_min)
    ev = tuple(env.items()) if isinstance(env, dict) else tuple(env)
    return FittedCandidateArtifact(candidate, disease, state_items(net), input_norm, target_norm,
                                   sm, best_epoch_zero_based, best_epoch_zero_based + 1, n_epochs_executed, ev)
