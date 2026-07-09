"""Trust-region primitives for TALOS low-dimensional adapters."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from talos_eeg.data.source_state import SourceState, source_logits


@dataclass(frozen=True)
class TrustRegionBounds:
    tau_log_t: float = 0.25
    tau_beta: float = 1.50
    tau_a: float = 0.35
    tau_c: float = 1.50

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


@dataclass(frozen=True)
class AdapterState:
    variant: str
    diag: np.ndarray
    shift: np.ndarray
    beta: np.ndarray
    log_t: float
    boundary_hits: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "variant": self.variant,
            "diag": self.diag.astype(float).tolist(),
            "shift": self.shift.astype(float).tolist(),
            "beta": self.beta.astype(float).tolist(),
            "log_t": float(self.log_t),
            "boundary_hits": list(self.boundary_hits),
        }

    def hash(self) -> str:
        raw = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":")).encode()
        return hashlib.sha256(raw).hexdigest()


def identity_adapter(variant: str, n_features: int, n_classes: int) -> AdapterState:
    return AdapterState(
        variant=variant,
        diag=np.ones(int(n_features), dtype=np.float64),
        shift=np.zeros(int(n_features), dtype=np.float64),
        beta=np.zeros(int(n_classes), dtype=np.float64),
        log_t=0.0,
    )


def array_hash(values: np.ndarray) -> str:
    arr = np.ascontiguousarray(np.asarray(values))
    h = hashlib.sha256()
    h.update(str(arr.shape).encode())
    h.update(str(arr.dtype).encode())
    h.update(arr.tobytes())
    return h.hexdigest()


def stable_payload_hash(payload: dict[str, Any]) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(raw).hexdigest()


def clip_l2(values: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    vec = np.asarray(values, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm <= float(limit) or norm <= 1e-12:
        return vec.copy(), False
    return vec * (float(limit) / norm), True


def clip_abs(values: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    vec = np.asarray(values, dtype=np.float64)
    clipped = np.clip(vec, -float(limit), float(limit))
    return clipped, bool(np.any(np.abs(vec) > float(limit)))


def clip_scalar(value: float, limit: float) -> tuple[float, bool]:
    val = float(value)
    clipped = float(np.clip(val, -float(limit), float(limit)))
    return clipped, bool(abs(val) > float(limit))


def adapted_features(z: np.ndarray, adapter: AdapterState) -> np.ndarray:
    z = np.asarray(z, dtype=np.float64)
    return z * adapter.diag[None, :] + adapter.shift[None, :]


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_logits(state: SourceState, z: np.ndarray, adapter: AdapterState) -> np.ndarray:
    z_prime = adapted_features(z, adapter)
    logits = source_logits(state, z_prime) + adapter.beta[None, :]
    temperature = float(np.exp(adapter.log_t))
    return logits / max(temperature, 1e-8)


def predict_proba(state: SourceState, z: np.ndarray, adapter: AdapterState) -> np.ndarray:
    return softmax(predict_logits(state, z, adapter))


def trust_region_report(adapter: AdapterState, bounds: TrustRegionBounds) -> dict[str, Any]:
    diag_delta = adapter.diag - 1.0
    report = {
        "variant": adapter.variant,
        "log_t_abs": abs(float(adapter.log_t)),
        "beta_norm": float(np.linalg.norm(adapter.beta)),
        "diag_max_abs_delta": float(np.max(np.abs(diag_delta))) if len(diag_delta) else 0.0,
        "shift_norm": float(np.linalg.norm(adapter.shift)),
        "bounds": bounds.to_dict(),
        "boundary_hits": list(adapter.boundary_hits),
    }
    report["within_bounds"] = bool(
        report["log_t_abs"] <= bounds.tau_log_t + 1e-12
        and report["beta_norm"] <= bounds.tau_beta + 1e-12
        and report["diag_max_abs_delta"] <= bounds.tau_a + 1e-12
        and report["shift_norm"] <= bounds.tau_c + 1e-12
        and not adapter.boundary_hits
    )
    return report
