"""FIT-only, mask-aware normalizers for ACAR v3 (HSCR). DESIGN/DEV stage — SYNTHETIC only.

Two immutable, hashable, serializable normalizers fit on FIT data ONLY (notes/ACAR_V3_FREEZE_SKELETON.md S2/S5):
  - InputNormalizer: per-window-feature + per-context-feature mean/SD over AVAILABLE slots (mask=1). transform()
    standardizes available slots and keeps masked slots at exact 0 (mask preserved). SD floor frozen.
  - TargetNormalizer: per-disease ΔR mean/SD (FIT). standardize()/destandardize() round-trip; SD floor frozen.
Both are part of the FittedCandidate artifact and enter its SHA-256.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import numpy as np

from .set_features import (WindowActionSet, PER_WINDOW_FEATURES, CONTEXT_FEATURES, action_index)

SD_FLOOR = 1e-6
_F = len(PER_WINDOW_FEATURES)
_C = len(CONTEXT_FEATURES)


@dataclass(frozen=True, slots=True)
class InputNormalizer:
    win_mean: np.ndarray   # [F]
    win_sd: np.ndarray     # [F]
    ctx_mean: np.ndarray   # [C]
    ctx_sd: np.ndarray     # [C]

    def __post_init__(self):
        for a, n in ((self.win_mean, _F), (self.win_sd, _F), (self.ctx_mean, _C), (self.ctx_sd, _C)):
            arr = np.ascontiguousarray(np.asarray(a, float))
            if arr.shape != (n,) or not np.all(np.isfinite(arr)):
                raise ValueError("InputNormalizer arrays malformed")
        if np.any(self.win_sd < SD_FLOOR) or np.any(self.ctx_sd < SD_FLOOR):
            raise ValueError("SD below floor — fit() must clamp")
        for f in ("win_mean", "win_sd", "ctx_mean", "ctx_sd"):
            arr = np.ascontiguousarray(np.asarray(getattr(self, f), float)); arr.flags.writeable = False
            object.__setattr__(self, f, arr)

    @staticmethod
    def fit(sets) -> "InputNormalizer":
        sets = list(sets)
        if not sets:
            raise ValueError("InputNormalizer.fit needs >=1 set")
        wsum = np.zeros(_F); wsq = np.zeros(_F); wn = np.zeros(_F)
        csum = np.zeros(_C); csq = np.zeros(_C); cn = np.zeros(_C)
        for s in sets:
            v, m = s.values, s.availability_mask
            wsum += (v * m).sum(0); wsq += ((v ** 2) * m).sum(0); wn += m.sum(0)
            cm = s.context_mask
            csum += s.context_values * cm; csq += (s.context_values ** 2) * cm; cn += cm
        wn = np.maximum(wn, 1); cn = np.maximum(cn, 1)
        wm = wsum / wn; wsd = np.sqrt(np.maximum(wsq / wn - wm ** 2, 0.0))
        cmn = csum / cn; csd = np.sqrt(np.maximum(csq / cn - cmn ** 2, 0.0))
        return InputNormalizer(wm, np.maximum(wsd, SD_FLOOR), cmn, np.maximum(csd, SD_FLOOR))

    def transform(self, was: WindowActionSet) -> WindowActionSet:
        v = np.where(was.availability_mask == 1, (was.values - self.win_mean) / self.win_sd, 0.0)
        cv = np.where(was.context_mask == 1, (was.context_values - self.ctx_mean) / self.ctx_sd, 0.0)
        return WindowActionSet(v, was.availability_mask.copy(), cv, was.context_mask.copy(),
                               was.action_name, was.action_index, was.window_keys)

    def digest_update(self, h):
        for f in ("win_mean", "win_sd", "ctx_mean", "ctx_sd"):
            h.update(f.encode()); h.update(np.ascontiguousarray(getattr(self, f), dtype="<f8").tobytes())


@dataclass(frozen=True, slots=True)
class TargetNormalizer:
    mean: float
    sd: float

    def __post_init__(self):
        if not (np.isfinite(self.mean) and np.isfinite(self.sd)) or self.sd < SD_FLOOR:
            raise ValueError("TargetNormalizer malformed (sd must be >= floor)")

    @staticmethod
    def fit(delta_r_values) -> "TargetNormalizer":
        d = np.asarray(list(delta_r_values), float)
        if d.size == 0 or not np.all(np.isfinite(d)):
            raise ValueError("TargetNormalizer.fit needs finite ΔR values")
        return TargetNormalizer(float(d.mean()), float(max(d.std(), SD_FLOOR)))

    def standardize(self, y):
        return (np.asarray(y, float) - self.mean) / self.sd

    def destandardize(self, y_std):
        return np.asarray(y_std, float) * self.sd + self.mean

    def digest_update(self, h):
        h.update(b"tgt"); h.update(np.array([self.mean, self.sd], dtype="<f8").tobytes())
