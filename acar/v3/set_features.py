"""Per-window paired-set extraction for ACAR v3 (HSCR). DESIGN/DEV stage — SYNTHETIC inputs only until DEV_DESIGN_LOCK.
Hardened per the 93f417c + 685a526 reviews (notes/ACAR_V3_AMENDMENT_2.md, _3.md).

Data contract:
    WindowActionSet (frozen + slots + validated; arrays read-only)
      values [n,F], availability_mask [n,F]{0,1}, context_values [C], context_mask [C]{0,1},
      action_name in NON_IDENTITY, action_index == ACTION_VOCAB.index(action_name), window_keys (unique).
    FallbackBatchRecord (frozen): forced-identity provenance for <MIN_BATCH batches (no adapter is ever called).

Hardening invariants: canonical row order BEFORE adapters; exact full-64 SHA-256 digest (no rounding); object-level
immutability; canonical action execution order; action capability map; probability/shape validation; identity
reference computed ONCE per batch; disambiguated structured WindowKey encoding. No labels, no ΔR, no metrics here.
"""
from __future__ import annotations
from dataclasses import dataclass
from collections import namedtuple
import hashlib
import json
import numpy as np

from acar.config import MIN_BATCH, B
from acar.actions import apply_action
from ._util import frozen_array
from acar.features import (_entropy, _margin, _jsd, _bures_w2, _fisher_ratio, _ess, _maha2, _shrink)

SCHEMA_VERSION = "acar-v3-set/2"

ACTION_VOCAB = ("identity", "matched_coral", "spdim", "t3a")
NON_IDENTITY = tuple(a for a in ACTION_VOCAB if a != "identity")
ACTION_GEOMETRY = {"matched_coral": True, "spdim": True, "t3a": False}

PER_WINDOW_FEATURES = ("ent0", "enta", "d_ent", "margin0", "margina", "d_margin",
                       "flip", "js", "conf_change", "embed_disp")
GEOM_WINDOW_FEATURES = ("embed_disp",)
CONTEXT_FEATURES = ("bures", "post_sep", "n_eff", "g_unc", "s_support", "s_sep", "pr_cmi_proxy")
GEOM_CONTEXT_FEATURES = ("bures", "post_sep", "s_support", "s_sep", "pr_cmi_proxy")

@dataclass(frozen=True, slots=True)
class WindowKey:
    dataset_id: str
    subject_id: str
    recording_id: str
    window_index: int

    def __post_init__(self):
        for f in (self.dataset_id, self.subject_id, self.recording_id):
            if not isinstance(f, str) or f == "":
                raise ValueError("WindowKey ids must be non-empty str")
        if isinstance(self.window_index, bool) or not isinstance(self.window_index, int) or self.window_index < 0:
            raise ValueError("window_index must be a non-negative int (no bool, no coercion)")


def canon_key(k) -> str:
    """Disambiguated canonical serialization. Structured WindowKey -> 'WK'+canonical JSON array (no delimiter
    collision); toy string -> 'S'+string. The real loader must emit WindowKey (strings are toy-compat only)."""
    if isinstance(k, WindowKey):
        return "WK" + json.dumps([str(k.dataset_id), str(k.subject_id), str(k.recording_id), int(k.window_index)],
                                 separators=(",", ":"))
    if isinstance(k, str):
        return "S" + k
    raise TypeError(f"window key must be WindowKey or str; got {type(k).__name__}")


def action_index(name: str) -> int:
    return ACTION_VOCAB.index(name)


def canonical_tie_break(candidates):
    return min(candidates, key=action_index)


def _as_bin(m):
    m = np.asarray(m)
    if not np.all(np.isin(m, (0, 1))):
        raise ValueError("availability mask must be binary {0,1}")
    return np.ascontiguousarray(m.astype(np.uint8))


@dataclass(frozen=True, slots=True)
class WindowActionSet:
    values: np.ndarray
    availability_mask: np.ndarray
    context_values: np.ndarray
    context_mask: np.ndarray
    action_name: str
    action_index: int
    window_keys: tuple

    def __post_init__(self):
        v = np.ascontiguousarray(np.asarray(self.values, dtype=np.float64))
        m = _as_bin(self.availability_mask)
        if v.ndim != 2 or m.shape != v.shape:
            raise ValueError("values/availability_mask must be 2-D and same shape")
        if v.shape[1] != len(PER_WINDOW_FEATURES):
            raise ValueError(f"values must have {len(PER_WINDOW_FEATURES)} per-window features")
        if not (MIN_BATCH <= v.shape[0] <= B):
            raise ValueError(f"WindowActionSet n_windows must be in [{MIN_BATCH}, {B}] (got {v.shape[0]}) — a real "
                             f"eligible batch; <MIN_BATCH is a FallbackBatchRecord, not a WindowActionSet")
        cv = np.ascontiguousarray(np.asarray(self.context_values, dtype=np.float64))
        cm = _as_bin(self.context_mask)
        if cv.shape != (len(CONTEXT_FEATURES),) or cm.shape != (len(CONTEXT_FEATURES),):
            raise ValueError(f"context must have length {len(CONTEXT_FEATURES)}")
        if not np.all(np.isfinite(v)) or not np.all(np.isfinite(cv)):
            raise ValueError("non-finite value in WindowActionSet")
        if not np.array_equal(v[m == 0], np.zeros_like(v[m == 0])):
            raise ValueError("masked-unavailable slots must be exactly 0.0")
        if not np.array_equal(cv[cm == 0], np.zeros_like(cv[cm == 0])):
            raise ValueError("masked-unavailable context slots must be exactly 0.0")
        if self.action_name not in NON_IDENTITY:
            raise ValueError(f"action_name must be a non-identity action; got {self.action_name!r}")
        if self.action_index != ACTION_VOCAB.index(self.action_name):
            raise ValueError("action_index inconsistent with action_name")
        keys = tuple(self.window_keys)
        if len(keys) != v.shape[0]:
            raise ValueError("window_keys length != n_windows")
        ck = [canon_key(k) for k in keys]
        if len(set(ck)) != len(ck):
            raise ValueError("duplicate window_keys")
        object.__setattr__(self, "values", frozen_array(v))
        object.__setattr__(self, "availability_mask", frozen_array(m))
        object.__setattr__(self, "context_values", frozen_array(cv))
        object.__setattr__(self, "context_mask", frozen_array(cm))
        object.__setattr__(self, "window_keys", tuple(keys))


@dataclass(frozen=True, slots=True)
class FallbackBatchRecord:
    forced_identity: bool
    reason: str
    window_keys: tuple
    canonical_input_digest: str
    n_windows: int

    def __post_init__(self):
        if self.forced_identity is not True:
            raise ValueError("FallbackBatchRecord.forced_identity must be True")
        if not (isinstance(self.canonical_input_digest, str) and len(self.canonical_input_digest) == 64
                and all(c in "0123456789abcdef" for c in self.canonical_input_digest)):
            raise ValueError("canonical_input_digest must be a full hex SHA-256")
        keys = tuple(self.window_keys)
        if self.n_windows != len(keys) or not (1 <= self.n_windows < MIN_BATCH):
            raise ValueError(f"FallbackBatchRecord n_windows must equal len(window_keys) and be in [1, {MIN_BATCH})")
        object.__setattr__(self, "window_keys", keys)


def _input_digest(z, keys):
    h = hashlib.sha256(); h.update(SCHEMA_VERSION.encode())
    order = sorted(range(len(keys)), key=lambda i: canon_key(keys[i]))
    for i in order:
        h.update(b"K\x00"); h.update(canon_key(keys[i]).encode())
        h.update(b"Z\x00"); h.update(np.ascontiguousarray(z[i], dtype="<f8").tobytes())
    return h.hexdigest()


def _validate_proba(p, n, n_cls, name):
    p = np.asarray(p, float)
    if p.shape != (n, n_cls):
        raise ValueError(f"{name} has shape {p.shape}, expected {(n, n_cls)}")
    if not np.all(np.isfinite(p)):
        raise ValueError(f"{name} has non-finite entries")
    if np.any(p < -1e-9):
        raise ValueError(f"{name} has negative probabilities")
    if not np.allclose(p.sum(1), 1.0, atol=1e-6):
        raise ValueError(f"{name} rows do not sum to 1")


def _context(state, z0, za, pa, geom):
    n_eff = _ess(pa); g_unc = float(_entropy(pa).mean())
    if geom:
        Winv = np.linalg.inv(_shrink(np.asarray(state["Sig_pool0"], float)))
        m = np.stack([_maha2(za, state["mu_y"][c], Winv) for c in range(state["n_cls"])], 1)
        mg = np.sort(m, 1)[:, 1] - np.sort(m, 1)[:, 0]
        readout = state["clf"].predict(za); proto = m.argmin(1)
        vals = np.array([_bures_w2(za, z0), _fisher_ratio(za, pa.argmax(1)), n_eff, g_unc,
                         float(_maha2(za, state["mu_pool"], Winv).mean()),
                         float((-np.abs(m[:, 0] - m[:, 1])).mean()),
                         float(((proto != readout).astype(float) * mg + 0.01 * mg).mean())], float)
        mask = np.ones(len(CONTEXT_FEATURES), np.uint8)
    else:
        vals = np.array([0.0, 0.0, n_eff, g_unc, 0.0, 0.0, 0.0], float)
        mask = np.array([0, 0, 1, 1, 0, 0, 0], np.uint8)
    return vals, mask


def _canonicalize(z, window_keys):
    z = np.asarray(z, float); keys = list(window_keys)
    if len(keys) != len(z):
        raise ValueError("window_keys length != n_windows")
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    ck = [canon_key(k) for k in keys]
    if len(set(ck)) != len(ck):
        raise ValueError("duplicate window_keys are not allowed (window identity must be unique)")
    order = sorted(range(len(keys)), key=lambda i: ck[i])
    return z[order], [keys[i] for i in order]


def _build_was(state, z, keys, action, p0, z0, pa, za) -> WindowActionSet:
    """Feature computation from PRECOMPUTED outputs (identity p0,z0 and action pa,za). The single-execution loader path
    captures the adapter outputs ONCE and calls this directly, so a WindowActionSet and the ΔR computed elsewhere are
    guaranteed to derive from the same execution (no second adapter pass)."""
    n, n_cls = len(z), state["n_cls"]
    if not (MIN_BATCH <= n <= B):
        raise ValueError(f"eligible batch n_windows must be in [{MIN_BATCH}, {B}]; got {n}")
    if (za is not None) != ACTION_GEOMETRY[action]:
        raise ValueError(f"action capability drift: {action} geometry={za is not None}, expected {ACTION_GEOMETRY[action]}")
    _validate_proba(p0, n, n_cls, "p0"); _validate_proba(pa, n, n_cls, "pa")
    if za is not None and np.asarray(za).shape != z.shape:
        raise ValueError("embedding shape mismatch")
    geom = za is not None
    ent0, enta = _entropy(p0), _entropy(pa)
    m0, ma = _margin(p0), _margin(pa)
    flip = (pa.argmax(1) != p0.argmax(1)).astype(float)
    js = _jsd(p0, pa); conf = pa.max(1) - p0.max(1)
    disp = np.linalg.norm(za - z0, axis=1) if geom else np.zeros(n)
    values = np.stack([ent0, enta, enta - ent0, m0, ma, ma - m0, flip, js, conf, disp], axis=1)
    mask = np.ones_like(values, np.uint8)
    if not geom:
        for f in GEOM_WINDOW_FEATURES:
            mask[:, PER_WINDOW_FEATURES.index(f)] = 0
    cvals, cmask = _context(state, z0, za, pa, geom)
    if not np.all(np.isfinite(values[mask == 1])) or not np.all(np.isfinite(cvals[cmask == 1])):
        raise ValueError("non-finite available feature value")
    values = np.where(mask == 1, values, 0.0); cvals = np.where(cmask == 1, cvals, 0.0)
    return WindowActionSet(values, mask, cvals, cmask, action, action_index(action), tuple(keys))


def _extract_canonical(state, z, keys, action, p0, z0) -> WindowActionSet:
    """Build the set for ONE action; assumes z/keys already canonical and identity (p0,z0) already computed. Re-executes
    the adapter (used by the feature-only API); the loader uses _build_was on captured outputs instead."""
    n = len(z)
    if not (MIN_BATCH <= n <= B):
        raise ValueError(f"eligible batch n_windows must be in [{MIN_BATCH}, {B}]; got {n}")
    pa, za = apply_action(action, state, z)
    return _build_was(state, z, keys, action, p0, z0, pa, za)


def extract_action_set(state, z, window_keys, action) -> WindowActionSet:
    """Single-action public API (canonicalizes + computes identity once for this call)."""
    if action not in NON_IDENTITY:
        raise ValueError(f"extract_action_set is for non-identity actions; got {action!r}")
    z, keys = _canonicalize(z, window_keys)
    if not (MIN_BATCH <= len(z) <= B):                          # reject BEFORE any adapter (incl. identity)
        raise ValueError(f"extract_action_set requires {MIN_BATCH} <= n_windows <= {B}; got {len(z)}")
    p0, z0 = apply_action("identity", state, z)
    _validate_proba(p0, len(z), state["n_cls"], "p0")
    z0 = np.asarray(z0, float)
    if z0.shape != z.shape or not np.all(np.isfinite(z0)):
        raise ValueError("identity embedding malformed (shape/finiteness)")
    return _extract_canonical(state, z, keys, action, p0, z0)


def _validate_actions(actions):
    requested = list(actions)
    if not requested:
        raise ValueError("empty action selection")
    if any(a not in ACTION_VOCAB for a in requested):
        raise ValueError("unknown action requested")
    if "identity" in requested:
        raise ValueError("identity is the reference, not an extractable action")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate action requested")
    return tuple(a for a in NON_IDENTITY if a in set(requested))   # canonical execution order


def build_action_sets(state, z, window_keys, actions=NON_IDENTITY):
    """{action: WindowActionSet} in canonical order, computing identity ONCE; OR a FallbackBatchRecord when
    len(z) < MIN_BATCH (short-circuit BEFORE any adapter — actions/keys still validated)."""
    z = np.asarray(z, float)
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    ordered = _validate_actions(actions)
    keys = list(window_keys)
    if len(keys) != len(z):
        raise ValueError("window_keys length != n_windows")
    if len(set(canon_key(k) for k in keys)) != len(keys):
        raise ValueError("duplicate window_keys")
    if len(z) < MIN_BATCH:
        return FallbackBatchRecord(True, f"n_windows<{MIN_BATCH}", tuple(keys), _input_digest(z, keys), int(len(z)))
    if len(z) > B:
        raise ValueError(f"n_windows {len(z)} > B={B}")
    zc, kc = _canonicalize(z, keys)
    p0, z0 = apply_action("identity", state, zc)              # identity computed exactly ONCE
    _validate_proba(p0, len(zc), state["n_cls"], "p0")
    z0 = np.asarray(z0, float)
    if z0.shape != zc.shape or not np.all(np.isfinite(z0)):
        raise ValueError("identity embedding malformed (shape/finiteness)")
    return {a: _extract_canonical(state, zc, kc, a, p0, z0) for a in ordered}


def canonical_digest(was: WindowActionSet) -> str:
    """Full 64-char SHA-256 over schema header + raw float64-LE values/context + uint8 masks + canonical keys."""
    h = hashlib.sha256()
    header = json.dumps({"schema": SCHEMA_VERSION, "action_name": was.action_name,
                         "action_index": int(was.action_index), "vocab": list(ACTION_VOCAB),
                         "per_window": list(PER_WINDOW_FEATURES), "context": list(CONTEXT_FEATURES),
                         "shape": list(map(int, was.values.shape)), "n_context": int(len(was.context_values))},
                        sort_keys=True).encode()
    h.update(b"HDR\x00"); h.update(header)
    order = sorted(range(len(was.window_keys)), key=lambda i: canon_key(was.window_keys[i]))
    for i in order:
        h.update(b"K\x00"); h.update(canon_key(was.window_keys[i]).encode())
        h.update(b"V\x00"); h.update(np.ascontiguousarray(was.values[i], dtype="<f8").tobytes())
        h.update(b"M\x00"); h.update(np.ascontiguousarray(was.availability_mask[i], dtype=np.uint8).tobytes())
    h.update(b"C\x00"); h.update(np.ascontiguousarray(was.context_values, dtype="<f8").tobytes())
    h.update(b"CM\x00"); h.update(np.ascontiguousarray(was.context_mask, dtype=np.uint8).tobytes())
    return h.hexdigest()
