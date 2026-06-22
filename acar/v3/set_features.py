"""Per-window paired-set extraction for ACAR v3 (HSCR). DESIGN/DEV stage — SYNTHETIC inputs only until DEV_DESIGN_LOCK.
Hardened per the 93f417c review (notes/ACAR_V3_AMENDMENT_2.md).

Data contract:
    WindowActionSet (validated + immutable)
      values:             [n, F]  per-window paired features (canonical row order)
      availability_mask:  [n, F]  uint8 {0,1}; 1 = genuine value, 0 = structurally unavailable (slot forced exact 0)
      context_values:     [C]     batch-level features
      context_mask:       [C]     uint8 {0,1}
      action_name:        canonical string in NON_IDENTITY
      action_index:       == ACTION_VOCAB.index(action_name)
      window_keys:        unique stable identifiers (str or WindowKey), in canonical row order

Hardening invariants:
- **Canonical row order BEFORE adapters:** `(z, window_keys)` are stably sorted by `canon_key` *before* any adapter
  runs, so EA/CORAL/SPDIM/covariance/reductions execute in a fixed order → permutation invariance is a property of
  the execution path (byte-identical values), not of hash tolerance.
- **Exact, full 64-char digest:** SHA-256 over the schema header + raw float64-LE value/context bytes + uint8 masks +
  canonical keys. No rounding (single-ULP sensitive).
- **Validated immutable contract:** `WindowActionSet.__post_init__` enforces shapes, binary masks, masked-slots-are-
  exactly-0, finiteness, action_name/index consistency, unique keys; arrays are set read-only.
- **Canonical action execution order:** actions are validated and iterated as `ACTION_VOCAB` order, never caller order.
- **Action capability map:** matched_coral/spdim MUST yield geometry (z_tilde); t3a MUST NOT — asserted (drift guard).
- **Probability/shape validation** of `p0,pa,z0,za`. NaN/Inf rejected. `<MIN_BATCH` short-circuits to identity BEFORE
  any adapter. We do NOT call acar.features.feature_vector() (it collapses NaN->0).

No target labels, no DEV labels, no ΔR, no candidate/width/AUROC/router metric are computed here.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from collections import namedtuple
import hashlib
import json
import numpy as np

from acar.config import MIN_BATCH
from acar.actions import apply_action
from acar.features import (_entropy, _margin, _jsd, _bures_w2, _fisher_ratio, _ess, _maha2, _shrink)

SCHEMA_VERSION = "acar-v3-set/1"

ACTION_VOCAB = ("identity", "matched_coral", "spdim", "t3a")
NON_IDENTITY = tuple(a for a in ACTION_VOCAB if a != "identity")
# explicit capability map (geometry = returns a post-embedding z_tilde); drift here changes feature semantics.
ACTION_GEOMETRY = {"matched_coral": True, "spdim": True, "t3a": False}

PER_WINDOW_FEATURES = ("ent0", "enta", "d_ent", "margin0", "margina", "d_margin",
                       "flip", "js", "conf_change", "embed_disp")
GEOM_WINDOW_FEATURES = ("embed_disp",)
CONTEXT_FEATURES = ("bures", "post_sep", "n_eff", "g_unc", "s_support", "s_sep", "pr_cmi_proxy")
GEOM_CONTEXT_FEATURES = ("bures", "post_sep", "s_support", "s_sep", "pr_cmi_proxy")

WindowKey = namedtuple("WindowKey", "dataset_id subject_id recording_id window_index")


def canon_key(k) -> str:
    """Canonical serialization of a window identifier (for sorting + digest). Structured keys preferred for the real
    loader: (dataset_id, subject_id, recording_id, window_index)."""
    if isinstance(k, WindowKey):
        return f"{k.dataset_id}|{k.subject_id}|{k.recording_id}|{int(k.window_index):09d}"
    return str(k)


def action_index(name: str) -> int:
    return ACTION_VOCAB.index(name)


def canonical_tie_break(candidates):
    """Deterministic action choice among tied candidates: lowest ACTION_VOCAB index (NOT caller order)."""
    return min(candidates, key=action_index)


def _as_bin(m):
    m = np.asarray(m)
    if not np.all(np.isin(m, (0, 1))):
        raise ValueError("availability mask must be binary {0,1}")
    return m.astype(np.uint8)


@dataclass
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
        if any(s == "" for s in ck):
            raise ValueError("empty window key")
        if len(set(ck)) != len(ck):
            raise ValueError("duplicate window_keys")
        v.flags.writeable = False; m.flags.writeable = False
        cv.flags.writeable = False; cm.flags.writeable = False
        self.values, self.availability_mask = v, m
        self.context_values, self.context_mask = cv, cm
        self.window_keys = keys


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


def extract_action_set(state, z, window_keys, action) -> WindowActionSet:
    """Build the WindowActionSet for ONE non-identity action. Label-free. Canonicalizes row order before adapters."""
    if action not in NON_IDENTITY:
        raise ValueError(f"extract_action_set is for non-identity actions; got {action!r}")
    z = np.asarray(z, float); keys = list(window_keys)
    if len(keys) != len(z):
        raise ValueError("window_keys length != n_windows")
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    ck = [canon_key(k) for k in keys]
    if len(set(ck)) != len(ck):
        raise ValueError("duplicate window_keys are not allowed (window identity must be unique)")
    order = sorted(range(len(keys)), key=lambda i: ck[i])      # CANONICAL ROW ORDER before any adapter
    z = z[order]; keys = [keys[i] for i in order]
    n = len(z); n_cls = state["n_cls"]

    p0, z0 = apply_action("identity", state, z)
    pa, za = apply_action(action, state, z)
    if (za is not None) != ACTION_GEOMETRY[action]:
        raise ValueError(f"action capability drift: {action} geometry={za is not None}, expected {ACTION_GEOMETRY[action]}")
    _validate_proba(p0, n, n_cls, "p0"); _validate_proba(pa, n, n_cls, "pa")
    z0 = np.asarray(z0, float)
    if z0.shape != z.shape or (za is not None and np.asarray(za).shape != z.shape):
        raise ValueError("embedding shape mismatch")
    geom = za is not None

    ent0, enta = _entropy(p0), _entropy(pa)
    m0, ma = _margin(p0), _margin(pa)
    flip = (pa.argmax(1) != p0.argmax(1)).astype(float)
    js = _jsd(p0, pa)
    conf = pa.max(1) - p0.max(1)
    disp = np.linalg.norm(za - z0, axis=1) if geom else np.zeros(n)
    values = np.stack([ent0, enta, enta - ent0, m0, ma, ma - m0, flip, js, conf, disp], axis=1)
    mask = np.ones_like(values, np.uint8)
    if not geom:
        for f in GEOM_WINDOW_FEATURES:
            mask[:, PER_WINDOW_FEATURES.index(f)] = 0
    cvals, cmask = _context(state, z0, za, pa, geom)
    if not np.all(np.isfinite(values[mask == 1])) or not np.all(np.isfinite(cvals[cmask == 1])):
        raise ValueError("non-finite available feature value")
    values = np.where(mask == 1, values, 0.0)
    cvals = np.where(cmask == 1, cvals, 0.0)
    return WindowActionSet(values, mask, cvals, cmask, action, action_index(action), tuple(keys))


def build_action_sets(state, z, window_keys, actions=NON_IDENTITY):
    """{action: WindowActionSet} in CANONICAL action order, OR {'__fallback__': 'identity'} when len(z) < MIN_BATCH
    (short-circuit BEFORE any adapter runs). `actions` selects a subset; iteration order is always ACTION_VOCAB."""
    z = np.asarray(z, float)
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    if len(z) < MIN_BATCH:
        return {"__fallback__": "identity"}
    requested = list(actions)
    if not requested:
        raise ValueError("empty action selection")
    if any(a not in ACTION_VOCAB for a in requested):
        raise ValueError("unknown action requested")
    if "identity" in requested:
        raise ValueError("identity is the reference, not an extractable action")
    if len(set(requested)) != len(requested):
        raise ValueError("duplicate action requested")
    ordered = tuple(a for a in NON_IDENTITY if a in set(requested))   # canonical execution order
    return {a: extract_action_set(state, z, window_keys, a) for a in ordered}


def canonical_digest(was: WindowActionSet) -> str:
    """Full 64-char SHA-256 over schema header + raw float64-LE values/context + uint8 masks + canonical keys.
    Row-order-INSENSITIVE (rows sorted by canon_key), content-SENSITIVE to single-ULP value changes, mask flips,
    key/action/schema changes. No rounding."""
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
