"""Per-window paired-set extraction for ACAR v3 (HSCR). DESIGN/DEV stage — SYNTHETIC inputs only until DEV_DESIGN_LOCK.

Data contract (notes/ACAR_V3_FREEZE_SKELETON.md §S8/§S12; review):

    WindowActionSet
      values:             [n_windows, n_per_window_features]   per-window paired features
      availability_mask:  [n_windows, n_per_window_features]    1 = genuine value, 0 = structurally unavailable
      context_values:     [n_context]                           batch-level features (Bures, post-sep, ESS, A0 ctx)
      context_mask:       [n_context]
      action_name:        canonical string (ACTION_VOCAB)
      action_index:       canonical vocabulary index (deterministic tie order)
      window_keys:        stable per-window identifiers (unique within a set)

Hard rules carried here:
- A structurally-unavailable slot stores numeric 0 but with mask=0; a genuine 0 has mask=1. Missing-zero and
  genuine-zero therefore have DIFFERENT (values, mask) and DIFFERENT canonical digests. We do NOT call
  acar.features.feature_vector() (it collapses NaN->0, erasing that distinction).
- T3A adjusts the classifier, not the embedding (no z_tilde) -> all geometry-derived features are masked unavailable,
  never faked as a real zero.
- Action iteration uses the canonical vocabulary; deployment-time ties are broken by canonical_tie_break (fixed
  action priority), never by incoming dict/list order.
- window_keys must be unique within a set; they enter the canonical digest. Duplicate keys are rejected.
- Inputs with NaN/Inf are rejected. <MIN_BATCH batches short-circuit to identity BEFORE any adapter/extractor runs.

This module computes NO target labels, NO ΔR, and NO candidate/width/AUROC/router metric.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import json
import numpy as np

from acar.config import MIN_BATCH
from acar.actions import apply_action
from acar.features import (_entropy, _margin, _jsd, _bures_w2, _fisher_ratio, _ess, _maha2, _shrink)

# ---- canonical action vocabulary + deterministic tie order (identity first = highest priority) ----
ACTION_VOCAB = ("identity", "matched_coral", "spdim", "t3a")
NON_IDENTITY = tuple(a for a in ACTION_VOCAB if a != "identity")

PER_WINDOW_FEATURES = ("ent0", "enta", "d_ent", "margin0", "margina", "d_margin",
                       "flip", "js", "conf_change", "embed_disp")
GEOM_WINDOW_FEATURES = ("embed_disp",)                     # require z_tilde (unavailable for t3a)
CONTEXT_FEATURES = ("bures", "post_sep", "n_eff", "g_unc", "s_support", "s_sep", "pr_cmi_proxy")
GEOM_CONTEXT_FEATURES = ("bures", "post_sep", "s_support", "s_sep", "pr_cmi_proxy")   # require z_tilde


def action_index(name: str) -> int:
    return ACTION_VOCAB.index(name)


def canonical_tie_break(candidates):
    """Deterministic action choice among tied candidates: lowest ACTION_VOCAB index (NOT input order)."""
    return min(candidates, key=action_index)


@dataclass
class WindowActionSet:
    values: np.ndarray             # [n, F]
    availability_mask: np.ndarray  # [n, F] in {0,1}
    context_values: np.ndarray     # [C]
    context_mask: np.ndarray       # [C] in {0,1}
    action_name: str
    action_index: int
    window_keys: tuple


def _context(state, z0, za, pa, geom):
    n_eff = _ess(pa)
    g_unc = float(_entropy(pa).mean())
    if geom:
        bures = _bures_w2(za, z0)
        post_sep = _fisher_ratio(za, pa.argmax(1))
        Winv = np.linalg.inv(_shrink(np.asarray(state["Sig_pool0"], float)))
        s_support = float(_maha2(za, state["mu_pool"], Winv).mean())
        m = np.stack([_maha2(za, state["mu_y"][c], Winv) for c in range(state["n_cls"])], 1)
        s_sep = float((-np.abs(m[:, 0] - m[:, 1])).mean())
        readout = state["clf"].predict(za); proto = m.argmin(1)
        mg = np.sort(m, 1)[:, 1] - np.sort(m, 1)[:, 0]
        pr = float(((proto != readout).astype(float) * mg + 0.01 * mg).mean())
        vals = np.array([bures, post_sep, n_eff, g_unc, s_support, s_sep, pr], float)
        mask = np.ones(len(CONTEXT_FEATURES), float)
    else:
        vals = np.array([0.0, 0.0, n_eff, g_unc, 0.0, 0.0, 0.0], float)
        mask = np.array([0, 0, 1, 1, 0, 0, 0], float)
    return vals, mask


def extract_action_set(state, z, window_keys, action, p0=None, z0=None) -> WindowActionSet:
    """Build the WindowActionSet for ONE non-identity action. Label-free (no y argument)."""
    z = np.asarray(z, float)
    keys = list(window_keys)
    if len(keys) != len(z):
        raise ValueError("window_keys length != n_windows")
    if len(set(map(str, keys))) != len(keys):
        raise ValueError("duplicate window_keys are not allowed (window identity must be unique)")
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    if action == "identity" or action not in NON_IDENTITY:
        raise ValueError(f"extract_action_set is for non-identity actions; got {action!r}")

    if p0 is None or z0 is None:
        p0, z0 = apply_action("identity", state, z)
    pa, za = apply_action(action, state, z)
    geom = za is not None

    ent0, enta = _entropy(p0), _entropy(pa)
    m0, ma = _margin(p0), _margin(pa)
    flip = (pa.argmax(1) != p0.argmax(1)).astype(float)
    js = _jsd(p0, pa)
    conf = pa.max(1) - p0.max(1)
    disp = np.linalg.norm(za - z0, axis=1) if geom else np.zeros(len(z))
    values = np.stack([ent0, enta, enta - ent0, m0, ma, ma - m0, flip, js, conf, disp], axis=1)

    mask = np.ones_like(values)
    if not geom:
        for f in GEOM_WINDOW_FEATURES:
            mask[:, PER_WINDOW_FEATURES.index(f)] = 0.0
    cvals, cmask = _context(state, z0, za, pa, geom)

    # finiteness on AVAILABLE slots only (masked-unavailable slots are exact 0 by construction)
    if not np.all(np.isfinite(values[mask == 1])) or not np.all(np.isfinite(cvals[cmask == 1])):
        raise ValueError("non-finite available feature value")
    values = np.where(mask == 1, values, 0.0)              # force masked slots to exact 0
    cvals = np.where(cmask == 1, cvals, 0.0)
    return WindowActionSet(values=values, availability_mask=mask, context_values=cvals, context_mask=cmask,
                           action_name=action, action_index=action_index(action), window_keys=tuple(keys))


def build_action_sets(state, z, window_keys, actions=NON_IDENTITY):
    """{action: WindowActionSet} for all non-identity actions, OR {'__fallback__': 'identity'} when
    len(z) < MIN_BATCH (short-circuit BEFORE any adapter/extractor runs — label-blind forced identity)."""
    z = np.asarray(z, float)
    if not np.all(np.isfinite(z)):
        raise ValueError("non-finite (NaN/Inf) input features")
    if len(z) < MIN_BATCH:
        return {"__fallback__": "identity"}
    p0, z0 = apply_action("identity", state, z)            # compute identity once (label-free)
    return {a: extract_action_set(state, z, window_keys, a, p0=p0, z0=z0) for a in actions}


def canonical_digest(was: WindowActionSet) -> str:
    """Row-order-INSENSITIVE, content-SENSITIVE digest. Sensitive to any value, mask, window-key, action, or context
    change; invariant to window reordering (rows sorted by key)."""
    order = sorted(range(len(was.window_keys)), key=lambda i: str(was.window_keys[i]))
    rows = [(str(was.window_keys[i]),
             tuple(round(float(v), 12) for v in was.values[i]),
             tuple(int(m) for m in was.availability_mask[i])) for i in order]
    payload = dict(action_index=int(was.action_index), rows=rows,
                   ctx=tuple(round(float(v), 12) for v in was.context_values),
                   ctx_mask=tuple(int(m) for m in was.context_mask),
                   keys=tuple(sorted(map(str, was.window_keys))))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:32]
