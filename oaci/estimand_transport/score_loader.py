"""C22 — read-only score loader. For each regime, builds the FROZEN C19 robust-core feature atlas (reusing
C19 feature_registry) and extracts per-candidate held-out LOTO scores from the FROZEN probe (reusing
frozen_probe.fit_predict). Joins epoch / candidate trajectory-order / training-log proxies (from the C18
cand_meta) and the diagnostic target label. NOTHING here refits or tunes the probe; scores are extracted, not
optimized. Two score modes: in-regime (LOTO within a regime) and cross-regime (fit on dev S0/S2/S3, predict a
held-out regime)."""
from __future__ import annotations

import numpy as np

from ..competence_probe import feature_registry
from ..probe_validation import frozen_probe
from ..support_stress import source_signal_recompute as ssr
from . import schema

_ROBUST = list(schema.ROBUST_CORE)
_LABEL = schema.DIAGNOSTIC_LABEL


def epoch_lookup(extract_dir, folds) -> dict:
    """(seed,target,level,model_hash) -> {epoch, order, R_src, train_surrogate, balanced_err}. order = rank of
    the checkpoint by epoch within its fold-level (trajectory position; 0 = earliest feasible OACI candidate)."""
    out = {}
    for (s, t) in folds:
        for level in ssr._levels(extract_dir, s, t):
            fld = ssr.load_fold_level(extract_dir, s, t, level)
            feas = [c for c in fld["cand_meta"] if not c["is_erm"] and c["feasible"]]
            for order, c in enumerate(sorted(feas, key=lambda c: (c["epoch"], c["model_hash"]))):
                out[(s, t, level, c["model_hash"])] = {"epoch": int(c["epoch"]), "order": order,
                                                       "R_src": float(c["R_src"]),
                                                       "train_surrogate": float(c["train_surrogate"]),
                                                       "balanced_err": float(c["balanced_err"])}
    return out


def atlas_by_regime(extract_dir, c10_dir, *, boundary_classes, leakage_cache, folds) -> dict:
    out = {}
    for regime in schema.ALL_REGIMES:
        lk = (lambda s, t, l, mh, rg=regime: leakage_cache.get((s, t, l, rg, mh), (None, None)))
        out[regime] = feature_registry.build_atlas(extract_dir, c10_dir, regime, boundary_classes=boundary_classes,
                                                    leakage_lookup=lk, folds=folds)
    return out


def _loto_scores_within(rows):
    """LOTO within a single regime: per candidate, fit frozen probe on OTHER targets, predict this candidate."""
    targets = sorted({r["target"] for r in rows}); scored = {}
    for t in targets:
        train = [r for r in rows if r["target"] != t]
        test = [r for r in rows if r["target"] == t]
        s, y, _, _ = frozen_probe.fit_predict(train, test, _ROBUST)
        if s is None:
            continue
        finite = [r for r in test if all(_finite(r.get(c)) for c in _ROBUST)]
        for r, sc in zip(finite, s):
            scored[(r["target"], r["model_hash"])] = float(sc)
    return scored


def _loto_scores_cross(dev_pool, val_rows):
    """Cross-regime: per held-out target, fit frozen probe on dev-pool OTHER targets, predict val_rows target."""
    targets = sorted({r["target"] for r in dev_pool}); scored = {}
    for t in targets:
        train = [r for r in dev_pool if r["target"] != t]
        test = [r for r in val_rows if r["target"] == t]
        s, y, _, _ = frozen_probe.fit_predict(train, test, _ROBUST)
        if s is None:
            continue
        finite = [r for r in test if all(_finite(r.get(c)) for c in _ROBUST)]
        for r, sc in zip(finite, s):
            scored[(r["target"], r["model_hash"])] = float(sc)
    return scored


def _finite(v):
    import math
    return v is not None and not (isinstance(v, float) and not math.isfinite(v))


def score_table(extract_dir, c10_dir, *, boundary_classes, leakage_cache, folds) -> list:
    """Flat per-candidate rows across regimes + score modes, with epoch/order/features/label."""
    atlas = atlas_by_regime(extract_dir, c10_dir, boundary_classes=boundary_classes, leakage_cache=leakage_cache,
                            folds=folds)
    epochs = epoch_lookup(extract_dir, folds)
    dev_pool = [r for reg in schema.IN_REGIME for r in atlas[reg]]
    rows = []

    def _emit(mode, regime, r, score):
        key = (r["seed"], r["target"], r["level"], r["model_hash"])
        e = epochs.get(key, {})
        rows.append({"mode": mode, "regime": regime, "seed": r["seed"], "target": r["target"],
                     "level": r["level"], "model_hash": r["model_hash"], "score": score,
                     "label": 1 if r[_LABEL] else 0, "epoch": e.get("epoch"), "order": e.get("order"),
                     "R_src": e.get("R_src"), "train_surrogate": e.get("train_surrogate"),
                     **{f"feat__{c}": r.get(c) for c in _ROBUST}})

    # in-regime scores (S0/S2/S3): within-regime LOTO -- where the C19 signal lives
    for regime in schema.IN_REGIME:
        sc = _loto_scores_within(atlas[regime])
        for r in atlas[regime]:
            k = (r["target"], r["model_hash"])
            if k in sc:
                _emit("in_regime", regime, r, sc[k])
    # cross-regime scores (S4-S7): dev S0/S2/S3 -> held-out
    for regime in schema.CROSS_REGIME:
        sc = _loto_scores_cross(dev_pool, atlas[regime])
        for r in atlas[regime]:
            k = (r["target"], r["model_hash"])
            if k in sc:
                _emit("cross_regime", regime, r, sc[k])
    return rows
