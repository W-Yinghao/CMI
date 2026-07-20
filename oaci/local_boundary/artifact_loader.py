"""C33 read-only loader and local diagnostic score construction."""
from __future__ import annotations

import numpy as np

from ..joint_good_localization import artifact_loader as c32_loader
from ..joint_good_localization import information_ladder as c32_ladder
from . import schema


def _finite(v) -> bool:
    try:
        return v is not None and np.isfinite(float(v))
    except (TypeError, ValueError):
        return False


def unit_key(r):
    return (r["seed"], r["target"], r["level"], r.get("regime", ""))


def units(rows) -> dict:
    out = {}
    for r in rows:
        out.setdefault(unit_key(r), []).append(r)
    for k in out:
        out[k] = sorted(out[k], key=lambda r: (r.get("order", r.get("epoch", 0)), r.get("epoch", 0)))
    return out


def _joint_margin(r, margin):
    acc = r.get("bacc_delta")
    calib = max(r.get("nll_improve"), r.get("ece_improve")) if _finite(r.get("nll_improve")) and _finite(r.get("ece_improve")) else None
    if not _finite(acc) or not _finite(calib):
        return None
    return float(min(acc - margin, calib - margin))


def _robust_core_score(rows):
    feats = [k for k in rows[0] if k.startswith("feat__")] if rows else []
    vals = {}
    for f in feats:
        col = np.array([r.get(f) for r in rows if _finite(r.get(f))], dtype=float)
        if len(col) and col.std() > 1e-9:
            vals[f] = (float(col.mean()), float(col.std()))
    out = {}
    for r in rows:
        zs = [(float(r[f]) - mu) / sd for f, (mu, sd) in vals.items() if _finite(r.get(f))]
        out[id(r)] = float(np.mean(zs)) if zs else 0.0
    return out


def _within_unit_rank(rows, key="score"):
    rank = {}
    for cs in units(rows).values():
        vals = np.array([float(c[key]) for c in cs], dtype=float)
        order = np.argsort(np.argsort(vals)).astype(float)
        denom = max(len(cs) - 1, 1)
        for r, o in zip(cs, order):
            rank[id(r)] = float(o / denom)
    return rank


def _target_grouped_score(rows):
    means = {t: float(np.mean([r["score"] for r in rows if r["target"] == t])) for t in sorted({r["target"] for r in rows})}
    return {id(r): float(r["score"] - means[r["target"]]) for r in rows}


def attach_local_scores(rows, margin):
    tu_scores = c32_ladder.ridge_loto_predict(rows, c32_ladder.tuf.target_unlabeled_feature_names())
    c30_rank = _within_unit_rank(rows, "score")
    robust_core = _robust_core_score(rows)
    grouped = _target_grouped_score(rows)
    for r in rows:
        r["source_score"] = float(r["score"])
        r["joint_margin"] = _joint_margin(r, margin)
        r["c30_source_rank"] = c30_rank.get(id(r), 0.0)
        r["robust_core_score"] = robust_core.get(id(r), 0.0)
        r["target_unlabeled_r3_score"] = tu_scores.get(id(r), np.nan)
        r["target_grouped_centered_score"] = grouped.get(id(r), np.nan)
        r["target_label_oracle_score"] = r["joint_margin"] if r["joint_margin"] is not None else np.nan
    return rows


def load_rows(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime", margin=None):
    margin = schema.PRIMARY_MARGIN if margin is None else margin
    rows, tu = c32_loader.load_rows(scores_sidecar, c10_dir, reinfer_sidecar, mode=mode, margin=margin)
    rows = [r for r in rows if all(_finite(r.get(k)) for k in ("score", "R_src", "bacc", "nll", "ece", "epoch"))]
    attach_local_scores(rows, margin)
    return rows, tu
