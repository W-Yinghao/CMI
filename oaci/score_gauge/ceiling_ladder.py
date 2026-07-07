"""C23 — calibration ceiling ladder: pooled AUC under raw / regime-centered / SOURCE-GAUGE (LOTO offset_hat) /
target-centered oracle / target-rank oracle, plus the within-target ceiling. The oracle rungs use target
identity and are ceilings, NOT deployable. gap_closed = (source_gauge - raw) / (target_centered_oracle - raw)."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc


def _pooled_auc(rows, score_key="score", subtract=None):
    y = np.array([r["label"] for r in rows])
    s = np.array([r[score_key] for r in rows], dtype=float)
    if subtract is not None:
        s = s - np.array([subtract(r) for r in rows], dtype=float)
    return _auc(y, s) if (0 < y.sum() < len(y)) else None


def _within_target_mean(rows):
    per = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        y = np.array([r["label"] for r in g]); s = np.array([r["score"] for r in g], float)
        if 0 < y.sum() < len(y):
            per.append(_auc(y, s))
    per = [p for p in per if p is not None]
    return float(np.mean(per)) if per else None


def ceiling_ladder(rows, mode, offset_hat_loto) -> dict:
    mr = [r for r in rows if r["mode"] == mode]
    tgt_mean = {t: float(np.mean([r["score"] for r in mr if r["target"] == t])) for t in {r["target"] for r in mr}}
    reg_mean = {g: float(np.mean([r["score"] for r in mr if r["regime"] == g])) for g in {r["regime"] for r in mr}}
    # target-rank oracle: rank within target -> [0,1]
    rank = {}
    for t in {r["target"] for r in mr}:
        g = [r for r in mr if r["target"] == t]
        order = np.argsort(np.argsort([r["score"] for r in g])).astype(float)
        for r, o in zip(g, order):
            rank[id(r)] = o / max(len(g) - 1, 1)
    raw = _pooled_auc(mr)
    reg = _pooled_auc(mr, subtract=lambda r: reg_mean[r["regime"]])
    gauge = _pooled_auc(mr, subtract=lambda r: offset_hat_loto.get(r["target"], 0.0)) if offset_hat_loto else None
    tgt = _pooled_auc(mr, subtract=lambda r: tgt_mean[r["target"]])
    for r in mr:
        r["_rank"] = rank[id(r)]
    tgt_rank = _pooled_auc(mr, score_key="_rank")
    within = _within_target_mean(mr)
    gap_closed = ((gauge - raw) / (tgt - raw)) if (gauge is not None and tgt is not None and (tgt - raw) > 1e-6) else None
    return {"raw_pooled": raw, "regime_centered": reg, "source_gauge_loto": gauge,
            "target_centered_oracle": tgt, "target_rank_oracle": tgt_rank, "within_target_ceiling": within,
            "gap_closed_source_gauge": gap_closed,
            "auc_improve_source_gauge": ((gauge - raw) if (gauge is not None and raw is not None) else None)}
