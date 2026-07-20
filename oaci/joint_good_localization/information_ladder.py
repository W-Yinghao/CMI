"""C32 source-only / target-unlabeled / target-grouped localization ladder."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from ..information_ladder import target_unlabeled_features as tuf
from . import schema, topk
from .regret import strategy_top1_regret


def _finite_rows(rows, names):
    return [r for r in rows if all(r.get(n) is not None and np.isfinite(float(r.get(n))) for n in names)]


def ridge_loto_predict(rows, names, label_key="joint_good", l2=None) -> dict:
    """Fixed-ridge leave-one-target-out diagnostic scorer.

    The held-out target's labels are never used to fit its scores. Feature names are fixed; there is no feature
    selection or tuning. The resulting scores are diagnostic aggregates, not a selector artifact.
    """
    l2 = schema.RIDGE_L2 if l2 is None else l2
    pred = {}
    targets = sorted({r["target"] for r in rows})
    for t in targets:
        train = _finite_rows([r for r in rows if r["target"] != t], names)
        test = _finite_rows([r for r in rows if r["target"] == t], names)
        if not train or not test:
            continue
        X = np.array([[r[n] for n in names] for r in train], dtype=float)
        y = np.array([r[label_key] for r in train], dtype=float)
        Xt = np.array([[r[n] for n in names] for r in test], dtype=float)
        mu = X.mean(0)
        sd = X.std(0) + 1e-9
        Xs = (X - mu) / sd
        Xts = (Xt - mu) / sd
        ym = y.mean()
        w = np.linalg.solve(Xs.T @ Xs + l2 * np.eye(len(names)), Xs.T @ (y - ym))
        p = Xts @ w + ym
        for r, pp in zip(test, p):
            pred[id(r)] = float(pp)
    return pred


def pooled_auc(rows, score_getter, label_key="joint_good"):
    y = np.array([r[label_key] for r in rows], dtype=int)
    s = np.array([score_getter(r) for r in rows], dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    return _auc(y, s) if (len(y) > 2 and 0 < y.sum() < len(y)) else None


def within_target_auc(rows, score_getter, label_key="joint_good"):
    per = []
    for t in sorted({r["target"] for r in rows}):
        g = [r for r in rows if r["target"] == t]
        y = np.array([r[label_key] for r in g], dtype=int)
        s = np.array([score_getter(r) for r in g], dtype=float)
        ok = np.isfinite(s)
        y = y[ok]
        s = s[ok]
        if len(y) > 2 and 0 < y.sum() < len(y) and s.std() > 1e-9:
            per.append(_auc(y, s))
    return float(np.mean(per)) if per else None


def _target_centered_score(rows):
    means = {t: float(np.mean([r["score"] for r in rows if r["target"] == t]))
             for t in sorted({r["target"] for r in rows})}
    return {id(r): float(r["score"] - means[r["target"]]) for r in rows}


def build_score_maps(rows):
    names = tuf.target_unlabeled_feature_names()
    tuf.assert_no_target_labels(names)
    target_unlabeled = ridge_loto_predict(rows, names)
    source_plus_tu = ridge_loto_predict(rows, ["score"] + names)
    target_centered = _target_centered_score(rows)
    return {
        "source_score": lambda r: float(r["score"]),
        "target_unlabeled_loto": lambda r: target_unlabeled.get(id(r), np.nan),
        "source_plus_target_unlabeled_loto": lambda r: source_plus_tu.get(id(r), np.nan),
        "target_grouped_centered_score": lambda r: target_centered.get(id(r), np.nan),
    }, {"target_unlabeled_feature_names": names,
        "target_unlabeled_missing_predictions": len(rows) - len(target_unlabeled),
        "source_plus_target_unlabeled_missing_predictions": len(rows) - len(source_plus_tu)}


def localization_ladder(rows, top_ks=None) -> dict:
    top_ks = schema.TOP_KS if top_ks is None else top_ks
    score_maps, meta = build_score_maps(rows)
    models = []
    topk_rows = []
    for name, getter in score_maps.items():
        tk = topk.topk_enrichment(rows, getter, top_ks)
        for r in tk["topk"]:
            topk_rows.append({"strategy": name, **r})
        models.append({
            "strategy": name,
            "pooled_auc": pooled_auc(rows, getter),
            "within_target_auc": within_target_auc(rows, getter),
            "top1_hit_rate": next(r["hit_rate"] for r in tk["topk"] if r["k"] == 1),
            "top5_hit_rate": next(r["hit_rate"] for r in tk["topk"] if r["k"] == 5),
        })
    regrets = strategy_top1_regret(rows, score_maps)
    raw = next(m for m in models if m["strategy"] == "source_score")
    tu = next(m for m in models if m["strategy"] == "target_unlabeled_loto")
    grouped = next(m for m in models if m["strategy"] == "target_grouped_centered_score")
    meta.update({
        "target_unlabeled_pooled_auc_gain_over_source": (
            tu["pooled_auc"] - raw["pooled_auc"] if tu["pooled_auc"] is not None and raw["pooled_auc"] is not None else None),
        "target_unlabeled_top1_gain_over_source": (
            tu["top1_hit_rate"] - raw["top1_hit_rate"] if tu["top1_hit_rate"] is not None and raw["top1_hit_rate"] is not None else None),
        "target_grouped_pooled_auc_gain_over_source": (
            grouped["pooled_auc"] - raw["pooled_auc"] if grouped["pooled_auc"] is not None and raw["pooled_auc"] is not None else None),
    })
    return {"models": models, "topk": topk_rows, "strategy_top1_regret": regrets, "meta": meta}
