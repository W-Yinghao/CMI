"""C33 local information-ladder diagnostics inside frozen selected neighborhoods."""
from __future__ import annotations

import numpy as np

from ..identifiability.multivariate_probe import _auc
from . import schema
from .artifact_loader import units


STRATEGIES = (
    ("source_score", "source_only"),
    ("R_src", "source_only_R_src"),
    ("robust_core_score", "C19_robust_core_scalar"),
    ("c30_source_rank", "C30_source_rank"),
    ("target_unlabeled_r3_score", "target_unlabeled_R3_non_source_only"),
    ("target_grouped_centered_score", "target_grouped_non_deployable"),
    ("target_label_oracle_score", "target_label_oracle_non_deployable"),
)


def _selected_index(cs):
    selected = [i for i, c in enumerate(cs) if c.get("selected_oaci")]
    return selected[0] if len(selected) == 1 else None


def _nearest_joint_index(cs, si):
    joints = [i for i, c in enumerate(cs) if c["joint_good"]]
    if not joints:
        return None
    return min(joints, key=lambda i: abs(i - si))


def _joint_cluster(cs, ji, si):
    if ji is None:
        return [cs[si]]
    lo = hi = ji
    while lo > 0 and cs[lo - 1]["joint_good"]:
        lo -= 1
    while hi + 1 < len(cs) and cs[hi + 1]["joint_good"]:
        hi += 1
    idx = set(range(lo, hi + 1))
    idx.add(si)
    return [cs[i] for i in sorted(idx)]


def _neighborhoods(cs, si):
    out = {}
    for k in schema.ORDER_NEIGHBORHOODS:
        lo = max(0, si - k)
        hi = min(len(cs), si + k + 1)
        out[f"pm{k}"] = cs[lo:hi]
    ji = _nearest_joint_index(cs, si)
    out["same_joint_good_cluster"] = _joint_cluster(cs, ji, si)
    se = float(cs[si]["epoch"])
    out[f"epoch_pm{schema.EPOCH_WINDOW}"] = [c for c in cs if abs(float(c["epoch"]) - se) <= schema.EPOCH_WINDOW]
    return out


def _auc_or_none(cands, score_key):
    y = np.array([c["joint_good"] for c in cands], dtype=int)
    s = np.array([c.get(score_key, np.nan) for c in cands], dtype=float)
    ok = np.isfinite(s)
    y = y[ok]
    s = s[ok]
    if len(y) < 2 or y.sum() == 0 or y.sum() == len(y) or s.std() <= 1e-9:
        return None
    return _auc(y, s)


def _top1_hit(cands, score_key):
    cands = [c for c in cands if np.isfinite(float(c.get(score_key, np.nan)))]
    if not cands:
        return None
    return int(max(cands, key=lambda c: c[score_key])["joint_good"])


def local_random_and_ladder(rows):
    random_rows, strategy_rows = [], []
    for key, cs in units(rows).items():
        si = _selected_index(cs)
        if si is None:
            continue
        for name, cands in _neighborhoods(cs, si).items():
            if not cands:
                continue
            base = float(np.mean([c["joint_good"] for c in cands]))
            random_rows.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                                "neighborhood": name, "n": len(cands), "random_top1_hit_rate": base,
                                "contains_joint_good": int(base > 0)})
            for score_key, info_class in STRATEGIES:
                hit = _top1_hit(cands, score_key)
                auc = _auc_or_none(cands, score_key)
                strategy_rows.append({"seed": key[0], "target": key[1], "level": key[2], "regime": key[3],
                                      "neighborhood": name, "strategy": score_key, "info_class": info_class,
                                      "n": len(cands), "local_pairwise_auc": auc, "local_top1_hit": hit,
                                      "local_random_top1": base,
                                      "top1_enrichment": ((hit / base) if hit is not None and base > 0 else None)})
    agg = []
    for score_key, info_class in STRATEGIES:
        for nname in sorted({r["neighborhood"] for r in random_rows}):
            rows_ = [r for r in strategy_rows if r["strategy"] == score_key and r["neighborhood"] == nname]
            aucs = [r["local_pairwise_auc"] for r in rows_ if r["local_pairwise_auc"] is not None]
            hits = [r["local_top1_hit"] for r in rows_ if r["local_top1_hit"] is not None]
            bases = [r["local_random_top1"] for r in rows_ if r["local_random_top1"] is not None]
            agg.append({"strategy": score_key, "info_class": info_class, "neighborhood": nname,
                        "mean_local_pairwise_auc": float(np.mean(aucs)) if aucs else None,
                        "top1_hit_rate": float(np.mean(hits)) if hits else None,
                        "local_random_top1_hit_rate": float(np.mean(bases)) if bases else None,
                        "top1_enrichment": ((float(np.mean(hits)) / float(np.mean(bases)))
                                            if hits and bases and float(np.mean(bases)) > 0 else None)})
    summary = {
        "target_unlabeled_pm1_top1_gain_vs_source": _gain(agg, "target_unlabeled_r3_score", "source_score", "pm1"),
        "target_unlabeled_pm2_top1_gain_vs_source": _gain(agg, "target_unlabeled_r3_score", "source_score", "pm2"),
        "target_grouped_pm1_top1_gain_vs_source": _gain(agg, "target_grouped_centered_score", "source_score", "pm1"),
        "source_pm1_enrichment": _enrichment(agg, "source_score", "pm1"),
    }
    return {"summary": summary, "random_rows": random_rows, "strategy_rows": strategy_rows, "aggregate": agg}


def _row(agg, strategy, neighborhood):
    for r in agg:
        if r["strategy"] == strategy and r["neighborhood"] == neighborhood:
            return r
    return None


def _gain(agg, a, b, neighborhood):
    ra, rb = _row(agg, a, neighborhood), _row(agg, b, neighborhood)
    if not ra or not rb or ra["top1_hit_rate"] is None or rb["top1_hit_rate"] is None:
        return None
    return float(ra["top1_hit_rate"] - rb["top1_hit_rate"])


def _enrichment(agg, strategy, neighborhood):
    r = _row(agg, strategy, neighborhood)
    return None if not r else r["top1_enrichment"]
