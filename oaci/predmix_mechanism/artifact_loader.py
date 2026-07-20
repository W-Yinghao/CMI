"""C26 — read-only loaders + shared gauge builder. Reuses the C22 score sidecar + C24 target-unlabeled sidecar
(per-candidate aggregate geometry). build_gauge() constructs a per-target gauge from an arbitrary per-candidate
feature function (mean over the target's candidates) so Q2/Q3/Q4 can score signed / symmetric / confidence /
rotated / residualized feature sets without re-inference. Target labels are NEVER loaded here."""
from __future__ import annotations

import numpy as np

from ..information_ladder import artifact_loader as il
from ..unlabeled_gauge import artifact_loader as c25
from . import schema

load_scores = il.load
load_target_unlabeled = c25.load_target_unlabeled
per_candidate_join = c25.per_candidate_join
_finite = il._finite


def raw_oracle(rows, mode="in_regime"):
    from ..score_gauge.ceiling_ladder import _pooled_auc, _within_target_mean
    mr = [r for r in rows if r["mode"] == mode]
    tmean = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    raw = _pooled_auc(mr); oracle = _pooled_auc(mr, subtract=lambda r: tmean[r["target"]])
    return raw, oracle, _within_target_mean(mr), tmean


def build_gauge(joined, rows, mode, feature_fn):
    """Per-target gauge = mean over the target's candidates of feature_fn(candidate); offset = per-target mean
    score. feature_fn(candidate_dict) -> {feature_name: value}."""
    mr = [r for r in rows if r["mode"] == mode]
    tscore = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    by_t = {}
    for c in joined:
        by_t.setdefault(c["target"], []).append(c)
    names = None; table = {}
    for t, cands in by_t.items():
        vecs = [feature_fn(c) for c in cands]
        if names is None:
            names = sorted(vecs[0])
        gv = {n: float(np.mean([v[n] for v in vecs if v.get(n) is not None])) for n in names}
        table[t] = {"gauge": gv, "offset": tscore.get(t, 0.0)}
    return table, names


def predmix_vector(c):
    return np.array([float(c[k]) for k in schema.PRED_PROP], dtype=np.float64)
