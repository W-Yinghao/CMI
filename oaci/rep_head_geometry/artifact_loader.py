"""C29 — read-only loaders + join. Target logits (C26 re-persistence) + head params (C29 head sidecar: b,
weight norms, class angles) + source class factors (C28 source_factor_builder) + offsets (C22). The
representation projection W.z is recovered as (target logit - b); no target-z re-persistence is needed for the
offset-relevant contribution. build_gauge/recover reuse the C24-C28 LOTO machinery."""
from __future__ import annotations

import json
import os

import numpy as np

from ..information_ladder import artifact_loader as il
from ..logit_geometry import artifact_loader as c27_al
from ..logit_geometry import factor_registry as c27_fr
from ..score_gauge.ceiling_ladder import _pooled_auc, _within_target_mean
from ..source_target_homology import source_factor_builder as c28_sfb
from . import schema

load_scores = il.load
candidate_features = c27_fr.candidate_features             # IDENTICAL carrier definition (C27/C28)
target_labels = c27_al.labels_by_fold


def head_available(head_sidecar=None) -> bool:
    return os.path.exists(head_sidecar or schema.C29_HEAD_SIDECAR)


def load(scores_sidecar=None, head_sidecar=None, repersist_dir=None, extract_dir=None):
    score_rows = il.load(scores_sidecar)
    hp = json.load(open(head_sidecar or schema.C29_HEAD_SIDECAR))["head_params"]
    head = {(h["seed"], h["target"], h["level"], h["model_hash"]): h for h in hp}
    tgt = c27_al.load_logits(repersist_dir)                 # per-candidate target 'L'
    src = c28_sfb.build_source_factors(extract_dir)         # per-candidate src_{role}_feats (effective source bias etc.)
    srcmap = {(s["seed"], s["target"], s["level"], s["model_hash"]): s for s in src}
    keys = {(r["seed"], r["target"], r["level"], r["model_hash"]) for r in score_rows if r["mode"] == "in_regime"}
    cands = []
    for r in tgt:
        k = (r["seed"], r["target"], r["level"], r["model_hash"])
        if k in keys and k in head and k in srcmap:
            b = np.asarray(head[k]["bias"], dtype=np.float64)
            cands.append({"seed": k[0], "target": k[1], "level": k[2], "model_hash": k[3], "L": r["L"], "b": b,
                          "weight_norms": np.asarray(head[k]["weight_norms"], dtype=np.float64),
                          "src_feats": srcmap[k]["src_source_guard_feats"]})
    return cands, score_rows


def raw_oracle(score_rows, mode="in_regime"):
    mr = [r for r in score_rows if r["mode"] == mode]
    tmean = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    raw = _pooled_auc(mr); oracle = _pooled_auc(mr, subtract=lambda r: tmean[r["target"]])
    return raw, oracle, _within_target_mean(mr), tmean


def build_gauge(cands, score_rows, mode, feature_fn):
    mr = [r for r in score_rows if r["mode"] == mode]
    tscore = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    by_t = {}
    for c in cands:
        by_t.setdefault(c["target"], []).append(c)
    names = None; table = {}
    for t, cs in by_t.items():
        vecs = [feature_fn(c) for c in cs]
        if names is None:
            names = sorted(vecs[0])
        table[t] = {"gauge": {n: float(np.mean([v[n] for v in vecs])) for n in names}, "offset": tscore.get(t, 0.0)}
    return table, names


def recover(cands, score_rows, mode, raw, oracle, feature_fn):
    from ..information_ladder import target_unlabeled_features as tuf
    table, names = build_gauge(cands, score_rows, mode, feature_fn)
    perm = tuf.r3_loto_permutation(score_rows, table, names, mode, raw, oracle)
    return {"gap_closed": perm["gap_closed"], "perm_p": perm["auc_improve_perm_p"],
            "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]}


# ---- carrier feature-fns over the (possibly transformed) target logits ----
def carrier_from_logits(transform=None):
    """class-conditioned confidence (conf_c0..c3) from the candidate's target logits, optionally transformed."""
    def fn(c):
        L = c["L"] if transform is None else transform(c)
        f = candidate_features(L)
        return {n: f[n] for n in schema.CARRIER_NAMES}
    return fn


def vec_gauge(key):
    """A 4-dim gauge directly from a per-candidate 4-vector stored under `key` (e.g. effective bias)."""
    return lambda c: {schema.CARRIER_NAMES[k]: float(c[key][k]) for k in range(schema.N_CLASSES)}
