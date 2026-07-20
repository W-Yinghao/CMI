"""C28 — read-only loaders + join. Source factors (C18 extract) + target factors (C26 re-persistence logits,
IDENTICAL C27 definition) + score offsets, merged per candidate. build_gauge/recover reuse the C24-C27 LOTO
machinery. Target labels are exposed ONLY via target_labels() (post-hoc error geometry)."""
from __future__ import annotations

import numpy as np

from ..information_ladder import artifact_loader as il
from ..logit_geometry import artifact_loader as c27_al
from ..score_gauge.ceiling_ladder import _pooled_auc, _within_target_mean
from . import factor_registry, schema, source_factor_builder

load_scores = il.load
target_labels = c27_al.labels_by_fold


def load_target_factors(repersist_dir=None) -> list:
    rows = c27_al.load_logits(repersist_dir)
    for c in rows:
        c["tgt_feats"] = factor_registry.candidate_features(c["L"])
        del c["L"]
    return rows


def join(source_rows, target_rows, score_rows, mode="in_regime"):
    keys = {(r["seed"], r["target"], r["level"], r["model_hash"]) for r in score_rows if r["mode"] == mode}
    tmap = {(r["seed"], r["target"], r["level"], r["model_hash"]): r for r in target_rows}
    out = []
    for s in source_rows:
        k = (s["seed"], s["target"], s["level"], s["model_hash"])
        if k in keys and k in tmap:
            out.append({**s, "tgt_feats": tmap[k]["tgt_feats"]})
    return out


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
    return {"gap_closed": perm["gap_closed"], "auc_improve": perm["auc_improve"], "perm_p": perm["auc_improve_perm_p"],
            "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]}


def recover_table(table, names, score_rows, mode, raw, oracle):
    from ..information_ladder import target_unlabeled_features as tuf
    perm = tuf.r3_loto_permutation(score_rows, table, names, mode, raw, oracle)
    return {"gap_closed": perm["gap_closed"], "perm_p": perm["auc_improve_perm_p"],
            "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]}


# feature-fn helpers over the merged candidate dict
def src_carrier(role):
    return lambda c: {n: c[f"src_{role}_feats"][n] for n in schema.CARRIER_NAMES}


def src_family(role, *families):
    names = factor_registry.family_feature_names(*families)
    return lambda c: {n: c[f"src_{role}_feats"][n] for n in names}


def tgt_carrier():
    return lambda c: {n: c["tgt_feats"][n] for n in schema.CARRIER_NAMES}
