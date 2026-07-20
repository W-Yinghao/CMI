"""C27 — read-only loaders. Reads the C22 score sidecar (offsets) and the C26 re-persistence per-sample target
LOGITS npz (oaci-c26-repersist/*.unlabeled.npz) + the SEPARATE QUARANTINED labels npz. build_gauge() constructs
a per-target gauge from any per-candidate feature function of the (possibly transformed) logits, so every C27
factor / counterfactual is scored read-only. Quarantined labels are exposed ONLY via labels_by_fold()."""
from __future__ import annotations

import glob
import os
import re

import numpy as np

from ..information_ladder import artifact_loader as il
from ..score_gauge.ceiling_ladder import _pooled_auc, _within_target_mean
from . import schema

load_scores = il.load


def load_logits(repersist_dir=None):
    """Per-candidate rows with raw per-sample logits + split membership. NO labels here (unlabeled path)."""
    root = repersist_dir or schema.C26_REPERSIST_DIR
    rows = []
    for uf in sorted(glob.glob(os.path.join(root, "seed-*-target-*.unlabeled.npz"))):
        m = re.search(r"seed-(\d+)-target-(\d+)\.unlabeled\.npz", uf)
        seed, target = int(m.group(1)), int(m.group(2))
        u = np.load(uf, allow_pickle=True)
        logits = u["logits"]; mh = u["model_hash"]; lvl = u["level"]
        splits = {"half": u["split_half"], "odd_even": u["split_odd_even"], "bootstrap": u["split_bootstrap"]}
        for ci in range(len(mh)):
            rows.append({"seed": seed, "target": target, "level": int(lvl[ci]), "model_hash": str(mh[ci]),
                         "L": np.asarray(logits[ci], dtype=np.float64), "splits": splits})
    if not rows:
        raise FileNotFoundError(f"C27 requires the C26 re-persistence npz in {root} (run C26 Stage-3 first)")
    return rows


def labels_by_fold(repersist_dir=None):
    """QUARANTINED per-fold target labels (sample_id order matches the unlabeled logits). Loaded ONLY by the
    post-hoc label-alignment module."""
    root = repersist_dir or schema.C26_REPERSIST_DIR
    out = {}
    for lf in sorted(glob.glob(os.path.join(root, "seed-*-target-*.labels.npz"))):
        m = re.search(r"seed-(\d+)-target-(\d+)\.labels\.npz", lf)
        out[(int(m.group(1)), int(m.group(2)))] = np.asarray(np.load(lf, allow_pickle=True)["y"], dtype=np.int64)
    return out


def offset_join(logit_rows, score_rows, mode="in_regime"):
    """Attach the per-target score OFFSET population: keep only candidates present in the score table `mode`."""
    keys = {(r["seed"], r["target"], r["level"], r["model_hash"]) for r in score_rows if r["mode"] == mode}
    return [c for c in logit_rows if (c["seed"], c["target"], c["level"], c["model_hash"]) in keys]


def raw_oracle(score_rows, mode="in_regime"):
    mr = [r for r in score_rows if r["mode"] == mode]
    tmean = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    raw = _pooled_auc(mr); oracle = _pooled_auc(mr, subtract=lambda r: tmean[r["target"]])
    return raw, oracle, _within_target_mean(mr), tmean


def recover(logit_cands, score_rows, mode, raw, oracle, feature_fn):
    """Build a per-target gauge from feature_fn and score its LOTO offset recovery (gap_closed + permutation p +
    survives) with the shared C24/C26 r3_loto_permutation."""
    from ..information_ladder import target_unlabeled_features as tuf
    table, names = build_gauge(logit_cands, score_rows, mode, feature_fn)
    perm = tuf.r3_loto_permutation(score_rows, table, names, mode, raw, oracle)
    return {"gap_closed": perm["gap_closed"], "auc_improve": perm["auc_improve"], "perm_p": perm["auc_improve_perm_p"],
            "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]}


def recover_table(table, names, score_rows, mode, raw, oracle):
    from ..information_ladder import target_unlabeled_features as tuf
    perm = tuf.r3_loto_permutation(score_rows, table, names, mode, raw, oracle)
    return {"gap_closed": perm["gap_closed"], "auc_improve": perm["auc_improve"], "perm_p": perm["auc_improve_perm_p"],
            "survives_permutation": perm["survives_permutation"], "loto_r2": perm["loto_r2"]}


def build_gauge(logit_cands, score_rows, mode, feature_fn):
    """Per-target gauge = mean over the target's candidates of feature_fn(candidate); offset = per-target mean
    score. feature_fn(candidate_dict) -> {feature_name: value} (may transform candidate['L'])."""
    mr = [r for r in score_rows if r["mode"] == mode]
    tscore = {t: float(np.mean([c["score"] for c in mr if c["target"] == t])) for t in {r["target"] for r in mr}}
    by_t = {}
    for c in logit_cands:
        by_t.setdefault(c["target"], []).append(c)
    names = None; table = {}
    for t, cands in by_t.items():
        vecs = [feature_fn(c) for c in cands]
        if names is None:
            names = sorted(vecs[0])
        table[t] = {"gauge": {n: float(np.mean([v[n] for v in vecs])) for n in names}, "offset": tscore.get(t, 0.0)}
    return table, names
