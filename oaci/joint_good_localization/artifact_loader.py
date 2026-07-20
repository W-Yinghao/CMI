"""C32 read-only artifact loader.

Joins the C22 source-score sidecar, C10 replay target endpoints plus the method-selected OACI hash, and the C24
label-free target-unlabeled summaries. The selected OACI hash is used only to compute aggregate diagnostic regret;
no selected-checkpoint artifact is emitted.
"""
from __future__ import annotations

import glob
import json
import os
import re

from ..endpoint_geometry import artifact_loader as c31_loader
from ..endpoint_geometry import endpoint_labels
from ..information_ladder import artifact_loader as c24_loader
from ..information_ladder import target_unlabeled_features as tuf
from . import schema

load_scores = c31_loader.load_scores


def load_candidate_endpoints(c10_dir=None):
    """Per-candidate target endpoint metrics plus selected-OACI membership from the C10 replay."""
    root = c10_dir or schema.C10_REPLAY_DIR
    rows = []
    for f in sorted(glob.glob(os.path.join(root, "seed-*-target-*.json"))):
        m = re.search(r"seed-(\d+)-target-(\d+)", f)
        seed, target = int(m.group(1)), int(m.group(2))
        d = json.load(open(f))
        for L, lv in d["levels"].items():
            selected = lv.get("selected", {})
            selected_oaci = selected.get("OACI")
            selected_erm = selected.get("ERM")
            cs = lv["candidates"]
            erm = next(c for c in cs if c.get("is_erm"))
            eb = erm.get("target_worst_bacc")
            en = erm.get("target_worst_nll")
            ee = erm.get("target_worst_ece")
            for c in cs:
                if c.get("is_erm") or not c.get("feasible"):
                    continue
                rows.append({
                    "seed": seed,
                    "target": target,
                    "level": int(L),
                    "model_hash": c["model_hash"],
                    "bacc": c.get("target_worst_bacc"),
                    "nll": c.get("target_worst_nll"),
                    "ece": c.get("target_worst_ece"),
                    "erm_bacc": eb,
                    "erm_nll": en,
                    "erm_ece": ee,
                    "epoch": c.get("epoch"),
                    "lambda": c.get("lambda"),
                    "origin": c.get("origin"),
                    "selected_oaci": int(c["model_hash"] == selected_oaci),
                    "selected_erm_hash_present": bool(selected_erm),
                })
    return rows


def merge(score_rows, endpoint_rows, mode="in_regime"):
    """Merge source-score rows with C10 endpoint/selection rows."""
    emap = {(r["seed"], r["target"], r["level"], r["model_hash"]): r for r in endpoint_rows}
    out = []
    for s in score_rows:
        if s["mode"] != mode:
            continue
        k = (s["seed"], s["target"], s["level"], s["model_hash"])
        if k in emap:
            out.append({**s, **emap[k]})
    return out


def attach_target_unlabeled(rows, reinfer_sidecar=None):
    """Attach the fixed C24 label-free target-unlabeled feature registry to candidate rows."""
    names = tuf.target_unlabeled_feature_names()
    tuf.assert_no_target_labels(names)
    sidecar = c24_loader.load_target_unlabeled_sidecar(reinfer_sidecar or schema.C24_TARGET_UNLABELED_SIDECAR)
    cmap = {(c["seed"], c["target"], c["level"], c["model_hash"]): c["target_unlabeled"]
            for c in sidecar["per_candidate"]}
    missing = 0
    for r in rows:
        key = (r["seed"], r["target"], r["level"], r["model_hash"])
        feats = cmap.get(key)
        if feats is None:
            missing += 1
        else:
            r.update(feats)
    return {"feature_names": names, "missing_rows": missing, "sidecar_config_hash": sidecar.get("config_hash")}


def load_rows(scores_sidecar=None, c10_dir=None, reinfer_sidecar=None, mode="in_regime", margin=None):
    score_rows = load_scores(scores_sidecar)
    endpoint_rows = load_candidate_endpoints(c10_dir)
    rows = merge(score_rows, endpoint_rows, mode=mode)
    endpoint_labels.attach_labels(rows, margin=schema.IMPROVE_MARGIN if margin is None else margin)
    tu = attach_target_unlabeled(rows, reinfer_sidecar)
    return rows, tu
