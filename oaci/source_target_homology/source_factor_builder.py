"""C28 — build the SOURCE-side analog of the C27 target factor from the C18 extract per-candidate source logits
(logits-source_{guard,audit}.npy + units npz). Uses the IDENTICAL factor definition as C27 (factor_registry.
candidate_features). Also computes source per-class RECALL (source labels are source-side observable) and a
per-domain factor-variance summary. NO target labels here."""
from __future__ import annotations

import glob
import json
import os
import re

import numpy as np

from . import factor_registry, schema


def _recall(pred, y, C):
    return [float((pred[y == k] == k).mean()) if np.any(y == k) else 0.0 for k in range(C)]


def _softmax(z):
    z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)


def build_source_factors(extract_dir=None) -> list:
    root = extract_dir or schema.C18_EXTRACT_DIR
    C = schema.N_CLASSES
    out = []
    for cmf in sorted(glob.glob(os.path.join(root, "seed-*-target-*", "level-*", "cand_meta.json"))):
        m = re.search(r"seed-(\d+)-target-(\d+)[/\\]level-(\d+)", cmf)
        seed, target, level = int(m.group(1)), int(m.group(2)), int(m.group(3))
        base = os.path.dirname(cmf)
        cand_meta = json.load(open(cmf))
        role_data = {}
        for role in schema.SOURCE_ROLES:
            lg = np.load(os.path.join(base, f"logits-{role}.npy"))                 # [Ncand, Nunits, C]
            u = np.load(os.path.join(base, f"units-{role}.npz"), allow_pickle=True)
            role_data[role] = (lg, np.asarray(u["y"], dtype=np.int64), np.array([str(x) for x in u["domain_raw"]]))
        for c in cand_meta:
            ci, mh = c["index"], c["model_hash"]
            row = {"seed": seed, "target": target, "level": level, "model_hash": mh,
                   "is_erm": bool(c["is_erm"]), "feasible": bool(c["feasible"])}
            for role in schema.SOURCE_ROLES:
                lg, y, dom = role_data[role]
                L = np.asarray(lg[ci], dtype=np.float64)
                row[f"src_{role}_feats"] = factor_registry.candidate_features(L)
                pred = _softmax(L).argmax(1)
                row[f"src_{role}_recall"] = _recall(pred, y, C)
                # per-domain carrier-factor variance (domain-stability of class-conditioned confidence)
                per_dom = []
                for d in np.unique(dom):
                    mk = dom == d
                    if mk.sum() >= C:
                        f = factor_registry.candidate_features(L[mk])
                        per_dom.append([f[n] for n in schema.CARRIER_NAMES])
                pd = np.array(per_dom) if per_dom else np.zeros((1, C))
                row[f"src_{role}_domain_std"] = float(pd.std(0).mean())
            out.append(row)
    return out
