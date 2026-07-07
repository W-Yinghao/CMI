"""C26 Q1 — is predicted-class-mix a STABLE target decision-occupancy signal or finite-sample noise? Needs
per-sample target logits to split the held-out target's samples (half / odd-even / bootstrap) and recompute the
mix per split. The C24 sidecar stores per-candidate AGGREGATES only, so this rung is availability-gated: it
returns REQUIRES_REPERSISTENCE_REINFERENCE when the split sidecar is absent (NOT proxied, NOT deferred). When
present (Stage-2, produced by the P0-validated forward persisting per-split mix summaries), it computes per-
target pred_prop split reliability + the split-half recovery ratio."""
from __future__ import annotations

import os

import numpy as np

from ..information_ladder import target_unlabeled_features as tuf
from . import artifact_loader, schema


def availability(split_sidecar=None) -> dict:
    path = split_sidecar or schema.C26_SPLIT_SIDECAR
    ready = os.path.exists(path)
    return {"path": path, "per_split_ready": ready,
            "status": schema.STATUS_OK if ready else schema.STATUS_REQUIRES_REINFERENCE,
            "reason": ("" if ready else "per-sample target logits not persisted; the C24 sidecar stores per-"
                       "candidate AGGREGATES only. Split-stability needs a scoped re-PERSISTENCE re-inference "
                       "(P0-validated forward, persist per-split mix summaries). NOT proxied from method-final.")}


def split_stability(rows, mode, raw, oracle, split_sidecar=None) -> dict:
    av = availability(split_sidecar)
    if not av["per_split_ready"]:
        return {"status": schema.STATUS_REQUIRES_REINFERENCE, "reason": av["reason"], "splits": None}
    import json
    d = json.load(open(av["path"]))
    if d.get("config_hash") != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C26 split sidecar config {d.get('config_hash')} != {schema.LOCKED_C19_CONFIG_HASH}")
    percand = {(c["seed"], c["target"], c["level"], c["model_hash"]): c["splits"] for c in d["per_candidate"]}
    mr = [r for r in rows if r["mode"] == mode]
    joined = [{"seed": r["seed"], "target": r["target"], "level": r["level"], "model_hash": r["model_hash"],
               "splits": percand[(r["seed"], r["target"], r["level"], r["model_hash"])]}
              for r in mr if (r["seed"], r["target"], r["level"], r["model_hash"]) in percand]
    out = []
    for split in schema.SPLITS:
        a_key, b_key = f"{split}_a", f"{split}_b"
        # per-candidate pred_prop on each half; per-target reliability = corr(mean pred_prop halfA, halfB)
        by_t = {}
        for c in joined:
            if a_key in c["splits"] and b_key in c["splits"]:
                by_t.setdefault(c["target"], []).append(c)
        corrs = []
        for t, cs in by_t.items():
            A = np.array([[c["splits"][a_key][k] for k in schema.PRED_PROP] for c in cs]).mean(0)
            B = np.array([[c["splits"][b_key][k] for k in schema.PRED_PROP] for c in cs]).mean(0)
            if A.std() > 1e-9 and B.std() > 1e-9:
                corrs.append(float(np.corrcoef(A, B)[0, 1]))
        reliability = float(np.mean(corrs)) if corrs else None

        def _fn(half):
            return lambda c: {k: float(c["splits"][half][k]) for k in schema.PRED_PROP}
        tbl_a, names = artifact_loader.build_gauge([{**c, **{k: c["splits"][a_key][k] for k in schema.PRED_PROP}} for c in joined if a_key in c["splits"]], rows, mode, lambda c: {k: c[k] for k in schema.PRED_PROP})
        rec_a = tuf.r3_loto_permutation(rows, tbl_a, names, mode, raw, oracle)
        out.append({"split": split, "predprop_reliability": reliability,
                    "split_half_gap": rec_a["gap_closed"], "survives_permutation": rec_a["survives_permutation"]})
    stable = all(s["predprop_reliability"] is not None and s["predprop_reliability"] >= schema.SPLIT_STABLE_CORR for s in out)
    return {"status": schema.STATUS_OK, "splits": out, "split_stable": bool(stable),
            "note": "predicted-class mix is split-stable (target decision-occupancy signal, not finite-sample noise)"
                    if stable else "predicted-class mix NOT split-stable -> finite-sample artifact risk"}
