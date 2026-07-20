"""C29 Q3 — source vs target representation-projection residual, read through the SAME head. Both source and
target logits pass through the same W/b, so W.z_source vs W.z_target isolates the representation-cloud shift. We
decompose the target projection mean into a source-explained component + a target residual and ask which carries
the offset (mirrors C28 at the representation-projection level)."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def source_target_residual(cands, score_rows, mode, raw, oracle) -> dict:
    K = schema.N_CLASSES
    # per-candidate target projection mean (mean logit - b) and source projection mean (src effective bias - b)
    T = np.array([[c["L"].mean(0)[k] - c["b"][k] for k in range(K)] for c in cands], dtype=np.float64)
    S = np.array([[c["src_feats"][f"bias_c{k}"] - c["b"][k] for k in range(K)] for c in cands], dtype=np.float64)
    pred = np.zeros_like(T)
    for k in range(K):
        if S[:, k].std() > 1e-9:
            a, b = np.polyfit(S[:, k], T[:, k], 1); pred[:, k] = a * S[:, k] + b
        else:
            pred[:, k] = T[:, k].mean()
    resid = T - pred
    for i, c in enumerate(cands):
        c["_tgt_projmean"] = T[i]; c["_src_explained_projmean"] = pred[i]; c["_resid_projmean"] = resid[i]

    def rec(key):
        return artifact_loader.recover(cands, score_rows, mode, raw, oracle, artifact_loader.vec_gauge(key))
    full = rec("_tgt_projmean"); source_explained = rec("_src_explained_projmean"); residual = rec("_resid_projmean")
    rg, sg = residual["gap_closed"], source_explained["gap_closed"]
    residual_carries = bool(rg is not None and full["gap_closed"] and rg >= schema.CARRIES_FRACTION * full["gap_closed"] and residual["survives_permutation"])
    residual_over_source = bool(rg is not None and sg is not None and rg > 0 and rg > sg)
    return {"target_projection_mean": full, "source_explained": source_explained, "target_residual": residual,
            "residual_carries_offset": residual_carries, "residual_over_source_explained": residual_over_source,
            "note": ("the target projection-mean RESIDUAL (after removing its source-explained component) carries "
                     "the offset -> the missing gauge is a target-specific representation-projection shift, not the "
                     "source-visible component" if (residual_carries or residual_over_source) else
                     "neither the source-explained projection nor its residual cleanly carries the offset")}
