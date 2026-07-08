"""C28 Q5 — decompose the target carrier into a SOURCE-explained component + a TARGET RESIDUAL, and ask which
carries the offset. target_conf_k ~= a_k*source_conf_k + b_k (per-class pooled fit); residual = target - predicted.
If the residual carries most of the offset recovery, the missing gauge is target-specific decision occupancy, not
source-visible logit geometry (the pooled fit is a decomposition, not a held-out predictor -- noted)."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def residual_decomposition(cands, score_rows, mode, raw, oracle, role) -> dict:
    K = schema.N_CLASSES; nm = schema.CARRIER_NAMES
    S = np.array([[c[f"src_{role}_feats"][n] for n in nm] for c in cands], dtype=np.float64)
    T = np.array([[c["tgt_feats"][n] for n in nm] for c in cands], dtype=np.float64)
    pred = np.zeros_like(T)
    for k in range(K):
        if S[:, k].std() > 1e-9:
            a, b = np.polyfit(S[:, k], T[:, k], 1); pred[:, k] = a * S[:, k] + b
        else:
            pred[:, k] = T[:, k].mean()
    resid = T - pred
    for i, c in enumerate(cands):
        c["_pred_carrier"] = {nm[k]: float(pred[i, k]) for k in range(K)}
        c["_resid_carrier"] = {nm[k]: float(resid[i, k]) for k in range(K)}
        c["_full_carrier"] = {nm[k]: float(T[i, k]) for k in range(K)}

    def _rec(key):
        return artifact_loader.recover(cands, score_rows, mode, raw, oracle, lambda c: dict(c[key]))
    full = _rec("_full_carrier"); source_explained = _rec("_pred_carrier"); residual = _rec("_resid_carrier")
    rc = bool(residual["gap_closed"] is not None and full["gap_closed"] is not None and full["gap_closed"] > 1e-6
              and residual["gap_closed"] >= schema.RESIDUAL_CARRIES_FRACTION * full["gap_closed"]
              and residual["survives_permutation"])
    se_carries = bool(source_explained["gap_closed"] is not None and full["gap_closed"] is not None and full["gap_closed"] > 1e-6
                      and source_explained["gap_closed"] >= schema.RESIDUAL_CARRIES_FRACTION * full["gap_closed"]
                      and source_explained["survives_permutation"])
    rg = residual["gap_closed"]; sg = source_explained["gap_closed"]
    residual_over_source_explained = bool(rg is not None and sg is not None and rg > 0 and rg > sg)
    return {"role": role, "full_target_carrier": full, "source_explained": source_explained, "target_residual": residual,
            "residual_carries_offset": rc, "source_explained_carries_offset": se_carries,
            "residual_over_source_explained": residual_over_source_explained,
            "note": ("the TARGET RESIDUAL (target carrier minus its source-explained component) carries the offset "
                     "recovery -> the missing gauge is target-specific decision occupancy, not source-visible logit "
                     "geometry" if rc else
                     "the source-explained component carries the offset" if se_carries else
                     "neither the source-explained component nor the residual cleanly dominates the offset recovery")}
