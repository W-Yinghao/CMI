"""C29 Q2 — target representation geometry, read through the linear head. Because logit = W.z + b, the ONLY
part of the 800-d representation z that affects the logits (and hence the offset) is the projection W.z =
(logit - b); any z-component orthogonal to W's row space is offset-IRRELEVANT. So the offset-relevant
representation geometry is fully captured read-only by the projection W.z. This module summarizes the target
projection geometry (per-class projection mean/std, margins) and scores its offset recovery."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def _proj_summary(c):
    proj = c["L"] - c["b"]                                   # W.z (offset-relevant representation projection)
    out = {}
    for k in range(schema.N_CLASSES):
        out[f"proj_mean_c{k}"] = float(proj[:, k].mean())
        out[f"proj_std_c{k}"] = float(proj[:, k].std())
    return out


def representation_geometry(cands, score_rows, mode, raw, oracle) -> dict:
    # gauge from the full projection summary (mean + std per class)
    def full_proj(c):
        return _proj_summary(c)
    proj = artifact_loader.recover(cands, score_rows, mode, raw, oracle, full_proj)
    # gauge from projection MEANS only (the effective-bias part)
    def proj_mean(c):
        s = _proj_summary(c); return {f"proj_mean_c{k}": s[f"proj_mean_c{k}"] for k in range(schema.N_CLASSES)}
    pm = artifact_loader.recover(cands, score_rows, mode, raw, oracle, proj_mean)
    return {"projection_full_geometry": proj, "projection_mean_only": pm,
            "note": ("logit = W.z + b, so the projection W.z = (logit - b) is the COMPLETE offset-relevant "
                     "representation summary; z-components orthogonal to W are offset-irrelevant (a full 800-d z "
                     "re-persistence would add only offset-orthogonal descriptive geometry).")}
