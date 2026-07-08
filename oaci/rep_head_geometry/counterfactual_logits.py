"""C29 Q4 — deterministic logit counterfactuals on the frozen W/b/logits (NO retraining). Which head/
representation intervention DESTROYS the C27 offset recovery? Baseline = the raw class-conditioned confidence
carrier (+0.524). Interventions isolate parameter bias, effective (projection) mean, weight norm / global scale,
and source-vs-target representation projection."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def _temp():
    pass


def _weight_norm_normalized(c):
    # new_logit_k = (W_k/||W_k||).z + b_k = (logit_k - b_k)/||W_k|| + b_k
    return (c["L"] - c["b"]) / (c["weight_norms"][None, :] + 1e-9) + c["b"]


def _source_mean_centered_projection(c):
    # W.z_target - mean(W.z_target) + mean(W.z_source), then + b  (source proj mean from src effective bias - b)
    src_eff = np.array([c["src_feats"][f"bias_c{k}"] for k in range(schema.N_CLASSES)], dtype=np.float64)
    src_proj_mean = src_eff - c["b"]
    proj = c["L"] - c["b"]
    return (proj - proj.mean(0) + src_proj_mean) + c["b"]


_TRANSFORMS = {
    "parameter_bias_removed": lambda c: c["L"] - c["b"],
    "effective_mean_removed": lambda c: c["L"] - c["L"].mean(0),
    "projection_only": lambda c: c["L"] - c["b"],
    "weight_norm_normalized": _weight_norm_normalized,
    "global_scale_removed": lambda c: c["L"] / (np.linalg.norm(c["L"], axis=1, keepdims=True) + 1e-9),
    "source_mean_centered_projection": _source_mean_centered_projection,
}


def counterfactuals(cands, score_rows, mode, raw, oracle) -> dict:
    base = artifact_loader.recover(cands, score_rows, mode, raw, oracle, artifact_loader.carrier_from_logits())
    results = {"raw": base}
    for name, tf in _TRANSFORMS.items():
        results[name] = artifact_loader.recover(cands, score_rows, mode, raw, oracle,
                                                artifact_loader.carrier_from_logits(tf))
    bg = base["gap_closed"]
    for name, r in results.items():
        r["destroys_recovery"] = bool(name != "raw" and bg is not None and r["gap_closed"] is not None
                                      and r["gap_closed"] < bg * (1 - schema.DESTROYS_FRACTION))
    destroyers = [n for n, r in results.items() if r.get("destroys_recovery")]
    return {"baseline_gap": bg, "per_intervention": results, "destroyers": destroyers,
            "note": "interventions that destroy the offset recovery: %s" % (", ".join(destroyers) if destroyers else "none")}
