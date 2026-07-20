"""C29 Q1 — decompose the target logits into parameter head-bias b and representation projection W.z = (logit -
b), and ask which carries the C27 class-conditioned-confidence offset gauge. C27 found removing the per-class
EFFECTIVE bias (mean logit) destroys the carrier; the effective bias = mean(W.z) + b. This module tests whether
that effective bias is driven by the parameter b (R1) or by the representation-projection mean mean(W.z) (R2)."""
from __future__ import annotations

import numpy as np

from . import artifact_loader, schema


def _attach(cands):
    for c in cands:
        eff = c["L"].mean(0)                                 # effective class bias = mean logit per class
        c["_eff_bias"] = eff; c["_b"] = c["b"]; c["_proj_mean"] = eff - c["b"]   # proj_mean = mean(W.z)


def logit_decomposition(cands, score_rows, mode, raw, oracle) -> dict:
    _attach(cands)

    def rec(fn):
        return artifact_loader.recover(cands, score_rows, mode, raw, oracle, fn)
    full = rec(artifact_loader.carrier_from_logits())                            # C27 carrier (+0.524 ref)
    bias_removed = rec(artifact_loader.carrier_from_logits(lambda c: c["L"] - c["b"]))       # remove parameter b -> W.z
    eff_removed = rec(artifact_loader.carrier_from_logits(lambda c: c["L"] - c["L"].mean(0)))  # C27 class_bias_center
    eff_bias_g = rec(artifact_loader.vec_gauge("_eff_bias"))                     # effective bias 4-vec
    b_g = rec(artifact_loader.vec_gauge("_b"))                                   # parameter bias 4-vec (R1)
    proj_mean_g = rec(artifact_loader.vec_gauge("_proj_mean"))                   # representation-projection mean (R2)

    # DECISIVE test is the counterfactual on the ACTUAL (nonlinear softmax) carrier, not the linear 4-vec gauges:
    # the carrier is class-conditioned CONFIDENCE, so linear logit-mean gauges (b / projection-mean) do NOT
    # isolate it (disclosed). Removing the parameter bias b (-> W.z) PRESERVING the carrier => b is not the driver;
    # removing the effective per-class mean DESTROYING it => the representation-projection mean mean(W.z) drives it.
    fg = full["gap_closed"]
    bias_removed_preserves = bool(bias_removed["gap_closed"] is not None and fg and bias_removed["gap_closed"] >= 0.5 * fg
                                  and bias_removed["survives_permutation"])
    eff_removed_destroys = bool(eff_removed["gap_closed"] is not None and fg and eff_removed["gap_closed"] < 0.5 * fg)
    param_bias_drives = bool(not bias_removed_preserves)
    representation_projection_drives = bool(bias_removed_preserves and eff_removed_destroys)
    linear_gauges_recover = bool(proj_mean_g["gap_closed"] is not None and proj_mean_g["gap_closed"] >= schema.SUCCESS_GAP_CLOSED)
    return {"full_carrier": full, "parameter_bias_removed_carrier": bias_removed, "effective_mean_removed_carrier": eff_removed,
            "effective_bias_gauge": eff_bias_g, "parameter_bias_gauge": b_g, "projection_mean_gauge": proj_mean_g,
            "parameter_bias_drives_offset": param_bias_drives,
            "representation_projection_drives_offset": representation_projection_drives,
            "carrier_is_nonlinear_linear_gauges_fail": bool(representation_projection_drives and not linear_gauges_recover),
            "note": ("removing the parameter head-bias b (-> W.z) PRESERVES the carrier (%.3f vs full %.3f) while "
                     "removing the per-class EFFECTIVE mean DESTROYS it -> the offset-carrying effective class bias "
                     "is the representation-projection mean mean(W.z), NOT the parameter head-bias b (R2). The "
                     "carrier is a NONLINEAR softmax confidence, so the LINEAR b/projection-mean 4-vec gauges do "
                     "not isolate it (b-gauge %.3f, projmean-gauge %.3f) -- evidence is the counterfactual on the "
                     "actual carrier, not the linear gauges."
                     % (bias_removed["gap_closed"] or 0, fg or 0, b_g["gap_closed"] or 0, proj_mean_g["gap_closed"] or 0)
                     if representation_projection_drives else
                     "removing the parameter head-bias b DESTROYS the carrier -> b drives it (R1)" if param_bias_drives else
                     "no clean head/representation isolation")}
