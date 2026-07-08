"""C29 — deterministic representation-head origin taxonomy R1-R8. Decisive axis: does the offset-carrying
effective class bias come from the parameter head-bias b (R1) or the representation-projection mean mean(W.z)
(R2)? Then whether a target-specific projection SHIFT / residual carries it (R3/R7), whether scale/norm are
insufficient (R4), and whether an interaction is required (R5) or no clean decomposition exists (R8)."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(decomp, cf, resid, full_survives) -> dict:
    param_drives = bool(decomp.get("parameter_bias_drives_offset"))
    rep_drives = bool(decomp.get("representation_projection_drives_offset"))
    destroyers = set(cf.get("destroyers", []))
    scale_insufficient = bool("weight_norm_normalized" not in destroyers and "global_scale_removed" not in destroyers)
    rep_shift = bool("source_mean_centered_projection" in destroyers or resid.get("residual_over_source_explained"))
    residual_carries = bool(resid.get("residual_carries_offset") or resid.get("residual_over_source_explained"))
    source_explained_fails = not bool((resid.get("source_explained") or {}).get("survives_permutation"))

    if not full_survives:
        primary = schema.R8
    elif rep_drives:
        primary = schema.R2
    elif param_drives:
        primary = schema.R1
    else:
        primary = schema.R8
    established = [primary]
    if rep_shift and primary != schema.R1:
        established.append(schema.R3)
    if scale_insufficient:
        established.append(schema.R4)
    if source_explained_fails and primary != schema.R1:
        established.append(schema.R6)
    if residual_carries and primary != schema.R1:
        established.append(schema.R7)
    interp = {
        schema.R1: "the parameter head-bias b drives the offset carrier.",
        schema.R2: "the offset-carrying effective class bias is an EFFECTIVE logit bias induced by the target representation projection mean(W.z), NOT the parameter head-bias b.",
        schema.R3: "a TARGET-SPECIFIC representation-projection shift drives the offset (source-mean-centering the projection destroys recovery / the residual carries it).",
        schema.R4: "weight-norm / global-scale interventions do NOT destroy the recovery -> scale is not the driver (consistent with C27 scale-invariance).",
        schema.R5: "neither the head parameters nor the representation projection alone suffices; their interaction is required.",
        schema.R6: "the source representation projection does not carry the target offset (tracks source error geometry only).",
        schema.R7: "after removing the source-explained projection component, the TARGET RESIDUAL carries the offset -> the missing gauge is a target-specific representation-projection shift.",
        schema.R8: "the offset carrier is real but no tested head/representation factorization isolates it cleanly.",
    }[primary]
    return {"primary_case": primary, "established": established, "destroyers": sorted(destroyers),
            "parameter_bias_drives": param_drives, "representation_projection_drives": rep_drives,
            "scale_insufficient": scale_insufficient, "target_representation_shift": rep_shift,
            "target_residual_carries": residual_carries, "interpretation": interp, "diagnostic_only_non_deployable": True}
