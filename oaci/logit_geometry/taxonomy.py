"""C27 — deterministic logit-geometry taxonomy L1-L7. Primary = which logit factor carries the confidence-
occupancy interaction (class-conditioned confidence / class-bias-occupancy / global scale / sample-level
coupling), or whether it is fingerprint-only / not cleanly factorized. L5 (identity fingerprint) is always
disclosed (C26 entanglement); L6 (error-geometry coupling) is an established secondary."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(cc, cf, full_recovery, full_id_acc, label) -> dict:
    destroyers = set(cf.get("destroyers", []))
    cc_explains = bool(cc.get("class_conditioned_confidence_explains"))
    full_survives = bool(full_recovery.get("survives_permutation"))
    id_entangled = bool(full_id_acc is not None and full_id_acc > schema.IDENTITY_SIGNATURE_CEILING)
    l2 = bool({"class_bias_center", "class_uniformize"} & destroyers)
    l3 = bool({"temperature", "logit_norm_normalize"} & destroyers)
    l4 = bool({"confidence_shuffle", "class_shuffle"} & destroyers)
    l6 = bool(label.get("offset_and_alignment_coupled"))

    if not full_survives:
        primary = schema.L5                                 # recovery not permutation-robust -> fingerprint/none
    elif cc_explains:
        primary = schema.L1
    elif l2 and not l3:
        primary = schema.L2
    elif l3:
        primary = schema.L3
    elif l4:
        primary = schema.L4
    else:
        primary = schema.L7                                 # real but no tested factorization explains it
    established = [primary]
    if l6:
        established.append(schema.L6)
    if id_entangled and primary != schema.L5:
        established.append(schema.L5)                       # entanglement disclosed as co-finding
    interp = {
        schema.L1: "the interaction is CLASS-CONDITIONED CONFIDENCE (how confidently the model occupies each predicted class): it is a SINGLE sufficient factor (recovers alone, survives permutation), which REVISES C26's 'irreducible synergy' -- the synergy was an artifact of class-agnostic global confidence being too coarse; global confidence SCALE is NOT necessary (temperature/logit-norm survive) while the class-specific structure + per-target occupancy-confidence coupling ARE. Still IDENTITY-ENTANGLED (the per-class confidence profile is also the target fingerprint) and only PARTIALLY error-geometry-coupled.",
        schema.L2: "removing per-class logit bias / uniformizing occupancy destroys the recovery -> class-bias / occupancy drives the offset.",
        schema.L3: "temperature / logit-norm interventions destroy the recovery -> a global logit scale (confidence magnitude) drives the offset.",
        schema.L4: "shuffling the per-target coupling between occupancy and confidence destroys the recovery -> the offset needs their joint (sample/target-level) coupling, not either aggregate alone.",
        schema.L5: "the recovery is not permutation-robust / is explained only by target-identity fingerprinting.",
        schema.L6: "some interventions that destroy offset recovery also destroy target error-geometry alignment -> partly shared logit structure (see coupling_partial for whether a decoupling intervention exists).",
        schema.L7: "the confidence-occupancy interaction is real (permutation-robust) but no tested single factorization explains it cleanly.",
    }[primary]
    return {"primary_case": primary, "established": established, "destroyers": sorted(destroyers),
            "class_conditioned_confidence_explains": cc_explains, "full_recovery_survives": full_survives,
            "identity_entangled": id_entangled, "full_id_accuracy": full_id_acc,
            "error_geometry_coupled": l6, "error_geometry_coupling_partial": bool(label.get("coupling_partial")),
            "interpretation": interp, "diagnostic_only_non_deployable": True}
