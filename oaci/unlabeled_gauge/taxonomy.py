"""C25 — deterministic mechanism taxonomy U1-U7. The primary case is the R3 information-source verdict (which
family carries the weak recovery, or whether identity dominates, or whether it is a distributed weak multi-
family signal). U6 (source interference) and U7 (grouping separate problem class) are established secondaries."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(shap, identity, r4, grouping) -> dict:
    established = []
    # primary: information source of the weak R3 recovery
    if identity.get("identity_signature_dominates"):
        primary = schema.U4
    elif shap.get("single_family_dominates"):
        dom = shap["dominant_family"]
        primary = {"confidence_entropy": schema.U1, "margin_logitnorm": schema.U1,
                   "pred_class_prop": schema.U2}.get(dom, schema.U5)
    else:
        primary = schema.U5                                  # weak, distributed across families
    established.append(primary)
    # secondaries
    if r4.get("r4_gap") is not None and r4.get("r3_gap") is not None and r4["r4_gap"] < r4["r3_gap"]:
        established.append(schema.U6)                         # source interference confirmed
    if grouping.get("grouping_is_separate_problem_class"):
        established.append(schema.U7)
    interp = {
        schema.U1: "confidence/entropy (and/or margin/logit-norm) geometry carries most of the weak R3 offset recovery.",
        schema.U2: "predicted-class proportions carry most of the weak R3 offset recovery (Shapley-dominant + necessary: leave-one-family-out on it destroys recovery), though NOT sufficient alone (needs a confidence/margin-geometry scaffold) and ENTANGLED with target identity (same family is most identity-predictive) -- credited as a transferable marginal relationship only because it survives the LOTO permutation control.",
        schema.U3: "target feature-geometry families carry the recovery (only reachable if families 4-6 were computed; NOT in scope here).",
        schema.U4: "the R3 recovery mostly tracks a target-identity fingerprint (fails the identity/permutation control).",
        schema.U5: "no single family dominates; the weak R3 recovery is a DISTRIBUTED target-marginal signal across confidence-geometry families (permutation-robust, not identity).",
        schema.U6: "adding source features DESTROYS the R3 recovery (source is nuisance for the offset).",
        schema.U7: "the 0-label target-grouping oracle is a SEPARATE problem class (target grouping + held-out target's own scores), not source-only DG and not a target-label oracle.",
    }[primary]
    return {"primary_case": primary, "established": established,
            "single_family_dominates": bool(shap.get("single_family_dominates")),
            "dominant_family": shap.get("dominant_family"), "dominant_share": shap.get("dominant_share"),
            "identity_signature_dominates": bool(identity.get("identity_signature_dominates")),
            "source_interference_confirmed": schema.U6 in established,
            "grouping_separate_problem_class": schema.U7 in established,
            "r4_mechanism": r4.get("mechanism"), "interpretation": interp,
            "diagnostic_only_non_deployable": True}
