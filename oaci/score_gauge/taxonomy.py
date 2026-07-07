"""C23 — deterministic target-free score-gauge taxonomy. The identity-leakage audit gates a positive claim: a
gauge that only works because source features carry target identity (and does NOT generalize LOTO) is G3, not
G1. Epoch/order residual gates G6. Otherwise the case is set by the LOTO gap closure and the robust-core-vs-
risk-family comparison."""
from __future__ import annotations

from . import schema


def gauge_taxonomy(ladder, fit, identity, diag, risk, epoch_residual) -> dict:
    raw = ladder["raw_pooled"]; gauge = ladder["source_gauge_loto"]
    improve = ladder["auc_improve_source_gauge"]; gap = ladder["gap_closed_source_gauge"]
    loto_generalizes = bool(diag.get("loto_beats_permutation") and (fit.get("loto_r2") or -1) > 0)
    identity_laden = bool(identity.get("source_features_identity_separable"))
    meets_primary = bool(improve is not None and improve >= schema.SUCCESS_AUC_IMPROVE
                         and gap is not None and gap >= schema.SUCCESS_GAP_CLOSED and loto_generalizes)
    risk_gap = risk.get("gap_closed"); robust_gap = gap
    risk_family_only = bool(risk_gap is not None and risk_gap >= schema.SUCCESS_GAP_CLOSED
                            and (robust_gap is None or robust_gap < schema.SUCCESS_GAP_CLOSED))

    if epoch_residual:
        case = schema.G6
    elif meets_primary and not identity_laden:
        case = schema.G1
    elif identity_laden and not loto_generalizes and (improve or 0) > 0:
        case = schema.G3                     # apparent calibration only via identity, no LOTO generalization
    elif risk_family_only:
        case = schema.G4
    elif improve is not None and improve > 0 and (loto_generalizes or (gap or 0) >= 0.10):
        case = schema.G2                     # partial source gauge
    else:
        case = schema.G5                     # offset source-unobservable from allowed summaries
    interp = {
        schema.G1: "a target-free source-only gauge closes a substantial oracle gap and improves pooled held-out AUC WITHOUT target-identity leakage -> score non-transport is source-correctable (diagnostic).",
        schema.G2: "the per-target offset is PARTIALLY predictable from target-free source summaries; pooled transport improves but remains weak/marginal (diagnostic).",
        schema.G3: "apparent calibration works only because source features encode target identity; it does NOT generalize leave-one-target-out -> not a target-free gauge.",
        schema.G4: "a static source-risk scalar (R_src) explains the offset better than the robust-core gauge -> risk-family, not a robust-core deletion-robust gauge.",
        schema.G5: "the per-target offset is NOT predictable from the allowed target-free source summaries -> within-target competence is source-visible, but cross-target calibration is not identifiable from tested source evidence.",
        schema.G6: "the calibration improvement is explained by epoch/order trajectory position (despite the C22 baseline).",
    }[case]
    nxt = {
        schema.G1: "future work (diagnostic-only): a pre-registered target-free score-calibration estimand. NOT a selector.",
        schema.G2: "characterize the partially-predictable offset component; remains diagnostic-only.",
        schema.G3: "the offset is target-identity-bound; target-free calibration is not established.",
        schema.G4: "report the risk-family gauge as secondary; robust-core gauge does not carry the offset.",
        schema.G5: "strong mechanism boundary: cross-target calibration is not identifiable from tested source summaries; package as final.",
        schema.G6: "re-examine the trajectory-position residual before any calibration claim.",
    }[case]
    return {"primary_case": case, "meets_primary_success": meets_primary, "loto_generalizes": loto_generalizes,
            "identity_laden": identity_laden, "auc_improve": improve, "gap_closed": gap,
            "risk_family_only": risk_family_only, "interpretation": interp, "next_science": nxt,
            "diagnostic_only_non_deployable": True}
