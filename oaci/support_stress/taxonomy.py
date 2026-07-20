"""C18 — support-severity response taxonomy. Reason-codes every regime's feature loss (enum) BEFORE choosing
an overall case, so an accuracy-ENDPOINT non-estimability collapse (bAcc->NaN under cell deletion, while the
weak signal survives cell-present stress) is NOT mislabeled as a signal-level calibration-only collapse. Cases:
case_III_preserved / collapsed_by_accuracy_endpoint_nonestimability / collapsed_to_case_II_calibration_only_
signal / collapsed_to_case_IV_source_unidentifiable / boundary_visibility_destroyed / support_abstention_
dominant / support_stress_inconclusive_due_to_feature_loss. No target information enters the decision; the case
is a statement about SOURCE-side observability under controlled support degradation, not a deployment claim."""
from __future__ import annotations

from . import schema

# cell-PRESENT stress keeps every (d,y) cell present (bAcc endpoint stays computable); cell-DELETION
# stress removes whole cells (a domain loses a class -> worst-domain reference bAcc -> NaN, endpoint drops).
_CELL_PRESENT = ("S2_rare_cells", "S3_nonestimable_cells", "S5_block_class_by_domain")
_CELL_DELETION = ("S4_missing_cells", "S6_boundary_aligned_mask", "S7_random_matched_mask")
_DELETING = _CELL_PRESENT + _CELL_DELETION           # kept for the calibration-vs-accuracy contrast


def _frac(vals):
    vals = [v for v in vals if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def regime_reason(regime, p, *, s0_nfeat, s0_loto) -> str:
    """collapse-reason ENUM for one regime (reason-code BEFORE taxonomy; no free text)."""
    lo = p.get("loto_auc")
    if regime == "S1_label_marginal_skew" and lo is not None and s0_loto is not None and abs(lo - s0_loto) < 1e-9:
        return "implemented_noop"                    # identical to S0 -> row-based recompute no-op
    if p.get("beats_permutation"):
        return "none"
    nf, acc_vis, nused = p.get("n_features"), p.get("accuracy_visibility"), p.get("n_used")
    if acc_vis is None or (s0_nfeat is not None and nf is not None and nf < s0_nfeat):
        return "endpoint_metric_nonestimability"     # accuracy endpoint (bAcc) became NaN under deletion
    if nused is not None and nused < 50:
        return "insufficient_rows"
    return "signal_loss"                             # features present, but the multivariate advantage is gone


def severity_taxonomy(*, probe_by_regime, axis_by_regime=None, boundary_s6_s7=None,
                      leakage_by_regime=None, estimability_by_regime=None, s0_reproduces_c17=None) -> dict:
    """probe_by_regime[regime] must carry loto_auc, beats_permutation, n_features, accuracy_visibility,
    n_used. Distinguishes a genuine signal-level calibration-only collapse from an accuracy-ENDPOINT
    non-estimability collapse (bAcc->NaN under cell deletion) that leaves the weak signal intact under
    cell-present stress."""
    ev = {}
    s0 = probe_by_regime.get("S0_full_support", {})
    s0_ok = s0.get("beats_permutation"); s0_nfeat = s0.get("n_features"); s0_loto = s0.get("loto_auc")
    ev["s0_beats_permutation"] = s0_ok
    reasons = {r: regime_reason(r, probe_by_regime.get(r, {}), s0_nfeat=s0_nfeat, s0_loto=s0_loto)
               for r in schema.REGIME_ORDER}
    ev["regime_collapse_reason"] = reasons
    present_preserved = _frac([probe_by_regime.get(r, {}).get("beats_permutation") for r in _CELL_PRESENT])
    deletion_endpoint = _frac([reasons[r] == "endpoint_metric_nonestimability" for r in _CELL_DELETION])
    ev["cell_present_preserved_fraction"] = present_preserved
    ev["cell_deletion_endpoint_nonestimability_fraction"] = deletion_endpoint

    # inconclusive: most deleting regimes lost ALL comparable classes -> nothing left to identify
    if estimability_by_regime:
        frac_comp = _frac([estimability_by_regime.get(r, {}).get("any_comparable_remaining") for r in _DELETING])
        ev["deleting_regimes_with_comparable_remaining"] = frac_comp
        if frac_comp is not None and frac_comp < 0.5:
            return _verdict(schema.CASE_INCONCLUSIVE, ev, reasons,
                            "over half the deleting regimes lost all comparable classes; identifiability undefined")

    # H5 abstention
    abstention_dominant = False
    if leakage_by_regime:
        frac_estim = _frac([leakage_by_regime.get(r, {}).get("any_estimable") for r in _DELETING])
        ev["deleting_regimes_leakage_estimable_fraction"] = frac_estim
        abstention_dominant = frac_estim is not None and frac_estim < 0.5

    # H4 boundary-specific destruction
    boundary_specific = False
    if boundary_s6_s7:
        s6, s7 = boundary_s6_s7.get("s6_corr"), boundary_s6_s7.get("s7_corr")
        ev["boundary_s6_corr"], ev["boundary_s7_corr"] = s6, s7
        if s6 is not None and s7 is not None:
            boundary_specific = (abs(s6) < 0.3) and (abs(s7) - abs(s6) > 0.15)

    # H3 calibration-vs-accuracy visibility (over all masked regimes with a finite accuracy axis)
    calib_over_acc = False
    if axis_by_regime:
        acc = [axis_by_regime.get(r, {}).get("accuracy_visibility") for r in _DELETING]
        cal = [axis_by_regime.get(r, {}).get("calibration_visibility") for r in _DELETING]
        acc = [a for a in acc if a is not None]; cal = [c for c in cal if c is not None]
        if acc and cal:
            ma, mc = sum(acc) / len(acc), sum(cal) / len(cal)
            ev["mean_accuracy_visibility_deleting"], ev["mean_calibration_visibility_deleting"] = ma, mc
            calib_over_acc = (mc - ma) > 0.05

    # ---- decision (most specific first) ----
    pp = present_preserved is not None and present_preserved >= 0.5
    de = deletion_endpoint is not None and deletion_endpoint >= 0.5
    if pp and de:
        return _verdict(schema.CASE_ENDPOINT_NONESTIMABILITY, ev, reasons,
                        "weak signal survives cell-present stress; collapses under cell DELETION via accuracy-endpoint non-estimability")
    if boundary_specific:
        return _verdict(schema.CASE_BOUNDARY_DESTROYED, ev, reasons,
                        "boundary-aligned masking specifically destroys the source-visible class-boundary mirror")
    if abstention_dominant:
        return _verdict(schema.CASE_ABSTENTION_DOMINANT, ev, reasons,
                        "support-aware leakage abstains (non-estimable) rather than smoothing unsupported cells")
    if pp:
        return _verdict(schema.CASE_PRESERVED, ev, reasons,
                        "weak source-only multivariate identifiability survives support degradation that keeps cells present")
    if calib_over_acc:
        return _verdict(schema.CASE_COLLAPSED_II, ev, reasons,
                        "even cell-present stress collapses accuracy identifiability while calibration stays visible (signal-level)")
    return _verdict(schema.CASE_COLLAPSED_IV, ev, reasons,
                    "support degradation destroys source-side observability of target accuracy (features present, signal gone)")


def _verdict(case, evidence, reasons, summary) -> dict:
    next_science = {
        schema.CASE_PRESERVED: "Case III is not a full-support artifact -> a low-freedom source-only competence probe MAY be pre-registered; still diagnostic, NOT deployable.",
        schema.CASE_ENDPOINT_NONESTIMABILITY: "the limiter is estimator-level accuracy-endpoint availability under cell deletion, not signal loss; a pre-registered competence probe should use deletion-robust (calibration/leakage) observables + report endpoint estimability. Still diagnostic, NOT deployable.",
        schema.CASE_COLLAPSED_II: "push the mechanism: source evidence tracks softness/calibration, not discriminative transfer, once support degrades.",
        schema.CASE_COLLAPSED_IV: "support-mismatch mechanism section: support mismatch destroys source-side competence observability.",
        schema.CASE_BOUNDARY_DESTROYED: "focus on class-boundary support coverage as the missing observability condition.",
        schema.CASE_ABSTENTION_DOMINANT: "measurement contribution: the framework refuses unsupported conditional-invariance claims rather than smoothing.",
        schema.CASE_INCONCLUSIVE: "reduce deletion severity or add source domains; current stress removes too much support to decide.",
    }[case]
    return {"case_label": case, "interpretation": schema.TAXONOMY_INTERPRETATION[case],
            "summary": summary, "evidence": evidence, "regime_collapse_reason": reasons,
            "next_science": next_science, "diagnostic_only_non_deployable": True}
