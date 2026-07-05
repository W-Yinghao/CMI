"""C18 — support-severity response taxonomy. Deterministically maps the per-regime H2/H3/H4/H5 summaries to
one of: case_III_preserved / collapsed_to_case_II_calibration_only / collapsed_to_case_IV_source_
unidentifiable / boundary_visibility_destroyed / support_abstention_dominant / inconclusive_due_to_
estimability_loss. No target information enters the decision; the case is a statement about SOURCE-side
observability under controlled support degradation, not a deployment claim."""
from __future__ import annotations

from . import schema

_DELETING = ("S3_nonestimable_cells", "S4_missing_cells", "S5_block_class_by_domain",
             "S6_boundary_aligned_mask", "S7_random_matched_mask")


def _frac(vals):
    vals = [v for v in vals if v is not None]
    return (sum(1 for v in vals if v) / len(vals)) if vals else None


def severity_taxonomy(*, probe_by_regime, axis_by_regime=None, boundary_s6_s7=None,
                      leakage_by_regime=None, estimability_by_regime=None,
                      s0_reproduces_c17=None) -> dict:
    """probe_by_regime[regime] = {loto_auc, beats_permutation}; axis_by_regime[regime] =
    {accuracy_visibility, calibration_visibility}; boundary_s6_s7 = {s6_corr, s7_corr};
    leakage_by_regime[regime] = {any_estimable}; estimability_by_regime[regime] = {any_comparable_remaining}."""
    ev = {}
    # H2: does the LOTO advantage survive support degradation on the deleting regimes?
    del_beats = [probe_by_regime.get(r, {}).get("beats_permutation") for r in _DELETING]
    frac_beats = _frac(del_beats)
    ev["deleting_regimes_beats_perm_fraction"] = frac_beats
    s0_ok = probe_by_regime.get("S0_full_support", {}).get("beats_permutation")
    ev["s0_beats_permutation"] = s0_ok
    probe_preserved = bool(s0_ok) and frac_beats is not None and frac_beats >= 0.5
    probe_collapsed = frac_beats is not None and frac_beats < 0.5

    # inconclusive: most deleting regimes lost ALL comparable classes -> nothing left to identify
    if estimability_by_regime:
        any_comp = [estimability_by_regime.get(r, {}).get("any_comparable_remaining") for r in _DELETING]
        frac_comp = _frac(any_comp)
        ev["deleting_regimes_with_comparable_remaining"] = frac_comp
        if frac_comp is not None and frac_comp < 0.5:
            return _verdict(schema.CASE_INCONCLUSIVE, ev,
                            "over half the deleting regimes lost all comparable classes; identifiability undefined")

    # H5: support-aware leakage abstains (non-estimable) under degradation?
    abstention_dominant = False
    if leakage_by_regime:
        del_estim = [leakage_by_regime.get(r, {}).get("any_estimable") for r in _DELETING]
        frac_estim = _frac(del_estim)
        ev["deleting_regimes_leakage_estimable_fraction"] = frac_estim
        abstention_dominant = frac_estim is not None and frac_estim < 0.5

    # H4: boundary-aligned S6 destroys the mirror specifically vs severity-matched random S7?
    boundary_specific = False
    if boundary_s6_s7:
        s6, s7 = boundary_s6_s7.get("s6_corr"), boundary_s6_s7.get("s7_corr")
        ev["boundary_s6_corr"], ev["boundary_s7_corr"] = s6, s7
        if s6 is not None and s7 is not None:
            boundary_specific = (abs(s6) < 0.3) and (abs(s7) - abs(s6) > 0.15)

    # H3: accuracy visibility collapses while calibration survives?
    calib_over_acc = False
    if axis_by_regime:
        acc = [axis_by_regime.get(r, {}).get("accuracy_visibility") for r in _DELETING]
        cal = [axis_by_regime.get(r, {}).get("calibration_visibility") for r in _DELETING]
        acc = [a for a in acc if a is not None]; cal = [c for c in cal if c is not None]
        if acc and cal:
            ma, mc = sum(acc) / len(acc), sum(cal) / len(cal)
            ev["mean_accuracy_visibility_deleting"], ev["mean_calibration_visibility_deleting"] = ma, mc
            calib_over_acc = (mc - ma) > 0.05

    # ---- decision tree (ordered) ----
    if probe_preserved and not probe_collapsed:
        return _verdict(schema.CASE_PRESERVED, ev,
                        "weak source-only multivariate identifiability survives controlled support degradation")
    if probe_collapsed and calib_over_acc:
        return _verdict(schema.CASE_COLLAPSED_II, ev,
                        "accuracy observability collapses under support stress; calibration remains source-visible")
    if boundary_specific:
        return _verdict(schema.CASE_BOUNDARY_DESTROYED, ev,
                        "boundary-aligned masking specifically destroys the source-visible class-boundary mirror")
    if abstention_dominant:
        return _verdict(schema.CASE_ABSTENTION_DOMINANT, ev,
                        "support-aware leakage abstains (non-estimable) rather than smoothing unsupported cells")
    if probe_collapsed:
        return _verdict(schema.CASE_COLLAPSED_IV, ev,
                        "support degradation destroys source-side observability of target accuracy entirely")
    return _verdict(schema.CASE_PRESERVED, ev, "no collapse detected on the tested regimes")


def _verdict(case, evidence, summary) -> dict:
    next_science = {
        schema.CASE_PRESERVED: "Case III is not a full-support artifact -> a low-freedom source-only competence probe MAY be pre-registered; still diagnostic, NOT deployable.",
        schema.CASE_COLLAPSED_II: "push the mechanism: source evidence tracks softness/calibration, not discriminative transfer, once support degrades.",
        schema.CASE_COLLAPSED_IV: "support-mismatch mechanism section: support mismatch destroys source-side competence observability.",
        schema.CASE_BOUNDARY_DESTROYED: "focus on class-boundary support coverage as the missing observability condition.",
        schema.CASE_ABSTENTION_DOMINANT: "measurement contribution: the framework refuses unsupported conditional-invariance claims rather than smoothing.",
        schema.CASE_INCONCLUSIVE: "reduce deletion severity or add source domains; current stress removes too much support to decide.",
    }[case]
    return {"case_label": case, "interpretation": schema.TAXONOMY_INTERPRETATION[case],
            "summary": summary, "evidence": evidence, "next_science": next_science,
            "diagnostic_only_non_deployable": True}
