"""ACAR V6-A0 — report schema + continuation gate.

DIAGNOSTIC-ONLY / EXPLORATORY. The report is a diagnostic; it MUST NOT contain any selection / routing / gate-pass / later-stage
key (candidate_id, threshold, route, G1–G6, Stage-4, external, held-out, ASZED, lockbox). `continuation_gate` applies the
EVAL-primary, BOTH-diseases V6_CONTINUE test; V6_CONTINUE authorizes ONLY drafting a V6 protocol — never policy fitting, candidate
selection, external read, or lockbox. Pure/stdlib. CODE + SYNTHETIC TESTS ONLY.
"""
from __future__ import annotations

CONTINUE = "V6_CONTINUE"
STOP = "V6_STOP"
RED_UPPER_MIN = 0.02
COVERAGE_MIN = 0.15
AUROC_MIN = 0.60
PERM_P_MAX = 0.05
DISEASES = ("PD", "SCZ")
# the gate reads EXACTLY these subject-balanced fields (V6-A0a2): no ambiguity about which metric fed the decision
_REQUIRED_METRICS = ("oracle_red_upper", "beneficial_coverage_subject_macro", "sign_auroc_subject_balanced",
                     "perm_p_subject_block")
# exact lowercased keys that may NOT appear ANYWHERE in the report (a diagnostic never selects / routes / certifies / goes external)
FORBIDDEN_REPORT_KEYS = ("candidate_id", "selected_candidate_id", "selected_candidate", "threshold", "thresholds", "route",
                         "routing", "g1", "g2", "g3", "g4", "g5", "g6", "stage4", "stage_4", "s1", "s2", "s3",
                         "external", "held_out", "aszed", "lockbox")


class V6A0ReportError(RuntimeError):
    """Raised when a V6-A0 diagnostic report is malformed or carries a forbidden selection/routing/later-stage key."""


def continuation_gate(per_disease_eval):
    """EVAL-primary. Returns (decision, detail). V6_CONTINUE iff BOTH diseases pass ALL FOUR sub-gates; else V6_STOP."""
    detail, ok_all = {}, True
    for d in DISEASES:
        m = per_disease_eval.get(d)
        if not isinstance(m, dict) or any(k not in m for k in _REQUIRED_METRICS):
            detail[d] = {"pass": False, "reason": "missing/incomplete disease metrics"}
            ok_all = False
            continue
        ru, cov = m["oracle_red_upper"], m["beneficial_coverage_subject_macro"]
        auroc, pp = m["sign_auroc_subject_balanced"], m["perm_p_subject_block"]
        checks = {                                                              # NaN fails every threshold (v == v guard)
            "oracle_red_upper_gt_0.02": bool(ru == ru and ru > RED_UPPER_MIN),
            "beneficial_coverage_subject_macro_ge_0.15": bool(cov == cov and cov >= COVERAGE_MIN),
            "sign_auroc_subject_balanced_ge_0.60": bool(auroc == auroc and auroc >= AUROC_MIN),
            "perm_p_subject_block_le_0.05": bool(pp == pp and pp <= PERM_P_MAX),
        }
        dp = all(checks.values())
        detail[d] = {"pass": dp, "checks": checks}
        ok_all = ok_all and dp
    return (CONTINUE if ok_all else STOP), detail


def _scan_forbidden(obj, path=""):
    bad = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            if str(k).lower() in FORBIDDEN_REPORT_KEYS:
                bad.append(f"{path}.{k}" if path else str(k))
            bad += _scan_forbidden(v, (f"{path}.{k}" if path else str(k)))
    elif isinstance(obj, (list, tuple)):
        for i, v in enumerate(obj):
            bad += _scan_forbidden(v, f"{path}[{i}]")
    return bad


def validate_v6a0_report(rep):
    """Fail-closed structural validation. Diagnostic-only: EVAL-primary, exploratory, no forbidden selection/routing/later keys."""
    if not isinstance(rep, dict):
        raise V6A0ReportError("report must be a dict")
    if rep.get("audit") != "ACAR_V6_A0_ACTION_VIABILITY":
        raise V6A0ReportError("audit tag mismatch")
    if rep.get("primary_split") != "EVAL":
        raise V6A0ReportError("primary_split must be 'EVAL' (the continuation gate is EVAL-only)")
    if rep.get("exploratory") is not True:
        raise V6A0ReportError("V6-A0 report must be exploratory=True")
    if rep.get("decision") not in (CONTINUE, STOP):
        raise V6A0ReportError("decision must be V6_CONTINUE or V6_STOP")
    bad = _scan_forbidden(rep)
    if bad:
        raise V6A0ReportError(f"report carries forbidden selection/routing/later-stage key(s): {sorted(set(bad))}")
    return True


def build_v6a0_report(*, per_disease_eval, descriptive=None, accounting=None, provenance_tags=None, meta=None):
    """Assemble + validate a V6-A0 diagnostic report. `descriptive` (FIT/CAL/per-action/calibration/coefficients) is reported but
    NEVER feeds `continuation_gate` (which reads per_disease_eval only). Raises if any forbidden key is present."""
    decision, gate_detail = continuation_gate(per_disease_eval)
    rep = {
        "audit": "ACAR_V6_A0_ACTION_VIABILITY", "primary_split": "EVAL", "exploratory": True, "decision": decision,
        "gate_thresholds": {"oracle_red_upper_min": RED_UPPER_MIN, "beneficial_coverage_min": COVERAGE_MIN,
                            "sign_auroc_min": AUROC_MIN, "perm_p_max": PERM_P_MAX},
        "per_disease_eval": per_disease_eval, "gate_detail": gate_detail,
        "descriptive_fit_cal_and_secondary": descriptive or {}, "accounting": accounting or {},
        "provenance_tags": list(provenance_tags or []), "meta": meta or {},
    }
    validate_v6a0_report(rep)
    return rep
