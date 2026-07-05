"""ACAR V5 Stage-2 selection REPORT schema (pure/stdlib; NO I/O). Defines the Stage-2 DEV result structure and a fail-closed
validator. The report is EITHER a SELECTED candidate (with its EVAL G1–G5 table) OR a DEV_STOP; it carries per-candidate CAL
certification + EVAL utility, the Holm-adjusted H1–H3 status, and the G5 comparator values — and NEVER any S1/S2/S3 robustness
or external/lockbox result. Stage-2B0 defines + tests this schema; no real report (real selected candidate) is produced.
"""
from __future__ import annotations

OUTCOME_SELECTED = "SELECTED"
OUTCOME_DEV_STOP = "DEV_STOP"
_OUTCOMES = (OUTCOME_SELECTED, OUTCOME_DEV_STOP)

# report keys that would smuggle a later-stage result into the Stage-2 DEV report — forbidden
FORBIDDEN_REPORT_KEYS = ("s1", "s2", "s3", "s1_robustness", "s2_robustness", "s3_robustness",
                         "external", "held_out", "lockbox", "aszed", "final_external", "stage4", "stage5")

_REQUIRED_TOP = ("outcome", "selected_candidate_id", "per_candidate", "per_disease", "macro",
                 "holm_family_alpha", "objective", "notes")


class Stage2ReportError(RuntimeError):
    """Raised when a Stage-2 selection report violates the schema (fail-closed)."""


def candidate_disease_result(*, coverage_lcb, l_harm_all_ucb, harm_among_adapted_ucb, h2_evaluable,
                             g1, g3, g4, holm_p_h1, holm_p_h2, holm_p_h3, cert_pass,
                             red, red_upper, v2_replay_red, g2_margin, g2, g5, g5_comparator):
    """One (candidate, disease) result row: CAL certification (G1/G3/G4 + Holm-adjusted H1–H3) + EVAL utility (G2/G5)."""
    return {
        "cal": {"coverage_lcb": coverage_lcb, "l_harm_all_ucb": l_harm_all_ucb,
                "harm_among_adapted_ucb": harm_among_adapted_ucb, "h2_evaluable": bool(h2_evaluable),
                "G1": bool(g1), "G3": bool(g3), "G4": bool(g4),
                "holm_adj_p": {"H1": holm_p_h1, "H2": holm_p_h2, "H3": holm_p_h3}, "cert_pass": bool(cert_pass)},
        "eval": {"red": red, "red_upper": red_upper, "v2_replay_red": v2_replay_red, "g2_margin": g2_margin,
                 "G2": bool(g2), "G5": bool(g5), "g5_comparator": g5_comparator},
    }


def build_selection_report(*, outcome, selected_candidate_id, per_candidate, per_disease, macro, holm_family_alpha,
                           objective, notes):
    """Assemble a Stage-2 selection report dict (validated). `per_candidate` = {candidate_id: {disease: candidate_disease_result}}."""
    rep = {
        "outcome": outcome,
        "selected_candidate_id": selected_candidate_id,
        "per_candidate": per_candidate,
        "per_disease": per_disease,
        "macro": macro,
        "holm_family_alpha": holm_family_alpha,
        "objective": objective,
        "notes": notes,
    }
    validate_selection_report(rep)
    return rep


def validate_selection_report(rep):
    """Fail-closed schema check: required keys present; outcome∈{SELECTED,DEV_STOP} consistent with selected_candidate_id; no
    forbidden (S1/S2/S3/external/lockbox) key anywhere at the top level of the report or its notes."""
    if not isinstance(rep, dict):
        raise Stage2ReportError("report must be a dict")
    missing = [k for k in _REQUIRED_TOP if k not in rep]
    if missing:
        raise Stage2ReportError(f"report missing required key(s) {missing}")
    if rep["outcome"] not in _OUTCOMES:
        raise Stage2ReportError(f"outcome must be one of {_OUTCOMES}")
    if rep["outcome"] == OUTCOME_SELECTED and not rep["selected_candidate_id"]:
        raise Stage2ReportError("SELECTED outcome requires a selected_candidate_id")
    if rep["outcome"] == OUTCOME_DEV_STOP and rep["selected_candidate_id"] is not None:
        raise Stage2ReportError("DEV_STOP outcome must have selected_candidate_id = None")
    bad = sorted(k for k in rep if any(tok == str(k).lower() for tok in FORBIDDEN_REPORT_KEYS))
    if bad:
        raise Stage2ReportError(f"report carries forbidden later-stage/external key(s) {bad}")
    notes_keys = set(rep["notes"]) if isinstance(rep["notes"], dict) else set()
    bad_notes = sorted(k for k in notes_keys if any(tok == str(k).lower() for tok in FORBIDDEN_REPORT_KEYS))
    if bad_notes:
        raise Stage2ReportError(f"report.notes carries forbidden later-stage/external key(s) {bad_notes}")
    return True
