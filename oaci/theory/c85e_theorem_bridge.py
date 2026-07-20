"""Fail-closed empirical applicability labels for the frozen C85 theorems."""
from __future__ import annotations

from typing import Any, Iterable, Mapping


ALLOWED_LABELS = {
    "EXACTLY_APPLICABLE", "DESCRIPTIVE_ANALOGUE_ONLY",
    "ASSUMPTIONS_NOT_IDENTIFIED", "NOT_APPLICABLE", "OPEN_THEOREM",
}
THEOREM_STATUSES = {
    "T1": "PROVED", "T2": "COUNTEREXAMPLE", "T3": "PROVED",
    "T4": "PROVED", "T5": "OPEN", "T6": "COUNTEREXAMPLE", "T7": "PROVED",
}


def theorem_applicability_matrix(
    exact_collapse_scopes: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    scopes = [dict(row) for row in exact_collapse_scopes if bool(row.get("exact_collapse"))]
    base = {
        "T1": ("ASSUMPTIONS_NOT_IDENTIFIED", "No empirical Blackwell order is identified from registered-policy risks."),
        "T2": ("DESCRIPTIVE_ANALOGUE_ONLY", "Registered-policy nonmonotonicity is illustrative; unrestricted experiment value is unidentified."),
        "T4": ("ASSUMPTIONS_NOT_IDENTIFIED", "Two-state laws, TV, decoder, and uniform Delta are not identified."),
        "T5": ("OPEN_THEOREM", "The frozen multi-state theorem remains OPEN."),
        "T6": ("DESCRIPTIVE_ANALOGUE_ONLY", "Mean/tail separation is a descriptive empirical analogue."),
        "T7": ("ASSUMPTIONS_NOT_IDENTIFIED", "The pairwise sub-Gaussian score-error MGF assumption is not identified."),
    }
    rows = [{
        "theorem_id": theorem,
        "formal_status": THEOREM_STATUSES[theorem],
        "applicability": label,
        "scope": "C84_FROZEN_FIELD",
        "dataset": None,
        "level": None,
        "method_id": None,
        "reference_id": None,
        "reason": reason,
        "theorem_status_changed": False,
        "result_tag": "POST_C84S_EXPLORATORY",
    } for theorem, (label, reason) in base.items()]
    if scopes:
        for scope in scopes:
            rows.append({
                "theorem_id": "T3",
                "formal_status": THEOREM_STATUSES["T3"],
                "applicability": "EXACTLY_APPLICABLE",
                "scope": str(scope.get("scope")),
                "dataset": scope.get("dataset"),
                "level": scope.get("level"),
                "method_id": scope.get("method_id"),
                "reference_id": scope.get("reference_id"),
                "reason": "The fixed policy and reference select the same canonical action in every context of this scope.",
                "theorem_status_changed": False,
                "result_tag": "POST_C84S_EXPLORATORY",
            })
    else:
        rows.append({
            "theorem_id": "T3", "formal_status": THEOREM_STATUSES["T3"],
            "applicability": "NOT_APPLICABLE", "scope": "NO_EXACT_COLLAPSE_SCOPE",
            "dataset": None, "level": None, "method_id": None, "reference_id": None,
            "reason": "Near-collapse is insufficient for T3.",
            "theorem_status_changed": False, "result_tag": "POST_C84S_EXPLORATORY",
        })
    if any(row["applicability"] not in ALLOWED_LABELS for row in rows):
        raise ValueError("unregistered theorem-applicability label")
    return sorted(rows, key=lambda row: (row["theorem_id"], row["scope"]))


def assumption_identification_ledger() -> list[dict[str, Any]]:
    return [
        {"theorem_id": "T1", "assumption": "Blackwell_order", "identified": False,
         "result_tag": "POST_C84S_EXPLORATORY"},
        {"theorem_id": "T3", "assumption": "exact_action_kernel_collapse", "identified": "SCOPE_DEPENDENT",
         "result_tag": "POST_C84S_EXPLORATORY"},
        {"theorem_id": "T4", "assumption": "two_state_TV_decoder_Delta", "identified": False,
         "result_tag": "POST_C84S_EXPLORATORY"},
        {"theorem_id": "T5", "assumption": "proved_multistate_statement", "identified": False,
         "result_tag": "POST_C84S_EXPLORATORY"},
        {"theorem_id": "T7", "assumption": "pairwise_subGaussian_MGF", "identified": False,
         "result_tag": "POST_C84S_EXPLORATORY"},
    ]


def forbidden_transfer_claims() -> list[dict[str, str]]:
    claims = (
        "C84 proves Blackwell dominance",
        "Q0 estimates unrestricted label value",
        "COTT or MaNo is minimax optimal",
        "The C84 field satisfies T4 or T7 assumptions",
        "C85E proves an information-theoretic lower bound",
    )
    return [{
        "claim": claim, "allowed": "false", "reason": "Frozen theorem-transfer boundary",
        "result_tag": "POST_C84S_EXPLORATORY",
    } for claim in claims]


__all__ = [
    "ALLOWED_LABELS", "THEOREM_STATUSES", "assumption_identification_ledger",
    "forbidden_transfer_claims", "theorem_applicability_matrix",
]
