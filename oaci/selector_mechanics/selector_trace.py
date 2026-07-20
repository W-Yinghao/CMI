"""C36 selector trace availability and registry helpers."""
from __future__ import annotations

from . import artifact_loader as al


REQUIRED_FIELDS = (
    ("candidate_id", "exact_candidate_identity_without_hash", "classification"),
    ("candidate_order", "exact_candidate_order", "classification"),
    ("selected_oaci", "exact_artifact_selected_flag", "classification"),
    ("feasible", "exact_risk_feasibility_for_replayed_candidates", "classification"),
    ("R_src", "exact_source_risk", "classification"),
    ("risk_slack_to_tau", "derived_tau_minus_R_src", "classification"),
    ("selection_leakage_point", "exact_selection_leakage_point_component", "classification"),
    ("per_candidate_selector_ucl_available", "actual_selector_score_ucl_per_candidate", "blocked"),
    ("audit_leakage_point", "exact_source_audit_leakage_point_component", "classification"),
    ("source_guard_worst_bacc", "source_guard_endpoint", "classification"),
    ("source_guard_worst_nll", "source_guard_endpoint", "classification"),
    ("source_guard_worst_ece", "source_guard_endpoint", "classification"),
    ("source_audit_worst_bacc", "source_audit_endpoint", "classification"),
    ("source_audit_worst_nll", "source_audit_endpoint", "classification"),
    ("source_audit_worst_ece", "source_audit_endpoint", "classification"),
    ("actual_selector_rank_known", "actual_selected_rank_known_only_for_selected", "limited"),
    ("tie_break_metadata_available", "exact_tie_break_metadata", "blocked"),
    ("checkpoint_hash_available", "checkpoint_hash_available_not_emitted", "classification"),
    ("checkpoint_hash_emitted", "checkpoint_hash_not_emitted", "gate"),
)


def _present(row, field):
    if field in ("per_candidate_selector_ucl_available", "tie_break_metadata_available"):
        return bool(row.get(field))
    if field == "actual_selector_rank_known":
        return bool(row.get(field))
    if field == "checkpoint_hash_emitted":
        return not bool(row.get(field))
    if field in {
        "R_src", "risk_slack_to_tau", "selection_leakage_point", "audit_leakage_point",
        "source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece",
        "source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
    }:
        return al.finite(row.get(field))
    return row.get(field) not in (None, "")


def availability_audit(registry):
    n = len(registry)
    rows = []
    for field, label, use in REQUIRED_FIELDS:
        got = sum(1 for r in registry if _present(r, field))
        if field == "checkpoint_hash_emitted":
            status = "pass_not_emitted" if got == n else "fail_hash_emitted"
        elif got == n:
            status = "complete"
        elif got == 0:
            status = "unavailable"
        else:
            status = "partial"
        rows.append({
            "field": field,
            "trace_item": label,
            "n_available": got,
            "n_total": n,
            "availability_fraction": (got / n if n else 0.0),
            "status": status,
            "used_for_classification": int(use == "classification" and got == n),
            "trace_use": use,
        })
    return rows


def registry_rows(trace):
    return trace["registry"]


def robust_pair_trace_resolves(pairs, trace):
    missing = []
    for p in pairs:
        key = (p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])
        alt = (p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])
        if key not in trace["by_key"] or alt not in trace["by_key"]:
            missing.append(p["pair_id"])
    return {"n_pairs": len(pairs), "n_missing": len(missing), "missing_pair_ids": missing[:20],
            "all_resolved": len(missing) == 0}


def selected_ucl_availability_for_pairs(pairs, trace):
    n = len(pairs)
    selected = better = 0
    for p in pairs:
        s = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["selected_order"])]
        b = trace["by_key"][(p["seed"], p["target"], p["level"], p["regime"], p["candidate_order"])]
        selected += int(al.finite(s.get("actual_selector_score_ucl")))
        better += int(al.finite(b.get("actual_selector_score_ucl")))
    return {"n_pairs": n, "selected_ucl_available": selected, "better_ucl_available": better,
            "pairwise_selector_score_delta_available": selected == n and better == n}
