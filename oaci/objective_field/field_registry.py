"""C41 objective-field availability and candidate registry."""
from __future__ import annotations

from . import artifact_loader as al


FIELD_SPECS = (
    ("selection_leakage_point", "candidate", "available", "lower"),
    ("selection_leakage_ucl", "candidate", "selected_only", "lower"),
    ("audit_leakage_point", "candidate", "available", "lower"),
    ("R_src", "candidate", "available", "lower"),
    ("balanced_err", "candidate", "available", "lower"),
    ("source_guard_worst_bacc", "candidate", "available", "higher"),
    ("source_guard_worst_nll", "candidate", "available", "lower"),
    ("source_guard_worst_ece", "candidate", "available", "lower"),
    ("source_audit_worst_bacc", "candidate", "available", "higher"),
    ("source_audit_worst_nll", "candidate", "available", "lower"),
    ("source_audit_worst_ece", "candidate", "available", "lower"),
    ("C30_source_rank_score", "aggregate_only", "available_aggregate_only", "higher"),
    ("target_unlabeled_R3", "local_pair_only", "available_local_pair_only_non_source_only", "higher"),
    ("target_gauge_factor", "local_pair_only", "available_local_pair_only_non_source_only", "higher"),
    ("target_utility_vector", "candidate", "available_diagnostic_only", "higher"),
    ("joint_good_label", "candidate", "available_diagnostic_only", "higher"),
    ("pareto_good_label", "candidate", "available_diagnostic_only", "higher"),
)


def availability(ctx):
    n = len(ctx["registry"])
    rows = []
    for field, scope, status, orientation in FIELD_SPECS:
        if scope == "candidate" and field in ctx["registry"][0]:
            available = sum(1 for r in ctx["registry"] if al.finite(r.get(field)))
        elif field in ("joint_good_label", "pareto_good_label"):
            available = n
        elif field == "target_utility_vector":
            available = sum(1 for r in ctx["registry"] if all(al.finite(r.get(k)) for k in (
                "target_bacc_delta", "target_nll_delta", "target_ece_delta")))
        elif scope == "aggregate_only":
            available = len(ctx["tables"]["c30"]["rank_gauge"])
        elif scope == "local_pair_only":
            available = len(ctx["tables"]["c34"]["selected_pairs"])
        else:
            available = 0
        rows.append({
            "field": field,
            "scope": scope,
            "status": status,
            "orientation_better": orientation,
            "n_available": available,
            "n_candidate_rows": n,
            "availability_fraction": available / n if n else None,
            "used_for_global_candidate_alignment": int(scope == "candidate" and "available" in status),
            "target_labels_diagnostic_only": int(field.startswith("target") or field.endswith("label")),
            "non_source_only": int(field in ("target_unlabeled_R3", "target_gauge_factor")),
            "proxy_used": 0,
        })
    summary = {
        "n_candidate_rows": n,
        "n_trajectories": len(ctx["by_traj"]),
        "candidate_registry_complete": n == 3804,
        "selection_ucl_global_available": False,
        "target_gauge_candidate_level_available": False,
        "c30_rank_candidate_level_available": False,
    }
    return {"rows": rows, "summary": summary}


def candidate_registry(ctx):
    rows = []
    cols = [
        "candidate_id", "seed", "target", "level", "regime", "trajectory_id", "candidate_order",
        "epoch", "selected_oaci", "feasible", "selection_leakage_point", "selection_leakage_ucl",
        "selection_leakage_ucl_available", "audit_leakage_point", "R_src", "balanced_err",
        "source_guard_worst_bacc", "source_guard_worst_nll", "source_guard_worst_ece",
        "source_audit_worst_bacc", "source_audit_worst_nll", "source_audit_worst_ece",
        "target_bacc_delta", "target_nll_delta", "target_ece_delta", "target_bacc_z",
        "target_nll_z", "target_ece_z", "continuous_joint_min_margin", "endpoint_vector_norm_regret",
        "pareto_distance", "dominated_hypervolume_regret", "primary_joint_good", "pareto_good",
        "preference_robust_better_candidate", "target_utility_score", "target_utility_finite",
        "target_labels_diagnostic_only",
    ]
    for r in ctx["registry"]:
        rows.append({c: r.get(c, "") for c in cols})
    return {"rows": rows, "columns": cols, "summary": {"n_rows": len(rows)}}

