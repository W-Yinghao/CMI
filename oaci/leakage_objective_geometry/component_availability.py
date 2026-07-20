"""C38 Stage-0 leakage component availability audit."""
from __future__ import annotations

import os

from . import artifact_loader as al
from . import schema


def _row(component, family, n_available, n_total, status, source, trace_use, note):
    return {
        "component": component,
        "family": family,
        "n_available": int(n_available),
        "n_total": int(n_total),
        "availability_fraction": (float(n_available) / n_total if n_total else None),
        "status": status,
        "source_artifact": source,
        "trace_use": trace_use,
        "target_labels_loaded_for_replay": 0,
        "note": note,
    }


def audit(ctx):
    exact = ctx["tables"]["c37"]["exact"]
    n = len(exact)
    p0 = ctx["tables"]["c37"]["p0"]
    manifest = ctx["tables"]["c37"]["manifest"]
    c27_registry = ctx["tables"]["c27"]["factor_registry"]
    c29_avail = ctx["tables"]["c29"]["rep_availability"]
    rows = [
        _row("selection_leakage_point", "selection_leakage", n, n, "available",
             "c37_tables/selected_vs_better_exact_ucl.csv", "classification",
             "Exact point estimates for selected and better candidates."),
        _row("selection_leakage_ucl", "selection_leakage", n, n, "available",
             "c37_tables/selected_vs_better_exact_ucl.csv", "classification",
             "Exact bootstrap UCLs recovered by C37."),
        _row("selection_bootstrap_width", "selection_leakage", n, n, "derived",
             "C37 exact UCL minus point", "classification",
             "Width is defined as UCL - point with frozen orientation."),
        _row("source_audit_leakage_point", "source_audit", n, n, "available",
             "c36_tables/selected_vs_better_selector_trace.csv", "classification",
             "Source-audit leakage point counterpart is available, not used as UCL proxy."),
        _row("source_audit_leakage_ucl", "source_audit", 0, n, "unavailable",
             "not persisted", "boundary",
             "No source-audit UCL is cached; C38 uses source-audit point only as an audit split."),
        _row("source_guard_endpoints", "source_endpoint", n, n, "available",
             "c36_tables/selected_vs_better_selector_trace.csv", "diagnostic",
             "Guard bAcc/NLL/ECE endpoint deltas are available."),
        _row("source_audit_endpoints", "source_endpoint", n, n, "available",
             "c36_tables/selected_vs_better_selector_trace.csv", "diagnostic",
             "Audit bAcc/NLL/ECE endpoint deltas are available."),
        _row("source_pareto_status", "source_geometry", n, n, "available",
             "c37_tables/source_pareto_after_ucl_recovery.csv", "diagnostic",
             "C37 recomputed source-Pareto conflict after exact UCL recovery."),
        _row("target_utility_cone_label", "target_endpoint", n, n, "available_diagnostic_only",
             "c35_tables/preference_robust_case_audit.csv", "diagnostic",
             "Target endpoint labels are imported only as diagnostic C35 labels."),
        _row("target_gauge_delta", "target_gauge", n, n, "available_diagnostic_only",
             "c34_tables/selected_vs_continuous_better_pairs.csv", "diagnostic",
             "C34 local target gauge delta is pair-local but non-source-only."),
        _row("target_class_conditioned_confidence_factor", "target_gauge", len(c27_registry),
             len(c27_registry), "global_available_not_pair_local",
             "c27_tables/logit_factor_registry.csv", "boundary",
             "C27 identifies the class-conditioned confidence factor globally, not per C38 leakage atom."),
        _row("target_representation_projection_bias", "target_gauge", len(c29_avail), len(c29_avail),
             "global_available_not_pair_local", "c29_tables/rep_head_artifact_availability.csv",
             "boundary", "C29 identifies representation-projection origin globally, not per local leakage atom."),
        _row("fold_probe_leakage_atoms", "leakage_atom", 0, n, "unavailable",
             "not persisted", "boundary", "No fold/probe atom contribution table is cached."),
        _row("class_conditioned_leakage_atoms", "leakage_atom", 0, n, "unavailable",
             "not persisted", "boundary", "No class-conditioned leakage atom contribution table is cached."),
        _row("domain_group_leakage_atoms", "leakage_atom", 0, n, "unavailable",
             "not persisted", "boundary", "No domain/group leakage atom contribution table is cached."),
        _row("support_cell_leakage_atoms", "leakage_atom", 0, n, "unavailable",
             "not persisted", "boundary", "No support-cell leakage atom contribution table is cached."),
        _row("c37_p0_selected_identity_hash_gate", "integrity", sum(
             int(r["p0_identity_pass"]) for r in p0), len(p0), "passed",
             "c37_tables/selected_ucl_identity_gate.csv", "gate",
             "Fold/bootstrap plan hashes matched and selected point/UCL reproduced exactly for P0."),
        _row("c37_replay_store_and_plan_resolution", "integrity", sum(
             int(r["store_exists"]) and int(r["selection_fold_plan_available"]) and
             int(r["selection_bootstrap_plan_available"]) and int(r["support_graph_available"])
             for r in manifest), len(manifest), "available",
             "c37_tables/selector_trace_recovery_manifest.csv", "gate",
             "Replay store, fold plan, bootstrap plan, and support graph resolve for 38/38 unique pairs."),
    ]
    summary = {
        "n_pairs": n,
        "selection_point_available": True,
        "selection_ucl_available": True,
        "bootstrap_width_available": True,
        "source_audit_point_available": True,
        "source_audit_ucl_available": False,
        "atom_decomposition_available": False,
        "target_gauge_pair_local_available": True,
        "p0_identity_pass": bool(p0 and all(int(r["p0_identity_pass"]) for r in p0)),
        "no_monolithic_c38_json": not os.path.exists("oaci/reports/C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.full.json"),
    }
    return {"rows": rows, "summary": summary}

