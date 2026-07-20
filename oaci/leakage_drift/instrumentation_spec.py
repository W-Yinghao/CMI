"""Future trace requirements for exact leakage atom attribution."""
from __future__ import annotations


FIELDS = (
    ("candidate_id", "candidate identity", "required", "binds atom trace to selector candidate without emitting checkpoint hash"),
    ("split_id", "population identity", "required", "distinguishes selection/source-audit/source-guard traces"),
    ("support_graph_hash", "support identity", "required", "binds comparable classes and support cells"),
    ("leakage_design_hash", "population design", "required", "binds sample ids, domain/class labels, groups, and sample mass"),
    ("fold_plan_hash", "crossfit identity", "required", "binds grouped fold assignment"),
    ("cell_weights", "cell estimand weights", "required", "persists p_ref, p(d|y), and overlap mass denominators"),
    ("per_fold_probe_nll_by_cell", "first divergent stage", "required", "locates drift before/after probe prediction and fold aggregation"),
    ("per_candidate_atom_table", "atom identity", "required", "persists exact domain x class point atoms"),
    ("aggregate_point_leakage", "aggregate identity", "required", "canonical sum target for atom table"),
    ("bootstrap_replicate_aggregate_leakage", "UCL boundary", "required_for_ucl_diagnostics", "keeps UCL quantile aggregate, not per-atom UCL"),
    ("bootstrap_replicate_order", "bootstrap identity", "required_for_ucl_diagnostics", "binds quantile ordering and accepted draw ids"),
    ("numeric_environment", "numeric reproducibility", "recommended", "records BLAS/sklearn/numpy versions and dtype pathway"),
)


def rows():
    out = []
    for i, (field, category, necessity, rationale) in enumerate(FIELDS, start=1):
        out.append({
            "field_order": i,
            "field_name": field,
            "category": category,
            "necessity": necessity,
            "rationale": rationale,
            "available_in_current_artifacts": 0 if field in (
                "per_fold_probe_nll_by_cell", "per_candidate_atom_table",
                "bootstrap_replicate_aggregate_leakage", "bootstrap_replicate_order",
                "numeric_environment") else 1,
        })
    return {"rows": out, "summary": {"n_required_fields": sum(1 for r in out if r["necessity"].startswith("required")),
                                       "n_currently_missing": sum(1 for r in out if not r["available_in_current_artifacts"]),
                                       "future_instrumentation_required": True}}
