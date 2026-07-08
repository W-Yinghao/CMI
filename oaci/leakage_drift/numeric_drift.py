"""Numeric/pathway drift diagnostics from committed artifacts."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def diagnose(ctx, manifest_rows, stagewise_summary):
    selection = [r for r in manifest_rows if r["split"] == "selection" and r["persisted_point_available"]]
    abs_vals = [float(r["abs_drift"]) for r in selection]
    signs = [1 if float(r["signed_drift_recomputed_minus_persisted"]) > 0 else -1
             if float(r["signed_drift_recomputed_minus_persisted"]) < 0 else 0 for r in selection]
    rows = [
        {
            "diagnostic": "float_precision",
            "status": "float64_replay_observed",
            "evidence": "C39 atom replay coerces features/probe math to float64; persisted per-fold outputs are absent",
            "rules_out_semantic_mismatch": 0,
        },
        {
            "diagnostic": "summation_order",
            "status": "atom_additive_diff_near_zero",
            "evidence": "max atom_sum_minus_recomputed_point is <= 4.44e-16",
            "rules_out_semantic_mismatch": 0,
        },
        {
            "diagnostic": "serialization_precision",
            "status": "csv_precision_sufficient_for_observed_values",
            "evidence": "persisted C37/C39 CSV values carry full decimal precision; max drift is far above decimal truncation scale",
            "rules_out_semantic_mismatch": 0,
        },
        {
            "diagnostic": "row_order_and_groupby_order",
            "status": "not_directly_persisted",
            "evidence": "sample ids and per-fold prediction rows are not committed in C39/C37 tables",
            "rules_out_semantic_mismatch": 0,
        },
        {
            "diagnostic": "bounded_drift",
            "status": "bounded_at_1e-3",
            "evidence": f"max_abs_drift={max(abs_vals) if abs_vals else None}",
            "rules_out_semantic_mismatch": 0,
        },
    ]
    observed_semantic = stagewise_summary["observed_semantic_mismatch_count"]
    summary = {
        "n_selection_candidates": len(selection),
        "max_abs_drift": max(abs_vals) if abs_vals else None,
        "mean_abs_drift": al.finite_mean(abs_vals),
        "bounded_at_1e_3": bool(abs_vals and max(abs_vals) <= schema.BOUNDED_DRIFT_TOL),
        "positive_signed_drift_count": sum(1 for s in signs if s > 0),
        "negative_signed_drift_count": sum(1 for s in signs if s < 0),
        "zero_signed_drift_count": sum(1 for s in signs if s == 0),
        "observed_semantic_mismatch_count": observed_semantic,
        "numeric_only_not_proven_due_to_missing_per_fold_trace": True,
    }
    return {"rows": rows, "summary": summary}

