"""Frozen C42 score registry."""
from __future__ import annotations


SCORES = (
    {"score": "actual_oaci_selector", "field": "", "orientation": "selected_only",
     "scope": "actual_selected_top1_only", "candidate_level_available": 0, "topk_eligible": 0,
     "selected_only": 1, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "actual selected candidate only; not converted into a candidate ranking"},
    {"score": "selection_leakage_point", "field": "selection_leakage_point", "orientation": "lower",
     "scope": "candidate_level_source", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "C41 global leakage field"},
    {"score": "source_audit_leakage", "field": "audit_leakage_point", "orientation": "lower",
     "scope": "candidate_level_source", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "source-audit leakage point"},
    {"score": "R_src", "field": "R_src", "orientation": "lower",
     "scope": "candidate_level_source", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "source risk; lower is better for actionability audit"},
    {"score": "C19_robust_core_score", "field": "c19_robust_core_score", "orientation": "higher",
     "scope": "candidate_level_source", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "frozen C19 robust-core probe score"},
    {"score": "C30_source_rank_score", "field": "source_rank_score", "orientation": "higher",
     "scope": "candidate_level_source", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "same frozen score, interpreted as C30 within-target source-rank axis"},
    {"score": "target_unlabeled_R3", "field": "", "orientation": "higher",
     "scope": "aggregate_or_local_non_source_only", "candidate_level_available": 0, "topk_eligible": 0,
     "selected_only": 0, "non_source_only": 1, "diagnostic_ceiling": 0,
     "note": "not an available candidate-level ranking field for C42; no proxy used"},
    {"score": "target_grouped_diagnostic_ceiling", "field": "target_utility_oracle_score", "orientation": "higher",
     "scope": "candidate_level_target_labeled_ceiling", "candidate_level_available": 1, "topk_eligible": 1,
     "selected_only": 0, "non_source_only": 1, "diagnostic_ceiling": 1,
     "note": "target-utility oracle ceiling for regret scale only; not a method"},
    {"score": "random_trajectory_conditioned", "field": "", "orientation": "random",
     "scope": "trajectory_conditioned_baseline", "candidate_level_available": 0, "topk_eligible": 0,
     "selected_only": 0, "non_source_only": 0, "diagnostic_ceiling": 0,
     "note": "within-trajectory random baseline"},
)


def registry(ctx):
    n = len(ctx["registry"])
    rows = []
    for s in SCORES:
        field = s["field"]
        available = sum(1 for r in ctx["registry"] if field and field in r) if field else 0
        rows.append({**s, "n_candidate_rows": n, "n_available": available, "proxy_used": 0})
    return {"rows": rows, "summary": {r["score"]: r for r in rows}}


def topk_scores():
    return [s for s in SCORES if s["topk_eligible"]]
