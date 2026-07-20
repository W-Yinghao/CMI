"""C49 deterministic sparse-ceiling taxonomy."""
from __future__ import annotations

import math

from . import artifact_loader as al
from . import schema


CONDITIONED_SCOPES = ("within_target", "within_trajectory", "within_target_seed", "within_target_level")


def _f(row, key, default=math.nan):
    return float(row[key]) if row and al.finite(row.get(key)) else default


def _best_conditioned(best_rows):
    rows = [
        r for r in best_rows
        if r["label"] == "primary_joint_good" and r["group_scope"] in CONDITIONED_SCOPES
    ]
    if not rows:
        return {}
    return max(
        rows,
        key=lambda r: (
            _f(r, "mean_local_bayes_top1_hit"),
            _f(r, "mean_local_bayes_enrichment"),
            _f(r, "mean_coverage"),
        ),
    )


def _reliable(row):
    return (
        _f(row, "mean_local_bayes_top1_hit") >= schema.RELIABLE_TOP1_HIT_GATE and
        _f(row, "mean_local_bayes_enrichment") >= schema.RELIABLE_ENRICHMENT_GATE
    )


def _best_reliable_at_threshold(reliability_rows, threshold):
    rows = [
        r for r in reliability_rows
        if r["label"] == "primary_joint_good" and r["group_scope"] in CONDITIONED_SCOPES and
        float(r["coverage_threshold"]) == float(threshold) and
        int(r["passes_reliability_with_coverage"]) == 1
    ]
    if not rows:
        return {}
    return max(
        rows,
        key=lambda r: (
            _f(r, "mean_coverage"),
            _f(r, "mean_local_bayes_top1_hit"),
            _f(r, "mean_local_bayes_enrichment"),
        ),
    )


def _max_underuse(summary_rows):
    if not summary_rows:
        return {}
    return max(summary_rows, key=lambda r: _f(r, "mean_underuse_gap"))


def _stability(summary_rows, grouping):
    rows = [r for r in summary_rows if r["stability_grouping"] == grouping]
    return rows[0] if rows else {}


def classify(coverage, stability, underuse, c48_summary):
    best = _best_conditioned(coverage["best_rows"])
    broad50 = _best_reliable_at_threshold(coverage["reliability_rows"], 0.50)
    broad75 = _best_reliable_at_threshold(coverage["reliability_rows"], 0.75)
    reliable_min2 = [
        r for r in coverage["summary_rows"]
        if r["label"] == "primary_joint_good" and r["group_scope"] in CONDITIONED_SCOPES and
        int(r["min_neighbor_count"]) >= 2 and _reliable(r)
    ]
    max_under = _max_underuse(underuse["summary_rows"])
    target_stab = _stability(stability["summary_rows"], "target")
    traj_stab = _stability(stability["summary_rows"], "trajectory")
    stable_target = (
        _f(target_stab, "min_hit") >= schema.STABILITY_WORST_HIT_GATE and
        _f(target_stab, "min_coverage") >= schema.STABILITY_WORST_COVERAGE_GATE
    )
    stable_traj = (
        _f(traj_stab, "min_hit") >= schema.STABILITY_WORST_HIT_GATE and
        _f(traj_stab, "min_coverage") >= schema.STABILITY_WORST_COVERAGE_GATE
    )
    any_reliable = _reliable(best)
    broad_conditioned = bool(broad50)
    sparse_only = any_reliable and not broad_conditioned
    underuse_gap = _f(max_under, "mean_underuse_gap")
    underuse_active = underuse_gap >= schema.UNDERUSE_GAP_GATE
    scores_match = underuse["summary_rows"] and underuse_gap <= schema.MATCH_GAP_GATE
    singleton_artifact = any_reliable and not reliable_min2
    escape_closed = not any_reliable
    future_hypothesis = any_reliable and underuse_active and not singleton_artifact
    established = {
        schema.SC1: sparse_only,
        schema.SC2: broad_conditioned,
        schema.SC3: underuse_active,
        schema.SC4: bool(scores_match),
        schema.SC5: any_reliable and (not stable_target or not stable_traj),
        schema.SC6: any_reliable and stable_target and stable_traj and underuse_active,
        schema.SC7: singleton_artifact,
        schema.SC8: future_hypothesis,
        schema.SC9: escape_closed,
        schema.SC10: False,
    }
    evidence = {
        schema.SC1: (
            f"best_hit={_f(best, 'mean_local_bayes_top1_hit')}, "
            f"best_coverage={_f(best, 'mean_coverage')}, "
            f"coverage50_reliable={bool(broad50)}"
        ),
        schema.SC2: (
            f"coverage50_best_scope={broad50.get('group_scope')}, "
            f"coverage50_hit={_f(broad50, 'mean_local_bayes_top1_hit')}, "
            f"coverage50_enrichment={_f(broad50, 'mean_local_bayes_enrichment')}, "
            f"coverage50={_f(broad50, 'mean_coverage')}, "
            f"coverage75_reliable={bool(broad75)}"
        ),
        schema.SC3: (
            f"score={max_under.get('score')}, underuse_gap={underuse_gap}, "
            f"score_top_hit={_f(max_under, 'mean_source_score_top_hit')}, "
            f"local_hit={_f(max_under, 'mean_local_bayes_top1_hit')}"
        ),
        schema.SC4: (
            f"max_underuse_gap={underuse_gap}, match_gate={schema.MATCH_GAP_GATE}"
        ),
        schema.SC5: (
            f"target_min_hit={_f(target_stab, 'min_hit')}, target_min_coverage={_f(target_stab, 'min_coverage')}, "
            f"trajectory_min_hit={_f(traj_stab, 'min_hit')}, "
            f"trajectory_min_coverage={_f(traj_stab, 'min_coverage')}"
        ),
        schema.SC6: (
            f"target_stable={stable_target}, trajectory_stable={stable_traj}, "
            f"underuse_gap={underuse_gap}, diagnostic_oracle_only=True"
        ),
        schema.SC7: (
            f"reliable_min_neighbor_ge_2_count={len(reliable_min2)}, "
            f"best_min_neighbor={best.get('min_neighbor_count')}"
        ),
        schema.SC8: (
            f"any_reliable={any_reliable}, underuse_active={underuse_active}, "
            f"singleton_artifact={singleton_artifact}, "
            f"c48_cases={','.join(c48_summary['taxonomy']['cases'])}"
        ),
        schema.SC9: (
            f"best_hit={_f(best, 'mean_local_bayes_top1_hit')}, "
            f"best_enrichment={_f(best, 'mean_local_bayes_enrichment')}"
        ),
        schema.SC10: "required C48/C47/C45 artifacts available",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
        "primary_metrics": {
            "best_conditioned_scope": best.get("group_scope"),
            "best_conditioned_source_space": best.get("source_space"),
            "best_conditioned_neighborhood": best.get("neighborhood"),
            "best_conditioned_min_neighbor_count": best.get("min_neighbor_count"),
            "best_conditioned_hit": _f(best, "mean_local_bayes_top1_hit"),
            "best_conditioned_enrichment": _f(best, "mean_local_bayes_enrichment"),
            "best_conditioned_coverage": _f(best, "mean_coverage"),
            "best_conditioned_mean_neighbor_count": _f(best, "mean_neighbor_count"),
            "coverage50_reliable": bool(broad50),
            "coverage50_best_scope": broad50.get("group_scope"),
            "coverage50_best_source_space": broad50.get("source_space"),
            "coverage50_best_neighborhood": broad50.get("neighborhood"),
            "coverage50_best_min_neighbor_count": broad50.get("min_neighbor_count"),
            "coverage50_best_hit": _f(broad50, "mean_local_bayes_top1_hit"),
            "coverage50_best_enrichment": _f(broad50, "mean_local_bayes_enrichment"),
            "coverage50_best_coverage": _f(broad50, "mean_coverage"),
            "coverage75_reliable": bool(broad75),
            "coverage75_best_scope": broad75.get("group_scope"),
            "coverage75_best_source_space": broad75.get("source_space"),
            "coverage75_best_neighborhood": broad75.get("neighborhood"),
            "coverage75_best_min_neighbor_count": broad75.get("min_neighbor_count"),
            "coverage75_best_hit": _f(broad75, "mean_local_bayes_top1_hit"),
            "coverage75_best_enrichment": _f(broad75, "mean_local_bayes_enrichment"),
            "coverage75_best_coverage": _f(broad75, "mean_coverage"),
            "max_underuse_score": max_under.get("score"),
            "max_underuse_gap": underuse_gap,
            "target_stability_min_hit": _f(target_stab, "min_hit"),
            "target_stability_min_coverage": _f(target_stab, "min_coverage"),
            "trajectory_stability_min_hit": _f(traj_stab, "min_hit"),
            "trajectory_stability_min_coverage": _f(traj_stab, "min_coverage"),
            "reliable_min_neighbor_ge_2_count": len(reliable_min2),
        },
    }
