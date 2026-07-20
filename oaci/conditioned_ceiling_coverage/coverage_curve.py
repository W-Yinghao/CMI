"""C49 coverage-accuracy and local-island diagnostics."""
from __future__ import annotations

import math

import numpy as np

from ..conditioned_actionability import score_registry as c47_scores
from ..conditioned_local_ceiling import local_ceiling as c48_local
from . import artifact_loader as al
from . import schema


def neighborhood_specs(eps_summary, source_space):
    specs = []
    for q in schema.EPSILON_QUANTILES:
        specs.append({
            "neighborhood": f"eps_q{int(q * 100):02d}",
            "neighborhood_kind": "epsilon",
            "neighborhood_value": eps_summary[source_space][q],
        })
    for k in schema.K_VALUES:
        specs.append({
            "neighborhood": f"knn_{k}",
            "neighborhood_kind": "knn",
            "neighborhood_value": k,
        })
    return specs


def _purity_counts_from_dist(dist, spec, labels, base_rate):
    if spec["neighborhood_kind"] == "knn":
        return c48_local._knn_purity(dist, labels, int(spec["neighborhood_value"]), base_rate)
    return c48_local._epsilon_purity(dist, labels, float(spec["neighborhood_value"]), base_rate)


def _purity_counts(space, rows, spec, labels, base_rate):
    dist = c48_local._distance_matrix(space, rows)
    return _purity_counts_from_dist(dist, spec, labels, base_rate)


def _expected_hit_from_mask(purity, labels, covered):
    if not np.any(covered):
        return math.nan, math.nan, 0, math.nan
    p = purity[covered]
    y = labels[covered]
    max_p = float(np.max(p))
    tied = np.where(np.abs(p - max_p) <= 1e-12)[0]
    hit = float(np.mean(y[tied])) if len(tied) else math.nan
    return hit, max_p, int(len(tied)), len(tied) / len(y)


def _scope_fields(group):
    vals = {}
    for key in ("target", "seed", "level", "regime", "trajectory_id"):
        unique = sorted({str(r[key]) for r in group})
        vals[key] = unique[0] if len(unique) == 1 else "mixed"
    return vals


def _group_metric(ctx, scope, group_key, source_space_name, spec, label, min_n, group, purity, counts):
    labels = np.asarray([int(r[label]) for r in group], dtype=int)
    covered = counts >= int(min_n)
    coverage = float(np.mean(covered)) if len(covered) else math.nan
    full_base = float(np.mean(labels)) if len(labels) else math.nan
    covered_base = float(np.mean(labels[covered])) if np.any(covered) else math.nan
    hit, max_purity, tie_count, tie_fraction = _expected_hit_from_mask(purity, labels, covered)
    enrichment = hit / covered_base if al.finite(hit) and al.finite(covered_base) and covered_base > 0 else math.nan
    actual = al.c47_actual_top1(ctx, scope, label)
    fields = _scope_fields(group)
    return {
        "group_scope": scope,
        "group_key": group_key,
        "source_space": source_space_name,
        "neighborhood": spec["neighborhood"],
        "neighborhood_kind": spec["neighborhood_kind"],
        "neighborhood_value": spec["neighborhood_value"],
        "min_neighbor_count": min_n,
        "label": label,
        "target": fields["target"],
        "seed": fields["seed"],
        "level": fields["level"],
        "regime": fields["regime"],
        "trajectory_id": fields["trajectory_id"],
        "n_candidates": len(group),
        "n_covered_candidates": int(np.sum(covered)),
        "coverage": coverage,
        "empty_fraction": float(np.mean(counts == 0)) if len(counts) else math.nan,
        "mean_neighbor_count": float(np.mean(counts)) if len(counts) else math.nan,
        "median_neighbor_count": float(np.median(counts)) if len(counts) else math.nan,
        "group_base_rate": full_base,
        "covered_base_rate": covered_base,
        "mean_local_purity_all": float(np.mean(purity)) if len(purity) else math.nan,
        "mean_local_purity_covered": float(np.mean(purity[covered])) if np.any(covered) else math.nan,
        "max_local_purity_covered": max_purity,
        "local_bayes_top1_expected_hit": hit,
        "local_bayes_top1_gain_vs_covered_random": (
            hit - covered_base if al.finite(hit) and al.finite(covered_base) else math.nan
        ),
        "local_bayes_top1_enrichment": enrichment,
        "local_bayes_top1_tie_count": tie_count,
        "local_bayes_top1_tie_fraction": tie_fraction,
        "c47_actual_strict_source_top1_hit": actual,
        "gap_vs_c47_actual_top1": hit - actual if al.finite(hit) and al.finite(actual) else math.nan,
        "self_label_excluded": 1,
        "target_labels_diagnostic_only": 1,
        "no_candidate_id_emitted": 1,
    }


def evaluate(ctx, spaces, eps_summary):
    rows = []
    for source_space_name, space in spaces.items():
        specs = neighborhood_specs(eps_summary, source_space_name)
        for scope in schema.GROUP_SCOPES:
            for group_key, group in sorted(al.group_rows(ctx, scope).items()):
                dist = c48_local._distance_matrix(space, group)
                for label in schema.LABELS:
                    labels = np.asarray([int(r[label]) for r in group], dtype=int)
                    base_rate = float(np.mean(labels)) if len(labels) else math.nan
                    for spec in specs:
                        purity, counts, _ = _purity_counts_from_dist(dist, spec, labels, base_rate)
                        for min_n in schema.MIN_NEIGHBOR_COUNTS:
                            rows.append(_group_metric(
                                ctx, scope, group_key, source_space_name, spec, label, min_n,
                                group, purity, counts))
    return rows


def summarize(rows):
    out = []
    keys = sorted({
        (r["group_scope"], r["source_space"], r["neighborhood"], r["min_neighbor_count"], r["label"])
        for r in rows
    })
    for scope, source_space, neighborhood, min_n, label in keys:
        rs = [
            r for r in rows
            if r["group_scope"] == scope and r["source_space"] == source_space and
            r["neighborhood"] == neighborhood and int(r["min_neighbor_count"]) == int(min_n) and
            r["label"] == label
        ]
        hit = al.finite_mean([r["local_bayes_top1_expected_hit"] for r in rs])
        base = al.finite_mean([r["covered_base_rate"] for r in rs])
        actual = al.finite_mean([r["c47_actual_strict_source_top1_hit"] for r in rs])
        first = rs[0]
        out.append({
            "group_scope": scope,
            "source_space": source_space,
            "neighborhood": neighborhood,
            "neighborhood_kind": first["neighborhood_kind"],
            "neighborhood_value": first["neighborhood_value"],
            "min_neighbor_count": min_n,
            "label": label,
            "n_groups": len(rs),
            "n_evaluable_groups": sum(1 for r in rs if al.finite(r.get("local_bayes_top1_expected_hit"))),
            "mean_n_candidates": al.finite_mean([r["n_candidates"] for r in rs]),
            "mean_coverage": al.finite_mean([r["coverage"] for r in rs]),
            "mean_empty_fraction": al.finite_mean([r["empty_fraction"] for r in rs]),
            "mean_neighbor_count": al.finite_mean([r["mean_neighbor_count"] for r in rs]),
            "mean_group_base_rate": al.finite_mean([r["group_base_rate"] for r in rs]),
            "mean_covered_base_rate": base,
            "mean_local_purity_all": al.finite_mean([r["mean_local_purity_all"] for r in rs]),
            "mean_local_purity_covered": al.finite_mean([r["mean_local_purity_covered"] for r in rs]),
            "mean_local_bayes_top1_hit": hit,
            "mean_local_bayes_gain_vs_covered_random": hit - base
            if al.finite(hit) and al.finite(base) else math.nan,
            "mean_local_bayes_enrichment": hit / base if al.finite(hit) and al.finite(base) and base > 0
            else math.nan,
            "mean_c47_actual_strict_source_top1_hit": actual,
            "mean_gap_vs_c47_actual_top1": hit - actual if al.finite(hit) and al.finite(actual) else math.nan,
            "target_labels_diagnostic_only": 1,
        })
    return out


def best_by_scope(summary_rows):
    out = []
    for scope in schema.GROUP_SCOPES:
        for label in schema.LABELS:
            rs = [r for r in summary_rows if r["group_scope"] == scope and r["label"] == label]
            if not rs:
                continue
            best = max(
                rs,
                key=lambda r: (
                    float(r["mean_local_bayes_top1_hit"]) if al.finite(r["mean_local_bayes_top1_hit"]) else -1.0,
                    float(r["mean_local_bayes_enrichment"]) if al.finite(r["mean_local_bayes_enrichment"]) else -1.0,
                    float(r["mean_coverage"]) if al.finite(r["mean_coverage"]) else -1.0,
                ),
            )
            nr = dict(best)
            nr["best_kind"] = "best_coverage_local_bayes"
            out.append(nr)
    return out


def reliability_rows(summary_rows):
    out = []
    for r in summary_rows:
        for threshold in schema.COVERAGE_THRESHOLDS:
            passed = (
                al.finite(r.get("mean_coverage")) and float(r["mean_coverage"]) >= threshold and
                al.finite(r.get("mean_local_bayes_top1_hit")) and
                float(r["mean_local_bayes_top1_hit"]) >= schema.RELIABLE_TOP1_HIT_GATE and
                al.finite(r.get("mean_local_bayes_enrichment")) and
                float(r["mean_local_bayes_enrichment"]) >= schema.RELIABLE_ENRICHMENT_GATE
            )
            nr = dict(r)
            nr["coverage_threshold"] = threshold
            nr["passes_reliability_with_coverage"] = int(passed)
            out.append(nr)
    return out


def audit(ctx, spaces, eps_summary):
    rows = evaluate(ctx, spaces, eps_summary)
    summary_rows = summarize(rows)
    best_rows = best_by_scope(summary_rows)
    rel_rows = reliability_rows(summary_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "best_rows": best_rows,
        "reliability_rows": rel_rows,
        "summary": {
            "n_group_detail_rows": len(rows),
            "n_summary_rows": len(summary_rows),
            "n_best_rows": len(best_rows),
            "n_reliability_rows": len(rel_rows),
            "self_label_excluded": True,
            "coverage_definition": "neighbor_count >= min_neighbor_count",
        },
    }


def _best_conditioned_primary(best_rows):
    conditioned = ("within_target", "within_trajectory", "within_target_seed", "within_target_level")
    rs = [r for r in best_rows if r["label"] == "primary_joint_good" and r["group_scope"] in conditioned]
    return max(
        rs,
        key=lambda r: (
            float(r["mean_local_bayes_top1_hit"]) if al.finite(r["mean_local_bayes_top1_hit"]) else -1.0,
            float(r["mean_local_bayes_enrichment"]) if al.finite(r["mean_local_bayes_enrichment"]) else -1.0,
            float(r["mean_coverage"]) if al.finite(r["mean_coverage"]) else -1.0,
        ),
    ) if rs else {}


def island_rows(detail_rows, best_row):
    if not best_row:
        return []
    rows = [
        r for r in detail_rows
        if r["group_scope"] == best_row["group_scope"] and
        r["source_space"] == best_row["source_space"] and
        r["neighborhood"] == best_row["neighborhood"] and
        int(r["min_neighbor_count"]) == int(best_row["min_neighbor_count"]) and
        r["label"] == best_row["label"] and
        al.finite(r.get("local_bayes_top1_expected_hit"))
    ]
    rows.sort(
        key=lambda r: (
            float(r["local_bayes_top1_expected_hit"]),
            float(r["coverage"]),
            float(r["local_bayes_top1_enrichment"]) if al.finite(r.get("local_bayes_top1_enrichment")) else -1.0,
        ),
        reverse=True,
    )
    out = []
    for r in rows:
        nr = dict(r)
        nr["is_high_ceiling_island"] = int(
            float(r["local_bayes_top1_expected_hit"]) >= schema.RELIABLE_TOP1_HIT_GATE and
            al.finite(r.get("local_bayes_top1_enrichment")) and
            float(r["local_bayes_top1_enrichment"]) >= schema.RELIABLE_ENRICHMENT_GATE
        )
        out.append(nr)
    return out


def _setup_purity(space, group, setup):
    labels = np.asarray([int(r[setup["label"]]) for r in group], dtype=int)
    base = float(np.mean(labels)) if len(labels) else math.nan
    spec = {
        "neighborhood": setup["neighborhood"],
        "neighborhood_kind": setup["neighborhood_kind"],
        "neighborhood_value": setup["neighborhood_value"],
    }
    purity, counts, _ = _purity_counts(space, group, spec, labels, base)
    covered = counts >= int(setup["min_neighbor_count"])
    if not np.any(covered):
        return labels, purity, counts, covered, np.asarray([], dtype=int), math.nan
    p = purity[covered]
    covered_idx = np.where(covered)[0]
    max_p = float(np.max(p))
    island_idx = covered_idx[np.where(np.abs(p - max_p) <= 1e-12)[0]]
    hit = float(np.mean(labels[island_idx])) if len(island_idx) else math.nan
    return labels, purity, counts, covered, island_idx, hit


def underuse_audit(ctx, spaces, best_row):
    if not best_row:
        return {"rows": [], "summary_rows": []}
    score_specs = c47_scores.registry(ctx)
    strict_specs = [
        s for s in score_specs["rows"] if int(s["source_only"]) == 1 and int(s["hindsight_diagnostic_only"]) == 0
    ]
    group_map = al.group_rows(ctx, best_row["group_scope"])
    rows = []
    for group_key, group in sorted(group_map.items()):
        labels, purity, counts, covered, island_idx, local_hit = _setup_purity(
            spaces[best_row["source_space"]], group, best_row)
        if len(island_idx) == 0:
            continue
        island_set = set(int(i) for i in island_idx.tolist())
        fields = _scope_fields(group)
        for spec in strict_specs:
            scores = c47_scores.score_values(group, spec, score_specs["best_scalarization"])
            vals = np.asarray([float(scores[id(r)]) for r in group], dtype=float)
            top_score = float(np.max(vals))
            top_idx = np.where(np.abs(vals - top_score) <= 1e-12)[0]
            top_hit = float(np.mean(labels[top_idx])) if len(top_idx) else math.nan
            top_island_fraction = len([i for i in top_idx if int(i) in island_set]) / len(top_idx) if len(top_idx) else 0
            island_scores = vals[list(island_set)]
            island_best_score = float(np.max(island_scores)) if len(island_scores) else math.nan
            score_range = float(np.max(vals) - np.min(vals)) if len(vals) else math.nan
            flat_gap = top_score - island_best_score if al.finite(island_best_score) else math.nan
            rows.append({
                "group_scope": best_row["group_scope"],
                "group_key": group_key,
                "score": spec["score"],
                "target": fields["target"],
                "seed": fields["seed"],
                "level": fields["level"],
                "regime": fields["regime"],
                "trajectory_id": fields["trajectory_id"],
                "source_space": best_row["source_space"],
                "neighborhood": best_row["neighborhood"],
                "min_neighbor_count": best_row["min_neighbor_count"],
                "label": best_row["label"],
                "coverage": float(np.mean(covered)),
                "local_bayes_top1_expected_hit": local_hit,
                "source_score_top_expected_hit": top_hit,
                "top_score_island_fraction": top_island_fraction,
                "underuse_gap": local_hit - top_hit if al.finite(local_hit) and al.finite(top_hit) else math.nan,
                "score_range": score_range,
                "top_minus_best_island_score": flat_gap,
                "score_flat_at_island": int(al.finite(flat_gap) and abs(flat_gap) <= 1e-12),
                "wrong_direction_miss": int(
                    al.finite(local_hit) and local_hit >= schema.RELIABLE_TOP1_HIT_GATE and
                    al.finite(top_hit) and top_hit < schema.RELIABLE_TOP1_HIT_GATE and
                    top_island_fraction <= 0.0
                ),
                "target_labels_diagnostic_only": 1,
                "no_candidate_id_emitted": 1,
            })
    summary = []
    for score in sorted({r["score"] for r in rows}):
        rs = [r for r in rows if r["score"] == score]
        local_hit = al.finite_mean([r["local_bayes_top1_expected_hit"] for r in rs])
        score_hit = al.finite_mean([r["source_score_top_expected_hit"] for r in rs])
        summary.append({
            "score": score,
            "n_groups": len(rs),
            "mean_coverage": al.finite_mean([r["coverage"] for r in rs]),
            "mean_local_bayes_top1_hit": local_hit,
            "mean_source_score_top_hit": score_hit,
            "mean_underuse_gap": local_hit - score_hit if al.finite(local_hit) and al.finite(score_hit) else math.nan,
            "mean_top_score_island_fraction": al.finite_mean([r["top_score_island_fraction"] for r in rs]),
            "score_flat_at_island_fraction": al.finite_mean([r["score_flat_at_island"] for r in rs]),
            "wrong_direction_miss_fraction": al.finite_mean([r["wrong_direction_miss"] for r in rs]),
            "target_labels_diagnostic_only": 1,
        })
    return {"rows": rows, "summary_rows": summary}
