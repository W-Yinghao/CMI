"""C47 group-conditioned top-k and regret actionability metrics."""
from __future__ import annotations

import math

import numpy as np

from . import artifact_loader as al
from . import schema
from . import score_registry


def draw_count(n_rows: int, k: int) -> int:
    return min(int(k), int(n_rows))


def random_any_hit(n_rows: int, n_positive: int, n_draw: int) -> float:
    if n_rows <= 0 or n_draw <= 0:
        return math.nan
    if n_positive <= 0:
        return 0.0
    if n_draw >= n_rows or n_positive == n_rows:
        return 1.0
    miss = al.comb(n_rows - n_positive, n_draw) / al.comb(n_rows, n_draw)
    return float(1.0 - miss)


def random_expected_best_utility(rows, n_draw: int) -> float:
    vals = sorted(float(r["target_utility_score"]) for r in rows if al.finite(r.get("target_utility_score")))
    n = len(vals)
    if n <= 0 or n_draw <= 0:
        return math.nan
    k = min(int(n_draw), n)
    denom = al.comb(n, k)
    if denom <= 0:
        return math.nan
    total = 0.0
    for i, v in enumerate(vals):
        if i < k - 1:
            continue
        total += v * (al.comb(i, k - 1) / denom)
    return float(total)


def _ordered_rows(rows, scores):
    finite_rows = [r for r in rows if al.finite(scores.get(id(r))) and al.finite(r.get("target_utility_score"))]
    return sorted(finite_rows, key=lambda r: (float(scores[id(r)]), -int(r["source_idx"])), reverse=True)


def _group_metric(scope, group_key, rows, spec, scores, label, top_k):
    ordered = _ordered_rows(rows, scores)
    n = len(ordered)
    if n <= 0:
        return None
    k = draw_count(n, top_k)
    chosen = ordered[:k]
    positives = sum(int(r[label]) for r in ordered)
    any_hit = int(any(int(r[label]) for r in chosen))
    precision = sum(int(r[label]) for r in chosen) / k
    random_any = random_any_hit(n, positives, k)
    random_precision = positives / n if n else math.nan
    oracle = max(float(r["target_utility_score"]) for r in ordered)
    top_best = max(float(r["target_utility_score"]) for r in chosen)
    selected_regret = oracle - top_best
    random_best = random_expected_best_utility(ordered, k)
    random_regret = oracle - random_best if al.finite(random_best) else math.nan
    abs_reduction = random_regret - selected_regret if al.finite(random_regret) else math.nan
    rel_reduction = abs_reduction / random_regret if al.finite(random_regret) and random_regret > 1e-12 else math.nan
    return {
        "group_scope": scope,
        "group_key": group_key,
        "score": spec["score"],
        "score_family": spec["family"],
        "score_variant": "raw",
        "label": label,
        "top_k": top_k,
        "n_draw": k,
        "n_candidates": n,
        "n_label_positive": positives,
        "base_rate": random_precision,
        "any_hit": any_hit,
        "random_any_hit": random_any,
        "any_hit_gain_vs_random": any_hit - random_any if al.finite(random_any) else math.nan,
        "any_hit_enrichment": any_hit / random_any if al.finite(random_any) and random_any > 0 else math.nan,
        "precision_at_k": precision,
        "random_precision_at_k": random_precision,
        "precision_gain_vs_random": precision - random_precision if al.finite(random_precision) else math.nan,
        "precision_enrichment": precision / random_precision if random_precision > 0 else math.nan,
        "target_utility_oracle": oracle,
        "topk_best_target_utility": top_best,
        "regret_vs_oracle": selected_regret,
        "random_expected_best_target_utility": random_best,
        "random_expected_regret": random_regret,
        "absolute_regret_reduction_vs_random": abs_reduction,
        "relative_regret_reduction_vs_random": rel_reduction,
        "source_only": int(spec["source_only"]),
        "hindsight_diagnostic_only": int(spec["hindsight_diagnostic_only"]),
        "target_label_used": int(spec["target_label_used"]),
        "diagnostic_ceiling": int(spec["diagnostic_ceiling"]),
        "target_labels_diagnostic_only": 1,
        "no_candidate_id_emitted": 1,
    }


def evaluate_groups(ctx, score_specs):
    rows = []
    for scope in schema.GROUP_SCOPES:
        for group_key, group in sorted(al.group_rows(ctx, scope).items()):
            for spec in score_specs["rows"]:
                scores = score_registry.score_values(group, spec, score_specs["best_scalarization"])
                for label in schema.LABELS:
                    for top_k in schema.TOP_KS:
                        r = _group_metric(scope, group_key, group, spec, scores, label, top_k)
                        if r is not None:
                            rows.append(r)
    return rows


def _mean(rows, key):
    return al.finite_mean([r[key] for r in rows])


def summarize(rows):
    out = []
    for scope in schema.GROUP_SCOPES:
        scope_rows = [r for r in rows if r["group_scope"] == scope]
        scores = sorted({r["score"] for r in scope_rows})
        for score in scores:
            for label in schema.LABELS:
                for top_k in schema.TOP_KS:
                    rs = [
                        r for r in scope_rows
                        if r["score"] == score and r["label"] == label and int(r["top_k"]) == int(top_k)
                    ]
                    if not rs:
                        continue
                    mean_any = _mean(rs, "any_hit")
                    mean_random_any = _mean(rs, "random_any_hit")
                    mean_regret = _mean(rs, "regret_vs_oracle")
                    mean_random_regret = _mean(rs, "random_expected_regret")
                    out.append({
                        "group_scope": scope,
                        "score": score,
                        "label": label,
                        "top_k": top_k,
                        "n_groups": len(rs),
                        "mean_n_candidates": _mean(rs, "n_candidates"),
                        "mean_base_rate": _mean(rs, "base_rate"),
                        "mean_any_hit": mean_any,
                        "mean_random_any_hit": mean_random_any,
                        "mean_any_hit_gain_vs_random": (
                            mean_any - mean_random_any if al.finite(mean_any) and al.finite(mean_random_any)
                            else math.nan
                        ),
                        "mean_any_hit_enrichment": (
                            mean_any / mean_random_any if al.finite(mean_any) and
                            al.finite(mean_random_any) and mean_random_any > 0 else math.nan
                        ),
                        "mean_precision_at_k": _mean(rs, "precision_at_k"),
                        "mean_random_precision_at_k": _mean(rs, "random_precision_at_k"),
                        "mean_precision_gain_vs_random": _mean(rs, "precision_gain_vs_random"),
                        "mean_precision_enrichment": _mean(rs, "precision_enrichment"),
                        "mean_regret_vs_oracle": mean_regret,
                        "mean_random_expected_regret": mean_random_regret,
                        "mean_absolute_regret_reduction_vs_random": (
                            mean_random_regret - mean_regret
                            if al.finite(mean_random_regret) and al.finite(mean_regret) else math.nan
                        ),
                        "mean_relative_regret_reduction_vs_random": _mean(
                            rs, "relative_regret_reduction_vs_random"),
                        "source_only": max(int(r["source_only"]) for r in rs),
                        "hindsight_diagnostic_only": max(int(r["hindsight_diagnostic_only"]) for r in rs),
                        "target_label_used": max(int(r["target_label_used"]) for r in rs),
                        "diagnostic_ceiling": max(int(r["diagnostic_ceiling"]) for r in rs),
                        "target_labels_diagnostic_only": 1,
                    })
    return out


def _best(rows, *, source_only=True, allow_hindsight=True):
    candidates = []
    for r in rows:
        if source_only and int(r["source_only"]) != 1:
            continue
        if not allow_hindsight and int(r["hindsight_diagnostic_only"]) == 1:
            continue
        candidates.append(r)
    if not candidates:
        return None
    return max(
        candidates,
        key=lambda r: (
            float(r["mean_any_hit_gain_vs_random"]) if al.finite(r.get("mean_any_hit_gain_vs_random")) else -1e9,
            float(r["mean_any_hit"]) if al.finite(r.get("mean_any_hit")) else -1e9,
            -float(r["mean_regret_vs_oracle"]) if al.finite(r.get("mean_regret_vs_oracle")) else -1e9,
        ),
    )


def scope_best_summary(summary_rows):
    rows = []
    for scope in schema.GROUP_SCOPES:
        for label in schema.LABELS:
            for top_k in schema.TOP_KS:
                rs = [r for r in summary_rows if r["group_scope"] == scope and r["label"] == label and
                      int(r["top_k"]) == int(top_k)]
                best_source = _best(rs, source_only=True, allow_hindsight=False)
                best_hindsight = _best(rs, source_only=True, allow_hindsight=True)
                target_ceiling = next((r for r in rs if r["score"] == "target_utility_oracle_ceiling"), None)
                for kind, r in (("best_strict_source", best_source),
                                ("best_source_or_hindsight", best_hindsight),
                                ("target_oracle_ceiling", target_ceiling)):
                    if r is None:
                        continue
                    nr = dict(r)
                    nr["best_kind"] = kind
                    rows.append(nr)
    return rows


def audit(ctx, score_specs):
    rows = evaluate_groups(ctx, score_specs)
    summary_rows = summarize(rows)
    best_rows = scope_best_summary(summary_rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "best_rows": best_rows,
        "summary": {
            "n_group_metric_rows": len(rows),
            "n_summary_rows": len(summary_rows),
            "n_best_rows": len(best_rows),
            "n_group_scopes": len(schema.GROUP_SCOPES),
            "top_ks": list(schema.TOP_KS),
            "labels": list(schema.LABELS),
        },
    }


def correlation_by_scope(summary_rows):
    """Compact scope ordering helper for reports."""
    rows = []
    for scope in schema.GROUP_SCOPES:
        rs = [
            r for r in summary_rows if r["group_scope"] == scope and r["label"] == "primary_joint_good" and
            int(r["top_k"]) == 1 and int(r["source_only"]) == 1 and int(r["hindsight_diagnostic_only"]) == 0
        ]
        if not rs:
            continue
        vals = [float(r["mean_any_hit_gain_vs_random"]) for r in rs
                if al.finite(r.get("mean_any_hit_gain_vs_random"))]
        rows.append({
            "group_scope": scope,
            "strict_source_top1_gain_mean": float(np.mean(vals)) if vals else math.nan,
            "strict_source_top1_gain_max": max(vals) if vals else math.nan,
        })
    return rows
