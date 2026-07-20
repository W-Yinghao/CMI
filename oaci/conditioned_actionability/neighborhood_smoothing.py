"""C47 source-neighborhood score smoothing diagnostics."""
from __future__ import annotations

import math

import numpy as np

from ..source_nonidentifiability import source_space
from . import artifact_loader as al
from . import group_actionability
from . import schema
from . import score_registry


def _neighbor_indices(rows, space, radius):
    if not rows:
        return []
    idx = np.asarray([int(r["source_idx"]) for r in rows], dtype=int)
    z = space["z"][idx, :]
    out = []
    for i in range(len(rows)):
        dist = np.linalg.norm(z - z[i], axis=1)
        out.append(np.where(dist <= radius + 1e-12)[0].tolist())
    return out


def _smooth_scores(rows, raw_scores, neighbors):
    out = {}
    for i, r in enumerate(rows):
        vals = [
            float(raw_scores[id(rows[j])])
            for j in neighbors[i]
            if al.finite(raw_scores.get(id(rows[j])))
        ]
        out[id(r)] = float(np.mean(vals)) if vals else raw_scores.get(id(r), math.nan)
    return out


def _metrics_for_variant(scope, group_key, rows, spec, scores, label, top_k, variant):
    r = group_actionability._group_metric(scope, group_key, rows, spec, scores, label, top_k)
    if r is None:
        return None
    r["score_variant"] = variant
    return r


def evaluate_groups(ctx, space, score_specs, radius):
    rows = []
    specs = [s for s in score_specs["rows"] if int(s["target_label_used"]) == 0]
    for scope in schema.GROUP_SCOPES:
        for group_key, group in sorted(al.group_rows(ctx, scope).items()):
            neighbors = _neighbor_indices(group, space, radius)
            for spec in specs:
                raw = score_registry.score_values(group, spec, score_specs["best_scalarization"])
                smooth = _smooth_scores(group, raw, neighbors)
                for label in schema.LABELS:
                    for top_k in schema.TOP_KS:
                        for variant, scores in (("raw", raw), ("q10_source_neighborhood_smoothed", smooth)):
                            r = _metrics_for_variant(scope, group_key, group, spec, scores, label, top_k, variant)
                            if r is not None:
                                r["q10_radius"] = radius
                                r["source_neighborhood_smoothing"] = int(variant != "raw")
                                rows.append(r)
    return rows


def summarize(rows):
    out = []
    for scope in schema.GROUP_SCOPES:
        for score in sorted({r["score"] for r in rows if r["group_scope"] == scope}):
            for label in schema.LABELS:
                for top_k in schema.TOP_KS:
                    raw = [
                        r for r in rows
                        if r["group_scope"] == scope and r["score"] == score and r["label"] == label and
                        int(r["top_k"]) == int(top_k) and r["score_variant"] == "raw"
                    ]
                    smooth = [
                        r for r in rows
                        if r["group_scope"] == scope and r["score"] == score and r["label"] == label and
                        int(r["top_k"]) == int(top_k) and
                        r["score_variant"] == "q10_source_neighborhood_smoothed"
                    ]
                    if not raw or not smooth:
                        continue
                    raw_hit = al.finite_mean([r["any_hit"] for r in raw])
                    smooth_hit = al.finite_mean([r["any_hit"] for r in smooth])
                    raw_gain = al.finite_mean([r["any_hit_gain_vs_random"] for r in raw])
                    smooth_gain = al.finite_mean([r["any_hit_gain_vs_random"] for r in smooth])
                    raw_regret = al.finite_mean([r["regret_vs_oracle"] for r in raw])
                    smooth_regret = al.finite_mean([r["regret_vs_oracle"] for r in smooth])
                    out.append({
                        "group_scope": scope,
                        "score": score,
                        "label": label,
                        "top_k": top_k,
                        "n_groups": len(smooth),
                        "q10_radius": raw[0].get("q10_radius"),
                        "raw_mean_any_hit": raw_hit,
                        "smoothed_mean_any_hit": smooth_hit,
                        "smoothing_any_hit_delta": (
                            smooth_hit - raw_hit if al.finite(raw_hit) and al.finite(smooth_hit) else math.nan
                        ),
                        "raw_mean_gain_vs_random": raw_gain,
                        "smoothed_mean_gain_vs_random": smooth_gain,
                        "smoothing_gain_delta": (
                            smooth_gain - raw_gain if al.finite(raw_gain) and al.finite(smooth_gain) else math.nan
                        ),
                        "raw_mean_regret_vs_oracle": raw_regret,
                        "smoothed_mean_regret_vs_oracle": smooth_regret,
                        "smoothing_regret_delta": (
                            smooth_regret - raw_regret if al.finite(raw_regret) and al.finite(smooth_regret)
                            else math.nan
                        ),
                        "hindsight_diagnostic_only": max(int(r["hindsight_diagnostic_only"]) for r in smooth),
                        "target_label_used": 0,
                        "target_labels_diagnostic_only": 1,
                    })
    return out


def audit(ctx, score_specs):
    space = source_space.build_space(ctx)
    radius = float(ctx["c46_summary"]["q10_radius"])
    rows = evaluate_groups(ctx, space, score_specs, radius)
    summary_rows = summarize(rows)
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "summary": {
            "q10_radius": radius,
            "n_smoothing_rows": len(rows),
            "n_smoothing_summary_rows": len(summary_rows),
            "distance_metric": schema.PRIMARY_DISTANCE,
            "target_oracle_ceiling_smoothed": False,
        },
    }
