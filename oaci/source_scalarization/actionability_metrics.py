"""C43 scalarization actionability metrics."""
from __future__ import annotations

import math
from collections import defaultdict

from . import artifact_loader as al
from . import objective_registry, scalarization_grid, schema


def _draw_count(n, spec):
    if isinstance(spec, int):
        return min(spec, n)
    return max(1, int(math.ceil(n * float(spec))))


def _oriented(row, field, orientation):
    return objective_registry.oriented_value(row, field, orientation)


def _normalize_by_objective(rows):
    specs = objective_registry.scalar_objective_specs()
    out = {}
    for obj, spec in specs.items():
        vals = [(r["candidate_order"], _oriented(r, spec["field"], spec["orientation"])) for r in rows
                if al.finite(r.get(spec["field"]))]
        raw = [v for _, v in vals]
        mn, mx = min(raw), max(raw)
        norm = {}
        for order, v in vals:
            norm[order] = 0.5 if mx - mn <= 1e-12 else (v - mn) / (mx - mn)
        out[obj] = norm
    return out


def scalar_scores(rows, scalar_row):
    weights = scalarization_grid.weights(scalar_row)
    norms = _normalize_by_objective(rows)
    scores = {}
    for r in rows:
        total = 0.0
        for obj, w in weights.items():
            if w > 0:
                total += w * norms[obj][r["candidate_order"]]
        scores[r["candidate_order"]] = total
    return scores


def pairwise_auc(rows, scores):
    total = correct = ties = 0
    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            du = float(rows[i]["target_utility_score"]) - float(rows[j]["target_utility_score"])
            if abs(du) <= 1e-12:
                continue
            ds = scores[rows[i]["candidate_order"]] - scores[rows[j]["candidate_order"]]
            total += 1
            prod = du * ds
            if prod > 0:
                correct += 1
            elif abs(prod) <= 1e-12:
                ties += 1
    return (correct + 0.5 * ties) / total if total else math.nan


def random_baseline(ctx):
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        oracle = max(float(r["target_utility_score"]) for r in rs)
        mean_utility = al.finite_mean([r["target_utility_score"] for r in rs])
        for rule, spec in schema.TOPK_RULES:
            n = _draw_count(len(rs), spec)
            for label in schema.LABELS:
                rows.append({
                    "trajectory_id": tid,
                    "seed": seed,
                    "target": target,
                    "level": level,
                    "regime": regime,
                    "selection_rule": rule,
                    "label": label,
                    "n_candidates": len(rs),
                    "n_draw": n,
                    "trajectory_random_hit_rate": sum(int(r[label]) for r in rs) / len(rs),
                    "trajectory_random_expected_regret": oracle - mean_utility,
                })
    return {"rows": rows}


def audit(ctx, grid):
    rows = []
    top1_rows = []
    per_target = []
    baseline = random_baseline(ctx)
    for scalar in grid["rows"]:
        sid = scalar["scalarization_id"]
        target_groups = defaultdict(list)
        for tid, rs in sorted(ctx["by_traj"].items()):
            seed, target, level, regime = tid.split("|")
            scores = scalar_scores(rs, scalar)
            auc = pairwise_auc(rs, scores)
            target_groups[target].append((auc, rs, scores))
            ordered = sorted(rs, key=lambda r: (scores[r["candidate_order"]], -int(r["candidate_order"])),
                             reverse=True)
            oaci = next(r for r in rs if int(r["selected_oaci"]) == 1)
            oracle = max(float(r["target_utility_score"]) for r in rs)
            top = ordered[0]
            top1_rows.append({
                "scalarization_id": sid,
                "trajectory_id": tid,
                "seed": seed,
                "target": target,
                "level": level,
                "regime": regime,
                "pairwise_auc": auc,
                "top1_joint_good": int(top["primary_joint_good"]),
                "top1_pareto_good": int(top["pareto_good"]),
                "top1_preference_robust": int(top["preference_robust_better_candidate"]),
                "top1_regret_vs_oracle": oracle - float(top["target_utility_score"]),
                "top1_target_utility_delta_vs_oaci": float(top["target_utility_score"]) -
                float(oaci["target_utility_score"]),
                "top1_target_better_than_oaci": int(
                    float(top["target_utility_score"]) > float(oaci["target_utility_score"]) + 1e-12),
                "random_joint_baseline": sum(int(r["primary_joint_good"]) for r in rs) / len(rs),
            })
            for rule, spec in schema.TOPK_RULES:
                n = _draw_count(len(ordered), spec)
                chosen = ordered[:n]
                best_utility = max(float(r["target_utility_score"]) for r in chosen)
                for label in schema.LABELS:
                    hit = sum(int(r[label]) for r in chosen) / len(chosen)
                    base = sum(int(r[label]) for r in rs) / len(rs)
                    rows.append({
                        "scalarization_id": sid,
                        "grid_family": scalar["grid_family"],
                        "selection_rule": rule,
                        "label": label,
                        "trajectory_id": tid,
                        "pairwise_auc_vs_target_utility": auc,
                        "hit_rate": hit,
                        "trajectory_random_baseline": base,
                        "enrichment_ratio": hit / base if base > 0 else "",
                        "regret_vs_target_oracle": oracle - best_utility,
                        "target_utility_delta_vs_oaci": float(top["target_utility_score"]) -
                        float(oaci["target_utility_score"]),
                        "hindsight_diagnostic_only": 1,
                    })
        for target, vals in sorted(target_groups.items(), key=lambda kv: int(kv[0])):
            aucs = [v[0] for v in vals]
            top_hits = []
            for _, rs, scores in vals:
                top = max(rs, key=lambda r: (scores[r["candidate_order"]], -int(r["candidate_order"])))
                top_hits.append(int(top["primary_joint_good"]))
            per_target.append({
                "scalarization_id": sid,
                "target": target,
                "n_trajectories": len(vals),
                "mean_pairwise_auc": al.finite_mean(aucs),
                "top1_joint_good_rate": al.finite_mean(top_hits),
                "auc_positive_side": int((al.finite_mean(aucs) or 0) >= 0.5),
            })
    summary_rows = []
    for scalar in grid["rows"]:
        sid = scalar["scalarization_id"]
        for rule, _ in schema.TOPK_RULES:
            for label in schema.LABELS:
                rs = [r for r in rows if r["scalarization_id"] == sid and r["selection_rule"] == rule and
                      r["label"] == label]
                summary_rows.append({
                    "scalarization_id": sid,
                    "grid_family": scalar["grid_family"],
                    "selection_rule": rule,
                    "label": label,
                    "n_trajectories": len(rs),
                    "mean_pairwise_auc_vs_target_utility": al.finite_mean(
                        [r["pairwise_auc_vs_target_utility"] for r in rs]),
                    "mean_hit_rate": al.finite_mean([r["hit_rate"] for r in rs]),
                    "mean_random_baseline": al.finite_mean([r["trajectory_random_baseline"] for r in rs]),
                    "mean_enrichment_ratio": al.finite_mean([r["enrichment_ratio"] for r in rs]),
                    "mean_regret_vs_target_oracle": al.finite_mean([r["regret_vs_target_oracle"] for r in rs]),
                    "mean_target_utility_delta_vs_oaci": al.finite_mean(
                        [r["target_utility_delta_vs_oaci"] for r in rs]),
                    "hindsight_diagnostic_only": 1,
                })
    summary = {(r["scalarization_id"], r["selection_rule"], r["label"]): r for r in summary_rows}
    return {"rows": summary_rows, "trajectory_rows": top1_rows, "per_target_rows": per_target,
            "summary": summary, "random_baseline": baseline}
