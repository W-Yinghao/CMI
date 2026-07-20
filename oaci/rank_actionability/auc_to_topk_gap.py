"""C42 AUC-to-top-k gap calculations."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import schema, score_registry


def _draw_count(n, spec):
    if isinstance(spec, int):
        return min(spec, n)
    return max(1, int(math.ceil(n * float(spec))))


def _oriented(row, spec):
    v = float(row[spec["field"]])
    return -v if spec["orientation"] == "lower" else v


def order_rows(rows, spec):
    return sorted([r for r in rows if spec["field"] and al.finite(r.get(spec["field"]))],
                  key=lambda r: (_oriented(r, spec), -int(r["candidate_order"])), reverse=True)


def pairwise_auc(rows, spec):
    vals = [(r, _oriented(r, spec), float(r["target_utility_score"])) for r in rows
            if spec["field"] and al.finite(r.get(spec["field"])) and al.finite(r.get("target_utility_score"))]
    total = correct = ties = 0
    for i in range(len(vals)):
        for j in range(i + 1, len(vals)):
            du = vals[i][2] - vals[j][2]
            df = vals[i][1] - vals[j][1]
            if abs(du) <= 1e-12:
                continue
            total += 1
            prod = du * df
            if prod > 0:
                correct += 1
            elif abs(prod) <= 1e-12:
                ties += 1
    return (correct + 0.5 * ties) / total if total else math.nan


def _topk_hit(rows, label):
    return sum(int(r[label]) for r in rows) / len(rows) if rows else math.nan


def _oracle_utility(rows):
    return max(float(r["target_utility_score"]) for r in rows if al.finite(r["target_utility_score"]))


def random_baseline(ctx):
    rows = []
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        oracle = _oracle_utility(rs)
        mean_utility = al.finite_mean([r["target_utility_score"] for r in rs])
        for rule, spec in schema.TOPK_RULES:
            n = _draw_count(len(rs), spec)
            for label in schema.LABELS:
                base = sum(int(r[label]) for r in rs) / len(rs)
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
                    "trajectory_random_hit_rate": base,
                    "trajectory_random_expected_regret": oracle - mean_utility,
                })
    summary = {}
    for rule, _ in schema.TOPK_RULES:
        for label in schema.LABELS:
            rs = [r for r in rows if r["selection_rule"] == rule and r["label"] == label]
            summary[(rule, label)] = {
                "mean_hit_rate": al.finite_mean([r["trajectory_random_hit_rate"] for r in rs]),
                "mean_expected_regret": al.finite_mean([r["trajectory_random_expected_regret"] for r in rs]),
            }
    return {"rows": rows, "summary": summary}


def audit(ctx):
    baseline = random_baseline(ctx)
    rows = []
    pairwise_by_score = defaultdict(list)
    top_specs = [s for s in score_registry.topk_scores()]
    for score in top_specs:
        if score["diagnostic_ceiling"] and score["score"] != "target_grouped_diagnostic_ceiling":
            continue
        for tid, rs in sorted(ctx["by_traj"].items()):
            auc = pairwise_auc(rs, score)
            pairwise_by_score[score["score"]].append(auc)
            ordered = order_rows(rs, score)
            oracle = _oracle_utility(rs)
            for rule, draw_spec in schema.TOPK_RULES:
                n = _draw_count(len(ordered), draw_spec)
                chosen = ordered[:n]
                best_chosen = max([float(r["target_utility_score"]) for r in chosen], default=math.nan)
                for label in schema.LABELS:
                    hit = _topk_hit(chosen, label)
                    base = sum(int(r[label]) for r in rs) / len(rs)
                    rows.append({
                        "score": score["score"],
                        "selection_rule": rule,
                        "label": label,
                        "trajectory_id": tid,
                        "n_candidates": len(rs),
                        "n_selected_by_score": n,
                        "pairwise_auc_vs_target_utility": auc,
                        "hit_rate": hit,
                        "trajectory_random_baseline": base,
                        "enrichment_ratio": hit / base if base > 0 else "",
                        "regret_vs_target_oracle": oracle - best_chosen if al.finite(best_chosen) else "",
                        "non_source_only": score["non_source_only"],
                        "diagnostic_ceiling": score["diagnostic_ceiling"],
                    })
    summary_rows = []
    for score in {r["score"] for r in rows}:
        srows = [r for r in rows if r["score"] == score]
        for rule, _ in schema.TOPK_RULES:
            for label in schema.LABELS:
                rs = [r for r in srows if r["selection_rule"] == rule and r["label"] == label]
                summary_rows.append({
                    "score": score,
                    "selection_rule": rule,
                    "label": label,
                    "n_trajectories": len(rs),
                    "mean_pairwise_auc_vs_target_utility": al.finite_mean(
                        [r["pairwise_auc_vs_target_utility"] for r in rs]),
                    "mean_hit_rate": al.finite_mean([r["hit_rate"] for r in rs]),
                    "mean_random_baseline": al.finite_mean([r["trajectory_random_baseline"] for r in rs]),
                    "mean_enrichment_ratio": al.finite_mean([r["enrichment_ratio"] for r in rs]),
                    "mean_regret_vs_target_oracle": al.finite_mean([r["regret_vs_target_oracle"] for r in rs]),
                    "non_source_only": rs[0]["non_source_only"] if rs else "",
                    "diagnostic_ceiling": rs[0]["diagnostic_ceiling"] if rs else "",
                })
    summary = {(r["score"], r["selection_rule"], r["label"]): r for r in summary_rows}
    return {"rows": rows, "summary_rows": summary_rows, "summary": summary, "random_baseline": baseline}
