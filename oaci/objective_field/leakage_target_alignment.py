"""C41 leakage-target rank alignment metrics."""
from __future__ import annotations

import math
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import schema


FIELDS = (
    "selection_leakage_point",
    "audit_leakage_point",
    "R_src",
    "balanced_err",
    "source_guard_worst_bacc",
    "source_guard_worst_nll",
    "source_guard_worst_ece",
    "source_audit_worst_bacc",
    "source_audit_worst_nll",
    "source_audit_worst_ece",
)


def _oriented_value(field, value):
    v = float(value)
    return -v if field in schema.LOWER_IS_BETTER_FIELDS else v


def _rankdata(vals):
    order = sorted(range(len(vals)), key=lambda i: vals[i])
    ranks = [0.0] * len(vals)
    i = 0
    while i < len(vals):
        j = i
        while j + 1 < len(vals) and vals[order[j + 1]] == vals[order[i]]:
            j += 1
        rank = (i + j) / 2.0
        for k in range(i, j + 1):
            ranks[order[k]] = rank
        i = j + 1
    return np.asarray(ranks, dtype=float)


def spearman(x, y):
    if len(x) < 3:
        return math.nan
    rx, ry = _rankdata(x), _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def pairwise_auc(oriented_field, utility):
    n = len(oriented_field)
    total = correct = ties = 0
    for i in range(n):
        for j in range(i + 1, n):
            du = utility[i] - utility[j]
            df = oriented_field[i] - oriented_field[j]
            if abs(du) <= 1e-12:
                continue
            total += 1
            prod = du * df
            if prod > 0:
                correct += 1
            elif abs(prod) <= 1e-12:
                ties += 1
    return ((correct + 0.5 * ties) / total if total else math.nan, total)


def align(ctx):
    rows = []
    by_field = defaultdict(list)
    for tid, rs in sorted(ctx["by_traj"].items()):
        seed, target, level, regime = tid.split("|")
        utility = [float(r["target_utility_score"]) for r in rs if al.finite(r["target_utility_score"])]
        if len(utility) < 3:
            continue
        for field in FIELDS:
            pairs = [(r, _oriented_value(field, r[field]), float(r["target_utility_score"])) for r in rs
                     if al.finite(r.get(field)) and al.finite(r.get("target_utility_score"))]
            if len(pairs) < 3:
                continue
            fvals = [p[1] for p in pairs]
            uvals = [p[2] for p in pairs]
            auc, n_pairs = pairwise_auc(fvals, uvals)
            rho = spearman(fvals, uvals)
            row = {
                "trajectory_id": tid,
                "seed": seed,
                "target": target,
                "level": level,
                "regime": regime,
                "field": field,
                "field_orientation": "lower_better" if field in schema.LOWER_IS_BETTER_FIELDS else "higher_better",
                "n_candidates": len(pairs),
                "spearman_oriented_field_vs_target_utility": rho,
                "pairwise_auc_oriented_field_ranks_target_utility": auc,
                "n_pairwise_comparisons": n_pairs,
                "sign_class": _sign_class(auc),
                "target_labels_diagnostic_only": 1,
            }
            rows.append(row)
            by_field[field].append(row)
    summary_rows = []
    for field, frs in sorted(by_field.items()):
        aucs = [al.as_float(r["pairwise_auc_oriented_field_ranks_target_utility"]) for r in frs]
        rhos = [al.as_float(r["spearman_oriented_field_vs_target_utility"]) for r in frs]
        summary_rows.append({
            "field": field,
            "n_trajectories": len(frs),
            "mean_spearman": al.finite_mean(rhos),
            "median_spearman": al.finite_median(rhos),
            "mean_pairwise_auc": al.finite_mean(aucs),
            "median_pairwise_auc": al.finite_median(aucs),
            "alignment_fraction": sum(1 for a in aucs if al.finite(a) and a >= schema.ALIGNMENT_AUC_HIGH) / len(aucs),
            "anti_alignment_fraction": sum(1 for a in aucs if al.finite(a) and a <= schema.ALIGNMENT_AUC_LOW) / len(aucs),
            "decoupled_fraction": sum(1 for a in aucs if al.finite(a) and schema.ALIGNMENT_AUC_LOW < a < schema.ALIGNMENT_AUC_HIGH) / len(aucs),
        })
    summary = {r["field"]: r for r in summary_rows}
    return {"rows": rows, "summary_rows": summary_rows, "summary": summary}


def _sign_class(auc):
    if not al.finite(auc):
        return "unavailable"
    if auc >= schema.ALIGNMENT_AUC_HIGH:
        return "aligned"
    if auc <= schema.ALIGNMENT_AUC_LOW:
        return "anti_aligned"
    return "decoupled"
