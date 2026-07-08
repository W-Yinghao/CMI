"""C44 source-objective effective dimension and conflict audit."""
from __future__ import annotations

import itertools

import numpy as np

from . import artifact_loader as al
from . import objective_registry


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


def _spearman(x, y):
    rx, ry = _rankdata(x), _rankdata(y)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _effective_rank(evals):
    vals = np.asarray([v for v in evals if v > 1e-12], dtype=float)
    p = vals / vals.sum()
    return float(np.exp(-np.sum(p * np.log(p)))) if len(p) else None


def _matrix(ctx, specs):
    rows = ctx["registry"]
    mat = np.asarray([[objective_registry.oriented(r, s) for s in specs] for r in rows], dtype=float)
    ranked = np.vstack([_rankdata(mat[:, j]) for j in range(mat.shape[1])]).T
    ranked = (ranked - ranked.mean(axis=0)) / (ranked.std(axis=0) + 1e-12)
    return ranked


def audit(ctx):
    specs = objective_registry.source_pareto_specs(ctx)
    mat = _matrix(ctx, specs)
    corr = np.corrcoef(mat, rowvar=False)
    evals = np.linalg.eigvalsh(corr)
    evals = np.sort(np.maximum(evals, 0.0))[::-1]
    pca = evals / evals.sum()
    eff = _effective_rank(evals)
    rows = [{
        "scope": "all_source_pareto_objectives",
        "n_objectives": len(specs),
        "effective_rank": eff,
        "pca_var1": float(pca[0]),
        "pca_var2": float(pca[1]) if len(pca) > 1 else "",
        "pca_var3": float(pca[2]) if len(pca) > 2 else "",
        "pca_cum3": float(pca[:3].sum()),
        "mean_abs_correlation": float(np.mean(np.abs(corr[np.triu_indices_from(corr, 1)]))),
        "negative_pair_fraction": float(np.mean(corr[np.triu_indices_from(corr, 1)] < 0)),
        "redundancy_not_issue": int(eff >= 3.0),
    }]
    families = sorted({s["family"] for s in specs})
    family_rows = []
    for fa, fb in itertools.product(families, families):
        vals = []
        for i, sa in enumerate(specs):
            for j, sb in enumerate(specs):
                if i >= j and fa == fb:
                    continue
                if sa["family"] == fa and sb["family"] == fb:
                    vals.append(float(corr[i, j]))
        family_rows.append({
            "family_a": fa,
            "family_b": fb,
            "n_pairs": len(vals),
            "mean_spearman": al.finite_mean(vals),
            "mean_abs_spearman": al.finite_mean([abs(v) for v in vals]),
            "negative_fraction": al.finite_mean([int(v < 0) for v in vals]),
            "opposition": int((al.finite_mean(vals) or 0.0) < -0.10),
        })
    leakage_rank = next((r for r in family_rows if r["family_a"] == "leakage" and r["family_b"] == "source_rank"),
                        None)
    summary = {
        "n_objectives": len(specs),
        "effective_rank": eff,
        "pca_var1": rows[0]["pca_var1"],
        "pca_cum3": rows[0]["pca_cum3"],
        "negative_pair_fraction": rows[0]["negative_pair_fraction"],
        "mean_abs_correlation": rows[0]["mean_abs_correlation"],
        "leakage_rank_mean_spearman": leakage_rank["mean_spearman"] if leakage_rank else None,
    }
    return {"rows": rows, "family_rows": family_rows, "summary": summary}
