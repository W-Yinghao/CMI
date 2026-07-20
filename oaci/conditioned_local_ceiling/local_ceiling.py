"""C48 local source-space Bayes ceiling diagnostics."""
from __future__ import annotations

import math

import numpy as np

from . import artifact_loader as al
from . import schema


def _entropy01_from_p(p):
    p = np.asarray(p, dtype=float)
    out = np.zeros_like(p)
    mask = (p > 1e-12) & (p < 1.0 - 1e-12)
    out[mask] = -(p[mask] * np.log2(p[mask]) + (1.0 - p[mask]) * np.log2(1.0 - p[mask]))
    return out


def _distance_matrix(space, rows):
    idx = np.asarray([int(r["source_idx"]) for r in rows], dtype=int)
    z = space["z"][idx, :]
    sq = np.sum(z * z, axis=1, keepdims=True)
    d2 = np.maximum(sq + sq.T - 2.0 * z.dot(z.T), 0.0)
    dist = np.sqrt(d2)
    np.fill_diagonal(dist, np.inf)
    return dist


def _knn_purity(dist, labels, k, base_rate):
    n = len(labels)
    if n <= 1:
        return np.full(n, base_rate, dtype=float), np.zeros(n, dtype=int), None
    kk = min(int(k), n - 1)
    neigh = np.argpartition(dist, kth=kk - 1, axis=1)[:, :kk]
    purity = labels[neigh].mean(axis=1)
    return purity.astype(float), np.full(n, kk, dtype=int), neigh


def _epsilon_purity(dist, labels, radius, base_rate):
    n = len(labels)
    if n <= 1:
        return np.full(n, base_rate, dtype=float), np.zeros(n, dtype=int), None
    mask = dist <= float(radius) + 1e-12
    counts = mask.sum(axis=1).astype(int)
    sums = mask.dot(labels.astype(float))
    purity = np.full(n, base_rate, dtype=float)
    nz = counts > 0
    purity[nz] = sums[nz] / counts[nz]
    return purity, counts, mask


def _bayes_top1_expected_hit(purity, labels):
    if len(labels) == 0:
        return math.nan, math.nan, 0, math.nan
    max_p = float(np.max(purity))
    tied = np.where(np.abs(purity - max_p) <= 1e-12)[0]
    expected_hit = float(np.mean(labels[tied])) if len(tied) else math.nan
    tie_fraction = len(tied) / len(labels)
    return expected_hit, max_p, len(tied), tie_fraction


def _permutation_top1_hit(labels, structure, kind, base_rate, rng):
    n = len(labels)
    if n <= 1 or structure is None:
        return base_rate
    vals = []
    for _ in range(schema.PERMUTATION_REPS):
        perm = rng.permutation(labels)
        if kind == "knn":
            purity = perm[structure].mean(axis=1)
        else:
            counts = structure.sum(axis=1).astype(int)
            sums = structure.dot(perm.astype(float))
            purity = np.full(n, base_rate, dtype=float)
            nz = counts > 0
            purity[nz] = sums[nz] / counts[nz]
        hit, _, _, _ = _bayes_top1_expected_hit(purity, labels)
        vals.append(hit)
    return float(np.mean(vals)) if vals else math.nan


def _group_label_row(scope, group_key, source_space, neighborhood, rows, labels, purity, counts, c47_actual):
    n = len(rows)
    positives = int(np.sum(labels))
    base_rate = positives / n if n else math.nan
    hit, max_purity, tie_count, tie_fraction = _bayes_top1_expected_hit(purity, labels)
    entropy = _entropy01_from_p(purity)
    gain = hit - base_rate if al.finite(hit) and al.finite(base_rate) else math.nan
    enrichment = hit / base_rate if al.finite(hit) and base_rate > 0 else math.nan
    return {
        "group_scope": scope,
        "group_key": group_key,
        "source_space": source_space,
        "neighborhood": neighborhood["neighborhood"],
        "neighborhood_kind": neighborhood["kind"],
        "neighborhood_value": neighborhood["value"],
        "label": neighborhood["label"],
        "n_candidates": n,
        "n_label_positive": positives,
        "group_random_top1_baseline": base_rate,
        "mean_neighbor_count": float(np.mean(counts)) if len(counts) else math.nan,
        "empty_neighborhood_fraction": float(np.mean(counts == 0)) if len(counts) else math.nan,
        "mean_local_purity": float(np.mean(purity)) if len(purity) else math.nan,
        "median_local_purity": float(np.median(purity)) if len(purity) else math.nan,
        "max_local_purity": max_purity,
        "mean_conditional_entropy_bits": float(np.mean(entropy)) if len(entropy) else math.nan,
        "local_bayes_top1_expected_hit": hit,
        "local_bayes_top1_gain_vs_random": gain,
        "local_bayes_top1_enrichment": enrichment,
        "local_bayes_top1_tie_count": tie_count,
        "local_bayes_top1_tie_fraction": tie_fraction,
        "c47_actual_strict_source_top1_hit": c47_actual,
        "gap_vs_c47_actual_top1": hit - c47_actual if al.finite(hit) and al.finite(c47_actual) else math.nan,
        "self_label_excluded": 1,
        "target_labels_diagnostic_only": 1,
        "no_candidate_id_emitted": 1,
    }


def _neighborhood_specs(eps_summary, source_space):
    specs = []
    for k in schema.K_VALUES:
        specs.append({"neighborhood": f"knn_{k}", "kind": "knn", "value": k})
    for q in schema.EPSILON_QUANTILES:
        specs.append({
            "neighborhood": f"eps_q{int(q * 100):02d}",
            "kind": "epsilon",
            "value": eps_summary[source_space][q],
        })
    return specs


def evaluate(ctx, spaces, eps_summary):
    rows = []
    for source_space, space in spaces.items():
        n_specs = _neighborhood_specs(eps_summary, source_space)
        for scope in schema.GROUP_SCOPES:
            for group_key, group in sorted(al.group_rows(ctx, scope).items()):
                dist = _distance_matrix(space, group)
                for label in schema.LABELS:
                    labels = np.asarray([int(r[label]) for r in group], dtype=int)
                    base_rate = float(np.mean(labels)) if len(labels) else math.nan
                    c47_actual = al.c47_actual_top1(ctx, scope, label)
                    for spec in n_specs:
                        spec_label = dict(spec)
                        spec_label["label"] = label
                        if spec["kind"] == "knn":
                            purity, counts, structure = _knn_purity(dist, labels, int(spec["value"]), base_rate)
                        else:
                            purity, counts, structure = _epsilon_purity(dist, labels, float(spec["value"]), base_rate)
                        rows.append(_group_label_row(
                            scope, group_key, source_space, spec_label, group, labels, purity, counts, c47_actual))
    return rows


def summarize(rows):
    out = []
    for scope in schema.GROUP_SCOPES:
        scope_rows = [r for r in rows if r["group_scope"] == scope]
        for source_space in sorted({r["source_space"] for r in scope_rows}):
            for label in schema.LABELS:
                for neighborhood in sorted({r["neighborhood"] for r in scope_rows}):
                    rs = [
                        r for r in scope_rows
                        if r["source_space"] == source_space and r["label"] == label and
                        r["neighborhood"] == neighborhood
                    ]
                    if not rs:
                        continue
                    top1 = al.finite_mean([r["local_bayes_top1_expected_hit"] for r in rs])
                    base = al.finite_mean([r["group_random_top1_baseline"] for r in rs])
                    actual = al.finite_mean([r["c47_actual_strict_source_top1_hit"] for r in rs])
                    out.append({
                        "group_scope": scope,
                        "source_space": source_space,
                        "neighborhood": neighborhood,
                        "neighborhood_kind": rs[0]["neighborhood_kind"],
                        "neighborhood_value": rs[0]["neighborhood_value"],
                        "label": label,
                        "n_groups": len(rs),
                        "mean_n_candidates": al.finite_mean([r["n_candidates"] for r in rs]),
                        "mean_group_base_rate": base,
                        "mean_neighbor_count": al.finite_mean([r["mean_neighbor_count"] for r in rs]),
                        "mean_empty_neighborhood_fraction": al.finite_mean(
                            [r["empty_neighborhood_fraction"] for r in rs]),
                        "mean_local_purity": al.finite_mean([r["mean_local_purity"] for r in rs]),
                        "mean_max_local_purity": al.finite_mean([r["max_local_purity"] for r in rs]),
                        "mean_conditional_entropy_bits": al.finite_mean(
                            [r["mean_conditional_entropy_bits"] for r in rs]),
                        "mean_local_bayes_top1_hit": top1,
                        "mean_random_top1_baseline": base,
                        "mean_local_bayes_gain_vs_random": top1 - base if al.finite(top1) and al.finite(base)
                        else math.nan,
                        "mean_local_bayes_enrichment": top1 / base if al.finite(top1) and base > 0 else math.nan,
                        "mean_c47_actual_strict_source_top1_hit": actual,
                        "mean_gap_vs_c47_actual_top1": top1 - actual if al.finite(top1) and al.finite(actual)
                        else math.nan,
                        "target_labels_diagnostic_only": 1,
                    })
    return out


def best_rows(summary_rows):
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
                    float(r["mean_gap_vs_c47_actual_top1"]) if al.finite(r["mean_gap_vs_c47_actual_top1"]) else -1.0,
                ),
            )
            nr = dict(best)
            nr["best_kind"] = "best_local_bayes_ceiling"
            out.append(nr)
    return out


def _purity_for_spec(dist, labels, neighborhood, base_rate):
    if neighborhood["neighborhood_kind"] == "knn":
        return _knn_purity(dist, labels, int(float(neighborhood["neighborhood_value"])), base_rate)
    return _epsilon_purity(dist, labels, float(neighborhood["neighborhood_value"]), base_rate)


def _permutation_for_best_row(ctx, spaces, row, rng):
    source_space = row["source_space"]
    scope = row["group_scope"]
    label = row["label"]
    hits = []
    for _, group in sorted(al.group_rows(ctx, scope).items()):
        labels = np.asarray([int(r[label]) for r in group], dtype=int)
        base_rate = float(np.mean(labels)) if len(labels) else math.nan
        dist = _distance_matrix(spaces[source_space], group)
        _, _, structure = _purity_for_spec(dist, labels, row, base_rate)
        hits.append(_permutation_top1_hit(labels, structure, row["neighborhood_kind"], base_rate, rng))
    return al.finite_mean(hits)


def attach_permutation_baselines(ctx, spaces, best):
    rng = np.random.default_rng(schema.PERMUTATION_SEED)
    out = []
    for row in best:
        nr = dict(row)
        perm = _permutation_for_best_row(ctx, spaces, nr, rng)
        hit = float(nr["mean_local_bayes_top1_hit"]) if al.finite(nr.get("mean_local_bayes_top1_hit")) else math.nan
        nr["mean_permutation_local_bayes_top1_hit"] = perm
        nr["mean_local_bayes_gap_vs_permutation"] = hit - perm if al.finite(hit) and al.finite(perm) else math.nan
        nr["permutation_reps"] = schema.PERMUTATION_REPS
        nr["permutation_seed"] = schema.PERMUTATION_SEED
        out.append(nr)
    return out


def audit(ctx, spaces, eps_summary):
    rows = evaluate(ctx, spaces, eps_summary)
    summary_rows = summarize(rows)
    best = attach_permutation_baselines(ctx, spaces, best_rows(summary_rows))
    return {
        "rows": rows,
        "summary_rows": summary_rows,
        "best_rows": best,
        "summary": {
            "n_group_detail_rows": len(rows),
            "n_summary_rows": len(summary_rows),
            "n_best_rows": len(best),
            "self_label_excluded": True,
            "empty_neighborhood_fallback": "group_base_rate",
        },
    }
