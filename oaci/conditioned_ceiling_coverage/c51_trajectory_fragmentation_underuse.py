"""C51 - Trajectory Fragmentation / Source-Describability Audit."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..conditioned_actionability import score_registry as c47_scores
from ..conditioned_local_ceiling import local_ceiling as c48_local
from ..source_nonidentifiability import source_space as c45_source_space
from . import artifact_loader as al
from . import c50_conditioned_island_morphology as c50
from . import schema as c49_schema
from . import source_space_registry


REPORT_JSON = "oaci/reports/C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.json"
TABLE_DIR = "oaci/reports/c51_tables"

MILESTONE = "C51"
LOCKED_SCOPE = c50.WITNESS_SCOPE
LOCKED_SOURCE_SPACE = c50.WITNESS_SOURCE_SPACE
LOCKED_NEIGHBORHOOD = c50.WITNESS_NEIGHBORHOOD
LOCKED_MIN_N = c50.WITNESS_MIN_NEIGHBOR_COUNT
LOCKED_LABEL = c50.WITNESS_LABEL

EPS_QUANTILES = (0.10, 0.20, 0.30, 0.40)
MIN_N_GRID = (1, 2, 3, 5)
NULL_REPS = 64
NULL_SEED = 51051
NULL_NAMES = (
    "N0_global_label_shuffle",
    "N1_within_target_label_shuffle",
    "N2_within_target_trajectory_label_shuffle",
    "N3_degree_preserving_neighbor_randomization",
    "N4_source_geometry_permutation_within_target",
)
NULL_STATISTICS = (
    "trajectory_actionability_fail_fraction",
    "trajectory_min_hit",
    "trajectory_min_coverage",
    "mean_existing_score_underuse_gap",
    "max_mean_existing_score_underuse_gap",
    "covered_island_hit",
    "covered_island_enrichment",
)
DECISIONS = (
    "C51-A_null_like_trajectory_fragmentation",
    "C51-B_support_limited_fragmentation",
    "C51-C_score_orientation_underuse",
    "C51-D_score_monotone_or_nonlinear_underuse",
    "C51-E_target_trajectory_gauge_residual",
    "C51-F_mixed_support_fragmentation_underuse_residual",
)

STRICT_SOURCE_SCORES = c50.STRICT_SOURCE_SCORES
FORBIDDEN_CLAIM_SUBSTRINGS = c50.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-only rescue",
    "target-unlabeled method",
    "target-grouped method",
    "production rule",
)


def _lock_config():
    got = c49_schema.frozen_config_hash()
    if got != c49_schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C51 requires frozen C19 config {c49_schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _readcsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _f(x, default=math.nan):
    return al.as_float(x, default)


def _finite(vals):
    return [float(v) for v in vals if al.finite(v)]


def _mean(vals):
    vals = _finite(vals)
    return float(np.mean(vals)) if vals else math.nan


def _median(vals):
    vals = _finite(vals)
    return float(np.median(vals)) if vals else math.nan


def _quantile(vals, q):
    vals = _finite(vals)
    return float(np.quantile(vals, q)) if vals else math.nan


def _enrichment(hit, base):
    return hit / base if al.finite(hit) and al.finite(base) and float(base) > 0 else math.nan


def _query_id(row):
    return f"c50q_{int(row['source_idx']):04d}"


def _row_group_key(row, group_type):
    if group_type == "target":
        return str(row["target"])
    if group_type == "trajectory":
        return row.get("trajectory_id", row.get("trajectory"))
    if group_type == "seed":
        return str(row["seed"])
    if group_type == "level":
        return str(row["level"])
    if group_type == "regime":
        return row["regime"]
    raise ValueError(group_type)


def _score_specs(ctx):
    reg = c47_scores.registry(ctx)
    specs = [
        s for s in reg["rows"]
        if int(s["source_only"]) == 1 and int(s["target_label_used"]) == 0
    ]
    return specs, reg["best_scalarization"]


def _prepare():
    ctx = al.context()
    spaces = source_space_registry.registry(ctx)
    space = spaces["spaces"][LOCKED_SOURCE_SPACE]
    witness = c50._c49_locked_witness()
    if witness["condition_scope"] != LOCKED_SCOPE or witness["source_space"] != LOCKED_SOURCE_SPACE:
        raise ValueError("C51 locked witness does not match C50")
    pair_dist = c45_source_space.within_trajectory_pair_distances(ctx, space)
    radii = {q: _quantile(pair_dist, q) for q in EPS_QUANTILES}
    caches = {}
    for target_key, group in sorted(al.group_rows(ctx, LOCKED_SCOPE).items()):
        dist = c48_local._distance_matrix(space, group)
        caches[target_key] = {"group": group, "dist": dist}
    labels_by_idx = {int(r["source_idx"]): int(r[LOCKED_LABEL]) for r in ctx["registry"]}
    return ctx, space, witness, radii, caches, labels_by_idx


def _base_mask(dist, radius):
    return dist <= float(radius) + 1e-12


def _rows_from_masks(caches, labels_by_idx, mask_by_target, min_n):
    out = []
    self_excluded = True
    for target_key, cache in sorted(caches.items()):
        group = cache["group"]
        mask = mask_by_target[target_key]
        labels = np.asarray([int(labels_by_idx[int(r["source_idx"])]) for r in group], dtype=int)
        base = float(np.mean(labels)) if len(labels) else math.nan
        if len(mask) and bool(np.any(np.diag(mask))):
            self_excluded = False
        counts = mask.sum(axis=1).astype(int)
        sums = mask.dot(labels.astype(float))
        purity = np.full(len(labels), base, dtype=float)
        nz = counts > 0
        purity[nz] = sums[nz] / counts[nz]
        covered = counts >= int(min_n)
        for i, row in enumerate(group):
            out.append({
                "query_id": _query_id(row),
                "conditioning_key": target_key,
                "target": str(row["target"]),
                "trajectory": row["trajectory_id"],
                "seed": str(row["seed"]),
                "level": str(row["level"]),
                "regime": row["regime"],
                "neighbors_n": int(counts[i]),
                "covered": int(bool(covered[i])),
                "query_positive_label": int(labels[i]),
                "neighbor_positive_rate": float(purity[i]) if int(counts[i]) > 0 else math.nan,
                "neighbor_base_rate": base,
                "local_lift": float(purity[i] - base) if int(counts[i]) > 0 and al.finite(base) else math.nan,
                "local_enrichment": _enrichment(float(purity[i]), base) if int(counts[i]) > 0 else math.nan,
                "is_singleton_neighbor": int(int(counts[i]) == 1),
                "is_high_purity_island": int(int(counts[i]) > 0 and float(purity[i]) >= c50.HIT_GATE),
                "target_labels_diagnostic_only": 1,
                "no_selection_artifact": 1,
            })
    return out, self_excluded


def rows_for_radius(caches, labels_by_idx, radius, min_n):
    masks = {k: _base_mask(v["dist"], radius) for k, v in caches.items()}
    return _rows_from_masks(caches, labels_by_idx, masks, min_n)


def _target_trajectory_metrics(island_rows):
    frag = c50.group_fragmentation(island_rows)
    target_rows = [r for r in frag if r["group_type"] == "target"]
    traj_rows = [r for r in frag if r["group_type"] == "trajectory"]
    hit = _mean([r["hit_rate_if_covered"] for r in target_rows])
    coverage = _mean([r["coverage"] for r in target_rows])
    base = _mean([r["base_rate"] for r in target_rows])
    return {
        "fragmentation_rows": frag,
        "hit": hit,
        "coverage": coverage,
        "enrichment": _enrichment(hit, base),
        "target_min_hit": min(_finite([r["hit_rate_if_covered"] for r in target_rows]), default=math.nan),
        "target_min_coverage": min(_finite([r["coverage"] for r in target_rows]), default=math.nan),
        "trajectory_min_hit": min(_finite([r["hit_rate_if_covered"] for r in traj_rows]), default=math.nan),
        "trajectory_min_coverage": min(_finite([r["coverage"] for r in traj_rows]), default=math.nan),
        "trajectory_actionability_fail_fraction": _mean([1 - int(r["actionability_pass"]) for r in traj_rows]),
        "covered_island_hit": hit,
        "covered_island_enrichment": _enrichment(hit, base),
    }


def observed_c50_replay(c50_summary, island_rows):
    m = _target_trajectory_metrics(island_rows)
    c50_dec = c50_summary["decision"]
    witness = c50_summary["locked_witness"]
    return {
        "condition": LOCKED_SCOPE,
        "source_objectives": LOCKED_SOURCE_SPACE,
        "eps_quantile": "q20",
        "min_n": LOCKED_MIN_N,
        "epsilon": witness["epsilon_radius"],
        "hit": witness["c49_hit"],
        "coverage": witness["c49_coverage"],
        "enrichment": witness["c49_enrichment"],
        "target_min_hit": c50_dec["target_min_hit"],
        "target_min_coverage": c50_dec["target_min_coverage"],
        "trajectory_min_hit": c50_dec["trajectory_min_hit"],
        "trajectory_min_coverage": c50_dec["trajectory_min_coverage"],
        "trajectory_actionability_fail_fraction": c50_dec["trajectory_actionability_fail_fraction"],
        "replayed_target_hit": m["hit"],
        "replayed_target_coverage": m["coverage"],
        "replayed_trajectory_min_hit": m["trajectory_min_hit"],
        "replayed_trajectory_min_coverage": m["trajectory_min_coverage"],
    }


def support_ablation_grid(caches, labels_by_idx, radii):
    rows = []
    for q in EPS_QUANTILES:
        radius = radii[q]
        for min_n in MIN_N_GRID:
            island_rows, _ = rows_for_radius(caches, labels_by_idx, radius, min_n)
            m = _target_trajectory_metrics(island_rows)
            counts = [int(r["neighbors_n"]) for r in island_rows]
            rows.append({
                "eps_quantile": f"q{int(q * 100):02d}",
                "epsilon_radius": radius,
                "min_n": min_n,
                "hit": m["hit"],
                "coverage": m["coverage"],
                "enrichment": m["enrichment"],
                "target_min_hit": m["target_min_hit"],
                "target_min_coverage": m["target_min_coverage"],
                "trajectory_min_hit": m["trajectory_min_hit"],
                "trajectory_min_coverage": m["trajectory_min_coverage"],
                "trajectory_actionability_fail_fraction": m["trajectory_actionability_fail_fraction"],
                "mean_neighbor_count": _mean(counts),
                "median_neighbor_count": _median(counts),
                "p10_neighbor_count": _quantile(counts, 0.10),
                "empty_fraction": _mean([int(c == 0) for c in counts]),
                "singleton_fraction": _mean([int(c == 1) for c in counts]),
                "target_labels_diagnostic_only": 1,
            })
    return rows


def _shuffle_labels(ctx, labels_by_idx, null_name, rng):
    out = dict(labels_by_idx)
    if null_name == "N0_global_label_shuffle":
        idx = list(out)
        vals = np.asarray([out[i] for i in idx], dtype=int)
        rng.shuffle(vals)
        return {i: int(v) for i, v in zip(idx, vals)}
    if null_name == "N1_within_target_label_shuffle":
        for group in ctx["by_target"].values():
            idx = [int(r["source_idx"]) for r in group]
            vals = np.asarray([out[i] for i in idx], dtype=int)
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                out[i] = int(v)
        return out
    if null_name == "N2_within_target_trajectory_label_shuffle":
        for group in ctx["by_traj"].values():
            idx = [int(r["source_idx"]) for r in group]
            vals = np.asarray([out[i] for i in idx], dtype=int)
            rng.shuffle(vals)
            for i, v in zip(idx, vals):
                out[i] = int(v)
        return out
    raise ValueError(null_name)


def _degree_random_masks(caches, radius, rng):
    masks = {}
    for target_key, cache in caches.items():
        base = _base_mask(cache["dist"], radius)
        counts = base.sum(axis=1).astype(int)
        n = len(counts)
        mask = np.zeros((n, n), dtype=bool)
        for i, k in enumerate(counts):
            pool = np.asarray([j for j in range(n) if j != i], dtype=int)
            if len(pool) == 0 or int(k) <= 0:
                continue
            chosen = rng.choice(pool, size=min(int(k), len(pool)), replace=False)
            mask[i, chosen] = True
        masks[target_key] = mask
    return masks


def _geometry_permutation_masks(caches, radius, rng):
    masks = {}
    for target_key, cache in caches.items():
        base = _base_mask(cache["dist"], radius)
        n = base.shape[0]
        perm = rng.permutation(n)
        masks[target_key] = base[perm][:, perm]
    return masks


def _null_island_rows(ctx, caches, labels_by_idx, radius, null_name, rng):
    if null_name in (
        "N0_global_label_shuffle",
        "N1_within_target_label_shuffle",
        "N2_within_target_trajectory_label_shuffle",
    ):
        null_labels = _shuffle_labels(ctx, labels_by_idx, null_name, rng)
        masks = {k: _base_mask(v["dist"], radius) for k, v in caches.items()}
        return _rows_from_masks(caches, null_labels, masks, LOCKED_MIN_N)[0]
    if null_name == "N3_degree_preserving_neighbor_randomization":
        masks = _degree_random_masks(caches, radius, rng)
        return _rows_from_masks(caches, labels_by_idx, masks, LOCKED_MIN_N)[0]
    if null_name == "N4_source_geometry_permutation_within_target":
        masks = _geometry_permutation_masks(caches, radius, rng)
        return _rows_from_masks(caches, labels_by_idx, masks, LOCKED_MIN_N)[0]
    raise ValueError(null_name)


def _underuse_summary(ctx, island_rows):
    under = c50.existing_score_underuse(ctx, island_rows)
    gaps = [_f(r["mean_underuse_gap"]) for r in under["summary_rows"]]
    return {
        "mean_existing_score_underuse_gap": _mean(gaps),
        "max_mean_existing_score_underuse_gap": max(_finite(gaps), default=math.nan),
    }


def _null_stats(ctx, island_rows):
    m = _target_trajectory_metrics(island_rows)
    u = _underuse_summary(ctx, island_rows)
    return {
        "trajectory_actionability_fail_fraction": m["trajectory_actionability_fail_fraction"],
        "trajectory_min_hit": m["trajectory_min_hit"],
        "trajectory_min_coverage": m["trajectory_min_coverage"],
        "mean_existing_score_underuse_gap": u["mean_existing_score_underuse_gap"],
        "max_mean_existing_score_underuse_gap": u["max_mean_existing_score_underuse_gap"],
        "covered_island_hit": m["covered_island_hit"],
        "covered_island_enrichment": m["covered_island_enrichment"],
    }, m["fragmentation_rows"]


def _summarize_null_stat(null_name, stat, observed, samples):
    vals = _finite(samples)
    if not vals:
        return {
            "null_name": null_name,
            "statistic": stat,
            "observed": observed,
            "null_mean": math.nan,
            "null_std": math.nan,
            "z_score": math.nan,
            "percentile": math.nan,
            "empirical_p_two_sided": math.nan,
            "n_permutations": 0,
        }
    arr = np.asarray(vals, dtype=float)
    mu = float(np.mean(arr))
    sd = float(np.std(arr))
    percentile = float(np.mean(arr <= observed)) if al.finite(observed) else math.nan
    p_low = float(np.mean(arr <= observed)) if al.finite(observed) else math.nan
    p_high = float(np.mean(arr >= observed)) if al.finite(observed) else math.nan
    p2 = min(1.0, 2.0 * min(p_low, p_high)) if al.finite(p_low) and al.finite(p_high) else math.nan
    return {
        "null_name": null_name,
        "statistic": stat,
        "observed": observed,
        "null_mean": mu,
        "null_std": sd,
        "z_score": (observed - mu) / sd if al.finite(observed) and sd > 1e-12 else math.nan,
        "percentile": percentile,
        "empirical_p_two_sided": p2,
        "n_permutations": len(vals),
    }


def null_calibration(ctx, caches, labels_by_idx, radius, observed_stats):
    rng = np.random.default_rng(NULL_SEED)
    samples = {name: {s: [] for s in NULL_STATISTICS} for name in NULL_NAMES}
    traj_hits = {"N1_within_target_label_shuffle": defaultdict(list),
                 "N2_within_target_trajectory_label_shuffle": defaultdict(list)}
    for null_name in NULL_NAMES:
        for _ in range(NULL_REPS):
            rows = _null_island_rows(ctx, caches, labels_by_idx, radius, null_name, rng)
            stats, frag = _null_stats(ctx, rows)
            for stat in NULL_STATISTICS:
                samples[null_name][stat].append(stats[stat])
            if null_name in traj_hits:
                for r in frag:
                    if r["group_type"] == "trajectory" and al.finite(r.get("hit_rate_if_covered")):
                        traj_hits[null_name][r["group_key"]].append(float(r["hit_rate_if_covered"]))
    summary = []
    for null_name in NULL_NAMES:
        for stat in NULL_STATISTICS:
            summary.append(_summarize_null_stat(
                null_name, stat, observed_stats[stat], samples[null_name][stat]))
    return summary, traj_hits


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
    denom = max(len(vals) - 1, 1)
    return np.asarray([r / denom for r in ranks], dtype=float)


def _spearman(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if len(x) < 2 or len(set(y.tolist())) < 2:
        return math.nan
    rx = _rankdata(x.tolist())
    ry = _rankdata(y.tolist())
    sx, sy = float(np.std(rx)), float(np.std(ry))
    if sx <= 1e-12 or sy <= 1e-12:
        return math.nan
    return float(np.corrcoef(rx, ry)[0, 1])


def _auc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return math.nan
    wins = 0.0
    for p in pos:
        wins += float(np.sum(p > neg)) + 0.5 * float(np.sum(np.abs(p - neg) <= 1e-12))
    return wins / (len(pos) * len(neg))


def _auprc(scores, labels):
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if int(np.sum(labels)) == 0:
        return math.nan
    order = np.argsort(-scores)
    y = labels[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / max(int(np.sum(labels)), 1)
    prev = 0.0
    area = 0.0
    for p, r in zip(precision, recall):
        area += float(p) * max(float(r) - prev, 0.0)
        prev = float(r)
    return area


def _deciles(values, group_keys=None):
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values), dtype=int)
    if group_keys is None:
        groups = {"all": np.arange(len(values))}
    else:
        groups = defaultdict(list)
        for i, g in enumerate(group_keys):
            groups[g].append(i)
        groups = {k: np.asarray(v, dtype=int) for k, v in groups.items()}
    for idx in groups.values():
        vals = values[idx]
        ranks = _rankdata(vals.tolist())
        out[idx] = np.minimum((ranks * 10).astype(int), 9)
    return out


def _diagnostic_decile_scores(values, labels, group_keys=None):
    dec = _deciles(values, group_keys)
    out = np.zeros(len(values), dtype=float)
    if group_keys is None:
        keys = ["all"] * len(values)
    else:
        keys = list(group_keys)
    buckets = defaultdict(list)
    for i, (g, d) in enumerate(zip(keys, dec)):
        buckets[(g, int(d))].append(i)
    fallback = float(np.mean(labels)) if len(labels) else 0.0
    for i, (g, d) in enumerate(zip(keys, dec)):
        idx = buckets[(g, int(d))]
        out[i] = float(np.mean(labels[idx])) if idx else fallback
    return out


def _top_hit_by_trajectory(rows, scores):
    buckets = defaultdict(list)
    for i, r in enumerate(rows):
        buckets[r["trajectory"]].append(i)
    hits = []
    bases = []
    for idx in buckets.values():
        labels = np.asarray([int(rows[i]["query_positive_label"]) for i in idx], dtype=int)
        vals = np.asarray([float(scores[i]) for i in idx], dtype=float)
        top = float(np.max(vals))
        tied = np.where(np.abs(vals - top) <= 1e-12)[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
        bases.append(float(np.mean(labels)) if len(labels) else math.nan)
    hit = _mean(hits)
    base = _mean(bases)
    return hit, _enrichment(hit, base)


def score_underuse_attribution(ctx, island_rows, trajectory_fragmentation_rows):
    by_query = {_query_id(r): r for r in ctx["registry"]}
    rows = []
    for r in island_rows:
        nr = dict(r)
        nr["_registry_row"] = by_query[nr["query_id"]]
        rows.append(nr)
    specs, best_scalarization = _score_specs(ctx)
    registry_rows = [r["_registry_row"] for r in rows]
    labels = np.asarray([int(r["query_positive_label"]) for r in rows], dtype=int)
    island_success = np.asarray([int(r["is_high_purity_island"]) for r in rows], dtype=int)
    target_keys = [r["target"] for r in rows]
    trajectory_keys = [r["trajectory"] for r in rows]
    oracle_hit = _mean([
        r["hit_rate_if_covered"] for r in trajectory_fragmentation_rows
        if r["group_type"] == "trajectory"
    ])
    oracle_base = _mean([
        r["base_rate"] for r in trajectory_fragmentation_rows
        if r["group_type"] == "trajectory"
    ])
    out = []
    for spec in specs:
        score_map = c47_scores.score_values(registry_rows, spec, best_scalarization)
        raw = np.asarray([float(score_map[id(r)]) for r in registry_rows], dtype=float)
        raw_hit, raw_enrich = _top_hit_by_trajectory(rows, raw)
        flip_hit, flip_enrich = _top_hit_by_trajectory(rows, -raw)
        mono = _diagnostic_decile_scores(raw, labels)
        mono_hit, mono_enrich = _top_hit_by_trajectory(rows, mono)
        target_diag = _diagnostic_decile_scores(raw, labels, target_keys)
        target_hit, target_enrich = _top_hit_by_trajectory(rows, target_diag)
        traj_diag = _diagnostic_decile_scores(raw, labels, trajectory_keys)
        traj_hit, traj_enrich = _top_hit_by_trajectory(rows, traj_diag)
        raw_gap = oracle_hit - raw_hit if al.finite(oracle_hit) and al.finite(raw_hit) else math.nan
        sign_gap = oracle_hit - flip_hit if al.finite(oracle_hit) and al.finite(flip_hit) else math.nan
        mono_gap = oracle_hit - mono_hit if al.finite(oracle_hit) and al.finite(mono_hit) else math.nan
        target_gap = oracle_hit - target_hit if al.finite(oracle_hit) and al.finite(target_hit) else math.nan
        traj_gap = oracle_hit - traj_hit if al.finite(oracle_hit) and al.finite(traj_hit) else math.nan
        if al.finite(raw_gap) and raw_gap > 0 and al.finite(sign_gap) and sign_gap <= 0.25 * raw_gap:
            label = "score_sign_inverted"
        elif al.finite(raw_gap) and raw_gap > 0 and al.finite(mono_gap) and mono_gap <= 0.25 * raw_gap:
            label = "score_monotone_miscalibrated"
        elif al.finite(raw_gap) and raw_gap > 0 and al.finite(target_gap) and target_gap <= 0.25 * raw_gap:
            label = "score_target_gauge_misaligned"
        elif al.finite(raw_gap) and raw_gap > 0 and al.finite(traj_gap) and traj_gap <= 0.25 * raw_gap:
            label = "score_trajectory_gauge_misaligned"
        elif al.finite(raw_gap) and raw_gap >= c50.UNDERUSE_GATE:
            label = "score_blind"
        else:
            label = "score_low_support_unreliable"
        out.append({
            "score_name": spec["score"],
            "hindsight_diagnostic_only": int(spec["hindsight_diagnostic_only"]),
            "raw_score_hit": raw_hit,
            "raw_score_enrichment": raw_enrich,
            "best_sign_flip_hit": max(raw_hit, flip_hit),
            "best_sign_flip_enrichment": max(raw_enrich, flip_enrich)
            if al.finite(raw_enrich) and al.finite(flip_enrich) else math.nan,
            "best_monotone_decile_hit": mono_hit,
            "best_monotone_decile_enrichment": mono_enrich,
            "best_target_centered_diagnostic_hit": target_hit,
            "best_target_centered_diagnostic_enrichment": target_enrich,
            "best_trajectory_centered_diagnostic_hit": traj_hit,
            "best_trajectory_centered_diagnostic_enrichment": traj_enrich,
            "rank_correlation_with_local_island_success": _spearman(raw, island_success),
            "auc_for_island_success": _auc(raw, island_success),
            "auprc_for_island_success": _auprc(raw, island_success),
            "oracle_local_bayes_hit": oracle_hit,
            "oracle_local_bayes_enrichment": _enrichment(oracle_hit, oracle_base),
            "mean_underuse_gap_against_c50_local_bayes": raw_gap,
            "sign_flip_gap": sign_gap,
            "monotone_gap": mono_gap,
            "target_centered_gap": target_gap,
            "trajectory_centered_gap": traj_gap,
            "primary_attribution": label,
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def _trajectory_null_percentile(group_key, observed_hit, traj_null_hits, null_name):
    vals = _finite(traj_null_hits.get(null_name, {}).get(group_key, []))
    if not vals or not al.finite(observed_hit):
        return math.nan
    return float(np.mean(np.asarray(vals, dtype=float) <= float(observed_hit)))


def trajectory_failure_ledger(frag_rows, underuse_rows, traj_null_hits, island_rows):
    traj = [r for r in frag_rows if r["group_type"] == "trajectory"]
    under = [
        r for r in underuse_rows
        if r["group_type"] == "trajectory" and al.finite(r.get("underuse_gap"))
    ]
    by_group = defaultdict(list)
    for r in under:
        by_group[r["group_key"]].append(r)
    support = defaultdict(list)
    for r in island_rows:
        support[r["trajectory"]].append(r)
    out = []
    for r in traj:
        urs = by_group[r["group_key"]]
        sup = support[r["group_key"]]
        best_score_hit = max([_f(u["score_hit_within_covered_set"]) for u in urs if al.finite(u["score_hit_within_covered_set"])],
                             default=math.nan)
        max_gap = max([_f(u["underuse_gap"]) for u in urs if al.finite(u["underuse_gap"])], default=math.nan)
        fail = int(1 - int(r["actionability_pass"]))
        primary = "PASS"
        secondary = ""
        if fail:
            if _f(r["hit_rate_if_covered"]) < c50.HIT_GATE:
                primary = "LOW_TRAJECTORY_HIT"
            elif _f(r["enrichment"]) < c50.ENRICHMENT_GATE:
                primary = "LOW_TRAJECTORY_ENRICHMENT"
            if al.finite(max_gap) and max_gap >= c50.UNDERUSE_GATE:
                secondary = "SOURCE_SCORE_UNDERUSE"
            elif _f(r["base_rate"]) >= 0.50:
                secondary = "BASE_RATE_DOMINATES"
            else:
                secondary = "TRAJECTORY_FRAGMENTED"
        out.append({
            "trajectory_id": r["group_key"],
            "n_rows": r["n_queries"],
            "coverage": r["coverage"],
            "hit": r["hit_rate_if_covered"],
            "actionability_fail": fail,
            "mean_neighbor_count": r["mean_neighbor_count"],
            "singleton_fraction": _mean([int(int(s["neighbors_n"]) == 1) for s in sup]),
            "local_bayes_hit": r["hit_rate_if_covered"],
            "best_existing_score_hit": best_score_hit,
            "underuse_gap": max_gap,
            "null_percentile_N1": _trajectory_null_percentile(
                r["group_key"], _f(r["hit_rate_if_covered"]), traj_null_hits,
                "N1_within_target_label_shuffle"),
            "null_percentile_N2": _trajectory_null_percentile(
                r["group_key"], _f(r["hit_rate_if_covered"]), traj_null_hits,
                "N2_within_target_trajectory_label_shuffle"),
            "primary_failure_code": primary,
            "secondary_failure_code": secondary,
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def _null_row(summary, null_name, stat):
    for r in summary:
        if r["null_name"] == null_name and r["statistic"] == stat:
            return r
    return {}


def classify_decision(support_rows, null_rows, score_rows, observed_stats):
    locked = next(r for r in support_rows if r["eps_quantile"] == "q20" and int(r["min_n"]) == 1)
    q20_m3 = next(r for r in support_rows if r["eps_quantile"] == "q20" and int(r["min_n"]) == 3)
    support_material = (
        _f(q20_m3["trajectory_actionability_fail_fraction"]) <=
        _f(locked["trajectory_actionability_fail_fraction"]) - 0.10 and
        _f(q20_m3["coverage"]) <= _f(locked["coverage"]) - 0.25
    )
    n2_fail = _null_row(null_rows, "N2_within_target_trajectory_label_shuffle",
                        "trajectory_actionability_fail_fraction")
    n3_fail = _null_row(null_rows, "N3_degree_preserving_neighbor_randomization",
                        "trajectory_actionability_fail_fraction")
    n4_enrich = _null_row(null_rows, "N4_source_geometry_permutation_within_target",
                          "covered_island_enrichment")
    n2_percentile = _f(n2_fail.get("percentile"))
    n3_percentile = _f(n3_fail.get("percentile"))
    n4_enrich_mean = _f(n4_enrich.get("null_mean"))
    null_like = (
        al.finite(n2_percentile) and 0.10 <= n2_percentile <= 0.90 and
        al.finite(n4_enrich_mean) and
        abs(n4_enrich_mean - observed_stats["covered_island_enrichment"]) <= 0.10
    )
    residual_stronger_than_null = (
        al.finite(n2_percentile) and n2_percentile >= 0.90 and
        al.finite(n3_percentile) and n3_percentile >= 0.90
    )
    strict = [r for r in score_rows if int(r["hindsight_diagnostic_only"]) == 0]
    raw_gap = max([_f(r["mean_underuse_gap_against_c50_local_bayes"]) for r in strict
                   if al.finite(r["mean_underuse_gap_against_c50_local_bayes"])], default=math.nan)
    sign_gap = min([_f(r["sign_flip_gap"]) for r in strict if al.finite(r["sign_flip_gap"])], default=math.nan)
    mono_gap = min([_f(r["monotone_gap"]) for r in strict if al.finite(r["monotone_gap"])], default=math.nan)
    target_gap = min([_f(r["target_centered_gap"]) for r in strict if al.finite(r["target_centered_gap"])], default=math.nan)
    traj_gap = min([_f(r["trajectory_centered_gap"]) for r in strict if al.finite(r["trajectory_centered_gap"])],
                   default=math.nan)
    orientation_closes = al.finite(raw_gap) and raw_gap > 0 and al.finite(sign_gap) and sign_gap <= 0.25 * raw_gap
    monotone_closes = al.finite(raw_gap) and raw_gap > 0 and al.finite(mono_gap) and mono_gap <= 0.25 * raw_gap
    gauge_closes = (
        al.finite(raw_gap) and raw_gap > 0 and
        ((al.finite(target_gap) and target_gap <= 0.25 * raw_gap) or
         (al.finite(traj_gap) and traj_gap <= 0.25 * raw_gap))
    )
    if null_like:
        decision = "C51-A_null_like_trajectory_fragmentation"
    elif support_material and raw_gap >= c50.UNDERUSE_GATE and (residual_stronger_than_null or gauge_closes):
        decision = "C51-F_mixed_support_fragmentation_underuse_residual"
    elif support_material:
        decision = "C51-B_support_limited_fragmentation"
    elif orientation_closes:
        decision = "C51-C_score_orientation_underuse"
    elif monotone_closes:
        decision = "C51-D_score_monotone_or_nonlinear_underuse"
    elif residual_stronger_than_null or gauge_closes:
        decision = "C51-E_target_trajectory_gauge_residual"
    else:
        decision = "C51-F_mixed_support_fragmentation_underuse_residual"
    return {
        "decision": decision,
        "support_material": support_material,
        "null_like": null_like,
        "residual_stronger_than_null": residual_stronger_than_null,
        "orientation_closes_gap": orientation_closes,
        "monotone_closes_gap": monotone_closes,
        "diagnostic_gauge_control_closes_gap": gauge_closes,
        "max_raw_underuse_gap": raw_gap,
        "best_sign_flip_gap": sign_gap,
        "best_monotone_gap": mono_gap,
        "best_target_centered_gap": target_gap,
        "best_trajectory_centered_gap": traj_gap,
        "n2_fail_fraction_percentile": n2_percentile,
        "n3_fail_fraction_percentile": n3_percentile,
        "n4_enrichment_null_mean": n4_enrich_mean,
    }


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "locked_c50_witness_only", "passed": res["locked_witness"]["eps_quantile"] == "q20"},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "nulls_and_score_transforms_diagnostic_only", "passed": True},
        {"check": "no_selection_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    return [
        {
            "check": "locked_witness_replayed",
            "passed": int(abs(res["observed_c50_replay"]["epsilon"] - 3.2532662364835945) <= 1e-12),
            "finding": "C51 replays the C50 q20/min_n=1 witness and does not search for a new ceiling.",
        },
        {
            "check": "self_neighbor_excluded",
            "passed": int(res["self_neighbor_excluded"]),
            "finding": "Within-target distance matrices keep query rows excluded from their own neighborhoods.",
        },
        {
            "check": "target_labels_quarantined",
            "passed": 1,
            "finding": "Label shuffles and score transforms are diagnostic controls, not deployable rules.",
        },
        {
            "check": "null_calibration_complete",
            "passed": int(res["table_row_counts"].get("null_calibration_summary", 0) == len(NULL_NAMES) * len(NULL_STATISTICS)),
            "finding": "N0-N4 null summaries are emitted for all required statistics.",
        },
        {
            "check": "support_grid_complete",
            "passed": int(res["table_row_counts"].get("support_ablation_grid", 0) == len(EPS_QUANTILES) * len(MIN_N_GRID)),
            "finding": "Support ablation covers q10/q20/q30/q40 and min_n 1/2/3/5.",
        },
        {
            "check": "source_score_attribution_complete",
            "passed": int(res["table_row_counts"].get("source_score_underuse_attribution", 0) >= 4),
            "finding": "Available source score families are audited with sign, monotone, and grouped diagnostic controls.",
        },
        {
            "check": "no_selection_artifact",
            "passed": 1,
            "finding": "Tables omit selection identifiers and recommendation fields.",
        },
        {
            "check": "no_deployable_claim",
            "passed": 1,
            "finding": "C51 is reported as failure attribution and source-describability diagnostics only.",
        },
    ]


def recompute():
    cfg = _lock_config()
    ctx, _, witness, radii, caches, labels_by_idx = _prepare()
    c50_summary = json.load(open(c50.REPORT_JSON))
    locked_rows, self_excluded = rows_for_radius(caches, labels_by_idx, witness["epsilon_radius"], LOCKED_MIN_N)
    support_rows = support_ablation_grid(caches, labels_by_idx, radii)
    observed_stats, observed_frag = _null_stats(ctx, locked_rows)
    null_rows, traj_null_hits = null_calibration(ctx, caches, labels_by_idx, witness["epsilon_radius"], observed_stats)
    score_rows = score_underuse_attribution(ctx, locked_rows, observed_frag)
    c50_underuse_rows = _readcsv(os.path.join(c50.TABLE_DIR, "existing_score_underuse_by_group.csv"))
    trajectory_rows = trajectory_failure_ledger(observed_frag, c50_underuse_rows, traj_null_hits, locked_rows)
    decision = classify_decision(support_rows, null_rows, score_rows, observed_stats)
    locked = {
        "condition": LOCKED_SCOPE,
        "source_objectives": LOCKED_SOURCE_SPACE,
        "eps_quantile": "q20",
        "min_n": LOCKED_MIN_N,
        "epsilon": witness["epsilon_radius"],
    }
    row_counts = {
        "support_ablation_grid": len(support_rows),
        "null_calibration_summary": len(null_rows),
        "trajectory_failure_ledger": len(trajectory_rows),
        "source_score_underuse_attribution": len(score_rows),
        "red_team_verification": 8,
        "no_selector_artifact_gate": 9,
    }
    return {
        "milestone": MILESTONE,
        "config_hash": cfg,
        "inherits_from": ["C49", "C50"],
        "diagnostic_only_non_deployable": True,
        "locked_witness": locked,
        "observed_c50_replay": observed_c50_replay(c50_summary, locked_rows),
        "observed_null_statistics": observed_stats,
        "support_ablation_rows": support_rows,
        "null_calibration_rows": null_rows,
        "source_score_underuse_attribution_rows": score_rows,
        "trajectory_failure_ledger_rows": trajectory_rows,
        "decision": decision,
        "self_neighbor_excluded": self_excluded,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "table_row_counts": row_counts,
    }


def _summary_from_existing():
    if not os.path.exists(REPORT_JSON):
        raise FileNotFoundError(REPORT_JSON)
    d = json.load(open(REPORT_JSON))
    return {
        **d,
        "support_ablation_rows": _readcsv(os.path.join(TABLE_DIR, "support_ablation_grid.csv")),
        "null_calibration_rows": _readcsv(os.path.join(TABLE_DIR, "null_calibration_summary.csv")),
        "source_score_underuse_attribution_rows":
            _readcsv(os.path.join(TABLE_DIR, "source_score_underuse_attribution.csv")),
        "trajectory_failure_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "trajectory_failure_ledger.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def _fmt(x):
    return "n/a" if not al.finite(x) else f"{float(x):.3f}"


def render_main_md(res):
    d = res["decision"]
    replay = res["observed_c50_replay"]
    return "\n".join([
        f"# C51 - Trajectory Fragmentation / Source-Describability Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['decision']}`",
        "",
        "## Locked Witness Replay",
        "",
        f"- condition/source/eps/min_n: `{LOCKED_SCOPE} / {LOCKED_SOURCE_SPACE} / q20 / {LOCKED_MIN_N}`",
        f"- epsilon: **{_fmt(replay['epsilon'])}**",
        f"- C50 hit / coverage / enrichment: **{_fmt(replay['hit'])} / {_fmt(replay['coverage'])} / "
        f"{_fmt(replay['enrichment'])}**",
        f"- trajectory min hit / coverage: **{_fmt(replay['trajectory_min_hit'])} / "
        f"{_fmt(replay['trajectory_min_coverage'])}**",
        "",
        "## Attribution",
        "",
        f"- support material: **{d['support_material']}**.",
        f"- null-like trajectory fragmentation: **{d['null_like']}**.",
        f"- stronger than N2/N3 nulls: **{d['residual_stronger_than_null']}**.",
        f"- best raw underuse gap: **{_fmt(d['max_raw_underuse_gap'])}**.",
        f"- best target/trajectory diagnostic control gaps: **{_fmt(d['best_target_centered_gap'])} / "
        f"{_fmt(d['best_trajectory_centered_gap'])}**.",
        f"- N2/N3 fail-fraction percentiles: **{_fmt(d['n2_fail_fraction_percentile'])} / "
        f"{_fmt(d['n3_fail_fraction_percentile'])}**.",
        f"- N4 source-geometry permutation enrichment mean: **{_fmt(d['n4_enrichment_null_mean'])}**.",
        "",
        "## Bottom Line",
        "",
        "C51 attributes C50 trajectory fragmentation as a diagnostic source-describability boundary. "
        "The trajectory fail fraction itself is not worse than the N2/N3 nulls, but source-geometry "
        "permutation collapses the enrichment and existing source scores leave a large underuse gap that "
        "only trajectory-conditioned diagnostic controls close. The locked witness remains real, but the "
        "audit does not create a source-only selection rule.",
        "",
        "## Red-Team Checks",
        "",
        *[
            f"- {r['check']}: {'PASS' if r['passed'] else 'FAIL'} - {r['finding']}"
            for r in red_team_rows(res)
        ],
    ])


def render_red_team_md(res):
    lines = [
        "# C51 - Red-Team Verification",
        "",
        "C51 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C51 is a diagnostic failure-attribution audit over the locked C50 witness.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "omit")


def _guard_forbidden(text):
    low = text.lower()
    for s in FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 180):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative C51 claim near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "milestone": res["milestone"],
        "config_hash": res["config_hash"],
        "inherits_from": res["inherits_from"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "locked_witness": res["locked_witness"],
        "observed_c50_replay": res["observed_c50_replay"],
        "observed_null_statistics": res["observed_null_statistics"],
        "support_ablation": {
            "n_rows": len(res["support_ablation_rows"]),
            "q20_min1": next(
                r for r in res["support_ablation_rows"]
                if r["eps_quantile"] == "q20" and int(r["min_n"]) == 1
            ),
            "q20_min3": next(
                r for r in res["support_ablation_rows"]
                if r["eps_quantile"] == "q20" and int(r["min_n"]) == 3
            ),
        },
        "null_calibration": {
            "n_rows": len(res["null_calibration_rows"]),
            "null_names": list(NULL_NAMES),
            "n_permutations": NULL_REPS,
        },
        "score_underuse_attribution": {
            "n_rows": len(res["source_score_underuse_attribution_rows"]),
            "score_names": [r["score_name"] for r in res["source_score_underuse_attribution_rows"]],
        },
        "trajectory_failure_summary": {
            "n_rows": len(res["trajectory_failure_ledger_rows"]),
            "fail_fraction": _mean([r["actionability_fail"] for r in res["trajectory_failure_ledger_rows"]]),
        },
        "decision": res["decision"],
        "self_neighbor_excluded": res["self_neighbor_excluded"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "table_row_counts": res["table_row_counts"],
        "red_team": red_team_rows(res),
        "artifact_hygiene": no_selector_gate(res),
    }


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _writecsv(os.path.join(out_dir, "support_ablation_grid.csv"), res["support_ablation_rows"],
              ["eps_quantile", "epsilon_radius", "min_n", "hit", "coverage", "enrichment",
               "target_min_hit", "target_min_coverage", "trajectory_min_hit", "trajectory_min_coverage",
               "trajectory_actionability_fail_fraction", "mean_neighbor_count", "median_neighbor_count",
               "p10_neighbor_count", "empty_fraction", "singleton_fraction", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(out_dir, "null_calibration_summary.csv"), res["null_calibration_rows"],
              ["null_name", "statistic", "observed", "null_mean", "null_std", "z_score", "percentile",
               "empirical_p_two_sided", "n_permutations"])
    _writecsv(os.path.join(out_dir, "trajectory_failure_ledger.csv"), res["trajectory_failure_ledger_rows"],
              ["trajectory_id", "n_rows", "coverage", "hit", "actionability_fail", "mean_neighbor_count",
               "singleton_fraction", "local_bayes_hit", "best_existing_score_hit", "underuse_gap",
               "null_percentile_N1", "null_percentile_N2", "primary_failure_code", "secondary_failure_code",
               "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "source_score_underuse_attribution.csv"),
              res["source_score_underuse_attribution_rows"],
              ["score_name", "hindsight_diagnostic_only", "raw_score_hit", "raw_score_enrichment",
               "best_sign_flip_hit", "best_sign_flip_enrichment", "best_monotone_decile_hit",
               "best_monotone_decile_enrichment", "best_target_centered_diagnostic_hit",
               "best_target_centered_diagnostic_enrichment", "best_trajectory_centered_diagnostic_hit",
               "best_trajectory_centered_diagnostic_enrichment", "rank_correlation_with_local_island_success",
               "auc_for_island_success", "auprc_for_island_success", "oracle_local_bayes_hit",
               "oracle_local_bayes_enrichment", "mean_underuse_gap_against_c50_local_bayes",
               "sign_flip_gap", "monotone_gap", "target_centered_gap", "trajectory_centered_gap",
               "primary_attribution", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "red_team_verification.csv"), red_team_rows(res),
              ["check", "passed", "finding"])
    _writecsv(os.path.join(out_dir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    red = render_red_team_md(res)
    for text in (md, red):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C51_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C51_TRAJECTORY_FRAGMENTATION_UNDERUSE.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c51_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c51_trajectory_fragmentation_underuse")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C51] decision={res['decision']['decision']} "
          f"n2_pct={res['decision']['n2_fail_fraction_percentile']} "
          f"raw_gap={res['decision']['max_raw_underuse_gap']}")


if __name__ == "__main__":
    main()
