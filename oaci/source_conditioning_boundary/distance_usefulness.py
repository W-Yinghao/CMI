"""Source-distance usefulness before and after conditioning."""
from __future__ import annotations

import numpy as np

from ..source_nonidentifiability import source_space
from . import artifact_loader as al
from . import schema


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


def _spearman(xs, ys):
    pairs = [(float(x), float(y)) for x, y in zip(xs, ys) if al.finite(x) and al.finite(y)]
    if len(pairs) < 2:
        return None
    rx = _rankdata([p[0] for p in pairs])
    ry = _rankdata([p[1] for p in pairs])
    if float(np.std(rx)) <= 1e-12 or float(np.std(ry)) <= 1e-12:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def _scope_ok(a, b, scope):
    if scope == "within_trajectory":
        return a["trajectory_id"] == b["trajectory_id"]
    if scope == "within_target":
        return str(a["target"]) == str(b["target"])
    if scope == "within_seed":
        return str(a["seed"]) == str(b["seed"])
    if scope == "within_level":
        return str(a["level"]) == str(b["level"])
    if scope == "within_regime":
        return a["regime"] == b["regime"]
    if scope == "cross_target":
        return str(a["target"]) != str(b["target"])
    if scope == "cross_regime":
        return a["regime"] != b["regime"]
    raise ValueError(scope)


def _pair_sample(ctx, space, scope, scope_index):
    rows = ctx["registry"]
    pairs = []
    if scope == "within_trajectory":
        for cs in ctx["by_traj"].values():
            for i in range(len(cs)):
                for j in range(i + 1, len(cs)):
                    pairs.append((int(cs[i]["source_idx"]), int(cs[j]["source_idx"])))
        return pairs
    rng = np.random.RandomState(schema.PAIR_SAMPLE_SEED + 101 * scope_index)
    n = len(rows)
    attempts = 0
    seen = set()
    while len(pairs) < schema.PAIR_SAMPLE_MAX and attempts < schema.PAIR_SAMPLE_MAX * 100:
        i = int(rng.randint(0, n))
        j = int(rng.randint(0, n - 1))
        if j >= i:
            j += 1
        a, b = rows[i], rows[j]
        if not _scope_ok(a, b, scope):
            attempts += 1
            continue
        key = (min(i, j), max(i, j))
        if key not in seen:
            seen.add(key)
            pairs.append(key)
        attempts += 1
    return pairs


def audit(ctx, neighbor, space=None):
    space = space or neighbor["space"] or source_space.build_space(ctx)
    out = []
    nrows = ctx["registry"]
    for si, scope in enumerate(schema.CONDITIONING_SCOPES):
        pairs = _pair_sample(ctx, space, scope, si)
        dists = []
        gaps = []
        divs = []
        for i, j in pairs:
            a, b = nrows[i], nrows[j]
            d = source_space.distance(space, i, j)
            dists.append(d)
            gaps.append(abs(float(a["target_utility_score"]) - float(b["target_utility_score"])))
            divs.append(source_space.pair_metrics(a, b, space, metric_distance=d)["target_divergent"])
        ns = neighbor["summary"][scope]
        out.append({
            "scope": scope,
            "n_pairs": len(pairs),
            "pair_sample_max": schema.PAIR_SAMPLE_MAX,
            "source_distance_target_utility_gap_spearman": _spearman(dists, gaps),
            "sample_pair_target_divergent_rate": al.finite_mean(divs),
            "sample_pair_mean_source_distance": al.finite_mean(dists),
            "sample_pair_mean_target_utility_gap": al.finite_mean(gaps),
            "nearest_target_divergent_rate": ns["target_divergent_rate"],
            "nearest_q10_target_divergent_rate": ns["source_equivalent_q10_target_divergent_rate"],
            "nearest_joint_good_disagreement_rate": ns["joint_good_disagreement_rate"],
            "distance_usefulness_diagnostic_only": 1,
        })
    return {"rows": out, "summary": {r["scope"]: r for r in out}}
