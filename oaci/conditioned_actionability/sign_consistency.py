"""C47 pairwise sign-consistency diagnostics."""
from __future__ import annotations

import bisect
import math

import numpy as np

from . import artifact_loader as al
from . import schema
from . import score_registry


def _score_auc_for_pairs(pairs, scores):
    total = correct = ties = 0
    for a, b in pairs:
        du = float(a["target_utility_score"]) - float(b["target_utility_score"])
        if abs(du) <= 1e-12:
            continue
        da = scores.get(id(a), math.nan)
        db = scores.get(id(b), math.nan)
        if not al.finite(da) or not al.finite(db):
            continue
        ds = float(da) - float(db)
        total += 1
        prod = du * ds
        if prod > 0:
            correct += 1
        elif abs(prod) <= 1e-12:
            ties += 1
    auc = (correct + 0.5 * ties) / total if total else math.nan
    return total, correct, ties, auc


def _all_pairs(groups):
    for rows in groups:
        for i in range(len(rows)):
            for j in range(i + 1, len(rows)):
                yield rows[i], rows[j]


def _sample_pairs(groups, n_pairs, seed):
    counts = [al.comb(len(g), 2) for g in groups]
    cum = np.cumsum(counts)
    total = int(cum[-1]) if len(cum) else 0
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(min(n_pairs, total)):
        pick = int(rng.integers(0, total))
        gi = bisect.bisect_right(cum, pick)
        rows = groups[gi]
        n = len(rows)
        i = int(rng.integers(0, n))
        j = int(rng.integers(0, n - 1))
        if j >= i:
            j += 1
        if i > j:
            i, j = j, i
        out.append((rows[i], rows[j]))
    return out


def _pairs_for_scope(groups, seed):
    total_pairs = sum(al.comb(len(g), 2) for g in groups)
    if total_pairs <= schema.PAIR_SAMPLE_MAX:
        return list(_all_pairs(groups)), total_pairs, 0
    return _sample_pairs(groups, schema.PAIR_SAMPLE_MAX, seed), total_pairs, 1


def audit(ctx, score_specs):
    rows = []
    for scope_i, scope in enumerate(schema.GROUP_SCOPES):
        group_map = al.group_rows(ctx, scope)
        groups = [g for _, g in sorted(group_map.items()) if len(g) >= 2]
        pairs, total_pairs, sampled = _pairs_for_scope(groups, schema.PAIR_SAMPLE_SEED + scope_i * 1009)
        for spec in score_specs["rows"]:
            scores = {}
            for g in groups:
                scores.update(score_registry.score_values(g, spec, score_specs["best_scalarization"]))
            n_usable, n_correct, n_ties, auc = _score_auc_for_pairs(pairs, scores)
            rows.append({
                "group_scope": scope,
                "score": spec["score"],
                "score_family": spec["family"],
                "n_groups": len(groups),
                "n_possible_pairs": total_pairs,
                "n_sampled_pairs": len(pairs),
                "pair_sample_max": schema.PAIR_SAMPLE_MAX,
                "pair_sample_seed": schema.PAIR_SAMPLE_SEED + scope_i * 1009,
                "sampled_with_replacement": sampled,
                "n_usable_pairs": n_usable,
                "n_correct_pairs": n_correct,
                "n_score_tie_pairs": n_ties,
                "pairwise_auc_vs_target_utility": auc,
                "misranking_rate": 1.0 - auc if al.finite(auc) else math.nan,
                "source_only": int(spec["source_only"]),
                "hindsight_diagnostic_only": int(spec["hindsight_diagnostic_only"]),
                "target_label_used": int(spec["target_label_used"]),
                "diagnostic_ceiling": int(spec["diagnostic_ceiling"]),
                "target_labels_diagnostic_only": 1,
            })
    return {
        "rows": rows,
        "summary": {
            "n_sign_consistency_rows": len(rows),
            "pair_sample_max": schema.PAIR_SAMPLE_MAX,
            "pair_sample_seed": schema.PAIR_SAMPLE_SEED,
            "sampled_scopes": sorted({r["group_scope"] for r in rows if int(r["sampled_with_replacement"])}),
        },
    }
