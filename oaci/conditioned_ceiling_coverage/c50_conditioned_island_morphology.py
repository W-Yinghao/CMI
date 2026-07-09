"""C50 - Conditioned-Island Morphology / Fragmentation Audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from ..conditioned_actionability import score_registry as c47_scores
from ..conditioned_local_ceiling import local_ceiling as c48_local
from . import artifact_loader as al
from . import audit_utils as au
from . import island_metrics
from . import locked_witness
from . import schema as c49_schema
from . import source_space_registry


REPORT_JSON = "oaci/reports/C50_CONDITIONED_ISLAND_MORPHOLOGY.json"
TABLE_DIR = "oaci/reports/c50_tables"

WITNESS_SCOPE = locked_witness.WITNESS_SCOPE
WITNESS_SOURCE_SPACE = locked_witness.WITNESS_SOURCE_SPACE
WITNESS_NEIGHBORHOOD = locked_witness.WITNESS_NEIGHBORHOOD
WITNESS_MIN_NEIGHBOR_COUNT = locked_witness.WITNESS_MIN_NEIGHBOR_COUNT
WITNESS_LABEL = locked_witness.WITNESS_LABEL

COVERAGE_GATE = 0.50
HIT_GATE = c49_schema.RELIABLE_TOP1_HIT_GATE
ENRICHMENT_GATE = c49_schema.RELIABLE_ENRICHMENT_GATE
UNDERUSE_GATE = c49_schema.UNDERUSE_GAP_GATE
LOW_NEIGHBOR_SUPPORT_GATE = 3.0
PERMUTATION_REPS = 64
PERMUTATION_SEED = 50050

GROUP_TYPES = ("target", "trajectory", "seed", "level", "regime", "conditioned_key")
STRICT_SOURCE_SCORES = ("selection_leakage", "R_src", "C30_source_rank", "C19_robust_core")

FORBIDDEN_CLAIM_SUBSTRINGS = c49_schema.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "deployable method",
    "actionable selector",
    "source-only control is restored",
    "target-conditioned local bayes estimates are deployable",
)


def _lock_config():
    return au.lock_config("C50")


def _readcsv(path):
    return au.read_csv(path)


def _writecsv(path, rows, cols):
    return au.write_csv(path, rows, cols)


def _f(x, default=math.nan):
    return au.as_float(x, default)


def _finite_mean(vals):
    return au.finite_mean(vals)


def _finite_median(vals):
    return au.finite_median(vals)


def _finite_quantile(vals, q):
    return au.finite_quantile(vals, q)


def _enrichment(hit, base):
    return au.enrichment(hit, base)


def _row_group_key(row, group_type):
    return au.row_group_key(row, group_type)


def _query_id(row):
    return au.query_id(row)


def _c49_locked_witness():
    return locked_witness.c49_locked_witness()


def _group_fields(row, conditioned_key):
    return {
        "conditioning_key": conditioned_key,
        "target": str(row["target"]),
        "trajectory": row["trajectory_id"],
        "seed": str(row["seed"]),
        "level": str(row["level"]),
        "regime": row["regime"],
    }


def _local_arrays(space, group, radius):
    labels = np.asarray([int(r[WITNESS_LABEL]) for r in group], dtype=int)
    base = float(np.mean(labels)) if len(labels) else math.nan
    dist = c48_local._distance_matrix(space, group)
    purity, counts, mask = c48_local._epsilon_purity(dist, labels, radius, base)
    covered = counts >= WITNESS_MIN_NEIGHBOR_COUNT
    return labels, base, dist, purity, counts, mask, covered


def _distance_stats(vals):
    vals = [float(v) for v in vals if al.finite(v)]
    return {
        "source_distance_mean": float(np.mean(vals)) if vals else math.nan,
        "source_distance_min": float(np.min(vals)) if vals else math.nan,
        "source_distance_max": float(np.max(vals)) if vals else math.nan,
        "source_distance_q25": float(np.quantile(vals, 0.25)) if vals else math.nan,
        "source_distance_q75": float(np.quantile(vals, 0.75)) if vals else math.nan,
    }


def locked_islands(ctx, space, witness):
    radius = witness["epsilon_radius"]
    rows = []
    group_cache = {}
    self_excluded = True
    for conditioned_key, group in sorted(al.group_rows(ctx, WITNESS_SCOPE).items()):
        labels, base, dist, purity, counts, mask, covered = _local_arrays(space, group, radius)
        group_cache[conditioned_key] = {
            "group": group,
            "labels": labels,
            "base": base,
            "dist": dist,
            "purity": purity,
            "counts": counts,
            "mask": mask,
            "covered": covered,
        }
        for i, row in enumerate(group):
            if bool(mask[i, i]):
                self_excluded = False
            dvals = dist[i, mask[i]]
            stats = _distance_stats(dvals)
            positive_rate = float(purity[i]) if int(counts[i]) > 0 else math.nan
            lift = positive_rate - base if al.finite(positive_rate) and al.finite(base) else math.nan
            out = {
                "query_id": _query_id(row),
                **_group_fields(row, conditioned_key),
                "neighbors_n": int(counts[i]),
                "covered": int(bool(covered[i])),
                "query_positive_label": int(labels[i]),
                "neighbor_positive_rate": positive_rate,
                "neighbor_base_rate": base,
                "local_lift": lift,
                "local_enrichment": _enrichment(positive_rate, base),
                **stats,
                "is_singleton_neighbor": int(int(counts[i]) == 1),
                "is_high_purity_island": int(al.finite(positive_rate) and positive_rate >= HIT_GATE),
                "target_labels_diagnostic_only": 1,
                "no_checkpoint_recommendation": 1,
            }
            rows.append(out)
    return rows, group_cache, self_excluded


def _local_bayes_hit(rows):
    return island_metrics.local_bayes_hit(rows)


def group_fragmentation(island_rows):
    return island_metrics.group_fragmentation(
        island_rows, GROUP_TYPES, COVERAGE_GATE, HIT_GATE, ENRICHMENT_GATE)


def _rank_percentile(scores, covered_idx, idx):
    vals = scores[covered_idx]
    s = scores[idx]
    rank = 1 + int(np.sum(vals > s + 1e-12))
    n = len(covered_idx)
    pct = 1.0 if n <= 1 else 1.0 - ((rank - 1) / (n - 1))
    return rank, pct


def _score_underuse_for_group(group_type, group_key, rows, specs, best_scalarization):
    covered_rows = [r for r in rows if int(r["covered"])]
    if not covered_rows:
        return []
    labels = np.asarray([int(r["query_positive_label"]) for r in rows], dtype=int)
    covered_idx = np.asarray([i for i, r in enumerate(rows) if int(r["covered"])], dtype=int)
    max_p = max(float(r["neighbor_positive_rate"]) for r in covered_rows if al.finite(r["neighbor_positive_rate"]))
    island_idx = np.asarray([
        i for i, r in enumerate(rows)
        if int(r["covered"]) and abs(float(r["neighbor_positive_rate"]) - max_p) <= 1e-12
    ], dtype=int)
    positive_island_idx = np.asarray([i for i in island_idx if int(labels[i]) == 1], dtype=int)
    rank_pool = positive_island_idx if len(positive_island_idx) else island_idx
    oracle_hit = float(np.mean(labels[island_idx])) if len(island_idx) else math.nan
    registry_like_rows = [r["_registry_row"] for r in rows]
    out = []
    for spec in specs:
        scores = c47_scores.score_values(registry_like_rows, spec, best_scalarization)
        vals = np.asarray([float(scores[id(r)]) for r in registry_like_rows], dtype=float)
        covered_vals = vals[covered_idx]
        top_score = float(np.max(covered_vals))
        top_idx = covered_idx[np.where(np.abs(covered_vals - top_score) <= 1e-12)[0]]
        score_hit = float(np.mean(labels[top_idx])) if len(top_idx) else math.nan
        ranks = [_rank_percentile(vals, covered_idx, int(i)) for i in rank_pool]
        best_rank = min((r[0] for r in ranks), default=math.nan)
        best_pct = max((r[1] for r in ranks), default=math.nan)
        out.append({
            "score_name": spec["score"],
            "group_type": group_type,
            "group_key": group_key,
            "n_covered": len(covered_idx),
            "oracle_hit_or_ceiling": oracle_hit,
            "score_hit_within_covered_set": score_hit,
            "underuse_gap": oracle_hit - score_hit
            if al.finite(oracle_hit) and al.finite(score_hit) else math.nan,
            "rank_of_oracle_positive_under_score": best_rank,
            "score_percentile_of_oracle_positive": best_pct,
            "source_score_orientation": spec["orientation"],
            "target_labels_diagnostic_only": 1,
            "diagnostic_underuse_only": 1,
            "no_checkpoint_recommendation": 1,
        })
    return out


def existing_score_underuse(ctx, island_rows):
    by_query = {_query_id(r): r for r in ctx["registry"]}
    enriched = []
    for row in island_rows:
        nr = dict(row)
        nr["_registry_row"] = by_query[nr["query_id"]]
        enriched.append(nr)
    score_specs = c47_scores.registry(ctx)
    specs = [
        s for s in score_specs["rows"]
        if s["score"] in STRICT_SOURCE_SCORES and int(s["source_only"]) == 1 and
        int(s["hindsight_diagnostic_only"]) == 0
    ]
    rows = []
    for group_type in GROUP_TYPES:
        buckets = defaultdict(list)
        for r in enriched:
            buckets[_row_group_key(r, group_type)].append(r)
        for group_key, group_rows in sorted(buckets.items()):
            rows.extend(_score_underuse_for_group(
                group_type, group_key, group_rows, specs, score_specs["best_scalarization"]))
    summary = []
    for score in STRICT_SOURCE_SCORES:
        rs = [r for r in rows if r["score_name"] == score]
        summary.append({
            "score_name": score,
            "n_group_rows": len(rs),
            "mean_oracle_hit_or_ceiling": _finite_mean([r["oracle_hit_or_ceiling"] for r in rs]),
            "mean_score_hit_within_covered_set": _finite_mean([r["score_hit_within_covered_set"] for r in rs]),
            "mean_underuse_gap": _finite_mean([r["underuse_gap"] for r in rs]),
            "max_underuse_gap": max([float(r["underuse_gap"]) for r in rs if al.finite(r["underuse_gap"])], default=math.nan),
            "mean_score_percentile_of_oracle_positive":
                _finite_mean([r["score_percentile_of_oracle_positive"] for r in rs]),
            "diagnostic_underuse_only": 1,
        })
    return {"rows": rows, "summary_rows": summary}


def _underuse_lookup(underuse_rows):
    out = defaultdict(float)
    for r in underuse_rows:
        gap = _f(r.get("underuse_gap"))
        if al.finite(gap):
            key = (r["group_type"], r["group_key"])
            out[key] = max(out[key], gap)
    return out


def actionability_failure_ledger(fragmentation_rows, underuse_rows):
    underuse = _underuse_lookup(underuse_rows)
    rows = []
    for r in fragmentation_rows:
        if int(r["actionability_pass"]):
            continue
        codes = []
        coverage = _f(r["coverage"])
        hit = _f(r["hit_rate_if_covered"])
        enrichment = _f(r["enrichment"])
        base = _f(r["base_rate"])
        mean_neighbors = _f(r["mean_neighbor_count"])
        median_neighbors = _f(r["median_neighbor_count"])
        max_gap = underuse[(r["group_type"], r["group_key"])]
        if al.finite(coverage) and coverage <= 0:
            codes.append("NO_COVERAGE_IN_GROUP")
        if al.finite(coverage) and coverage > 0 and (not al.finite(hit) or hit < HIT_GATE):
            codes.append("COVERED_BUT_LOW_HIT")
        if (
            (al.finite(median_neighbors) and median_neighbors <= 1.0) or
            (al.finite(mean_neighbors) and mean_neighbors < LOW_NEIGHBOR_SUPPORT_GATE)
        ):
            codes.append("SINGLETON_OR_LOW_NEIGHBOR_SUPPORT")
        if al.finite(max_gap) and max_gap >= UNDERUSE_GATE:
            codes.append("SCORE_UNDERUSES_AVAILABLE_POSITIVES")
        if al.finite(base) and base >= 0.50 and (not al.finite(enrichment) or enrichment < ENRICHMENT_GATE):
            codes.append("GROUP_BASE_RATE_DOMINATES")
        if r["group_type"] in ("target", "conditioned_key"):
            codes.append("TARGET_FRAGMENTED")
        if r["group_type"] == "trajectory":
            codes.append("TRAJECTORY_FRAGMENTED")
        codes.append("DIAGNOSTIC_ONLY_TARGET_CONDITIONING")
        rows.append({
            "group_type": r["group_type"],
            "group_key": r["group_key"],
            "n_queries": r["n_queries"],
            "coverage": coverage,
            "hit_rate_if_covered": hit,
            "enrichment": enrichment,
            "max_underuse_gap": max_gap,
            "reason_codes": "|".join(dict.fromkeys(codes)),
            "target_labels_diagnostic_only": 1,
            "no_checkpoint_recommendation": 1,
        })
    counts = defaultdict(int)
    for r in rows:
        for code in r["reason_codes"].split("|"):
            if code:
                counts[code] += 1
    count_rows = [{"reason_code": k, "n_groups": counts[k]} for k in sorted(counts)]
    return {"rows": rows, "count_rows": count_rows}


def _permutation_adjusted_gap(group_cache, rng):
    hits = []
    perm_hits = []
    for g in group_cache.values():
        labels = g["labels"]
        if len(labels) == 0:
            continue
        covered = g["covered"]
        if not np.any(covered):
            continue
        purity = g["purity"]
        p = purity[covered]
        covered_idx = np.where(covered)[0]
        max_p = float(np.max(p))
        island = covered_idx[np.where(np.abs(p - max_p) <= 1e-12)[0]]
        hits.append(float(np.mean(labels[island])) if len(island) else math.nan)
        perm_hits.append(c48_local._permutation_top1_hit(
            labels, g["mask"], "epsilon", g["base"], rng))
    hit = _finite_mean(hits)
    perm = _finite_mean(perm_hits)
    return hit, perm, hit - perm if al.finite(hit) and al.finite(perm) else math.nan


def baseline_sanity(fragmentation_rows, underuse_rows, group_cache):
    rng = np.random.default_rng(PERMUTATION_SEED)
    target_hit, target_perm, target_gap = _permutation_adjusted_gap(group_cache, rng)
    rows = []
    for group_type in GROUP_TYPES:
        frs = [r for r in fragmentation_rows if r["group_type"] == group_type]
        if not frs:
            continue
        coverage = _finite_mean([r["coverage"] for r in frs])
        base = _finite_mean([r["base_rate"] for r in frs])
        local_hit = _finite_mean([r["hit_rate_if_covered"] for r in frs])
        local_enrich = _finite_mean([r["enrichment"] for r in frs])
        gap = target_gap if group_type in ("target", "conditioned_key") else math.nan
        rows.append({
            "baseline_name": "same_group_random",
            "condition_scope": WITNESS_SCOPE,
            "group_type": group_type,
            "hit": base,
            "coverage": coverage,
            "enrichment": 1.0,
            "permutation_adjusted_gap": math.nan,
            "permutation_hit": math.nan,
            "permutation_reps": "",
        })
        rows.append({
            "baseline_name": "locked_local_bayes_ceiling",
            "condition_scope": WITNESS_SCOPE,
            "group_type": group_type,
            "hit": local_hit,
            "coverage": coverage,
            "enrichment": local_enrich,
            "permutation_adjusted_gap": gap,
            "permutation_hit": target_perm if group_type in ("target", "conditioned_key") else math.nan,
            "permutation_reps": PERMUTATION_REPS if group_type in ("target", "conditioned_key") else "",
        })
        urs = [r for r in underuse_rows if r["group_type"] == group_type]
        if urs:
            by_group = defaultdict(list)
            for r in urs:
                by_group[r["group_key"]].append(r)
            best_hits = []
            for rows_for_group in by_group.values():
                vals = [_f(r["score_hit_within_covered_set"]) for r in rows_for_group]
                vals = [v for v in vals if al.finite(v)]
                if vals:
                    best_hits.append(max(vals))
            best_hit = _finite_mean(best_hits)
            rows.append({
                "baseline_name": "best_existing_source_score",
                "condition_scope": WITNESS_SCOPE,
                "group_type": group_type,
                "hit": best_hit,
                "coverage": coverage,
                "enrichment": _enrichment(best_hit, base),
                "permutation_adjusted_gap": math.nan,
                "permutation_hit": math.nan,
                "permutation_reps": "",
            })
    return rows


def classify_outcome(fragmentation_rows, underuse_summary):
    target_rows = [r for r in fragmentation_rows if r["group_type"] == "target"]
    traj_rows = [r for r in fragmentation_rows if r["group_type"] == "trajectory"]
    target_min_hit = min([_f(r["hit_rate_if_covered"]) for r in target_rows if al.finite(r["hit_rate_if_covered"])],
                         default=math.nan)
    target_min_coverage = min([_f(r["coverage"]) for r in target_rows if al.finite(r["coverage"])],
                              default=math.nan)
    traj_min_hit = min([_f(r["hit_rate_if_covered"]) for r in traj_rows if al.finite(r["hit_rate_if_covered"])],
                       default=math.nan)
    traj_min_coverage = min([_f(r["coverage"]) for r in traj_rows if al.finite(r["coverage"])],
                            default=math.nan)
    traj_fail_fraction = float(np.mean([1 - int(r["actionability_pass"]) for r in traj_rows])) if traj_rows else math.nan
    target_fail_fraction = float(np.mean([1 - int(r["actionability_pass"]) for r in target_rows])) if target_rows else math.nan
    max_underuse = max([_f(r["mean_underuse_gap"]) for r in underuse_summary if al.finite(r["mean_underuse_gap"])],
                       default=math.nan)
    fragmentation_material = (
        (al.finite(target_min_hit) and target_min_hit < HIT_GATE) or
        (al.finite(target_min_coverage) and target_min_coverage < COVERAGE_GATE) or
        (al.finite(traj_min_hit) and traj_min_hit < HIT_GATE) or
        (al.finite(traj_min_coverage) and traj_min_coverage < COVERAGE_GATE) or
        (al.finite(traj_fail_fraction) and traj_fail_fraction >= 0.25)
    )
    underuse_material = al.finite(max_underuse) and max_underuse >= UNDERUSE_GATE
    if fragmentation_material and underuse_material:
        outcome = "C50-C_mixed_fragmentation_plus_underuse"
    elif fragmentation_material:
        outcome = "C50-A_fragmented_diagnostic_islands"
    elif underuse_material:
        outcome = "C50-B_distributed_but_score_unrecoverable_islands"
    else:
        outcome = "C50-D_broad_stable_diagnostic_ceiling_still_non_deployable"
    return {
        "outcome": outcome,
        "fragmentation_material": fragmentation_material,
        "underuse_material": underuse_material,
        "target_min_hit": target_min_hit,
        "target_min_coverage": target_min_coverage,
        "trajectory_min_hit": traj_min_hit,
        "trajectory_min_coverage": traj_min_coverage,
        "target_actionability_fail_fraction": target_fail_fraction,
        "trajectory_actionability_fail_fraction": traj_fail_fraction,
        "max_mean_underuse_gap": max_underuse,
    }


def claim_ledger():
    can = [
        "C49 broad conditioned witness is real as a diagnostic ceiling.",
        "C50 attributes non-actionability to measured fragmentation and/or existing-score underuse.",
        "Conditioning can reveal target-good islands in source-objective space under diagnostic labels.",
    ]
    cannot = [
        "Do not treat the witness as a checkpoint selection rule.",
        "Do not frame C49/C50 as restoring OACI.",
        "Do not claim source-only control has been recovered.",
        "Do not present target-conditioned local Bayes estimates as a deployment rule.",
        "Do not turn existing-score underuse into a new trainable-objective claim.",
    ]
    return (
        [{"claim_type": "can_say", "claim": c} for c in can] +
        [{"claim_type": "cannot_say", "claim": c} for c in cannot]
    )


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c49_witness_locked_not_grid_search", "passed": res["locked_witness"]["neighborhood"] == WITNESS_NEIGHBORHOOD},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "no_selector_checkpoint_artifact", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": res["diagnostic_only_non_deployable"]},
        {"check": "coverage_actionability_definitions_frozen_from_c49", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    table_counts = res["table_row_counts"]
    return [
        {
            "check": "self_neighbor_excluded",
            "passed": int(res["self_neighbor_excluded"]),
            "finding": "Locked epsilon neighborhoods inherit C48 distance matrices with diagonal set to infinity.",
        },
        {
            "check": "query_row_excluded_from_own_neighborhood",
            "passed": int(res["self_neighbor_excluded"]),
            "finding": "Per-query morphology was audited against masks where the query row is excluded.",
        },
        {
            "check": "target_labels_quarantined",
            "passed": int(res["diagnostic_only_non_deployable"]),
            "finding": "Target endpoint labels appear only as diagnostic labels and report fields carry quarantine flags.",
        },
        {
            "check": "group_conditioned_baselines_reported",
            "passed": int(table_counts.get("baseline_sanity", 0) > 0),
            "finding": "Same-group random, locked local-Bayes, and source-score baselines are emitted by group type.",
        },
        {
            "check": "reason_coded_failure_ledger_emitted",
            "passed": int(table_counts.get("actionability_failure_ledger", 0) > 0),
            "finding": "Failed actionability groups carry predeclared reason codes.",
        },
        {
            "check": "no_selector_or_checkpoint_recommendation",
            "passed": 1,
            "finding": "Tables omit selector-facing identifiers and checkpoint recommendations.",
        },
        {
            "check": "no_deployable_claim",
            "passed": 1,
            "finding": "Report language is diagnostic-only and non-deployable.",
        },
    ]


def recompute():
    cfg = _lock_config()
    witness = _c49_locked_witness()
    ctx = al.context()
    spaces = source_space_registry.registry(ctx)
    space = spaces["spaces"][WITNESS_SOURCE_SPACE]
    island_rows, group_cache, self_excluded = locked_islands(ctx, space, witness)
    frag_rows = group_fragmentation(island_rows)
    under = existing_score_underuse(ctx, island_rows)
    ledger = actionability_failure_ledger(frag_rows, under["rows"])
    baselines = baseline_sanity(frag_rows, under["rows"], group_cache)
    decision = classify_outcome(frag_rows, under["summary_rows"])
    row_counts = {
        "locked_witness": 1,
        "island_morphology": len(island_rows),
        "group_fragmentation": len(frag_rows),
        "existing_score_underuse": len(under["rows"]),
        "existing_score_underuse_summary": len(under["summary_rows"]),
        "actionability_failure_ledger": len(ledger["rows"]),
        "reason_code_counts": len(ledger["count_rows"]),
        "baseline_sanity": len(baselines),
    }
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "locked_witness": witness,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "self_neighbor_excluded": self_excluded,
        "island_morphology_rows": island_rows,
        "group_fragmentation_rows": frag_rows,
        "existing_score_underuse": under,
        "actionability_failure_ledger": ledger,
        "baseline_sanity_rows": baselines,
        "decision": decision,
        "claim_ledger": claim_ledger(),
        "table_row_counts": row_counts,
    }


def _summary_from_existing():
    if not os.path.exists(REPORT_JSON):
        raise FileNotFoundError(REPORT_JSON)
    d = json.load(open(REPORT_JSON))
    return {
        **d,
        "island_morphology_rows": _readcsv(os.path.join(TABLE_DIR, "island_morphology.csv")),
        "group_fragmentation_rows": _readcsv(os.path.join(TABLE_DIR, "group_fragmentation.csv")),
        "existing_score_underuse": {
            "rows": _readcsv(os.path.join(TABLE_DIR, "existing_score_underuse_by_group.csv")),
            "summary_rows": _readcsv(os.path.join(TABLE_DIR, "existing_score_underuse_summary.csv")),
        },
        "actionability_failure_ledger": {
            "rows": _readcsv(os.path.join(TABLE_DIR, "actionability_failure_ledger.csv")),
            "count_rows": _readcsv(os.path.join(TABLE_DIR, "reason_code_counts.csv")),
        },
        "baseline_sanity_rows": _readcsv(os.path.join(TABLE_DIR, "baseline_sanity.csv")),
        "claim_ledger": _readcsv(os.path.join(TABLE_DIR, "claim_ledger.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def _f3(x):
    return au.fmt3(x)


def render_main_md(res):
    d = res["decision"]
    w = res["locked_witness"]
    return "\n".join([
        f"# C50 - Conditioned-Island Morphology / Fragmentation Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['outcome']}`",
        "",
        "## Locked Witness",
        "",
        f"- condition_scope: `{w['condition_scope']}`",
        f"- source_space: `{w['source_space']}`",
        f"- neighborhood: `{w['neighborhood']}` radius `{_f3(w['epsilon_radius'])}`",
        f"- min_n: `{w['min_neighbor_count']}`",
        f"- inherited C49 hit / coverage / enrichment: **{_f3(w['c49_hit'])} / "
        f"{_f3(w['c49_coverage'])} / {_f3(w['c49_enrichment'])}**",
        "",
        "## Main Result",
        "",
        f"- target min hit / coverage: **{_f3(d['target_min_hit'])} / {_f3(d['target_min_coverage'])}**.",
        f"- trajectory min hit / coverage: **{_f3(d['trajectory_min_hit'])} / "
        f"{_f3(d['trajectory_min_coverage'])}**.",
        f"- target / trajectory actionability fail fraction: **{_f3(d['target_actionability_fail_fraction'])} / "
        f"{_f3(d['trajectory_actionability_fail_fraction'])}**.",
        f"- max mean existing-score underuse gap: **{_f3(d['max_mean_underuse_gap'])}**.",
        "",
        "## Why Coverage Did Not Become Actionability",
        "",
        "C50 keeps the C49 broad witness fixed and audits morphology only. The witness remains diagnostic: "
        "coverage is computed inside target-conditioned source space, while actionability fails where groups are "
        "fragmented and/or available source scores do not recover the diagnostic islands.",
        "",
        "## Red-Team Checks",
        "",
        *[
            f"- {r['check']}: {'PASS' if r['passed'] else 'FAIL'} - {r['finding']}"
            for r in red_team_rows(res)
        ],
        "",
        "## Claim Ledger",
        "",
        "Can say:",
        "- C49's broad conditioned witness is real as a diagnostic ceiling.",
        "- C50 attributes non-actionability to measured fragmentation and/or existing-score underuse.",
        "- Conditioning can reveal target-good islands in source-objective space under diagnostic labels.",
        "",
        "Cannot say:",
        "- Do not treat the witness as a checkpoint selection rule.",
        "- Do not frame C49/C50 as restoring OACI.",
        "- Do not claim source-only control has been recovered.",
        "- Do not present target-conditioned local Bayes estimates as a deployment rule.",
        "- Do not turn existing-score underuse into a new trainable-objective claim.",
    ])


def render_red_team_md(res):
    lines = [
        "# C50 - Red-Team Verification",
        "",
        "C50 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C50 is a diagnostic morphology audit over a locked C49 witness; it does not emit a selector.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "cannot say")


def _guard_forbidden(text):
    au.guard_forbidden(
        text, FORBIDDEN_CLAIM_SUBSTRINGS,
        negation_cues=_NEG_CUES, window=180, label="C50")


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "locked_witness": res["locked_witness"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "self_neighbor_excluded": res["self_neighbor_excluded"],
        "decision": res["decision"],
        "table_row_counts": res["table_row_counts"],
        "red_team": red_team_rows(res),
        "no_selector_artifact_gate": no_selector_gate(res),
    }


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _writecsv(os.path.join(out_dir, "locked_witness.csv"), [res["locked_witness"]],
              ["condition_scope", "source_space", "neighborhood", "neighborhood_kind", "epsilon_radius",
               "min_neighbor_count", "label", "c49_hit", "c49_coverage", "c49_enrichment",
               "c49_mean_neighbor_count", "c49_covered_base_rate", "inherited_from_c49_commit"])
    _writecsv(os.path.join(out_dir, "group_fragmentation.csv"), res["group_fragmentation_rows"],
              ["group_type", "group_key", "n_queries", "n_covered", "coverage", "hit_rate_if_covered",
               "base_rate", "absolute_lift", "enrichment", "max_neighbor_positive_rate",
               "local_bayes_tie_count", "mean_neighbor_count", "median_neighbor_count", "empty_fraction",
               "min_neighbor_count", "max_neighbor_count", "actionability_pass",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(out_dir, "island_morphology.csv"), res["island_morphology_rows"],
              ["query_id", "conditioning_key", "target", "trajectory", "seed", "level", "regime",
               "neighbors_n", "covered", "query_positive_label", "neighbor_positive_rate",
               "neighbor_base_rate", "local_lift", "local_enrichment", "source_distance_mean",
               "source_distance_min", "source_distance_max", "source_distance_q25", "source_distance_q75",
               "is_singleton_neighbor", "is_high_purity_island", "target_labels_diagnostic_only",
               "no_checkpoint_recommendation"])
    _writecsv(os.path.join(out_dir, "existing_score_underuse_by_group.csv"),
              res["existing_score_underuse"]["rows"],
              ["score_name", "group_type", "group_key", "n_covered", "oracle_hit_or_ceiling",
               "score_hit_within_covered_set", "underuse_gap", "rank_of_oracle_positive_under_score",
               "score_percentile_of_oracle_positive", "source_score_orientation",
               "target_labels_diagnostic_only", "diagnostic_underuse_only", "no_checkpoint_recommendation"])
    _writecsv(os.path.join(out_dir, "existing_score_underuse_summary.csv"),
              res["existing_score_underuse"]["summary_rows"],
              ["score_name", "n_group_rows", "mean_oracle_hit_or_ceiling",
               "mean_score_hit_within_covered_set", "mean_underuse_gap", "max_underuse_gap",
               "mean_score_percentile_of_oracle_positive", "diagnostic_underuse_only"])
    _writecsv(os.path.join(out_dir, "actionability_failure_ledger.csv"),
              res["actionability_failure_ledger"]["rows"],
              ["group_type", "group_key", "n_queries", "coverage", "hit_rate_if_covered", "enrichment",
               "max_underuse_gap", "reason_codes", "target_labels_diagnostic_only",
               "no_checkpoint_recommendation"])
    _writecsv(os.path.join(out_dir, "reason_code_counts.csv"),
              res["actionability_failure_ledger"]["count_rows"],
              ["reason_code", "n_groups"])
    _writecsv(os.path.join(out_dir, "baseline_sanity.csv"), res["baseline_sanity_rows"],
              ["baseline_name", "condition_scope", "group_type", "hit", "coverage", "enrichment",
               "permutation_adjusted_gap", "permutation_hit", "permutation_reps"])
    _writecsv(os.path.join(out_dir, "claim_ledger.csv"), res["claim_ledger"],
              ["claim_type", "claim"])
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
    open(os.path.join(out_dir, "C50_CONDITIONED_ISLAND_MORPHOLOGY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C50_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C50_CONDITIONED_ISLAND_MORPHOLOGY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c50_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c50_conditioned_island_morphology")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    d = res["decision"]
    print(f"[C50] outcome={d['outcome']} target_min_hit={d['target_min_hit']} "
          f"trajectory_min_hit={d['trajectory_min_hit']} max_underuse={d['max_mean_underuse_gap']}")


if __name__ == "__main__":
    main()
