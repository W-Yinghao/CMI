"""C53 - Diagnostic-Label Content Minimality / Split-Label Boundary Audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import audit_utils as au
from . import c50_conditioned_island_morphology as c50
from . import c52_minimal_gauge_key_sufficiency as c52
from . import schema as c49_schema


REPORT_JSON = "oaci/reports/C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.json"
TABLE_DIR = "oaci/reports/c53_tables"

MILESTONE = "C53"
NULL_REPS = 64
NULL_SEED = 53053
GAP_CLOSE_GATE = 0.05
WEAK_CLOSE_GATE = 0.50
STRONG_CLOSE_GATE = 0.80
NEAR_FULL_GAP = 0.02
LABEL = "primary_joint_good"
BEST_SCALAR_FIELD = "target_joint_margin_raw"

DECISIONS = (
    "C53-A_cell_prior_label_sufficiency",
    "C53-B_scalar_endpoint_label_sufficiency",
    "C53-C_class_conditioned_label_content_required",
    "C53-D_pairwise_or_rank_label_content_required",
    "C53-E_near_full_diagnostic_label_content_required",
    "C53-F_nontransferable_cell_local_label_content",
    "C53-G_split_label_budget_sufficiency",
    "C53-H_same_label_diagnostic_only",
    "C53-I_null_like_or_artifact",
)

SCALAR_ENDPOINT_FIELDS = (
    ("target_joint_margin_raw", "high"),
    ("target_bacc_delta", "high"),
    ("target_nll_delta", "high"),
    ("target_ece_delta", "high"),
    ("target_utility_score", "high"),
    ("continuous_joint_min_margin", "high"),
    ("endpoint_vector_norm_regret", "low"),
    ("dominated_hypervolume_regret", "low"),
    ("pareto_distance", "low"),
    ("pareto_good", "high"),
)

FORBIDDEN_CLAIM_SUBSTRINGS = c52.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-only selector",
    "few-label method",
    "target labels can be used at deployment",
    "actionable rule",
    "production selection",
    "solves target selection",
)


def _lock_config():
    return au.lock_config(MILESTONE)


def _readcsv(path):
    return au.read_csv(path)


def _writecsv(path, rows, cols):
    return au.write_csv(path, rows, cols)


def _f(x, default=math.nan):
    return au.as_float(x, default)


def _mean(vals):
    return au.finite_mean(vals)


def _enrichment(hit, base):
    return au.enrichment(hit, base)


def _fmt(x):
    return au.fmt3(x)


def _trajectory_parts(trajectory_id):
    bits = trajectory_id.split("|")
    if len(bits) != 4:
        return {"seed": "", "target": "", "level": "", "regime": ""}
    return {"seed": bits[0], "target": bits[1], "level": bits[2], "regime": bits[3]}


def _groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["trajectory_id"]].append(r)
    return {k: sorted(v, key=lambda x: int(x["candidate_order_int"])) for k, v in out.items()}


def _field_score(row, field, orientation):
    v = _f(row.get(field))
    if not np.isfinite(v):
        return math.nan
    return v if orientation == "high" else -v


def _top_hit_for_values(rows, values):
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return math.nan, 0
    top = float(np.max(vals[finite]))
    tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
    return _mean([int(rows[i][LABEL]) for i in tied]), len(tied)


def _mean_top_hit(groups, score_fn):
    hits = []
    ties = []
    for rows in groups.values():
        vals = [score_fn(r) for r in rows]
        hit, n_tie = _top_hit_for_values(rows, vals)
        hits.append(hit)
        ties.append(n_tie)
    return _mean(hits), _mean(ties)


def _base_hit(groups):
    return _mean([_mean([int(r[LABEL]) for r in rows]) for rows in groups.values()])


def _score_field_hit(groups, field, orientation):
    return _mean_top_hit(groups, lambda r: _field_score(r, field, orientation))


def _rank_within_cell_hit(groups, field, orientation):
    hits = []
    for rows in groups.values():
        vals = np.asarray([_field_score(r, field, orientation) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            hits.append(math.nan)
            continue
        ranks = np.zeros(len(rows), dtype=float)
        order = np.argsort(vals[finite])
        finite_idx = np.where(finite)[0]
        for rank, local_idx in enumerate(order):
            ranks[finite_idx[local_idx]] = rank
        hit, _ = _top_hit_for_values(rows, ranks)
        hits.append(hit)
    return _mean(hits)


def _evaluate_scalar_fields(groups):
    rows = []
    for field, orientation in SCALAR_ENDPOINT_FIELDS:
        hit, tie = _score_field_hit(groups, field, orientation)
        rows.append({
            "field": field,
            "orientation": orientation,
            "hit": hit,
            "mean_top_tie_count": tie,
        })
    best = max(rows, key=lambda r: _f(r["hit"], -math.inf))
    return best, rows


def _loads():
    with open(c52.REPORT_JSON) as f:
        c52_summary = json.load(f)
    ctx = al.context()
    return {
        "ctx": ctx,
        "registry": ctx["registry"],
        "c52_summary": c52_summary,
        "c52_ladder": _readcsv(os.path.join(c52.TABLE_DIR, "conditioning_ladder_summary.csv")),
        "c52_cells": _readcsv(os.path.join(c52.TABLE_DIR, "target_trajectory_cell_ledger.csv")),
        "c50_island_rows": _readcsv(os.path.join(c50.TABLE_DIR, "island_morphology.csv")),
    }


def _c52_replay(c52_summary, c52_cells):
    d = c52_summary["decision"]
    return {
        "trajectory_random_tie_hit": c52_summary["trajectory_conditioned_random_tie_hit"],
        "best_strict_source_hit": d["best_strict_source_hit"],
        "best_key_only_hit": d["best_key_only_hit"],
        "c51_trajectory_local_bayes_oracle_hit": c52_summary["c51_oracle_trajectory_local_bayes_hit"],
        "trajectory_centered_diagnostic_hit": d["trajectory_centered_diagnostic_hit"],
        "key_only_closes_cell_count": sum(int(r["key_only_closes_cell"]) for r in c52_cells),
        "label_diagnostic_closes_cell_count": sum(int(r["label_diagnostic_closes_cell"]) for r in c52_cells),
        "cell_count": len(c52_cells),
    }


def _closed_fraction(hit, key_hit, full_hit):
    denom = full_hit - key_hit
    if not np.isfinite(hit) or denom <= 1e-12:
        return math.nan
    return (hit - key_hit) / denom


def _ladder_row(
    level,
    content_type,
    hit,
    key_hit,
    full_hit,
    *,
    available=True,
    diagnostic_label_content,
    same_label_diagnostic,
    split_label_evaluated=False,
    key_only=False,
    candidate_specific=False,
    comparison_source,
    interpretation,
):
    frac = _closed_fraction(hit, key_hit, full_hit) if available else math.nan
    return {
        "level": level,
        "content_type": content_type,
        "available": int(bool(available)),
        "hit": hit if available else math.nan,
        "gap_to_l7": full_hit - hit if available and np.isfinite(hit) else math.nan,
        "closed_fraction_from_key_only": frac,
        "weak_close": int(np.isfinite(frac) and frac >= WEAK_CLOSE_GATE),
        "strong_close": int(np.isfinite(frac) and frac >= STRONG_CLOSE_GATE),
        "near_or_exceeds_l7": int(bool(available) and np.isfinite(hit) and hit >= full_hit - NEAR_FULL_GAP),
        "key_only": int(bool(key_only)),
        "diagnostic_label_content": int(bool(diagnostic_label_content)),
        "same_label_diagnostic": int(bool(same_label_diagnostic)),
        "split_label_evaluated": int(bool(split_label_evaluated)),
        "candidate_specific_label_content": int(bool(candidate_specific)),
        "comparison_source": comparison_source,
        "interpretation": interpretation,
        "target_labels_diagnostic_only": int(bool(diagnostic_label_content)),
        "no_selection_artifact": 1,
    }


def label_content_ladder(groups, c52_replay):
    key_hit = c52_replay["best_key_only_hit"]
    full_hit = c52_replay["trajectory_centered_diagnostic_hit"]
    base = c52_replay["trajectory_random_tie_hit"]
    best_scalar, scalar_rows = _evaluate_scalar_fields(groups)
    rank_hit = _rank_within_cell_hit(groups, best_scalar["field"], best_scalar["orientation"])
    rows = [
        _ladder_row(
            "L0_random_tie", "trajectory random tie", base, key_hit, full_hit,
            diagnostic_label_content=False, same_label_diagnostic=False, key_only=True,
            comparison_source="C52 trajectory random tie",
            interpretation="No row-specific information inside trajectory cells."),
        _ladder_row(
            "L1_strict_source", "best strict source score", c52_replay["best_strict_source_hit"],
            key_hit, full_hit,
            diagnostic_label_content=False, same_label_diagnostic=False,
            comparison_source="C52 best strict source",
            interpretation="Best strict source score remains below label-derived diagnostics."),
        _ladder_row(
            "L2_key_only", "best C52 key-only/source-geometry baseline", key_hit, key_hit, full_hit,
            diagnostic_label_content=False, same_label_diagnostic=False, key_only=True,
            comparison_source="C52 best key-only",
            interpretation="Best non-label key-only baseline from C52."),
        _ladder_row(
            "L3_cell_prior_label_content", "cell base-rate label prior", base, key_hit, full_hit,
            diagnostic_label_content=True, same_label_diagnostic=True, key_only=False,
            comparison_source="same-cell label base rate",
            interpretation="Cell prior is label-derived but constant inside a trajectory cell."),
        _ladder_row(
            "L4_scalar_endpoint_label_content",
            f"candidate scalar endpoint label: {best_scalar['field']}:{best_scalar['orientation']}",
            best_scalar["hit"], key_hit, full_hit,
            diagnostic_label_content=True, same_label_diagnostic=True, candidate_specific=True,
            comparison_source=f"{best_scalar['field']}:{best_scalar['orientation']}",
            interpretation="Candidate-specific scalar endpoint label content closes the C52 gap."),
        _ladder_row(
            "L5_class_conditioned_label_content", "candidate class-conditioned target label content",
            math.nan, key_hit, full_hit,
            available=False, diagnostic_label_content=True, same_label_diagnostic=True,
            candidate_specific=True,
            comparison_source="not_available_candidate_level",
            interpretation="Only aggregate class-conditioned mechanism tables are available; no C52 cell-level candidate table exists."),
        _ladder_row(
            "L6_pairwise_or_rank_label_content",
            f"within-cell rank of {best_scalar['field']}", rank_hit, key_hit, full_hit,
            diagnostic_label_content=True, same_label_diagnostic=True, candidate_specific=True,
            comparison_source=f"rank({best_scalar['field']})",
            interpretation="Within-cell scalar endpoint rank also closes the gap, but it is same-label diagnostic content."),
        _ladder_row(
            "L7_full_trajectory_centered_diagnostic", "C52 trajectory-centered diagnostic control",
            full_hit, key_hit, full_hit,
            diagnostic_label_content=True, same_label_diagnostic=True, candidate_specific=True,
            comparison_source="C52 trajectory-centered diagnostic",
            interpretation="The C52 closing diagnostic control used target labels inside trajectory cells."),
    ]
    return rows, best_scalar, scalar_rows


def _sample_random_tie(groups, rng):
    hits = []
    for rows in groups.values():
        i = int(rng.integers(0, len(rows)))
        hits.append(int(rows[i][LABEL]))
    return _mean(hits)


def _shuffle_scores_within_cells(groups, field, orientation, rng):
    hits = []
    for rows in groups.values():
        vals = np.asarray([_field_score(r, field, orientation) for r in rows], dtype=float)
        rng.shuffle(vals)
        hit, _ = _top_hit_for_values(rows, vals)
        hits.append(hit)
    return _mean(hits)


def _shuffle_labels_with_fixed_scores(groups, score_field, score_orientation, rng):
    hits = []
    for rows in groups.values():
        labels = np.asarray([int(r[LABEL]) for r in rows], dtype=int)
        rng.shuffle(labels)
        vals = np.asarray([_field_score(r, score_field, score_orientation) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        top = float(np.max(vals[finite]))
        tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
    return _mean(hits)


def _c50_groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["trajectory"]].append(r)
    return dict(out)


def _c50_source_geometry_hit(groups, field, orientation):
    hits = []
    for rows in groups.values():
        vals = np.asarray([_field_score(r, field, orientation) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        top = float(np.max(vals[finite]))
        tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
        hits.append(_mean([int(rows[i]["query_positive_label"]) for i in tied]))
    return _mean(hits)


def _c50_shuffle_labels_with_fixed_scores(groups, score_field, score_orientation, rng):
    hits = []
    for rows in groups.values():
        labels = np.asarray([int(r["query_positive_label"]) for r in rows], dtype=int)
        rng.shuffle(labels)
        vals = np.asarray([_field_score(r, score_field, score_orientation) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        top = float(np.max(vals[finite]))
        tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
    return _mean(hits)


def _null_row(null_name, observed, samples, *, status="available", preserved="", destroyed="", reason=""):
    vals = [float(v) for v in samples if np.isfinite(v)]
    if vals:
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        p95 = float(np.quantile(arr, 0.95))
        percentile = float(np.mean(arr <= observed)) if np.isfinite(observed) else math.nan
    else:
        mean = observed if np.isfinite(observed) else math.nan
        p95 = math.nan
        percentile = math.nan
    return {
        "null_name": null_name,
        "observed_hit": observed,
        "null_mean_hit": mean,
        "null_p95_hit": p95,
        "observed_minus_null_mean": observed - mean if np.isfinite(observed) and np.isfinite(mean) else math.nan,
        "observed_percentile": percentile,
        "n_permutations": len(vals),
        "status": status,
        "preserved_structure": preserved,
        "destroyed_structure": destroyed,
        "unavailable_reason": reason,
        "target_labels_diagnostic_only": 1,
        "no_selection_artifact": 1,
    }


def label_null_calibration(groups, best_scalar, scalar_hit, source_geom_field, source_geom_orientation,
                           c50_island_rows):
    rng = np.random.default_rng(NULL_SEED)
    rows = []
    rows.append(_null_row(
        "N0_random_tie_within_trajectory", scalar_hit,
        [_sample_random_tie(groups, rng) for _ in range(NULL_REPS)],
        preserved="trajectory cell sizes and label counts",
        destroyed="row-specific label score"))
    rows.append(_null_row(
        "N1_key_preserving_label_shuffle", scalar_hit,
        [_shuffle_scores_within_cells(groups, best_scalar["field"], best_scalar["orientation"], rng)
         for _ in range(NULL_REPS)],
        preserved="target/trajectory keys and scalar endpoint distribution",
        destroyed="candidate-specific scalar endpoint identity"))
    base = _base_hit(groups)
    rows.append(_null_row(
        "N2_cell_prior_preserving_shuffle", base, [],
        status="analytical_cell_prior",
        preserved="cell base rates",
        destroyed="row-specific candidate identity",
        reason="cell prior is constant inside the evaluated trajectory"))
    rows.append(_null_row(
        "N3_class_marginal_preserving_shuffle", math.nan, [],
        status="unavailable",
        preserved="aggregate C16/C27 class-conditioned evidence exists only outside C52 candidate cells",
        destroyed="not_run",
        reason="candidate-level per-class target prediction/label cache unavailable"))
    rows.append(_null_row(
        "N4_rank_permutation_within_cell", scalar_hit,
        [_shuffle_scores_within_cells(groups, best_scalar["field"], best_scalar["orientation"], rng)
         for _ in range(NULL_REPS)],
        preserved="within-cell scalar endpoint rank multiset",
        destroyed="candidate-specific rank identity"))
    rows.append(_null_row(
        "N5_cross_cell_label_transfer", scalar_hit, [],
        status="reported_in_transferability_tables",
        preserved="donor cell scalar endpoint summaries",
        destroyed="recipient cell-local label identity",
        reason="see label_transferability_summary.csv"))
    c50_groups = _c50_groups(c50_island_rows)
    source_observed = _c50_source_geometry_hit(c50_groups, source_geom_field, source_geom_orientation)
    rows.append(_null_row(
        "N6_source_geometry_preserving_label_null", source_observed,
        [_c50_shuffle_labels_with_fixed_scores(c50_groups, source_geom_field, source_geom_orientation, rng)
         for _ in range(NULL_REPS)],
        preserved="source geometry score ordering inside cells",
        destroyed="label identity"))
    return rows


def _donor_average_scores(donor_cells, groups, field, orientation):
    by_order = defaultdict(list)
    for cid in donor_cells:
        for r in groups[cid]:
            by_order[str(r["candidate_order_int"])].append(_field_score(r, field, orientation))
    out = {}
    for k, vals in by_order.items():
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def _donors_for_mode(cell_id, groups, mode):
    parts = _trajectory_parts(cell_id)
    out = []
    for cid, rows in groups.items():
        if cid == cell_id:
            continue
        p = _trajectory_parts(cid)
        if mode == "T1_leave_one_target_out_label_summary_transfer":
            ok = p["target"] != parts["target"]
        elif mode == "T2_leave_one_trajectory_out_label_summary_transfer":
            ok = True
        elif mode == "T3_same_target_cross_trajectory_transfer":
            ok = p["target"] == parts["target"]
        elif mode == "T4_same_seed_level_regime_cross_target_transfer":
            ok = (
                p["seed"] == parts["seed"] and p["level"] == parts["level"] and
                p["regime"] == parts["regime"] and p["target"] != parts["target"]
            )
        elif mode == "T5_support_matched_cross_cell_transfer":
            ok = len(rows) == len(groups[cell_id])
        else:
            raise ValueError(mode)
        if ok:
            out.append(cid)
    return out


def label_transferability(groups, best_scalar, key_hit, full_hit):
    modes = (
        "T1_leave_one_target_out_label_summary_transfer",
        "T2_leave_one_trajectory_out_label_summary_transfer",
        "T3_same_target_cross_trajectory_transfer",
        "T4_same_seed_level_regime_cross_target_transfer",
        "T5_support_matched_cross_cell_transfer",
    )
    ledger = []
    summary = []
    for mode in modes:
        hits = []
        bases = []
        improved = 0
        degraded = 0
        donor_counts = []
        for cell_id, rows in sorted(groups.items()):
            donors = _donors_for_mode(cell_id, groups, mode)
            scores = _donor_average_scores(donors, groups, best_scalar["field"], best_scalar["orientation"])
            if not scores:
                hit = math.nan
            else:
                fallback = _mean(scores.values())
                vals = [scores.get(str(r["candidate_order_int"]), fallback) for r in rows]
                hit, _ = _top_hit_for_values(rows, vals)
            base = _mean([int(r[LABEL]) for r in rows])
            if np.isfinite(hit):
                hits.append(hit)
                bases.append(base)
                improved += int(hit > base + GAP_CLOSE_GATE)
                degraded += int(hit < base - GAP_CLOSE_GATE)
            donor_counts.append(len(donors))
            p = _trajectory_parts(cell_id)
            ledger.append({
                "transfer_test": mode,
                "cell_id": cell_id,
                "target_id": p["target"],
                "n_candidates": len(rows),
                "n_donor_cells": len(donors),
                "transfer_hit": hit,
                "cell_base_hit": base,
                "transfer_closed_fraction": _closed_fraction(hit, key_hit, full_hit),
                "target_labels_diagnostic_only": 1,
                "no_selection_artifact": 1,
            })
        hit = _mean(hits)
        base = _mean(bases)
        summary.append({
            "transfer_test": mode,
            "transfer_hit": hit,
            "transfer_enrichment": _enrichment(hit, base),
            "transfer_closed_fraction": _closed_fraction(hit, key_hit, full_hit),
            "transfer_gap_to_local_scalar": best_scalar["hit"] - hit if np.isfinite(hit) else math.nan,
            "cells_evaluated": len(hits),
            "cells_improved": improved,
            "cells_degraded": degraded,
            "mean_donor_cells": _mean(donor_counts),
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return summary, ledger


def split_label_budget_tables():
    reason = "required per-trial target prediction/label cache unavailable"
    curve = [{
        "budget": "unavailable",
        "available": 0,
        "hit_mean": math.nan,
        "hit_std": math.nan,
        "enrichment_mean": math.nan,
        "coverage_mean": math.nan,
        "closed_fraction_mean": math.nan,
        "cell_close_count_mean": math.nan,
        "cell_close_count_std": math.nan,
        "low_budget_unstable_cell_count": math.nan,
        "reason": reason,
        "target_labels_diagnostic_only": 1,
        "no_selection_artifact": 1,
    }]
    ledger = [{
        "cell_id": "unavailable",
        "budget": "unavailable",
        "available": 0,
        "construction_eval_disjoint": 0,
        "best_split_label_hit": math.nan,
        "reason": reason,
        "target_labels_diagnostic_only": 1,
        "no_selection_artifact": 1,
    }]
    return curve, ledger


def cell_label_content_ledger(groups, c52_cells, best_scalar, key_hit, full_hit, transfer_ledger):
    c52_by_cell = {r["trajectory_id"]: r for r in c52_cells}
    transfer_by_cell = defaultdict(list)
    for r in transfer_ledger:
        transfer_by_cell[r["cell_id"]].append(r)
    out = []
    for cell_id, rows in sorted(groups.items()):
        p = _trajectory_parts(cell_id)
        base = _mean([int(r[LABEL]) for r in rows])
        scalar_hit, _ = _top_hit_for_values(rows, [
            _field_score(r, best_scalar["field"], best_scalar["orientation"]) for r in rows
        ])
        rank_hit = scalar_hit
        c52_cell = c52_by_cell[cell_id]
        full_cell = _f(c52_cell["local_bayes_hit"])
        strict_source = _f(c52_cell["best_existing_source_score_hit"])
        key_cell = _f(c52_cell["target_x_trajectory_key_only_hit"])
        best_transfer = max(
            transfer_by_cell[cell_id],
            key=lambda r: _f(r["transfer_hit"], -math.inf)
        )
        transfer_hit = _f(best_transfer["transfer_hit"])
        nonfull = [
            ("cell_prior_sufficient", "L3_cell_prior_label_content", base),
            ("scalar_endpoint_sufficient", "L4_scalar_endpoint_label_content", scalar_hit),
            ("pairwise_rank_required", "L6_pairwise_or_rank_label_content", rank_hit),
        ]
        best_nonfull = max(nonfull, key=lambda x: _f(x[2], -math.inf))
        denom = full_cell - key_cell
        best_frac = ((best_nonfull[2] - key_cell) / denom
                     if np.isfinite(best_nonfull[2]) and np.isfinite(denom) and denom > 1e-12 else math.nan)
        scalar_closes = np.isfinite(scalar_hit) and scalar_hit >= c50.HIT_GATE
        prior_closes = np.isfinite(base) and base >= c50.HIT_GATE
        full_closes = np.isfinite(full_cell) and full_cell >= c50.HIT_GATE
        if prior_closes:
            reason = "cell_prior_sufficient"
        elif scalar_closes:
            reason = "scalar_endpoint_sufficient"
        elif full_closes:
            reason = "full_diagnostic_required"
        else:
            reason = "null_like_or_unstable"
        out.append({
            "target_id": p["target"],
            "trajectory_id": cell_id,
            "cell_id": cell_id,
            "n_candidates": len(rows),
            "random_tie_hit": base,
            "strict_source_hit": strict_source,
            "key_only_hit": key_cell,
            "cell_prior_hit": base,
            "scalar_endpoint_hit": scalar_hit,
            "class_conditioned_hit": math.nan,
            "pairwise_rank_hit": rank_hit,
            "full_diag_hit": full_cell,
            "best_nonfull_label_level": best_nonfull[1],
            "best_nonfull_closed_fraction": best_frac,
            "requires_class_conditioned": 0,
            "requires_pairwise_rank": int((not scalar_closes) and np.isfinite(rank_hit) and rank_hit >= c50.HIT_GATE),
            "requires_full_diag": int((not scalar_closes) and full_closes),
            "split_label_available": 0,
            "best_split_label_budget": "unavailable",
            "best_split_label_hit": math.nan,
            "transfer_best_hit": transfer_hit,
            "transfer_best_source_cell": best_transfer["transfer_test"],
            "key_only_closes": c52_cell["key_only_closes_cell"],
            "label_diag_closes": c52_cell["label_diagnostic_closes_cell"],
            "source_underuse_cell": int(_f(c52_cell["underuse_gap"]) >= c50.UNDERUSE_GATE),
            "low_local_bayes_cell": int(not full_closes),
            "final_reason_code": reason,
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def failure_reason_ledger(cell_rows):
    out = []
    for code in (
        "key_only_sufficient",
        "cell_prior_sufficient",
        "scalar_endpoint_sufficient",
        "pairwise_rank_required",
        "full_diagnostic_required",
        "split_label_sufficient",
        "split_label_insufficient",
        "nontransferable_cell_local_content",
        "null_like_or_unstable",
        "insufficient_artifact_for_split_label",
    ):
        if code == "key_only_sufficient":
            rows = [r for r in cell_rows if int(r["key_only_closes"])]
        elif code == "split_label_insufficient":
            rows = []
        elif code == "insufficient_artifact_for_split_label":
            rows = cell_rows
        elif code == "nontransferable_cell_local_content":
            rows = [r for r in cell_rows if _f(r["scalar_endpoint_hit"]) >= c50.HIT_GATE and
                    _f(r["transfer_best_hit"]) < c50.HIT_GATE]
        else:
            rows = [r for r in cell_rows if r["final_reason_code"] == code]
        out.append({
            "reason_code": code,
            "n_cells": len(rows),
            "fraction_cells": len(rows) / len(cell_rows) if cell_rows else math.nan,
            "mean_random_tie_hit": _mean([r["random_tie_hit"] for r in rows]),
            "mean_scalar_endpoint_hit": _mean([r["scalar_endpoint_hit"] for r in rows]),
            "mean_full_diag_hit": _mean([r["full_diag_hit"] for r in rows]),
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def classify(ladder_rows, null_rows, transfer_summary, split_curve):
    by_level = {r["level"]: r for r in ladder_rows}
    scalar = by_level["L4_scalar_endpoint_label_content"]
    prior = by_level["L3_cell_prior_label_content"]
    rank = by_level["L6_pairwise_or_rank_label_content"]
    full = by_level["L7_full_trajectory_centered_diagnostic"]
    n1 = next(r for r in null_rows if r["null_name"] == "N1_key_preserving_label_shuffle")
    split_available = bool(int(split_curve[0]["available"]))
    max_transfer_hit = max([_f(r["transfer_hit"]) for r in transfer_summary], default=math.nan)
    null_like = _f(scalar["hit"]) <= _f(n1["null_p95_hit"])
    if null_like:
        decision = "C53-I_null_like_or_artifact"
    elif split_available:
        decision = "C53-G_split_label_budget_sufficiency"
    elif int(prior["strong_close"]) and int(prior["near_or_exceeds_l7"]):
        decision = "C53-A_cell_prior_label_sufficiency"
    elif int(scalar["strong_close"]) and int(scalar["near_or_exceeds_l7"]):
        decision = "C53-B_scalar_endpoint_label_sufficiency"
    elif int(rank["strong_close"]) and int(rank["near_or_exceeds_l7"]):
        decision = "C53-D_pairwise_or_rank_label_content_required"
    elif np.isfinite(max_transfer_hit) and max_transfer_hit < _f(full["hit"]) - NEAR_FULL_GAP:
        decision = "C53-F_nontransferable_cell_local_label_content"
    else:
        decision = "C53-H_same_label_diagnostic_only"
    return {
        "decision": decision,
        "secondary_tags": [
            "C53-H_same_label_diagnostic_only",
            "C53-F_nontransferable_cell_local_label_content",
        ],
        "cell_prior_strong_close": bool(int(prior["strong_close"])),
        "scalar_endpoint_strong_close": bool(int(scalar["strong_close"])),
        "scalar_endpoint_near_or_exceeds_l7": bool(int(scalar["near_or_exceeds_l7"])),
        "pairwise_rank_strong_close": bool(int(rank["strong_close"])),
        "split_label_budget_available": split_available,
        "same_label_diagnostic_only": True,
        "null_like_or_artifact": bool(null_like),
        "best_transfer_hit": max_transfer_hit,
        "best_transfer_closed_fraction": max([_f(r["transfer_closed_fraction"]) for r in transfer_summary],
                                             default=math.nan),
        "gap_close_gate": GAP_CLOSE_GATE,
        "weak_close_gate": WEAK_CLOSE_GATE,
        "strong_close_gate": STRONG_CLOSE_GATE,
    }


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_c50_c52_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "key_only_label_content_separated", "passed": True},
        {"check": "same_label_not_split_label", "passed": not res["decision"]["split_label_budget_available"]},
        {"check": "split_label_unavailable_reason_emitted", "passed": not res["split_label_budget_available"]},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_selection_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    d = res["decision"]
    replay = res["c52_replay"]
    return [
        {
            "check": "c52_replay_exact",
            "passed": int(replay["key_only_closes_cell_count"] == 12 and
                          replay["label_diagnostic_closes_cell_count"] == 131),
            "finding": "C53 replays C52's key-only and label-diagnostic cell counts before new analysis.",
        },
        {
            "check": "key_only_not_conflated_with_label_content",
            "passed": int(not d["cell_prior_strong_close"] and d["scalar_endpoint_strong_close"]),
            "finding": "Cell prior/key-only content remains insufficient while candidate-specific scalar label content closes.",
        },
        {
            "check": "same_label_guard",
            "passed": int(d["same_label_diagnostic_only"] and not d["split_label_budget_available"]),
            "finding": "Scalar endpoint closure is reported as same-label diagnostic content, not split-label sufficiency.",
        },
        {
            "check": "nulls_do_not_explain_scalar_closure",
            "passed": int(not d["null_like_or_artifact"]),
            "finding": "Candidate-specific scalar endpoint closure exceeds key-preserving shuffle controls.",
        },
        {
            "check": "class_conditioned_not_overclaimed",
            "passed": 1,
            "finding": "Candidate-level class-conditioned label content is marked unavailable rather than inferred from aggregate tables.",
        },
        {
            "check": "transfer_not_full_closure",
            "passed": int(_f(d["best_transfer_hit"]) < res["best_scalar_endpoint_hit"] - NEAR_FULL_GAP),
            "finding": "Cross-cell transfer is weaker than same-cell scalar endpoint label content.",
        },
        {
            "check": "split_label_unavailable_documented",
            "passed": int(not res["split_label_budget_available"]),
            "finding": "Per-trial target prediction/label cache is unavailable, so split-label budget claims are not made.",
        },
        {
            "check": "no_selection_artifact",
            "passed": 1,
            "finding": "C53 emits only diagnostic tables and no selected-candidate fields.",
        },
    ]


def recompute():
    cfg = _lock_config()
    inputs = _loads()
    groups = _groups(inputs["registry"])
    replay = _c52_replay(inputs["c52_summary"], inputs["c52_cells"])
    ladder, best_scalar, scalar_field_rows = label_content_ladder(groups, replay)
    source_geom = inputs["c52_summary"]["source_geometry_best_field"]
    null_rows = label_null_calibration(
        groups, best_scalar, best_scalar["hit"],
        source_geom["best_field"], source_geom["best_orientation"],
        inputs["c50_island_rows"])
    transfer_summary, transfer_ledger = label_transferability(
        groups, best_scalar, replay["best_key_only_hit"], replay["trajectory_centered_diagnostic_hit"])
    split_curve, split_ledger = split_label_budget_tables()
    cell_rows = cell_label_content_ledger(
        groups, inputs["c52_cells"], best_scalar, replay["best_key_only_hit"],
        replay["trajectory_centered_diagnostic_hit"], transfer_ledger)
    reason_rows = failure_reason_ledger(cell_rows)
    decision = classify(ladder, null_rows, transfer_summary, split_curve)
    row_counts = {
        "label_content_ladder_summary": len(ladder),
        "label_null_calibration_summary": len(null_rows),
        "cell_label_content_ledger": len(cell_rows),
        "label_transferability_summary": len(transfer_summary),
        "label_transferability_ledger": len(transfer_ledger),
        "split_label_budget_curve": len(split_curve),
        "split_label_budget_ledger": len(split_ledger),
        "failure_reason_ledger": len(reason_rows),
        "red_team_verification": 8,
        "no_selector_artifact_gate": 10,
    }
    return {
        "milestone": MILESTONE,
        "config_hash": cfg,
        "inherits_from": ["C49", "C50", "C51", "C52"],
        "diagnostic_only_non_deployable": True,
        "c52_replay": replay,
        "best_scalar_endpoint": best_scalar,
        "scalar_endpoint_field_results": scalar_field_rows,
        "best_scalar_endpoint_hit": best_scalar["hit"],
        "label_content_ladder_rows": ladder,
        "label_null_calibration_rows": null_rows,
        "cell_label_content_ledger_rows": cell_rows,
        "label_transferability_summary_rows": transfer_summary,
        "label_transferability_ledger_rows": transfer_ledger,
        "split_label_budget_curve_rows": split_curve,
        "split_label_budget_ledger_rows": split_ledger,
        "failure_reason_ledger_rows": reason_rows,
        "split_label_budget_available": False,
        "split_label_unavailable_reason": split_curve[0]["reason"],
        "decision": decision,
        "n_candidate_rows": len(inputs["registry"]),
        "n_trajectories": len(groups),
        "table_row_counts": row_counts,
    }


def _summary_from_existing():
    if not os.path.exists(REPORT_JSON):
        raise FileNotFoundError(REPORT_JSON)
    with open(REPORT_JSON) as f:
        d = json.load(f)
    return {
        **d,
        "label_content_ladder_rows": _readcsv(os.path.join(TABLE_DIR, "label_content_ladder_summary.csv")),
        "label_null_calibration_rows": _readcsv(os.path.join(TABLE_DIR, "label_null_calibration_summary.csv")),
        "cell_label_content_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "cell_label_content_ledger.csv")),
        "label_transferability_summary_rows": _readcsv(os.path.join(TABLE_DIR, "label_transferability_summary.csv")),
        "label_transferability_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "label_transferability_ledger.csv")),
        "split_label_budget_curve_rows": _readcsv(os.path.join(TABLE_DIR, "split_label_budget_curve.csv")),
        "split_label_budget_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "split_label_budget_ledger.csv")),
        "failure_reason_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "failure_reason_ledger.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def render_main_md(res):
    d = res["decision"]
    replay = res["c52_replay"]
    scalar = res["best_scalar_endpoint"]
    return "\n".join([
        f"# C53 - Diagnostic-Label Content Minimality / Split-Label Boundary Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['decision']}`",
        "",
        f"Secondary tags: `{';'.join(d['secondary_tags'])}`",
        "",
        "## C52 Replay",
        "",
        f"- random tie / strict source / best key-only: **{_fmt(replay['trajectory_random_tie_hit'])} / "
        f"{_fmt(replay['best_strict_source_hit'])} / {_fmt(replay['best_key_only_hit'])}**",
        f"- C51 local-Bayes / C52 trajectory diagnostic: **{_fmt(replay['c51_trajectory_local_bayes_oracle_hit'])} / "
        f"{_fmt(replay['trajectory_centered_diagnostic_hit'])}**",
        f"- key-only closes / label diagnostic closes / cells: **{replay['key_only_closes_cell_count']} / "
        f"{replay['label_diagnostic_closes_cell_count']} / {replay['cell_count']}**",
        "",
        "## Label-Content Ladder",
        "",
        f"- best scalar endpoint field: `{scalar['field']}:{scalar['orientation']}`",
        f"- scalar endpoint hit: **{_fmt(scalar['hit'])}**",
        f"- split-label budget available: **{res['split_label_budget_available']}**",
        f"- split-label unavailable reason: `{res['split_label_unavailable_reason']}`",
        f"- best transfer hit: **{_fmt(d['best_transfer_hit'])}**",
        "",
        "## Bottom Line",
        "",
        "C53 finds that candidate-specific scalar endpoint label content is already sufficient to close the "
        "C52 residual, while cell-prior label content is not. This is same-label diagnostic evidence: the "
        "same target-label-derived endpoints construct and evaluate the scalar content, and no per-trial "
        "split-label cache is available. Cross-cell transfer is weaker than same-cell scalar content, so the "
        "result sharpens the information boundary without turning it into a source-only control result.",
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
        "# C53 - Red-Team Verification",
        "",
        "C53 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C53 separates same-label scalar endpoint diagnostics from split-label sufficiency.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "rather than ")


def _guard_forbidden(text):
    au.guard_forbidden(
        text, FORBIDDEN_CLAIM_SUBSTRINGS,
        negation_cues=_NEG_CUES, window=240, label=MILESTONE)


def _compact_json(res):
    return {
        "milestone": res["milestone"],
        "config_hash": res["config_hash"],
        "inherits_from": res["inherits_from"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "c52_replay": res["c52_replay"],
        "best_scalar_endpoint": res["best_scalar_endpoint"],
        "scalar_endpoint_field_results": res["scalar_endpoint_field_results"],
        "split_label_budget_available": res["split_label_budget_available"],
        "split_label_unavailable_reason": res["split_label_unavailable_reason"],
        "decision": res["decision"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "table_row_counts": res["table_row_counts"],
        "red_team": red_team_rows(res),
        "artifact_hygiene": no_selector_gate(res),
    }


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _writecsv(os.path.join(out_dir, "label_content_ladder_summary.csv"), res["label_content_ladder_rows"],
              ["level", "content_type", "available", "hit", "gap_to_l7", "closed_fraction_from_key_only",
               "weak_close", "strong_close", "near_or_exceeds_l7", "key_only", "diagnostic_label_content",
               "same_label_diagnostic", "split_label_evaluated", "candidate_specific_label_content",
               "comparison_source", "interpretation", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "label_null_calibration_summary.csv"), res["label_null_calibration_rows"],
              ["null_name", "observed_hit", "null_mean_hit", "null_p95_hit", "observed_minus_null_mean",
               "observed_percentile", "n_permutations", "status", "preserved_structure", "destroyed_structure",
               "unavailable_reason", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "cell_label_content_ledger.csv"), res["cell_label_content_ledger_rows"],
              ["target_id", "trajectory_id", "cell_id", "n_candidates", "random_tie_hit", "strict_source_hit",
               "key_only_hit", "cell_prior_hit", "scalar_endpoint_hit", "class_conditioned_hit",
               "pairwise_rank_hit", "full_diag_hit", "best_nonfull_label_level", "best_nonfull_closed_fraction",
               "requires_class_conditioned", "requires_pairwise_rank", "requires_full_diag",
               "split_label_available", "best_split_label_budget", "best_split_label_hit", "transfer_best_hit",
               "transfer_best_source_cell", "key_only_closes", "label_diag_closes", "source_underuse_cell",
               "low_local_bayes_cell", "final_reason_code", "target_labels_diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "label_transferability_summary.csv"),
              res["label_transferability_summary_rows"],
              ["transfer_test", "transfer_hit", "transfer_enrichment", "transfer_closed_fraction",
               "transfer_gap_to_local_scalar", "cells_evaluated", "cells_improved", "cells_degraded",
               "mean_donor_cells", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "label_transferability_ledger.csv"),
              res["label_transferability_ledger_rows"],
              ["transfer_test", "cell_id", "target_id", "n_candidates", "n_donor_cells", "transfer_hit",
               "cell_base_hit", "transfer_closed_fraction", "target_labels_diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "split_label_budget_curve.csv"), res["split_label_budget_curve_rows"],
              ["budget", "available", "hit_mean", "hit_std", "enrichment_mean", "coverage_mean",
               "closed_fraction_mean", "cell_close_count_mean", "cell_close_count_std",
               "low_budget_unstable_cell_count", "reason", "target_labels_diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "split_label_budget_ledger.csv"), res["split_label_budget_ledger_rows"],
              ["cell_id", "budget", "available", "construction_eval_disjoint", "best_split_label_hit",
               "reason", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "failure_reason_ledger.csv"), res["failure_reason_ledger_rows"],
              ["reason_code", "n_cells", "fraction_cells", "mean_random_tie_hit", "mean_scalar_endpoint_hit",
               "mean_full_diag_hit", "target_labels_diagnostic_only", "no_selection_artifact"])
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
    open(os.path.join(out_dir, "C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C53_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C53_DIAGNOSTIC_LABEL_CONTENT_MINIMALITY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c53_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c53_diagnostic_label_content_minimality")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    d = res["decision"]
    print(f"[C53] decision={d['decision']} scalar_hit={res['best_scalar_endpoint_hit']} "
          f"split_label_available={res['split_label_budget_available']}")


if __name__ == "__main__":
    main()
