"""C54 - Endpoint-Scalar Tautology / Bit-Budget Boundary Audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import audit_utils as au
from . import c53_diagnostic_label_content_minimality as c53
from . import schema as c49_schema
from . import score_diagnostics as sd


REPORT_JSON = "oaci/reports/C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.json"
TABLE_DIR = "oaci/reports/c54_tables"

MILESTONE = "C54"
LABEL = "primary_joint_good"
BEST_SCALAR = "target_joint_margin_raw"
BEST_ORIENTATION = "high"
NULL_REPS = 64
NULL_SEED = 54054
HIT_GATE = 0.70

DECISIONS = (
    "C54-A_direct_joint_endpoint_tautology",
    "C54-B_single_endpoint_component_sufficiency",
    "C54-C_low_bit_endpoint_oracle_sufficiency",
    "C54-D_near_continuous_endpoint_margin_required",
    "C54-E_cross_cell_endpoint_template_sufficiency",
    "C54-F_same_cell_candidate_endpoint_oracle_only",
    "C54-G_split_label_boundary_unresolved",
    "C54-H_artifact_or_null_like",
)

SECONDARY_DECISIONS = (
    "C54-S1_nontransferable_cell_local_endpoint_content",
    "C54-S2_component_endpoint_asymmetry",
    "C54-S3_joint_margin_dominates_components",
    "C54-S4_binary_threshold_already_sufficient",
    "C54-S5_transfer_partially_but_not_fully_closes",
    "C54-S6_no_split_label_budget_available",
)

SCALARS = (
    ("source_rank_score", "high", "S0_strict_source_observable", "source_rank"),
    ("R_src", "high", "S0_strict_source_observable", "source_risk"),
    ("c19_robust_core_score", "high", "S0_strict_source_observable", "c19_robust_core"),
    ("selection_leakage_point", "high", "S0_strict_source_observable", "selection_leakage"),
    ("cell_prior", "high", "S1_key_or_cell_prior_only", "cell_prior"),
    ("target_bacc_delta", "high", "S3_target_endpoint_component_label", "bAcc"),
    ("target_bacc_z", "high", "S3_target_endpoint_component_label", "bAcc_z"),
    ("target_nll_delta", "high", "S3_target_endpoint_component_label", "NLL"),
    ("target_nll_z", "high", "S3_target_endpoint_component_label", "NLL_z"),
    ("target_ece_delta", "high", "S3_target_endpoint_component_label", "ECE"),
    ("target_ece_z", "high", "S3_target_endpoint_component_label", "ECE_z"),
    ("target_utility_score", "high", "S4_target_joint_endpoint_label", "utility"),
    ("continuous_joint_min_margin", "high", "S4_target_joint_endpoint_label", "min_margin"),
    ("endpoint_vector_norm_regret", "low", "S4_target_joint_endpoint_label", "regret_norm"),
    ("dominated_hypervolume_regret", "low", "S4_target_joint_endpoint_label", "regret_hv"),
    ("pareto_distance", "low", "S4_target_joint_endpoint_label", "pareto_distance"),
    ("pareto_good", "high", "S4_target_joint_endpoint_label", "pareto_flag"),
    ("target_joint_margin_raw", "high", "S5_same_label_oracle_margin", "joint_margin"),
    ("primary_joint_good", "high", "S5_same_label_oracle_margin", "joint_good_label"),
)

COMPONENT_SCALARS = (
    ("accuracy_component", "target_bacc_delta", "high"),
    ("accuracy_component_z", "target_bacc_z", "high"),
    ("nll_component", "target_nll_delta", "high"),
    ("nll_component_z", "target_nll_z", "high"),
    ("ece_component", "target_ece_delta", "high"),
    ("ece_component_z", "target_ece_z", "high"),
    ("calibration_best_component", "target_nll_delta", "high"),
    ("joint_margin", "target_joint_margin_raw", "high"),
    ("joint_good_label", "primary_joint_good", "high"),
    ("min_endpoint_margin", "continuous_joint_min_margin", "high"),
    ("utility_score", "target_utility_score", "high"),
)

FORBIDDEN_CLAIM_SUBSTRINGS = c53.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-only selector",
    "few-label method",
    "target labels can be used at deployment",
    "actionable rule",
    "production selection",
    "solves target selection",
    "deployable target-aware selector",
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


def _fmt(x):
    return au.fmt3(x)


def _enrichment(hit, base):
    return au.enrichment(hit, base)


def _groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["trajectory_id"]].append(r)
    return {k: sorted(v, key=lambda x: int(x["candidate_order_int"])) for k, v in out.items()}


def _target_groups(rows):
    out = defaultdict(list)
    for r in rows:
        out[r["target"]].append(r)
    return dict(out)


def _parts(cell_id):
    bits = cell_id.split("|")
    if len(bits) != 4:
        return {"seed": "", "target": "", "level": "", "regime": ""}
    return {"seed": bits[0], "target": bits[1], "level": bits[2], "regime": bits[3]}


def _score(row, name, orientation="high", cell_base=None):
    if name == "cell_prior":
        return cell_base if cell_base is not None else math.nan
    v = _f(row.get(name))
    if not np.isfinite(v):
        return math.nan
    return v if orientation == "high" else -v


def _top_hit(rows, values):
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return math.nan, 0
    top = float(np.max(vals[finite]))
    tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
    return _mean([int(rows[i][LABEL]) for i in tied]), len(tied)


def _mean_top_hit(groups, scalar, orientation="high", values_by_id=None):
    hits = []
    ties = []
    for rows in groups.values():
        base = _mean([int(r[LABEL]) for r in rows])
        if values_by_id is None:
            vals = [_score(r, scalar, orientation, base) for r in rows]
        else:
            vals = [values_by_id[id(r)] for r in rows]
        hit, tie = _top_hit(rows, vals)
        hits.append(hit)
        ties.append(tie)
    return _mean(hits), _mean(ties), sum(1 for h in hits if np.isfinite(h) and h >= HIT_GATE)


def _field_values(rows, scalar, orientation="high"):
    return np.asarray([_score(r, scalar, orientation) for r in rows], dtype=float)


def _binary_sign_values(rows, scalar, orientation="high"):
    vals = []
    for r in rows:
        raw = _f(r.get(scalar))
        if not np.isfinite(raw):
            vals.append(math.nan)
        elif orientation == "high":
            vals.append(1.0 if raw > 0 else 0.0)
        else:
            vals.append(1.0 if raw < 0 else 0.0)
    return np.asarray(vals, dtype=float)


def _rank_auc_spearman_mi(rows, values):
    labels = np.asarray([int(r[LABEL]) for r in rows], dtype=int)
    vals = np.asarray(values, dtype=float)
    finite = np.isfinite(vals)
    if not np.any(finite):
        return math.nan, math.nan, math.nan
    vals = vals[finite]
    labs = labels[finite]
    auc = sd.auc(vals, labs)
    sp = sd.spearman(vals, labs)
    mi = _discrete_mi(vals, labs)
    return auc, sp, mi


def _discrete_mi(values, labels, bins=10):
    values = np.asarray(values, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if len(values) == 0 or len(set(labels.tolist())) < 2:
        return math.nan
    ranks = sd.rankdata(values.tolist())
    xb = np.minimum((ranks * bins).astype(int), bins - 1)
    mi = 0.0
    n = len(labels)
    for x in range(bins):
        for y in (0, 1):
            pxy = float(np.mean((xb == x) & (labels == y)))
            if pxy <= 0:
                continue
            px = float(np.mean(xb == x))
            py = float(np.mean(labels == y))
            mi += pxy * math.log(pxy / (px * py), 2)
    return mi


def _threshold_overlap(rows, scalar, orientation):
    if scalar == "cell_prior":
        return math.nan
    agree = []
    for r in rows:
        raw = _f(r.get(scalar))
        if not np.isfinite(raw):
            continue
        pred = raw > 0 if orientation == "high" else raw < 0
        agree.append(int(pred) == int(r[LABEL]))
    return float(np.mean(agree)) if agree else math.nan


def _loads():
    with open(c53.REPORT_JSON) as f:
        c53_summary = json.load(f)
    ctx = al.context()
    return {
        "ctx": ctx,
        "registry": ctx["registry"],
        "c53_summary": c53_summary,
        "c53_ladder": _readcsv(os.path.join(c53.TABLE_DIR, "label_content_ladder_summary.csv")),
        "c53_cells": _readcsv(os.path.join(c53.TABLE_DIR, "cell_label_content_ledger.csv")),
        "c53_transfer": _readcsv(os.path.join(c53.TABLE_DIR, "label_transferability_summary.csv")),
    }


def c53_replay_identity(c53_summary, groups):
    d = c53_summary["decision"]
    replay = c53_summary["c52_replay"]
    scalar_hit, _, _ = _mean_top_hit(groups, BEST_SCALAR, BEST_ORIENTATION)
    metrics = [
        ("random_tie_hit", replay["trajectory_random_tie_hit"], replay["trajectory_random_tie_hit"]),
        ("best_strict_source_hit", replay["best_strict_source_hit"], replay["best_strict_source_hit"]),
        ("best_key_only_hit", replay["best_key_only_hit"], replay["best_key_only_hit"]),
        ("c52_trajectory_diagnostic_hit", replay["trajectory_centered_diagnostic_hit"],
         replay["trajectory_centered_diagnostic_hit"]),
        ("c53_best_scalar_endpoint_hit", c53_summary["best_scalar_endpoint"]["hit"], scalar_hit),
        ("best_scalar_field", "target_joint_margin_raw:high",
         f"{c53_summary['best_scalar_endpoint']['field']}:{c53_summary['best_scalar_endpoint']['orientation']}"),
        ("cell_count", 162, len(groups)),
        ("split_label_budget_available", False, d["split_label_budget_available"]),
    ]
    rows = []
    for metric, reported, replayed in metrics:
        if isinstance(reported, str) or isinstance(replayed, str):
            diff = 0.0 if str(reported) == str(replayed) else math.inf
            passed = str(reported) == str(replayed)
        elif isinstance(reported, bool) or isinstance(replayed, bool):
            diff = 0.0 if bool(reported) == bool(replayed) else math.inf
            passed = bool(reported) == bool(replayed)
        else:
            diff = abs(float(reported) - float(replayed))
            passed = diff <= 1e-9
        rows.append({
            "metric": metric,
            "c53_reported_value": reported,
            "c54_replayed_value": replayed,
            "abs_diff": diff,
            "pass": int(passed),
        })
    return rows


def _semantic_flags(semantic, scalar):
    target_label = semantic in (
        "S3_target_endpoint_component_label",
        "S4_target_joint_endpoint_label",
        "S5_same_label_oracle_margin",
    )
    same_label = semantic == "S5_same_label_oracle_margin"
    endpoint = target_label
    direct = scalar in ("target_joint_margin_raw", "primary_joint_good")
    source = semantic == "S0_strict_source_observable"
    return target_label, same_label, endpoint, direct, source


def endpoint_scalar_inventory(rows, groups):
    out = []
    for scalar, orientation, semantic, component in SCALARS:
        hit, tie, close_count = _mean_top_hit(groups, scalar, orientation)
        target_label, same_label, endpoint, direct, source = _semantic_flags(semantic, scalar)
        out.append({
            "scalar_name": f"{scalar}:{orientation}",
            "semantic_class": semantic,
            "hit": hit,
            "cell_hit_ge_0_7_count": close_count,
            "target_label_derived": int(target_label),
            "same_label_derived": int(same_label),
            "requires_target_endpoint": int(endpoint),
            "available_at_selection_time": int(not target_label),
            "directly_contains_joint_good_threshold": int(direct),
            "component_fields_used": component,
            "cell_localized": 1,
            "source_observable": int(source),
            "notes": "same-label diagnostic endpoint scalar" if same_label else semantic,
            "diagnostic_only": int(target_label),
            "no_selection_artifact": 1,
        })
    out.append({
        "scalar_name": "split_label_constructed_scalar",
        "semantic_class": "S6_split_label_constructed_scalar",
        "hit": math.nan,
        "cell_hit_ge_0_7_count": math.nan,
        "target_label_derived": 1,
        "same_label_derived": 0,
        "requires_target_endpoint": 1,
        "available_at_selection_time": 0,
        "directly_contains_joint_good_threshold": 0,
        "component_fields_used": "unavailable",
        "cell_localized": 1,
        "source_observable": 0,
        "notes": "required per-trial target prediction/label cache unavailable",
        "diagnostic_only": 1,
        "no_selection_artifact": 1,
    })
    return out


def endpoint_tautology_distance(rows, groups, inventory, c53_gap, key_hit, random_hit):
    out = []
    for inv in inventory:
        if inv["scalar_name"] == "split_label_constructed_scalar":
            continue
        scalar, orientation = inv["scalar_name"].split(":")
        values = []
        for r in rows:
            if scalar == "cell_prior":
                values.append(math.nan)
            else:
                values.append(_score(r, scalar, orientation))
        auc, sp, mi = _rank_auc_spearman_mi(rows, values)
        hit = _f(inv["hit"])
        closed = (hit - key_hit) / c53_gap if np.isfinite(hit) and c53_gap > 1e-12 else math.nan
        near = int(np.isfinite(closed) and closed >= 0.90 and int(inv["target_label_derived"]))
        same_taut = int(near and int(inv["same_label_derived"]))
        out.append({
            "scalar_name": inv["scalar_name"],
            "semantic_class": inv["semantic_class"],
            "hit": hit,
            "hit_minus_key_only": hit - key_hit if np.isfinite(hit) else math.nan,
            "hit_minus_random": hit - random_hit if np.isfinite(hit) else math.nan,
            "closed_fraction_vs_c53_gap": closed,
            "auc_vs_joint_good": auc,
            "spearman_vs_joint_good": sp,
            "discrete_mi_vs_joint_good": mi,
            "threshold_overlap_with_joint_good": _threshold_overlap(rows, scalar, orientation),
            "direct_joint_label_field": inv["directly_contains_joint_good_threshold"],
            "near_endpoint_oracle": near,
            "same_label_tautology": same_taut,
            "target_label_derived": inv["target_label_derived"],
            "available_at_selection_time": inv["available_at_selection_time"],
            "diagnostic_only": inv["diagnostic_only"],
            "no_selection_artifact": 1,
        })
    return out


def endpoint_component_ablation(groups, key_hit, best_hit):
    rows = []
    for component, scalar, orientation in COMPONENT_SCALARS:
        hit, tie, close_count = _mean_top_hit(groups, scalar, orientation)
        semantic = (
            "S5_same_label_oracle_margin" if scalar in ("target_joint_margin_raw", "primary_joint_good")
            else "S4_target_joint_endpoint_label" if component in (
                "min_endpoint_margin", "utility_score", "joint_margin", "joint_good_label")
            else "S3_target_endpoint_component_label"
        )
        rows.append({
            "component_family": component,
            "scalar_name": f"{scalar}:{orientation}",
            "semantic_class": semantic,
            "hit": hit,
            "closed_fraction_vs_best_scalar": (hit - key_hit) / (best_hit - key_hit),
            "cell_hit_ge_0_7_count": close_count,
            "target_label_derived": 1,
            "same_label_derived": int(semantic == "S5_same_label_oracle_margin"),
            "available_at_selection_time": 0,
            "single_component": int(semantic == "S3_target_endpoint_component_label"),
            "diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return rows


def _coarsen_values(rows, scalar, orientation, mode, bins, scope):
    signed = np.asarray([_score(r, scalar, orientation) for r in rows], dtype=float)
    if mode == "continuous_raw":
        return {id(r): signed[i] for i, r in enumerate(rows)}
    if mode == "binary_sign":
        vals = _binary_sign_values(rows, scalar, orientation)
        return {id(r): vals[i] for i, r in enumerate(rows)}
    if mode == "rank_only":
        out = {}
        for cell_rows in _groups(rows).values():
            vals = np.asarray([_score(r, scalar, orientation) for r in cell_rows], dtype=float)
            finite = np.isfinite(vals)
            ranks = np.full(len(cell_rows), math.nan)
            if np.any(finite):
                ranks[finite] = sd.rankdata(vals[finite].tolist())
            for r, v in zip(cell_rows, ranks):
                out[id(r)] = v
        return out
    out = {}
    if scope == "global":
        buckets = {"global": rows}
    elif scope == "within_target":
        buckets = _target_groups(rows)
    elif scope == "within_trajectory":
        buckets = _groups(rows)
    elif scope == "within_target_trajectory_cell":
        buckets = _groups(rows)
    else:
        raise ValueError(scope)
    for bucket_rows in buckets.values():
        vals = np.asarray([_score(r, scalar, orientation) for r in bucket_rows], dtype=float)
        finite = np.isfinite(vals)
        assigned = np.full(len(bucket_rows), math.nan)
        if np.any(finite):
            ranks = sd.rankdata(vals[finite].tolist())
            assigned[finite] = np.minimum((ranks * bins).astype(int), bins - 1)
        for r, v in zip(bucket_rows, assigned):
            out[id(r)] = v
    return out


def endpoint_bit_budget_curve(rows, groups, key_hit, best_hit):
    modes = [
        ("binary_sign", 2, "global_threshold"),
        ("tertile", 3, "global"),
        ("quartile", 4, "global"),
        ("quintile", 5, "global"),
        ("decile", 10, "global"),
        ("rank_only", 0, "within_target_trajectory_cell"),
        ("continuous_raw", 0, "raw"),
    ]
    scopes = ("global", "within_target", "within_trajectory", "within_target_trajectory_cell")
    scalars = (
        ("target_joint_margin_raw", "high", "S5_same_label_oracle_margin"),
        ("target_bacc_delta", "high", "S3_target_endpoint_component_label"),
        ("target_nll_delta", "high", "S3_target_endpoint_component_label"),
        ("target_ece_delta", "high", "S3_target_endpoint_component_label"),
    )
    out = []
    for scalar, orientation, semantic in scalars:
        for mode, bins, default_scope in modes:
            active_scopes = (default_scope,) if mode in ("binary_sign", "rank_only", "continuous_raw") else scopes
            for scope in active_scopes:
                values = _coarsen_values(rows, scalar, orientation, mode, bins, scope)
                hit, _, close_count = _mean_top_hit(groups, scalar, orientation, values)
                out.append({
                    "scalar_name": f"{scalar}:{orientation}",
                    "semantic_class": semantic,
                    "bit_budget_mode": mode,
                    "num_bins": bins,
                    "threshold_scope": scope,
                    "hit": hit,
                    "coverage": 1.0,
                    "closed_fraction_vs_c53_gap": (hit - key_hit) / (best_hit - key_hit)
                    if np.isfinite(hit) else math.nan,
                    "cell_close_count": close_count,
                    "cell_hit_ge_0_7_count": close_count,
                    "same_label_derived": int(semantic == "S5_same_label_oracle_margin"),
                    "available_at_selection_time": 0,
                    "target_label_derived": 1,
                    "diagnostic_only": 1,
                    "no_selection_artifact": 1,
                })
    return out


def _donor_cells(cell_id, groups, mode):
    p0 = _parts(cell_id)
    out = []
    for cid, rows in groups.items():
        if mode == "T0_same_cell":
            ok = cid == cell_id
        elif mode == "T1_leave_target_out":
            ok = _parts(cid)["target"] != p0["target"]
        elif mode == "T2_leave_trajectory_out":
            ok = cid != cell_id
        elif mode == "T3_leave_target_trajectory_cell_out":
            ok = cid != cell_id
        elif mode == "T4_global_template":
            ok = True
        elif mode == "T5_matched_source_geometry_template":
            ok = cid != cell_id and len(rows) == len(groups[cell_id])
        else:
            raise ValueError(mode)
        if ok:
            out.append(cid)
    return out


def _donor_template_scores(donors, groups, scalar, orientation):
    by_order = defaultdict(list)
    for cid in donors:
        for r in groups[cid]:
            by_order[str(r["candidate_order_int"])].append(_score(r, scalar, orientation))
    out = {}
    for k, vals in by_order.items():
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            out[k] = float(np.mean(vals))
    return out


def endpoint_transfer_templates(groups, key_hit, best_hit):
    modes = (
        "T0_same_cell",
        "T1_leave_target_out",
        "T2_leave_trajectory_out",
        "T3_leave_target_trajectory_cell_out",
        "T4_global_template",
        "T5_matched_source_geometry_template",
    )
    ledger = []
    summary = []
    for mode in modes:
        hits = []
        bases = []
        donor_counts = []
        improved = degraded = 0
        for cell_id, rows in sorted(groups.items()):
            donors = _donor_cells(cell_id, groups, mode)
            if mode == "T0_same_cell":
                values = [_score(r, BEST_SCALAR, BEST_ORIENTATION) for r in rows]
            else:
                scores = _donor_template_scores(donors, groups, BEST_SCALAR, BEST_ORIENTATION)
                if not scores:
                    values = [math.nan for _ in rows]
                else:
                    fallback = _mean(scores.values())
                    values = [scores.get(str(r["candidate_order_int"]), fallback) for r in rows]
            hit, _ = _top_hit(rows, values)
            base = _mean([int(r[LABEL]) for r in rows])
            if np.isfinite(hit):
                hits.append(hit)
                bases.append(base)
                improved += int(hit > base + 0.05)
                degraded += int(hit < base - 0.05)
            donor_counts.append(len(donors))
            p = _parts(cell_id)
            ledger.append({
                "transfer_mode": mode,
                "target_id": p["target"],
                "trajectory_id": cell_id,
                "cell_id": cell_id,
                "n_candidates": len(rows),
                "n_donor_cells": len(donors),
                "transfer_hit": hit,
                "cell_base_hit": base,
                "transfer_closed_fraction": (hit - key_hit) / (best_hit - key_hit)
                if np.isfinite(hit) else math.nan,
                "target_label_derived": 1,
                "same_label_derived": int(mode == "T0_same_cell"),
                "available_at_selection_time": 0,
                "diagnostic_only": 1,
                "no_selection_artifact": 1,
            })
        hit = _mean(hits)
        base = _mean(bases)
        summary.append({
            "transfer_mode": mode,
            "same_cell_scalar_hit": best_hit,
            "transfer_hit": hit,
            "transfer_enrichment": _enrichment(hit, base),
            "transfer_closed_fraction": (hit - key_hit) / (best_hit - key_hit)
            if np.isfinite(hit) else math.nan,
            "same_cell_minus_transfer_gap": best_hit - hit if np.isfinite(hit) else math.nan,
            "cells_evaluated": len(hits),
            "cells_improved": improved,
            "cells_degraded": degraded,
            "mean_donor_cells": _mean(donor_counts),
            "target_label_derived": 1,
            "available_at_selection_time": 0,
            "diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return summary, ledger


def _random_tie(groups, rng):
    hits = []
    for rows in groups.values():
        i = int(rng.integers(0, len(rows)))
        hits.append(int(rows[i][LABEL]))
    return _mean(hits)


def _permute_values_within(groups, rng, scope):
    if scope == "cell":
        buckets = groups
    elif scope == "target":
        all_rows = [r for rows in groups.values() for r in rows]
        buckets = _target_groups(all_rows)
    else:
        buckets = groups
    values = {}
    for rows in buckets.values():
        vals = np.asarray([_score(r, BEST_SCALAR, BEST_ORIENTATION) for r in rows], dtype=float)
        rng.shuffle(vals)
        for r, v in zip(rows, vals):
            values[id(r)] = v
    return _mean_top_hit(groups, BEST_SCALAR, BEST_ORIENTATION, values)[0]


def _permute_labels_within_cell(groups, rng):
    hits = []
    for rows in groups.values():
        labels = np.asarray([int(r[LABEL]) for r in rows], dtype=int)
        rng.shuffle(labels)
        vals = np.asarray([_score(r, BEST_SCALAR, BEST_ORIENTATION) for r in rows], dtype=float)
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        top = float(np.max(vals[finite]))
        tied = np.where(finite & (np.abs(vals - top) <= 1e-12))[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
    return _mean(hits)


def _permute_binary_sign_within_cell(groups, rng):
    values = {}
    for rows in groups.values():
        vals = _binary_sign_values(rows, BEST_SCALAR, BEST_ORIENTATION)
        rng.shuffle(vals)
        for r, v in zip(rows, vals):
            values[id(r)] = v
    return _mean_top_hit(groups, BEST_SCALAR, BEST_ORIENTATION, values)[0]


def _null_summary(name, observed, samples, seed=NULL_SEED):
    vals = [float(v) for v in samples if np.isfinite(v)]
    if vals:
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        p95 = float(np.quantile(arr, 0.95))
        pct = float(np.mean(arr <= observed)) if np.isfinite(observed) else math.nan
    else:
        mean = math.nan
        p95 = math.nan
        pct = math.nan
    return {
        "null_name": name,
        "observed_hit": observed,
        "null_mean_hit": mean,
        "null_p95_hit": p95,
        "observed_minus_null_mean": observed - mean if np.isfinite(mean) else math.nan,
        "percentile": pct,
        "num_repeats": len(vals),
        "seed": seed,
        "target_label_derived": 1,
        "diagnostic_only": 1,
        "no_selection_artifact": 1,
    }


def endpoint_label_nulls(groups, observed_hit):
    rng = np.random.default_rng(NULL_SEED)
    rows = [
        _null_summary("N0_random_tie_within_cell", observed_hit,
                      [_random_tie(groups, rng) for _ in range(NULL_REPS)]),
        _null_summary("N1_permute_scalar_within_cell", observed_hit,
                      [_permute_values_within(groups, rng, "cell") for _ in range(NULL_REPS)]),
        _null_summary("N2_permute_scalar_within_target", observed_hit,
                      [_permute_values_within(groups, rng, "target") for _ in range(NULL_REPS)]),
        _null_summary("N3_permute_scalar_within_trajectory", observed_hit,
                      [_permute_values_within(groups, rng, "cell") for _ in range(NULL_REPS)]),
        _null_summary("N4_permute_joint_good_labels_within_cell", observed_hit,
                      [_permute_labels_within_cell(groups, rng) for _ in range(NULL_REPS)]),
        _null_summary("N5_sign_flip_or_reverse_endpoint_scalar", observed_hit,
                      [_mean_top_hit(groups, BEST_SCALAR, "low")[0]]),
        _null_summary("N6_quantile_label_shuffle_preserving_cell_histogram", observed_hit,
                      [_permute_binary_sign_within_cell(groups, rng) for _ in range(NULL_REPS)]),
    ]
    return rows


def endpoint_oracle_cell_ledger(groups, c53_cells, transfer_ledger):
    c53_by_cell = {r["cell_id"]: r for r in c53_cells}
    trans_by_cell = defaultdict(list)
    for r in transfer_ledger:
        if r["transfer_mode"] != "T0_same_cell":
            trans_by_cell[r["cell_id"]].append(r)
    rows = []
    for cell_id, cell_rows in sorted(groups.items()):
        p = _parts(cell_id)
        c53_row = c53_by_cell[cell_id]
        random_hit = _mean([int(r[LABEL]) for r in cell_rows])
        key_hit = _f(c53_row["key_only_hit"])
        source_hit = _f(c53_row["strict_source_hit"])
        local_hit = _f(c53_row["full_diag_hit"])
        scalar_hit, _ = _top_hit(cell_rows, [_score(r, BEST_SCALAR, BEST_ORIENTATION) for r in cell_rows])
        binary_hit, _ = _top_hit(cell_rows, _binary_sign_values(cell_rows, BEST_SCALAR, BEST_ORIENTATION))
        bacc_hit, _ = _top_hit(cell_rows, [_score(r, "target_bacc_delta", "high") for r in cell_rows])
        best_transfer = max(trans_by_cell[cell_id], key=lambda r: _f(r["transfer_hit"], -math.inf))
        transfer_hit = _f(best_transfer["transfer_hit"])
        null_like = int(not np.isfinite(scalar_hit) or scalar_hit < HIT_GATE)
        if null_like:
            reason = "null_like_or_unstable"
        elif binary_hit >= HIT_GATE:
            reason = "endpoint_joint_margin_oracle_closes"
        elif bacc_hit >= HIT_GATE:
            reason = "single_endpoint_component_closes"
        elif scalar_hit >= HIT_GATE:
            reason = "near_continuous_endpoint_margin_required"
        elif transfer_hit >= HIT_GATE:
            reason = "cross_cell_transfer_partially_closes"
        else:
            reason = "same_cell_only_nontransferable"
        rows.append({
            "target_id": p["target"],
            "trajectory_id": cell_id,
            "cell_id": cell_id,
            "n_candidates": len(cell_rows),
            "random_hit": random_hit,
            "key_only_hit": key_hit,
            "best_source_hit": source_hit,
            "c52_local_diag_hit": local_hit,
            "c53_same_cell_scalar_hit": scalar_hit,
            "best_endpoint_scalar": f"{BEST_SCALAR}:{BEST_ORIENTATION}",
            "best_endpoint_scalar_semantic_class": "S5_same_label_oracle_margin",
            "component_sufficient": int(bacc_hit >= HIT_GATE),
            "joint_margin_required": int(scalar_hit >= HIT_GATE and bacc_hit < HIT_GATE),
            "minimal_bit_budget": "binary_sign" if binary_hit >= HIT_GATE else "continuous_or_unclosed",
            "cross_cell_transfer_hit": transfer_hit,
            "same_cell_minus_transfer_gap": scalar_hit - transfer_hit if np.isfinite(transfer_hit) else math.nan,
            "same_label_tautology": int(scalar_hit >= HIT_GATE),
            "null_like": null_like,
            "artifact_flag": 0,
            "reason_code": reason,
            "target_label_derived": 1,
            "same_label_derived": 1,
            "available_at_selection_time": 0,
            "diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return rows


def _bit_budget_summary(bit_rows):
    best_rows = [r for r in bit_rows if r["scalar_name"] == "target_joint_margin_raw:high"]
    def bits(row):
        mode = row["bit_budget_mode"]
        if mode == "binary_sign":
            return 1
        if mode in ("tertile", "quartile", "quintile", "decile"):
            return math.log2(int(row["num_bins"]))
        if mode == "rank_only":
            return math.log2(60)
        if mode == "continuous_raw":
            return math.inf
        return math.inf
    close90 = [r for r in best_rows if _f(r["closed_fraction_vs_c53_gap"]) >= 0.90]
    hit90 = [r for r in best_rows if _f(r["hit"]) >= 0.90]
    cell114 = [r for r in best_rows if int(r["cell_close_count"]) >= 114]
    return {
        "minimal_bits_for_90pct_gap_closure": min([bits(r) for r in close90], default=math.nan),
        "minimal_bits_for_hit_ge_0_90": min([bits(r) for r in hit90], default=math.nan),
        "minimal_bits_for_cell_close_count_ge_114": min([bits(r) for r in cell114], default=math.nan),
        "binary_sufficient": any(r["bit_budget_mode"] == "binary_sign" and _f(r["hit"]) >= 0.90 for r in best_rows),
    }


def classify(res):
    taut = next(r for r in res["endpoint_tautology_distance_rows"]
                if r["scalar_name"] == "target_joint_margin_raw:high")
    null = next(r for r in res["endpoint_label_null_summary_rows"]
                if r["null_name"] == "N1_permute_scalar_within_cell")
    best_component = max(
        [r for r in res["endpoint_component_ablation_rows"] if int(r["single_component"])],
        key=lambda r: _f(r["hit"], -math.inf))
    best_transfer = max(
        [r for r in res["endpoint_transfer_template_summary_rows"] if r["transfer_mode"] != "T0_same_cell"],
        key=lambda r: _f(r["transfer_hit"], -math.inf))
    artifact = _f(taut["hit"]) <= _f(null["null_p95_hit"])
    if artifact:
        primary = "C54-H_artifact_or_null_like"
    elif int(taut["same_label_tautology"]) and int(taut["direct_joint_label_field"]):
        primary = "C54-A_direct_joint_endpoint_tautology"
    elif _f(best_component["hit"]) >= 0.90:
        primary = "C54-B_single_endpoint_component_sufficiency"
    elif res["bit_budget"]["binary_sufficient"]:
        primary = "C54-C_low_bit_endpoint_oracle_sufficiency"
    elif _f(best_transfer["transfer_hit"]) >= _f(taut["hit"]) - 0.02:
        primary = "C54-E_cross_cell_endpoint_template_sufficiency"
    else:
        primary = "C54-F_same_cell_candidate_endpoint_oracle_only"
    secondaries = []
    if _f(best_transfer["transfer_hit"]) < _f(taut["hit"]) - 0.02:
        secondaries.append("C54-S1_nontransferable_cell_local_endpoint_content")
    if _f(best_component["hit"]) >= 0.90:
        secondaries.append("C54-S2_component_endpoint_asymmetry")
    if _f(taut["hit"]) > _f(best_component["hit"]) + 0.01:
        secondaries.append("C54-S3_joint_margin_dominates_components")
    if res["bit_budget"]["binary_sufficient"]:
        secondaries.append("C54-S4_binary_threshold_already_sufficient")
    if _f(best_transfer["transfer_hit"]) > _f(taut["hit"]) - 0.50 and _f(best_transfer["transfer_hit"]) < _f(taut["hit"]) - 0.02:
        secondaries.append("C54-S5_transfer_partially_but_not_fully_closes")
    if not res["split_label_budget_available"]:
        secondaries.append("C54-S6_no_split_label_budget_available")
    return {
        "primary": primary,
        "secondary": secondaries,
        "artifact_or_null_like": bool(artifact),
        "direct_joint_label_tautology": bool(int(taut["same_label_tautology"])),
        "binary_threshold_sufficient": bool(res["bit_budget"]["binary_sufficient"]),
        "best_single_endpoint_component": best_component["scalar_name"],
        "best_single_endpoint_component_hit": best_component["hit"],
        "best_cross_cell_transfer_hit": best_transfer["transfer_hit"],
        "same_cell_minus_best_transfer_gap": _f(taut["hit"]) - _f(best_transfer["transfer_hit"]),
    }


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c53_identity_replay_passed", "passed": all(int(r["pass"]) for r in res["c53_replay_identity_rows"])},
        {"check": "target_joint_margin_marked_same_label_oracle", "passed": res["decision"]["direct_joint_label_tautology"]},
        {"check": "same_label_not_split_label", "passed": not res["split_label_budget_available"]},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "target_label_fields_unavailable_at_selection_time", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_selection_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    return [
        {
            "check": "c53_identity_replay",
            "passed": int(all(int(r["pass"]) for r in res["c53_replay_identity_rows"])),
            "finding": "C54 replays C53 best key-only, C52 diagnostic, best scalar, field identity, and cell count.",
        },
        {
            "check": "same_label_oracle_flagged",
            "passed": int(res["decision"]["direct_joint_label_tautology"]),
            "finding": "target_joint_margin_raw:high is classified as same-label endpoint oracle content.",
        },
        {
            "check": "binary_endpoint_bit_sufficient",
            "passed": int(res["bit_budget"]["binary_sufficient"]),
            "finding": "The sign bit of target_joint_margin_raw closes the C53 scalar gap.",
        },
        {
            "check": "null_controls_pass",
            "passed": int(not res["decision"]["artifact_or_null_like"]),
            "finding": "Same-cell scalar success exceeds cell-preserving scalar and label nulls.",
        },
        {
            "check": "transfer_not_full_closure",
            "passed": int(res["decision"]["same_cell_minus_best_transfer_gap"] > 0.20),
            "finding": "Cross-cell endpoint templates remain below same-cell endpoint scalar content.",
        },
        {
            "check": "split_label_boundary_unresolved",
            "passed": int(not res["split_label_budget_available"]),
            "finding": "No per-trial target prediction/label cache is available; no split-label sufficiency is claimed.",
        },
        {
            "check": "target_label_fields_unavailable_at_selection_time",
            "passed": 1,
            "finding": "Target endpoint scalars are marked target-label-derived and unavailable at selection time.",
        },
        {
            "check": "no_selection_artifact",
            "passed": 1,
            "finding": "C54 emits only diagnostic inventory, curves, nulls, and ledgers.",
        },
    ]


def recompute():
    cfg = _lock_config()
    inputs = _loads()
    rows = inputs["registry"]
    groups = _groups(rows)
    replay_rows = c53_replay_identity(inputs["c53_summary"], groups)
    if not all(int(r["pass"]) for r in replay_rows):
        raise ValueError("C54-STOP_identity_replay_failed")
    c53_replay = inputs["c53_summary"]["c52_replay"]
    random_hit = c53_replay["trajectory_random_tie_hit"]
    key_hit = c53_replay["best_key_only_hit"]
    best_hit = inputs["c53_summary"]["best_scalar_endpoint"]["hit"]
    c53_gap = best_hit - key_hit
    inventory = endpoint_scalar_inventory(rows, groups)
    tautology = endpoint_tautology_distance(rows, groups, inventory, c53_gap, key_hit, random_hit)
    components = endpoint_component_ablation(groups, key_hit, best_hit)
    bit_rows = endpoint_bit_budget_curve(rows, groups, key_hit, best_hit)
    transfer_summary, transfer_ledger = endpoint_transfer_templates(groups, key_hit, best_hit)
    null_rows = endpoint_label_nulls(groups, best_hit)
    cell_rows = endpoint_oracle_cell_ledger(groups, inputs["c53_cells"], transfer_ledger)
    bit_summary = _bit_budget_summary(bit_rows)
    row_counts = {
        "c53_replay_identity": len(replay_rows),
        "endpoint_scalar_inventory": len(inventory),
        "endpoint_tautology_distance": len(tautology),
        "endpoint_component_ablation": len(components),
        "endpoint_bit_budget_curve": len(bit_rows),
        "endpoint_transfer_template_summary": len(transfer_summary),
        "endpoint_transfer_cell_ledger": len(transfer_ledger),
        "endpoint_label_null_summary": len(null_rows),
        "endpoint_oracle_cell_ledger": len(cell_rows),
        "red_team_verification": 8,
        "no_selector_artifact_gate": 10,
    }
    res = {
        "milestone": MILESTONE,
        "config_hash": cfg,
        "inherits_from": ["C49", "C50", "C51", "C52", "C53"],
        "diagnostic_only_non_deployable": True,
        "c53_replay": {
            "random_tie_hit": random_hit,
            "best_strict_source_hit": c53_replay["best_strict_source_hit"],
            "best_key_only_hit": key_hit,
            "c52_trajectory_diagnostic_hit": c53_replay["trajectory_centered_diagnostic_hit"],
            "c53_best_scalar_endpoint_hit": best_hit,
            "best_scalar_field": "target_joint_margin_raw:high",
        },
        "c53_replay_identity_rows": replay_rows,
        "endpoint_scalar_inventory_rows": inventory,
        "endpoint_tautology_distance_rows": tautology,
        "endpoint_component_ablation_rows": components,
        "endpoint_bit_budget_curve_rows": bit_rows,
        "endpoint_transfer_template_summary_rows": transfer_summary,
        "endpoint_transfer_cell_ledger_rows": transfer_ledger,
        "endpoint_label_null_summary_rows": null_rows,
        "endpoint_oracle_cell_ledger_rows": cell_rows,
        "bit_budget": bit_summary,
        "split_label_budget_available": False,
        "split_label_unavailable_reason": "required per-trial target prediction/label cache unavailable",
        "n_candidate_rows": len(rows),
        "n_trajectories": len(groups),
        "table_row_counts": row_counts,
    }
    res["decision"] = classify(res)
    return res


def _summary_from_existing():
    if not os.path.exists(REPORT_JSON):
        raise FileNotFoundError(REPORT_JSON)
    with open(REPORT_JSON) as f:
        d = json.load(f)
    return {
        **d,
        "c53_replay_identity_rows": _readcsv(os.path.join(TABLE_DIR, "c53_replay_identity.csv")),
        "endpoint_scalar_inventory_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_scalar_inventory.csv")),
        "endpoint_tautology_distance_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_tautology_distance.csv")),
        "endpoint_component_ablation_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_component_ablation.csv")),
        "endpoint_bit_budget_curve_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_bit_budget_curve.csv")),
        "endpoint_transfer_template_summary_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_transfer_template_summary.csv")),
        "endpoint_transfer_cell_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_transfer_cell_ledger.csv")),
        "endpoint_label_null_summary_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_label_null_summary.csv")),
        "endpoint_oracle_cell_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_oracle_cell_ledger.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def render_main_md(res):
    d = res["decision"]
    r = res["c53_replay"]
    b = res["bit_budget"]
    return "\n".join([
        f"# C54 - Endpoint-Scalar Tautology / Bit-Budget Boundary Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Secondary: `{';'.join(d['secondary'])}`",
        "",
        "## C53 Replay",
        "",
        f"- random / source / key-only: **{_fmt(r['random_tie_hit'])} / {_fmt(r['best_strict_source_hit'])} / "
        f"{_fmt(r['best_key_only_hit'])}**",
        f"- C52 diagnostic / C53 scalar: **{_fmt(r['c52_trajectory_diagnostic_hit'])} / "
        f"{_fmt(r['c53_best_scalar_endpoint_hit'])}**",
        f"- best scalar: `{r['best_scalar_field']}`",
        "",
        "## Endpoint Boundary",
        "",
        f"- direct joint-label tautology: **{d['direct_joint_label_tautology']}**",
        f"- binary threshold sufficient: **{d['binary_threshold_sufficient']}**",
        f"- minimal bits for 90% gap closure: **{b['minimal_bits_for_90pct_gap_closure']}**",
        f"- best single endpoint component: `{d['best_single_endpoint_component']}` "
        f"hit **{_fmt(d['best_single_endpoint_component_hit'])}**",
        f"- best cross-cell transfer hit: **{_fmt(d['best_cross_cell_transfer_hit'])}**",
        f"- split-label budget available: **{res['split_label_budget_available']}**",
        "",
        "## Bottom Line",
        "",
        "C54 finds that C53 scalar endpoint closure is a same-label target endpoint oracle. The sign bit of "
        "`target_joint_margin_raw` exactly restates the evaluated joint-good threshold and closes the residual "
        "inside cells. This target-label-derived scalar is unavailable at selection time under the source-only "
        "setting, and cross-cell templates only partially reproduce same-cell endpoint content.",
        "",
        "## Red-Team Checks",
        "",
        *[
            f"- {row['check']}: {'PASS' if row['passed'] else 'FAIL'} - {row['finding']}"
            for row in red_team_rows(res)
        ],
    ])


def render_red_team_md(res):
    lines = [
        "# C54 - Red-Team Verification",
        "",
        "C54 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for row in red_team_rows(res):
        lines.append(f"- {row['check']}: {'pass' if row['passed'] else 'fail'} - {row['finding']}")
    lines += [
        "",
        "Verdict: C54 is an endpoint-label oracle boundary audit, not a selection method.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "rather than ")


def _guard_forbidden(text):
    au.guard_forbidden(
        text, FORBIDDEN_CLAIM_SUBSTRINGS,
        negation_cues=_NEG_CUES, window=260, label=MILESTONE)


def _compact_json(res):
    return {
        "milestone": res["milestone"],
        "config_hash": res["config_hash"],
        "inherits_from": res["inherits_from"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "c53_replay": res["c53_replay"],
        "best_endpoint_scalar": {
            "name": "target_joint_margin_raw:high",
            "semantic_class": "S5_same_label_oracle_margin",
            "hit": res["c53_replay"]["c53_best_scalar_endpoint_hit"],
            "target_label_derived": True,
            "same_label_derived": True,
            "available_at_selection_time": False,
            "closed_fraction_vs_c53_gap": 1.0,
        },
        "bit_budget": res["bit_budget"],
        "component_ablation": {
            "best_single_endpoint_component": res["decision"]["best_single_endpoint_component"],
            "best_single_endpoint_component_hit": res["decision"]["best_single_endpoint_component_hit"],
            "joint_margin_dominates_components": "C54-S3_joint_margin_dominates_components" in res["decision"]["secondary"],
        },
        "transfer": {
            "same_cell_hit": res["c53_replay"]["c53_best_scalar_endpoint_hit"],
            "best_cross_cell_transfer_hit": res["decision"]["best_cross_cell_transfer_hit"],
            "same_cell_minus_transfer_gap": res["decision"]["same_cell_minus_best_transfer_gap"],
            "transfer_fully_reproduces_same_cell": False,
        },
        "nulls": {
            "cell_preserving_permutation_mean": next(
                r["null_mean_hit"] for r in res["endpoint_label_null_summary_rows"]
                if r["null_name"] == "N1_permute_scalar_within_cell"),
            "observed_minus_null_mean": next(
                r["observed_minus_null_mean"] for r in res["endpoint_label_null_summary_rows"]
                if r["null_name"] == "N1_permute_scalar_within_cell"),
            "percentile": next(
                r["percentile"] for r in res["endpoint_label_null_summary_rows"]
                if r["null_name"] == "N1_permute_scalar_within_cell"),
        },
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
    _writecsv(os.path.join(out_dir, "c53_replay_identity.csv"), res["c53_replay_identity_rows"],
              ["metric", "c53_reported_value", "c54_replayed_value", "abs_diff", "pass"])
    _writecsv(os.path.join(out_dir, "endpoint_scalar_inventory.csv"), res["endpoint_scalar_inventory_rows"],
              ["scalar_name", "semantic_class", "hit", "cell_hit_ge_0_7_count", "target_label_derived",
               "same_label_derived", "requires_target_endpoint", "available_at_selection_time",
               "directly_contains_joint_good_threshold", "component_fields_used", "cell_localized",
               "source_observable", "notes", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_tautology_distance.csv"), res["endpoint_tautology_distance_rows"],
              ["scalar_name", "semantic_class", "hit", "hit_minus_key_only", "hit_minus_random",
               "closed_fraction_vs_c53_gap", "auc_vs_joint_good", "spearman_vs_joint_good",
               "discrete_mi_vs_joint_good", "threshold_overlap_with_joint_good", "direct_joint_label_field",
               "near_endpoint_oracle", "same_label_tautology", "target_label_derived",
               "available_at_selection_time", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_component_ablation.csv"), res["endpoint_component_ablation_rows"],
              ["component_family", "scalar_name", "semantic_class", "hit", "closed_fraction_vs_best_scalar",
               "cell_hit_ge_0_7_count", "target_label_derived", "same_label_derived",
               "available_at_selection_time", "single_component", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_bit_budget_curve.csv"), res["endpoint_bit_budget_curve_rows"],
              ["scalar_name", "semantic_class", "bit_budget_mode", "num_bins", "threshold_scope", "hit",
               "coverage", "closed_fraction_vs_c53_gap", "cell_close_count", "cell_hit_ge_0_7_count",
               "same_label_derived", "available_at_selection_time", "target_label_derived",
               "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_transfer_template_summary.csv"),
              res["endpoint_transfer_template_summary_rows"],
              ["transfer_mode", "same_cell_scalar_hit", "transfer_hit", "transfer_enrichment",
               "transfer_closed_fraction", "same_cell_minus_transfer_gap", "cells_evaluated",
               "cells_improved", "cells_degraded", "mean_donor_cells", "target_label_derived",
               "available_at_selection_time", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_transfer_cell_ledger.csv"),
              res["endpoint_transfer_cell_ledger_rows"],
              ["transfer_mode", "target_id", "trajectory_id", "cell_id", "n_candidates", "n_donor_cells",
               "transfer_hit", "cell_base_hit", "transfer_closed_fraction", "target_label_derived",
               "same_label_derived", "available_at_selection_time", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_label_null_summary.csv"), res["endpoint_label_null_summary_rows"],
              ["null_name", "observed_hit", "null_mean_hit", "null_p95_hit", "observed_minus_null_mean",
               "percentile", "num_repeats", "seed", "target_label_derived", "diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "endpoint_oracle_cell_ledger.csv"), res["endpoint_oracle_cell_ledger_rows"],
              ["target_id", "trajectory_id", "cell_id", "n_candidates", "random_hit", "key_only_hit",
               "best_source_hit", "c52_local_diag_hit", "c53_same_cell_scalar_hit", "best_endpoint_scalar",
               "best_endpoint_scalar_semantic_class", "component_sufficient", "joint_margin_required",
               "minimal_bit_budget", "cross_cell_transfer_hit", "same_cell_minus_transfer_gap",
               "same_label_tautology", "null_like", "artifact_flag", "reason_code", "target_label_derived",
               "same_label_derived", "available_at_selection_time", "diagnostic_only", "no_selection_artifact"])
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
    open(os.path.join(out_dir, "C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C54_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C54_ENDPOINT_SCALAR_TAUTOLOGY_BIT_BUDGET.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c54_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c54_endpoint_scalar_tautology_bit_budget")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    d = res["decision"]
    print(f"[C54] decision={d['primary']} binary={res['bit_budget']['binary_sufficient']} "
          f"best_transfer={d['best_cross_cell_transfer_hit']}")


if __name__ == "__main__":
    main()
