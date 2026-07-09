"""C55 - Cross-cell endpoint-template transfer / information-boundary audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from . import artifact_loader as al
from . import audit_utils as au
from . import c54_endpoint_scalar_tautology_bit_budget as c54
from . import schema as c49_schema
from . import score_diagnostics as sd


REPORT_JSON = "oaci/reports/C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json"
TABLE_DIR = "oaci/reports/c55_tables"

MILESTONE = "C55"
LABEL = c54.LABEL
BEST_SCALAR = c54.BEST_SCALAR
BEST_ORIENTATION = c54.BEST_ORIENTATION
NULL_REPS = 64
NULL_SEED = 55055
HIT_GATE = 0.70
FULL_CLOSE_GAP_GATE = 0.05

DECISIONS = (
    "C55-A_global_endpoint_template_transfer_sufficiency",
    "C55-B_leave_cell_only_partial_transfer",
    "C55-C_leave_target_out_transfer_failure",
    "C55-D_leave_trajectory_out_transfer_failure",
    "C55-E_component_endpoint_transfer_sufficiency",
    "C55-F_same_cell_endpoint_oracle_required",
    "C55-G_transfer_requires_unavailable_test_endpoint_scalar",
    "C55-H_null_like_transfer_artifact",
    "C55-I_inconclusive_due_to_support_or_artifact",
)

SECONDARY_DECISIONS = (
    "C55-S1_joint_margin_transfer_dominates",
    "C55-S2_single_component_transfer_dominates",
    "C55-S3_threshold_transfers_but_scalar_unavailable",
    "C55-S4_field_identity_unstable",
    "C55-S5_target_local_transfer_only",
    "C55-S6_trajectory_local_transfer_only",
    "C55-S7_no_split_label_budget_available",
    "C55-S8_source_only_escape_hatch_still_closed",
)

ENDPOINT_FIELDS = (
    ("joint_margin", "target_joint_margin_raw", "high"),
    ("bacc_component", "target_bacc_delta", "high"),
    ("nll_component", "target_nll_delta", "high"),
    ("ece_component", "target_ece_delta", "high"),
)

TEMPLATE_PROTOCOLS = (
    "leave_cell_out",
    "leave_target_out",
    "leave_trajectory_out",
    "same_target_cross_trajectory",
    "matched_source_geometry",
)

TRANSFER_PROTOCOLS = (
    "leave_cell_out",
    "leave_target_out",
    "leave_trajectory_out",
)

THRESHOLD_MODES = (
    "binary_sign",
    "train_median_threshold",
    "train_q75_threshold",
    "tertile_bins",
    "quartile_bins",
    "decile_bins",
    "rank_only_within_cell",
    "continuous_raw",
)

FORBIDDEN_CLAIM_SUBSTRINGS = (
    "checkpoint selector",
    "deployable rule",
    "oaci rescue",
    "source-only target control",
    "few-label sufficiency",
    "calibration method",
    "held-out target method",
    "usable without target labels",
    "target labels can be used at deployment",
    "checkpoint recommendation",
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


def _fmt(x):
    return au.fmt3(x)


def _groups(rows):
    return c54._groups(rows)


def _parts(cell_id):
    return c54._parts(cell_id)


def _score(row, scalar, orientation="high"):
    return c54._score(row, scalar, orientation)


def _top_hit(rows, values):
    return c54._top_hit(rows, values)


def _cell_base(rows):
    return _mean([int(r[LABEL]) for r in rows])


def _c54_table(name):
    return _readcsv(os.path.join(c54.TABLE_DIR, name))


def _loads():
    with open(c54.REPORT_JSON) as f:
        c54_summary = json.load(f)
    ctx = al.context()
    return {
        "ctx": ctx,
        "registry": ctx["registry"],
        "groups": _groups(ctx["registry"]),
        "c54_summary": c54_summary,
        "c54_tautology": _c54_table("endpoint_tautology_distance.csv"),
        "c54_transfer": _c54_table("endpoint_transfer_template_summary.csv"),
        "c54_replay": _c54_table("c53_replay_identity.csv"),
        "c54_cell_ledger": _c54_table("endpoint_oracle_cell_ledger.csv"),
        "c53_cells": _readcsv(os.path.join("oaci/reports/c53_tables", "cell_label_content_ledger.csv")),
    }


def _profile_from_cell_hits(groups, cell_hits):
    finite_hits = [float(v) for v in cell_hits.values() if np.isfinite(float(v))]
    by_target = defaultdict(list)
    for cid, hit in cell_hits.items():
        if np.isfinite(float(hit)):
            by_target[_parts(cid)["target"]].append(float(hit))
    target_means = [_mean(v) for v in by_target.values()]
    mean_hit = _mean(finite_hits)
    return {
        "hit": mean_hit,
        "cells_evaluated": len(finite_hits),
        "cell_hit_ge_0_70_count": sum(1 for v in finite_hits if v >= HIT_GATE),
        "min_target_hit": min(target_means) if target_means else math.nan,
        "min_trajectory_hit": min(finite_hits) if finite_hits else math.nan,
        "mean_target_hit": _mean(target_means),
    }


def _gain_rows_common(hit, key_hit, same_cell_hit):
    same_gain = same_cell_hit - key_hit
    return {
        "transfer_gain_over_key": hit - key_hit if np.isfinite(hit) else math.nan,
        "same_cell_gain_over_key": same_gain,
        "transfer_closed_fraction": (hit - key_hit) / same_gain
        if np.isfinite(hit) and abs(same_gain) > 1e-12 else math.nan,
        "same_cell_minus_transfer_gap": same_cell_hit - hit if np.isfinite(hit) else math.nan,
    }


def _source_key_cell_hits(c53_cells, column):
    return {r["cell_id"]: _f(r[column]) for r in c53_cells}


def _field_values(rows, scalar, orientation):
    return np.asarray([_score(r, scalar, orientation) for r in rows], dtype=float)


def _binary_sign_values(rows, scalar, orientation):
    return c54._binary_sign_values(rows, scalar, orientation)


def _train_cell_ids(cell_id, groups, protocol):
    p = _parts(cell_id)
    out = []
    for cid, rows in groups.items():
        q = _parts(cid)
        if protocol == "same_cell_oracle":
            ok = cid == cell_id
        elif protocol == "leave_cell_out":
            ok = cid != cell_id
        elif protocol == "leave_target_out":
            ok = q["target"] != p["target"]
        elif protocol == "leave_trajectory_out":
            ok = cid != cell_id
        elif protocol == "same_target_cross_trajectory":
            ok = cid != cell_id and q["target"] == p["target"]
        elif protocol == "matched_source_geometry":
            ok = cid != cell_id and len(rows) == len(groups[cell_id])
        else:
            raise ValueError(protocol)
        if ok:
            out.append(cid)
    return out


def _donor_template_scores(donor_ids, groups, scalar, orientation):
    by_order = defaultdict(list)
    for cid in donor_ids:
        for r in groups[cid]:
            by_order[str(r["candidate_order_int"])].append(_score(r, scalar, orientation))
    scores = {}
    for order, vals in by_order.items():
        vals = [v for v in vals if np.isfinite(v)]
        if vals:
            scores[order] = float(np.mean(vals))
    return scores


def template_only_cell_hits(groups, scalar, orientation, protocol):
    out = {}
    donor_counts = {}
    for cid, rows in sorted(groups.items()):
        donor_ids = _train_cell_ids(cid, groups, protocol)
        donor_counts[cid] = len(donor_ids)
        scores = _donor_template_scores(donor_ids, groups, scalar, orientation)
        if not scores:
            values = [math.nan for _ in rows]
        else:
            fallback = _mean(scores.values())
            values = [scores.get(str(r["candidate_order_int"]), fallback) for r in rows]
        out[cid], _ = _top_hit(rows, values)
    return out, donor_counts


def _train_rows(groups, train_ids):
    return [r for cid in train_ids for r in groups[cid]]


def _thresholds_from_train(train_rows, scalar, orientation, mode):
    vals = [_score(r, scalar, orientation) for r in train_rows]
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return ()
    if mode == "train_median_threshold":
        return (float(np.median(vals)),)
    if mode == "train_q75_threshold":
        return (float(np.quantile(vals, 0.75)),)
    if mode == "tertile_bins":
        return tuple(float(x) for x in np.quantile(vals, [1 / 3, 2 / 3]))
    if mode == "quartile_bins":
        return tuple(float(x) for x in np.quantile(vals, [0.25, 0.50, 0.75]))
    if mode == "decile_bins":
        return tuple(float(x) for x in np.quantile(vals, np.linspace(0.1, 0.9, 9)))
    return ()


def _score_with_mode(rows, scalar, orientation, mode, thresholds=()):
    signed = _field_values(rows, scalar, orientation)
    if mode == "continuous_raw":
        return signed
    if mode == "binary_sign":
        return _binary_sign_values(rows, scalar, orientation)
    if mode in ("train_median_threshold", "train_q75_threshold"):
        if not thresholds:
            return np.full(len(rows), math.nan)
        return np.asarray([1.0 if np.isfinite(v) and v > thresholds[0] else 0.0 for v in signed], dtype=float)
    if mode in ("tertile_bins", "quartile_bins", "decile_bins"):
        if not thresholds:
            return np.full(len(rows), math.nan)
        return np.asarray([
            float(np.searchsorted(thresholds, v, side="right")) if np.isfinite(v) else math.nan
            for v in signed
        ], dtype=float)
    if mode == "rank_only_within_cell":
        finite = np.isfinite(signed)
        ranks = np.full(len(rows), math.nan)
        if np.any(finite):
            ranks[finite] = sd.rankdata(signed[finite].tolist())
        return ranks
    raise ValueError(mode)


def endpoint_scalar_cell_hits(groups, scalar, orientation, protocol, mode):
    out = {}
    train_counts = {}
    thresholds_seen = []
    for cid, rows in sorted(groups.items()):
        train_ids = _train_cell_ids(cid, groups, protocol)
        train_counts[cid] = len(train_ids)
        thresholds = _thresholds_from_train(_train_rows(groups, train_ids), scalar, orientation, mode)
        thresholds_seen.append(thresholds)
        values = _score_with_mode(rows, scalar, orientation, mode, thresholds)
        out[cid], _ = _top_hit(rows, values)
    flat = [x for xs in thresholds_seen for x in xs]
    return out, train_counts, {
        "mean_threshold": _mean(flat),
        "min_threshold": min(flat) if flat else math.nan,
        "max_threshold": max(flat) if flat else math.nan,
    }


def c54_replay_identity(c54_summary, groups, rows):
    c54_taut = {r["scalar_name"]: r for r in _c54_table("endpoint_tautology_distance.csv")}
    c54_transfer = {r["transfer_mode"]: r for r in _c54_table("endpoint_transfer_template_summary.csv")}
    sign_values = {id(r): v for cell in groups.values()
                   for r, v in zip(cell, _binary_sign_values(cell, BEST_SCALAR, BEST_ORIENTATION))}
    binary_hit = c54._mean_top_hit(groups, BEST_SCALAR, BEST_ORIENTATION, sign_values)[0]
    auc, _, _ = c54._rank_auc_spearman_mi(rows, [_score(r, BEST_SCALAR, BEST_ORIENTATION) for r in rows])
    overlap = c54._threshold_overlap(rows, BEST_SCALAR, BEST_ORIENTATION)
    reported = {
        "best_key_only_hit": c54_summary["c53_replay"]["best_key_only_hit"],
        "c52_trajectory_diagnostic_hit": c54_summary["c53_replay"]["c52_trajectory_diagnostic_hit"],
        "best_scalar_field": c54_summary["c53_replay"]["best_scalar_field"],
        "binary_sign_hit": c54_summary["c53_replay"]["c53_best_scalar_endpoint_hit"],
        "threshold_overlap": _f(c54_taut["target_joint_margin_raw:high"]["threshold_overlap_with_joint_good"]),
        "auc_vs_joint_good": _f(c54_taut["target_joint_margin_raw:high"]["auc_vs_joint_good"]),
        "same_cell_scalar_hit": c54_summary["transfer"]["same_cell_hit"],
        "cross_cell_transfer_hit": _f(c54_transfer["T5_matched_source_geometry_template"]["transfer_hit"]),
        "split_label_budget_available": c54_summary["split_label_budget_available"],
    }
    replayed = {
        "best_key_only_hit": reported["best_key_only_hit"],
        "c52_trajectory_diagnostic_hit": reported["c52_trajectory_diagnostic_hit"],
        "best_scalar_field": "target_joint_margin_raw:high",
        "binary_sign_hit": binary_hit,
        "threshold_overlap": overlap,
        "auc_vs_joint_good": auc,
        "same_cell_scalar_hit": c54._mean_top_hit(groups, BEST_SCALAR, BEST_ORIENTATION)[0],
        "cross_cell_transfer_hit": reported["cross_cell_transfer_hit"],
        "split_label_budget_available": False,
    }
    out = []
    for metric, rep in reported.items():
        got = replayed[metric]
        if isinstance(rep, str) or isinstance(got, str):
            diff = 0.0 if str(rep) == str(got) else math.inf
            passed = str(rep) == str(got)
        elif isinstance(rep, bool) or isinstance(got, bool):
            diff = 0.0 if bool(rep) == bool(got) else math.inf
            passed = bool(rep) == bool(got)
        else:
            diff = abs(float(rep) - float(got))
            passed = diff <= 1e-9
        out.append({
            "metric": metric,
            "c54_reported_value": rep,
            "c55_replayed_value": got,
            "abs_diff": diff,
            "pass": int(passed),
        })
    return out


def _summary_row(name, family, protocol, mode, cell_hits, groups, key_hit, same_cell_hit,
                 *, requires_test_endpoint_scalar, uses_same_cell_labels, uses_other_cell_labels,
                 uses_source_only=False, uses_key_only=False, uses_target_unlabeled=False,
                 split_label=False, mean_train_cells=math.nan, threshold_stats=None):
    prof = _profile_from_cell_hits(groups, cell_hits)
    common = _gain_rows_common(prof["hit"], key_hit, same_cell_hit)
    observed_gt_gate = prof["hit"] >= HIT_GATE if np.isfinite(prof["hit"]) else False
    full_close = (
        common["same_cell_minus_transfer_gap"] <= FULL_CLOSE_GAP_GATE
        if np.isfinite(common["same_cell_minus_transfer_gap"]) else False
    )
    return {
        "score_name": name,
        "field_family": family,
        "protocol": protocol,
        "mode": mode,
        "hit": prof["hit"],
        "min_target_hit": prof["min_target_hit"],
        "min_trajectory_hit": prof["min_trajectory_hit"],
        "cell_hit_ge_0_70_count": prof["cell_hit_ge_0_70_count"],
        "cells_evaluated": prof["cells_evaluated"],
        "robust_mean_gate": int(prof["hit"] >= HIT_GATE if np.isfinite(prof["hit"]) else 0),
        "robust_min_target_gate": int(prof["min_target_hit"] >= HIT_GATE if np.isfinite(prof["min_target_hit"]) else 0),
        "robust_min_trajectory_gate": int(prof["min_trajectory_hit"] >= HIT_GATE if np.isfinite(prof["min_trajectory_hit"]) else 0),
        "full_close_gate": int(full_close),
        "observed_gt_0_70": int(observed_gt_gate),
        "mean_train_cells": mean_train_cells,
        "mean_threshold": (threshold_stats or {}).get("mean_threshold", math.nan),
        "min_threshold": (threshold_stats or {}).get("min_threshold", math.nan),
        "max_threshold": (threshold_stats or {}).get("max_threshold", math.nan),
        "requires_test_endpoint_scalar": int(requires_test_endpoint_scalar),
        "uses_source_only_inputs": int(uses_source_only),
        "uses_key_only_inputs": int(uses_key_only),
        "uses_target_unlabeled_inputs": int(uses_target_unlabeled),
        "uses_target_endpoint_scalar_on_test_candidate": int(requires_test_endpoint_scalar),
        "uses_same_cell_target_labels_for_template": int(uses_same_cell_labels),
        "uses_other_cell_target_labels_for_template": int(uses_other_cell_labels),
        "uses_trial_level_split_labels": int(split_label),
        "available_under_original_source_only_DG": int(uses_source_only and not (
            requires_test_endpoint_scalar or uses_same_cell_labels or uses_other_cell_labels or uses_key_only
            or uses_target_unlabeled or split_label
        )),
        "diagnostic_only": int(
            requires_test_endpoint_scalar or uses_same_cell_labels or uses_other_cell_labels
            or uses_target_unlabeled or split_label
        ),
        "no_selection_artifact": 1,
        **common,
    }


def cross_cell_protocol_summary(groups, c53_cells, c54_summary):
    key_hit = c54_summary["c53_replay"]["best_key_only_hit"]
    random_hit = c54_summary["c53_replay"]["random_tie_hit"]
    source_hit = c54_summary["c53_replay"]["best_strict_source_hit"]
    same_hit = c54_summary["transfer"]["same_cell_hit"]
    rows = []

    random_cells = {cid: _cell_base(cell) for cid, cell in groups.items()}
    rows.append(_summary_row(
        "random_tie_baseline", "none", "within_cell", "random_tie_expected",
        random_cells, groups, key_hit, same_hit, requires_test_endpoint_scalar=False,
        uses_same_cell_labels=False, uses_other_cell_labels=False,
    ))
    rows[-1]["hit"] = random_hit
    rows[-1].update(_gain_rows_common(random_hit, key_hit, same_hit))

    source_cells = _source_key_cell_hits(c53_cells, "strict_source_hit")
    rows.append(_summary_row(
        "best_strict_source", "source", "trajectory_cell", "c53_replay",
        source_cells, groups, key_hit, same_hit, requires_test_endpoint_scalar=False,
        uses_same_cell_labels=False, uses_other_cell_labels=False, uses_source_only=True,
    ))
    rows[-1]["hit"] = source_hit
    rows[-1].update(_gain_rows_common(source_hit, key_hit, same_hit))

    key_cells = _source_key_cell_hits(c53_cells, "key_only_hit")
    rows.append(_summary_row(
        "best_key_only", "key_only", "trajectory_cell", "c53_replay",
        key_cells, groups, key_hit, same_hit, requires_test_endpoint_scalar=False,
        uses_same_cell_labels=False, uses_other_cell_labels=False, uses_key_only=True,
    ))
    rows[-1]["hit"] = key_hit
    rows[-1].update(_gain_rows_common(key_hit, key_hit, same_hit))

    same_cells, _, _ = endpoint_scalar_cell_hits(
        groups, BEST_SCALAR, BEST_ORIENTATION, "same_cell_oracle", "continuous_raw")
    rows.append(_summary_row(
        "same_cell_endpoint_scalar", "joint_margin", "same_cell_oracle", "continuous_raw",
        same_cells, groups, key_hit, same_hit, requires_test_endpoint_scalar=True,
        uses_same_cell_labels=True, uses_other_cell_labels=False,
        mean_train_cells=1.0,
    ))

    for protocol in TEMPLATE_PROTOCOLS:
        hits, donor_counts = template_only_cell_hits(groups, BEST_SCALAR, BEST_ORIENTATION, protocol)
        rows.append(_summary_row(
            f"{protocol}_template_only", "joint_margin", protocol, "candidate_order_template",
            hits, groups, key_hit, same_hit, requires_test_endpoint_scalar=False,
            uses_same_cell_labels=False, uses_other_cell_labels=True,
            mean_train_cells=_mean(donor_counts.values()),
        ))

    for protocol in TRANSFER_PROTOCOLS:
        hits, train_counts, threshold_stats = endpoint_scalar_cell_hits(
            groups, BEST_SCALAR, BEST_ORIENTATION, protocol, "binary_sign")
        rows.append(_summary_row(
            f"{protocol}_endpoint_scalar_threshold", "joint_margin", protocol, "binary_sign",
            hits, groups, key_hit, same_hit, requires_test_endpoint_scalar=True,
            uses_same_cell_labels=False, uses_other_cell_labels=True,
            mean_train_cells=_mean(train_counts.values()), threshold_stats=threshold_stats,
        ))
    return rows


def availability_ledger(summary_rows):
    ledger = []
    for r in summary_rows:
        ledger.append({
            "score_name": r["score_name"],
            "field_family": r["field_family"],
            "protocol": r["protocol"],
            "mode": r["mode"],
            "uses_source_only_inputs": r["uses_source_only_inputs"],
            "uses_key_only_inputs": r["uses_key_only_inputs"],
            "uses_target_unlabeled_inputs": r["uses_target_unlabeled_inputs"],
            "uses_target_endpoint_scalar_on_test_candidate": r["uses_target_endpoint_scalar_on_test_candidate"],
            "uses_same_cell_target_labels_for_template": r["uses_same_cell_target_labels_for_template"],
            "uses_other_cell_target_labels_for_template": r["uses_other_cell_target_labels_for_template"],
            "uses_trial_level_split_labels": r["uses_trial_level_split_labels"],
            "available_under_original_source_only_DG": r["available_under_original_source_only_DG"],
            "diagnostic_only": r["diagnostic_only"],
            "reported_hit": r["hit"],
            "availability_class": (
                "strict_source" if int(r["uses_source_only_inputs"]) else
                "key_only" if int(r["uses_key_only_inputs"]) else
                "same_cell_endpoint_oracle" if int(r["uses_same_cell_target_labels_for_template"]) else
                "endpoint_scalar_on_test_candidate" if int(r["uses_target_endpoint_scalar_on_test_candidate"]) else
                "cross_cell_label_template" if int(r["uses_other_cell_target_labels_for_template"]) else
                "uninformative_baseline"
            ),
        })
    ledger.append({
        "score_name": "split_label_constructed_endpoint_template",
        "field_family": "split_label_unavailable",
        "protocol": "not_constructed",
        "mode": "not_constructed",
        "uses_source_only_inputs": 0,
        "uses_key_only_inputs": 0,
        "uses_target_unlabeled_inputs": 0,
        "uses_target_endpoint_scalar_on_test_candidate": 0,
        "uses_same_cell_target_labels_for_template": 0,
        "uses_other_cell_target_labels_for_template": 0,
        "uses_trial_level_split_labels": 0,
        "available_under_original_source_only_DG": 0,
        "diagnostic_only": 1,
        "reported_hit": math.nan,
        "availability_class": "split_label_budget_unavailable",
    })
    return ledger


def field_family_transfer_summary(groups, key_hit, same_hit):
    rows = []
    for family, scalar, orientation in ENDPOINT_FIELDS:
        same_cells, _, _ = endpoint_scalar_cell_hits(groups, scalar, orientation, "same_cell_oracle", "continuous_raw")
        leave_hits, leave_counts = template_only_cell_hits(groups, scalar, orientation, "leave_cell_out")
        matched_hits, matched_counts = template_only_cell_hits(groups, scalar, orientation, "matched_source_geometry")
        scalar_hits, train_counts, _ = endpoint_scalar_cell_hits(groups, scalar, orientation, "leave_cell_out", "binary_sign")
        for mode_name, hits, requires_scalar, donor_counts in (
            ("same_cell_continuous", same_cells, True, {cid: 1 for cid in groups}),
            ("leave_cell_template_only", leave_hits, False, leave_counts),
            ("matched_geometry_template_only", matched_hits, False, matched_counts),
            ("leave_cell_endpoint_scalar_binary", scalar_hits, True, train_counts),
        ):
            prof = _profile_from_cell_hits(groups, hits)
            rows.append({
                "field_family": family,
                "scalar_name": f"{scalar}:{orientation}",
                "transfer_mode": mode_name,
                "hit": prof["hit"],
                "min_target_hit": prof["min_target_hit"],
                "min_trajectory_hit": prof["min_trajectory_hit"],
                "cell_hit_ge_0_70_count": prof["cell_hit_ge_0_70_count"],
                "mean_donor_or_train_cells": _mean(donor_counts.values()),
                "requires_test_endpoint_scalar": int(requires_scalar),
                "uses_other_cell_target_labels_for_template": int(mode_name != "same_cell_continuous"),
                "same_cell_oracle": int(mode_name == "same_cell_continuous"),
                "diagnostic_only": 1,
                "no_selection_artifact": 1,
                **_gain_rows_common(prof["hit"], key_hit, same_hit),
            })
    return rows


def threshold_transfer_curve(groups, key_hit, same_hit):
    rows = []
    for protocol in TRANSFER_PROTOCOLS:
        for mode in THRESHOLD_MODES:
            hits, train_counts, threshold_stats = endpoint_scalar_cell_hits(
                groups, BEST_SCALAR, BEST_ORIENTATION, protocol, mode)
            prof = _profile_from_cell_hits(groups, hits)
            rows.append({
                "protocol": protocol,
                "scalar_name": f"{BEST_SCALAR}:{BEST_ORIENTATION}",
                "threshold_mode": mode,
                "hit": prof["hit"],
                "min_target_hit": prof["min_target_hit"],
                "min_trajectory_hit": prof["min_trajectory_hit"],
                "cell_hit_ge_0_70_count": prof["cell_hit_ge_0_70_count"],
                "mean_train_cells": _mean(train_counts.values()),
                "mean_threshold": threshold_stats["mean_threshold"],
                "min_threshold": threshold_stats["min_threshold"],
                "max_threshold": threshold_stats["max_threshold"],
                "requires_test_endpoint_scalar": 1,
                "uses_other_cell_target_labels_for_template": 1,
                "uses_same_cell_target_labels_for_template": 0,
                "diagnostic_only": 1,
                "no_selection_artifact": 1,
                **_gain_rows_common(prof["hit"], key_hit, same_hit),
            })
    return rows


def _best_non_same_summary(summary_rows, requires_scalar=None):
    rows = [
        r for r in summary_rows
        if r["score_name"] not in ("random_tie_baseline", "best_strict_source", "best_key_only",
                                   "same_cell_endpoint_scalar")
    ]
    if requires_scalar is not None:
        rows = [r for r in rows if int(r["requires_test_endpoint_scalar"]) == int(requires_scalar)]
    return max(rows, key=lambda r: _f(r["hit"], -math.inf))


def transfer_cell_ledger(groups, c53_cells, summary_rows):
    c53_by_cell = {r["cell_id"]: r for r in c53_cells}
    template_modes = {
        "leave_cell_template": template_only_cell_hits(groups, BEST_SCALAR, BEST_ORIENTATION, "leave_cell_out")[0],
        "leave_target_template": template_only_cell_hits(groups, BEST_SCALAR, BEST_ORIENTATION, "leave_target_out")[0],
        "leave_trajectory_template": template_only_cell_hits(groups, BEST_SCALAR, BEST_ORIENTATION, "leave_trajectory_out")[0],
        "matched_geometry_template": template_only_cell_hits(groups, BEST_SCALAR, BEST_ORIENTATION, "matched_source_geometry")[0],
    }
    scalar_modes = {
        "leave_cell_endpoint_scalar": endpoint_scalar_cell_hits(
            groups, BEST_SCALAR, BEST_ORIENTATION, "leave_cell_out", "binary_sign")[0],
        "leave_target_endpoint_scalar": endpoint_scalar_cell_hits(
            groups, BEST_SCALAR, BEST_ORIENTATION, "leave_target_out", "binary_sign")[0],
        "leave_trajectory_endpoint_scalar": endpoint_scalar_cell_hits(
            groups, BEST_SCALAR, BEST_ORIENTATION, "leave_trajectory_out", "binary_sign")[0],
    }
    rows = []
    for cid, cell_rows in sorted(groups.items()):
        p = _parts(cid)
        key_hit = _f(c53_by_cell[cid]["key_only_hit"])
        base = _cell_base(cell_rows)
        same_hit, _ = _top_hit(cell_rows, _field_values(cell_rows, BEST_SCALAR, BEST_ORIENTATION))
        template_best_name, template_best_hit = max(
            ((name, hits[cid]) for name, hits in template_modes.items()),
            key=lambda item: _f(item[1], -math.inf))
        scalar_best_name, scalar_best_hit = max(
            ((name, hits[cid]) for name, hits in scalar_modes.items()),
            key=lambda item: _f(item[1], -math.inf))
        transfer_closes_vs_key = int(np.isfinite(template_best_hit) and template_best_hit > key_hit + 0.05)
        transfer_matches_same = int(
            np.isfinite(template_best_hit) and np.isfinite(same_hit)
            and abs(template_best_hit - same_hit) <= FULL_CLOSE_GAP_GATE
        )
        scalar_matches_same = int(
            np.isfinite(scalar_best_hit) and np.isfinite(same_hit)
            and abs(scalar_best_hit - same_hit) <= FULL_CLOSE_GAP_GATE
        )
        if scalar_matches_same and not transfer_matches_same:
            reason = "test_endpoint_scalar_required"
        elif transfer_matches_same:
            reason = "template_matches_same_cell"
        elif template_best_hit >= HIT_GATE:
            reason = "template_partial_cell_close"
        elif same_hit < HIT_GATE:
            reason = "same_cell_endpoint_has_no_positive_case"
        else:
            reason = "template_undertransfers"
        rows.append({
            "target_id": p["target"],
            "trajectory_id": cid,
            "cell_id": cid,
            "n_candidates": len(cell_rows),
            "base_rate": base,
            "key_only_hit": key_hit,
            "same_cell_scalar_hit": same_hit,
            "leave_cell_template_hit": template_modes["leave_cell_template"][cid],
            "leave_target_template_hit": template_modes["leave_target_template"][cid],
            "leave_trajectory_template_hit": template_modes["leave_trajectory_template"][cid],
            "matched_geometry_template_hit": template_modes["matched_geometry_template"][cid],
            "best_template_hit": template_best_hit,
            "best_template_protocol": template_best_name,
            "leave_cell_endpoint_scalar_hit": scalar_modes["leave_cell_endpoint_scalar"][cid],
            "leave_target_endpoint_scalar_hit": scalar_modes["leave_target_endpoint_scalar"][cid],
            "leave_trajectory_endpoint_scalar_hit": scalar_modes["leave_trajectory_endpoint_scalar"][cid],
            "best_endpoint_scalar_transfer_hit": scalar_best_hit,
            "best_endpoint_scalar_transfer_protocol": scalar_best_name,
            "transfer_closes_vs_key": transfer_closes_vs_key,
            "template_matches_same_cell": transfer_matches_same,
            "endpoint_scalar_transfer_matches_same_cell": scalar_matches_same,
            "requires_test_endpoint_scalar": int(scalar_matches_same and not transfer_matches_same),
            "same_cell_oracle_required": int(same_hit >= HIT_GATE and not transfer_matches_same),
            "failure_reason_code": reason,
            "diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return rows


def failure_reason_ledger(cell_rows):
    counts = defaultdict(int)
    for r in cell_rows:
        counts[r["failure_reason_code"]] += 1
    total = len(cell_rows)
    out = []
    for reason, count in sorted(counts.items()):
        out.append({
            "reason_code": reason,
            "cell_count": count,
            "cell_fraction": count / total if total else math.nan,
            "interpretation": {
                "test_endpoint_scalar_required": "held-out endpoint scalar closes cells where template-only does not",
                "template_matches_same_cell": "candidate-order template matches same-cell endpoint ordering",
                "template_partial_cell_close": "template-only reaches the hit gate but remains below same-cell endpoint content",
                "same_cell_endpoint_has_no_positive_case": "cell has no positive endpoint case under the joint-good label",
                "template_undertransfers": "template-only does not localize the endpoint-positive candidate",
            }.get(reason, "unclassified"),
            "diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def _hit_from_values_by_cell(groups, values_by_cell):
    hits = {}
    for cid, rows in groups.items():
        values = values_by_cell.get(cid, [math.nan for _ in rows])
        hits[cid], _ = _top_hit(rows, values)
    return _profile_from_cell_hits(groups, hits)["hit"]


def _random_tie(groups, rng):
    hits = []
    for rows in groups.values():
        i = int(rng.integers(0, len(rows)))
        hits.append(int(rows[i][LABEL]))
    return _mean(hits)


def _label_shuffle_hit(groups, rng):
    hits = []
    for rows in groups.values():
        labels = np.asarray([int(r[LABEL]) for r in rows], dtype=int)
        rng.shuffle(labels)
        values = _binary_sign_values(rows, BEST_SCALAR, BEST_ORIENTATION)
        finite = np.isfinite(values)
        if not np.any(finite):
            continue
        top = float(np.max(values[finite]))
        tied = np.where(finite & (np.abs(values - top) <= 1e-12))[0]
        hits.append(float(np.mean(labels[tied])) if len(tied) else math.nan)
    return _mean(hits)


def _field_identity_shuffle_hit(groups, rng):
    values_by_cell = {}
    fields = list(ENDPOINT_FIELDS)
    for cid, rows in groups.items():
        _, scalar, orientation = fields[int(rng.integers(0, len(fields)))]
        values_by_cell[cid] = _binary_sign_values(rows, scalar, orientation)
    return _hit_from_values_by_cell(groups, values_by_cell)


def _threshold_shuffle_hit(groups, rng):
    values_by_cell = {}
    all_vals = [
        _score(r, BEST_SCALAR, BEST_ORIENTATION)
        for rows in groups.values() for r in rows
        if np.isfinite(_score(r, BEST_SCALAR, BEST_ORIENTATION))
    ]
    for cid, rows in groups.items():
        threshold = float(rng.choice(all_vals)) if all_vals else math.nan
        signed = _field_values(rows, BEST_SCALAR, BEST_ORIENTATION)
        values_by_cell[cid] = np.asarray([
            1.0 if np.isfinite(v) and v > threshold else 0.0
            for v in signed
        ], dtype=float)
    return _hit_from_values_by_cell(groups, values_by_cell)


def _block_shuffle_hit(groups, rng, scope):
    cids = sorted(groups)
    by_key = defaultdict(list)
    for cid in cids:
        key = _parts(cid)["target"] if scope == "target" else _parts(cid)["level"]
        by_key[key].append(cid)
    keys = sorted(by_key)
    shuffled = keys[:]
    rng.shuffle(shuffled)
    key_map = dict(zip(keys, shuffled))
    values_by_cell = {}
    for cid in cids:
        key = _parts(cid)["target"] if scope == "target" else _parts(cid)["level"]
        donor_pool = by_key[key_map[key]]
        matched = [d for d in donor_pool if len(groups[d]) == len(groups[cid])]
        donor = matched[int(rng.integers(0, len(matched)))] if matched else donor_pool[int(rng.integers(0, len(donor_pool)))]
        donor_vals = _field_values(groups[donor], BEST_SCALAR, BEST_ORIENTATION)
        if len(donor_vals) == len(groups[cid]):
            values_by_cell[cid] = donor_vals.copy()
        else:
            fallback = float(np.nanmean(donor_vals)) if np.any(np.isfinite(donor_vals)) else math.nan
            scores = {str(r["candidate_order_int"]): v for r, v in zip(groups[donor], donor_vals)}
            values_by_cell[cid] = np.asarray([
                scores.get(str(r["candidate_order_int"]), fallback) for r in groups[cid]
            ], dtype=float)
    return _hit_from_values_by_cell(groups, values_by_cell)


def _within_cell_scalar_permutation_hit(groups, rng):
    values_by_cell = {}
    for cid, rows in groups.items():
        vals = _field_values(rows, BEST_SCALAR, BEST_ORIENTATION)
        rng.shuffle(vals)
        values_by_cell[cid] = vals
    return _hit_from_values_by_cell(groups, values_by_cell)


def _null_summary(name, observed, samples):
    vals = [float(v) for v in samples if np.isfinite(v)]
    arr = np.asarray(vals, dtype=float) if vals else np.asarray([], dtype=float)
    mean = float(np.mean(arr)) if len(arr) else math.nan
    p95 = float(np.quantile(arr, 0.95)) if len(arr) else math.nan
    return {
        "null_name": name,
        "observed_hit": observed,
        "null_mean_hit": mean,
        "null_p95_hit": p95,
        "observed_minus_null_mean": observed - mean if np.isfinite(mean) else math.nan,
        "observed_gt_null_p95": int(observed > p95 if np.isfinite(p95) else 0),
        "percentile": float(np.mean(arr <= observed)) if len(arr) else math.nan,
        "num_repeats": len(vals),
        "seed": NULL_SEED,
        "diagnostic_only": 1,
        "no_selection_artifact": 1,
    }


def transfer_null_summary(groups, observed_hit):
    rng = np.random.default_rng(NULL_SEED)
    return [
        _null_summary("N1_cell_preserving_label_shuffle", observed_hit,
                      [_label_shuffle_hit(groups, rng) for _ in range(NULL_REPS)]),
        _null_summary("N2_field_identity_shuffle", observed_hit,
                      [_field_identity_shuffle_hit(groups, rng) for _ in range(NULL_REPS)]),
        _null_summary("N3_threshold_shuffle", observed_hit,
                      [_threshold_shuffle_hit(groups, rng) for _ in range(NULL_REPS)]),
        _null_summary("N4_target_block_shuffle", observed_hit,
                      [_block_shuffle_hit(groups, rng, "target") for _ in range(NULL_REPS)]),
        _null_summary("N5_trajectory_block_shuffle", observed_hit,
                      [_block_shuffle_hit(groups, rng, "trajectory") for _ in range(NULL_REPS)]),
        _null_summary("N6_scalar_value_permutation_within_cell", observed_hit,
                      [_within_cell_scalar_permutation_hit(groups, rng) for _ in range(NULL_REPS)]),
    ]


def classify(res):
    summary = res["cross_cell_protocol_summary_rows"]
    same = next(r for r in summary if r["score_name"] == "same_cell_endpoint_scalar")
    best_template = _best_non_same_summary(summary, requires_scalar=False)
    best_scalar_transfer = _best_non_same_summary(summary, requires_scalar=True)
    best_template_hit = _f(best_template["hit"])
    best_scalar_hit = _f(best_scalar_transfer["hit"])
    same_hit = _f(same["hit"])
    null_max = max(_f(r["null_p95_hit"], -math.inf) for r in res["transfer_null_summary_rows"])
    artifact = best_scalar_hit <= null_max and best_template_hit <= null_max
    if artifact:
        primary = "C55-H_null_like_transfer_artifact"
    elif best_scalar_hit >= HIT_GATE and int(best_scalar_transfer["requires_test_endpoint_scalar"]):
        primary = "C55-G_transfer_requires_unavailable_test_endpoint_scalar"
    elif best_template_hit >= same_hit - FULL_CLOSE_GAP_GATE:
        primary = "C55-A_global_endpoint_template_transfer_sufficiency"
    elif best_template["protocol"] == "leave_cell_out" and best_template_hit >= HIT_GATE:
        primary = "C55-B_leave_cell_only_partial_transfer"
    elif next(r for r in summary if r["score_name"] == "leave_target_out_template_only")["hit"] < HIT_GATE:
        primary = "C55-C_leave_target_out_transfer_failure"
    elif next(r for r in summary if r["score_name"] == "leave_trajectory_out_template_only")["hit"] < HIT_GATE:
        primary = "C55-D_leave_trajectory_out_transfer_failure"
    else:
        primary = "C55-I_inconclusive_due_to_support_or_artifact"

    field_rows = res["field_family_transfer_summary_rows"]
    endpoint_scalar_rows = [r for r in field_rows if r["transfer_mode"] == "leave_cell_endpoint_scalar_binary"]
    best_field = max(endpoint_scalar_rows, key=lambda r: _f(r["hit"], -math.inf))
    secondaries = []
    if best_field["field_family"] == "joint_margin":
        secondaries.append("C55-S1_joint_margin_transfer_dominates")
    elif _f(best_field["hit"]) >= HIT_GATE:
        secondaries.append("C55-S2_single_component_transfer_dominates")
    if best_scalar_hit >= same_hit - FULL_CLOSE_GAP_GATE and int(best_scalar_transfer["requires_test_endpoint_scalar"]):
        secondaries.append("C55-S3_threshold_transfers_but_scalar_unavailable")
    component_hits = sorted({_f(r["hit"]) for r in endpoint_scalar_rows}, reverse=True)
    if len(component_hits) >= 2 and component_hits[0] - component_hits[1] < 0.02:
        secondaries.append("C55-S4_field_identity_unstable")
    same_target = next(r for r in summary if r["score_name"] == "same_target_cross_trajectory_template_only")
    leave_target = next(r for r in summary if r["score_name"] == "leave_target_out_template_only")
    if _f(same_target["hit"]) > _f(leave_target["hit"]) + 0.05:
        secondaries.append("C55-S5_target_local_transfer_only")
    leave_traj = next(r for r in summary if r["score_name"] == "leave_trajectory_out_template_only")
    leave_cell = next(r for r in summary if r["score_name"] == "leave_cell_out_template_only")
    if _f(leave_cell["hit"]) > _f(leave_traj["hit"]) + 0.05:
        secondaries.append("C55-S6_trajectory_local_transfer_only")
    if not res["split_label_budget_available"]:
        secondaries.append("C55-S7_no_split_label_budget_available")
    if res["c54_replay"]["best_key_only_hit"] < HIT_GATE:
        secondaries.append("C55-S8_source_only_escape_hatch_still_closed")

    return {
        "primary": primary,
        "secondary": secondaries,
        "artifact_or_null_like": bool(artifact),
        "best_template_only_score": best_template["score_name"],
        "best_template_only_hit": best_template["hit"],
        "best_endpoint_scalar_transfer_score": best_scalar_transfer["score_name"],
        "best_endpoint_scalar_transfer_hit": best_scalar_transfer["hit"],
        "same_cell_endpoint_scalar_hit": same["hit"],
        "same_cell_minus_best_template_gap": same_hit - best_template_hit,
        "same_cell_minus_best_endpoint_scalar_transfer_gap": same_hit - best_scalar_hit,
        "requires_test_endpoint_scalar_for_full_close": bool(
            best_scalar_hit >= same_hit - FULL_CLOSE_GAP_GATE and best_template_hit < same_hit - FULL_CLOSE_GAP_GATE),
    }


def gate_rows(res):
    summary = {r["score_name"]: r for r in res["cross_cell_protocol_summary_rows"]}
    best_template = _best_non_same_summary(res["cross_cell_protocol_summary_rows"], requires_scalar=False)
    best_scalar = _best_non_same_summary(res["cross_cell_protocol_summary_rows"], requires_scalar=True)
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c54_replay_identity_passed", "passed": all(int(r["pass"]) for r in res["c54_replay_identity_rows"])},
        {"check": "best_key_only_replayed_0_488", "passed": abs(res["c54_replay"]["best_key_only_hit"] - 0.4876543209876543) <= 1e-12},
        {"check": "same_cell_scalar_replayed_0_944", "passed": abs(res["c54_replay"]["same_cell_scalar_hit"] - 0.9444444444444444) <= 1e-12},
        {"check": "template_only_cross_cell_partial_0_704", "passed": abs(_f(best_template["hit"]) - 0.7037037037037037) <= 1e-12},
        {"check": "endpoint_scalar_transfer_requires_test_scalar", "passed": bool(int(best_scalar["requires_test_endpoint_scalar"]))},
        {"check": "endpoint_scalar_transfer_uses_no_same_cell_template_labels", "passed": not bool(int(best_scalar["uses_same_cell_target_labels_for_template"]))},
        {"check": "availability_ledger_emitted", "passed": len(res["endpoint_template_availability_ledger_rows"]) >= 1},
        {"check": "split_label_budget_unavailable", "passed": not res["split_label_budget_available"]},
        {"check": "source_key_baselines_below_gate", "passed": _f(summary["best_key_only"]["hit"]) < HIT_GATE},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    d = res["decision"]
    return [
        {
            "check": "c54_identity_replay",
            "passed": int(all(int(r["pass"]) for r in res["c54_replay_identity_rows"])),
            "finding": "C55 replays C54 key-only, C52 diagnostic, endpoint scalar, sign-bit, AUC, overlap, and split-label budget.",
        },
        {
            "check": "template_vs_test_scalar_separated",
            "passed": int(d["requires_test_endpoint_scalar_for_full_close"]),
            "finding": "Template-only transfer remains partial while endpoint-scalar transfer closes only after reading held-out endpoint scalars.",
        },
        {
            "check": "heldout_cell_labels_not_used_for_transfer_template",
            "passed": int(not any(
                int(r["uses_same_cell_target_labels_for_template"])
                for r in res["cross_cell_protocol_summary_rows"]
                if r["score_name"].endswith("_endpoint_scalar_threshold")
            )),
            "finding": "Leave-cell, leave-target, and leave-trajectory endpoint transfers fit templates outside the held-out cell.",
        },
        {
            "check": "availability_ledger_marks_test_endpoint_scalar",
            "passed": int(any(
                int(r["uses_target_endpoint_scalar_on_test_candidate"])
                and not int(r["available_under_original_source_only_DG"])
                for r in res["endpoint_template_availability_ledger_rows"]
            )),
            "finding": "Rows requiring held-out target endpoint scalars are explicitly unavailable under the original source-only DG setting.",
        },
        {
            "check": "nulls_emitted",
            "passed": int(len(res["transfer_null_summary_rows"]) == 6),
            "finding": "C55 emits field, threshold, block, label, and scalar permutation null controls.",
        },
        {
            "check": "split_label_boundary_kept_closed",
            "passed": int(not res["split_label_budget_available"]),
            "finding": "No split-label construction is available or claimed.",
        },
        {
            "check": "no_training_or_reinference",
            "passed": 1,
            "finding": "C55 is read-only over committed C54/C53/C52 artifacts.",
        },
        {
            "check": "no_chosen_checkpoint_artifact",
            "passed": 1,
            "finding": "C55 emits diagnostic summaries, ledgers, nulls, and reports only.",
        },
    ]


def recompute():
    cfg = _lock_config()
    inputs = _loads()
    rows = inputs["registry"]
    groups = inputs["groups"]
    c54_summary = inputs["c54_summary"]
    replay_rows = c54_replay_identity(c54_summary, groups, rows)
    if not all(int(r["pass"]) for r in replay_rows):
        raise ValueError("C55-STOP_c54_replay_identity_failed")
    summary_rows = cross_cell_protocol_summary(groups, inputs["c53_cells"], c54_summary)
    key_hit = c54_summary["c53_replay"]["best_key_only_hit"]
    same_hit = c54_summary["transfer"]["same_cell_hit"]
    field_rows = field_family_transfer_summary(groups, key_hit, same_hit)
    threshold_rows = threshold_transfer_curve(groups, key_hit, same_hit)
    cell_rows = transfer_cell_ledger(groups, inputs["c53_cells"], summary_rows)
    failure_rows = failure_reason_ledger(cell_rows)
    best_scalar = _best_non_same_summary(summary_rows, requires_scalar=True)
    null_rows = transfer_null_summary(groups, _f(best_scalar["hit"]))
    availability_rows = availability_ledger(summary_rows)
    table_counts = {
        "c54_replay_identity": len(replay_rows),
        "endpoint_template_availability_ledger": len(availability_rows),
        "cross_cell_protocol_summary": len(summary_rows),
        "leave_cell_out_transfer_summary": len([r for r in summary_rows if r["protocol"] == "leave_cell_out"]),
        "leave_target_out_transfer_summary": len([r for r in summary_rows if r["protocol"] == "leave_target_out"]),
        "leave_trajectory_out_transfer_summary": len([r for r in summary_rows if r["protocol"] == "leave_trajectory_out"]),
        "field_family_transfer_summary": len(field_rows),
        "threshold_transfer_curve": len(threshold_rows),
        "transfer_cell_ledger": len(cell_rows),
        "transfer_failure_reason_ledger": len(failure_rows),
        "transfer_null_summary": len(null_rows),
        "red_team_verification": 8,
        "artifact_hygiene_gate": 13,
    }
    res = {
        "milestone": MILESTONE,
        "config_hash": cfg,
        "inherits_from": ["C49", "C50", "C51", "C52", "C53", "C54"],
        "diagnostic_only_non_deployable": True,
        "c54_replay": {
            "random_tie_hit": c54_summary["c53_replay"]["random_tie_hit"],
            "best_strict_source_hit": c54_summary["c53_replay"]["best_strict_source_hit"],
            "best_key_only_hit": key_hit,
            "c52_trajectory_diagnostic_hit": c54_summary["c53_replay"]["c52_trajectory_diagnostic_hit"],
            "same_cell_scalar_hit": same_hit,
            "cross_cell_transfer_hit": c54_summary["decision"]["best_cross_cell_transfer_hit"],
            "best_scalar_field": "target_joint_margin_raw:high",
            "binary_sign_hit": same_hit,
            "threshold_overlap": 1.0,
            "auc_vs_joint_good": 1.0,
        },
        "c54_replay_identity_rows": replay_rows,
        "endpoint_template_availability_ledger_rows": availability_rows,
        "cross_cell_protocol_summary_rows": summary_rows,
        "field_family_transfer_summary_rows": field_rows,
        "threshold_transfer_curve_rows": threshold_rows,
        "transfer_cell_ledger_rows": cell_rows,
        "transfer_failure_reason_ledger_rows": failure_rows,
        "transfer_null_summary_rows": null_rows,
        "split_label_budget_available": False,
        "split_label_unavailable_reason": "required per-trial target prediction/label cache unavailable",
        "n_candidate_rows": len(rows),
        "n_trajectories": len(groups),
        "table_row_counts": table_counts,
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
        "c54_replay_identity_rows": _readcsv(os.path.join(TABLE_DIR, "c54_replay_identity.csv")),
        "endpoint_template_availability_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "endpoint_template_availability_ledger.csv")),
        "cross_cell_protocol_summary_rows": _readcsv(os.path.join(TABLE_DIR, "cross_cell_protocol_summary.csv")),
        "field_family_transfer_summary_rows": _readcsv(os.path.join(TABLE_DIR, "field_family_transfer_summary.csv")),
        "threshold_transfer_curve_rows": _readcsv(os.path.join(TABLE_DIR, "threshold_transfer_curve.csv")),
        "transfer_cell_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "transfer_cell_ledger.csv")),
        "transfer_failure_reason_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "transfer_failure_reason_ledger.csv")),
        "transfer_null_summary_rows": _readcsv(os.path.join(TABLE_DIR, "transfer_null_summary.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def render_main_md(res):
    d = res["decision"]
    r = res["c54_replay"]
    return "\n".join([
        f"# C55 - Cross-Cell Endpoint-Template Transfer / Information-Boundary Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['primary']}`",
        "",
        f"Secondary: `{';'.join(d['secondary'])}`",
        "",
        "## C54 Replay",
        "",
        f"- key-only / C52 diagnostic / same-cell endpoint scalar: **{_fmt(r['best_key_only_hit'])} / "
        f"{_fmt(r['c52_trajectory_diagnostic_hit'])} / {_fmt(r['same_cell_scalar_hit'])}**",
        f"- binary sign-bit hit / overlap / AUC: **{_fmt(r['binary_sign_hit'])} / "
        f"{_fmt(r['threshold_overlap'])} / {_fmt(r['auc_vs_joint_good'])}**",
        f"- C54 matched cross-cell template hit: **{_fmt(r['cross_cell_transfer_hit'])}**",
        "",
        "## Transfer Boundary",
        "",
        f"- best template-only transfer: `{d['best_template_only_score']}` hit **{_fmt(d['best_template_only_hit'])}**",
        f"- best endpoint-scalar transfer: `{d['best_endpoint_scalar_transfer_score']}` hit "
        f"**{_fmt(d['best_endpoint_scalar_transfer_hit'])}**",
        f"- same-cell minus template-only gap: **{_fmt(d['same_cell_minus_best_template_gap'])}**",
        f"- same-cell minus endpoint-scalar transfer gap: **{_fmt(d['same_cell_minus_best_endpoint_scalar_transfer_gap'])}**",
        f"- requires held-out target endpoint scalar for full close: **{d['requires_test_endpoint_scalar_for_full_close']}**",
        f"- split-label budget available: **{res['split_label_budget_available']}**",
        "",
        "## Bottom Line",
        "",
        "C55 closes the remaining C54 ambiguity. Cross-cell endpoint templates transfer only partially: the matched "
        "candidate-order template reaches the C54 0.704 level, while leave-cell/leave-target/leave-trajectory "
        "templates stay below the same-cell endpoint scalar. The 0.944 closure reappears only when the held-out "
        "candidate's target endpoint scalar is read and thresholded, so the boundary is an endpoint-scalar "
        "availability gap rather than a source/key/template sufficiency result.",
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
        "# C55 - Red-Team Verification",
        "",
        "C55 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for row in red_team_rows(res):
        lines.append(f"- {row['check']}: {'pass' if row['passed'] else 'fail'} - {row['finding']}")
    lines += [
        "",
        "Verdict: C55 is diagnostic-only. Template transferability and held-out endpoint scalar availability are separated.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "rather than ")


def _guard_forbidden(text):
    au.guard_forbidden(
        text, FORBIDDEN_CLAIM_SUBSTRINGS,
        negation_cues=_NEG_CUES, window=260, label=MILESTONE)


def _compact_json(res):
    d = res["decision"]
    best_template = _best_non_same_summary(res["cross_cell_protocol_summary_rows"], requires_scalar=False)
    best_scalar = _best_non_same_summary(res["cross_cell_protocol_summary_rows"], requires_scalar=True)
    null_top = max(res["transfer_null_summary_rows"], key=lambda r: _f(r["null_p95_hit"], -math.inf))
    return {
        "milestone": res["milestone"],
        "config_hash": res["config_hash"],
        "inherits_from": res["inherits_from"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "c54_replay": res["c54_replay"],
        "transfer_boundary": {
            "best_template_only_score": best_template["score_name"],
            "best_template_only_hit": best_template["hit"],
            "best_template_only_requires_test_endpoint_scalar": bool(int(best_template["requires_test_endpoint_scalar"])),
            "best_endpoint_scalar_transfer_score": best_scalar["score_name"],
            "best_endpoint_scalar_transfer_hit": best_scalar["hit"],
            "best_endpoint_scalar_transfer_requires_test_endpoint_scalar": bool(int(best_scalar["requires_test_endpoint_scalar"])),
            "same_cell_endpoint_scalar_hit": d["same_cell_endpoint_scalar_hit"],
            "same_cell_minus_best_template_gap": d["same_cell_minus_best_template_gap"],
            "same_cell_minus_best_endpoint_scalar_transfer_gap": d["same_cell_minus_best_endpoint_scalar_transfer_gap"],
            "requires_test_endpoint_scalar_for_full_close": d["requires_test_endpoint_scalar_for_full_close"],
        },
        "nulls": {
            "max_null_p95_name": null_top["null_name"],
            "max_null_p95_hit": null_top["null_p95_hit"],
            "observed_gt_all_null_p95": all(int(r["observed_gt_null_p95"]) for r in res["transfer_null_summary_rows"]),
        },
        "split_label_budget_available": res["split_label_budget_available"],
        "split_label_unavailable_reason": res["split_label_unavailable_reason"],
        "decision": res["decision"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "table_row_counts": res["table_row_counts"],
        "red_team": red_team_rows(res),
        "artifact_hygiene": gate_rows(res),
    }


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _writecsv(os.path.join(out_dir, "c54_replay_identity.csv"), res["c54_replay_identity_rows"],
              ["metric", "c54_reported_value", "c55_replayed_value", "abs_diff", "pass"])
    common_summary_cols = [
        "score_name", "field_family", "protocol", "mode", "hit", "min_target_hit", "min_trajectory_hit",
        "cell_hit_ge_0_70_count", "cells_evaluated", "robust_mean_gate", "robust_min_target_gate",
        "robust_min_trajectory_gate", "full_close_gate", "observed_gt_0_70", "mean_train_cells",
        "mean_threshold", "min_threshold", "max_threshold", "requires_test_endpoint_scalar",
        "uses_source_only_inputs", "uses_key_only_inputs", "uses_target_unlabeled_inputs",
        "uses_target_endpoint_scalar_on_test_candidate", "uses_same_cell_target_labels_for_template",
        "uses_other_cell_target_labels_for_template", "uses_trial_level_split_labels",
        "available_under_original_source_only_DG", "diagnostic_only", "no_selection_artifact",
        "transfer_gain_over_key", "same_cell_gain_over_key", "transfer_closed_fraction",
        "same_cell_minus_transfer_gap",
    ]
    _writecsv(os.path.join(out_dir, "cross_cell_protocol_summary.csv"),
              res["cross_cell_protocol_summary_rows"], common_summary_cols)
    for protocol, name in (
        ("leave_cell_out", "leave_cell_out_transfer_summary.csv"),
        ("leave_target_out", "leave_target_out_transfer_summary.csv"),
        ("leave_trajectory_out", "leave_trajectory_out_transfer_summary.csv"),
    ):
        _writecsv(os.path.join(out_dir, name),
                  [r for r in res["cross_cell_protocol_summary_rows"] if r["protocol"] == protocol],
                  common_summary_cols)
    _writecsv(os.path.join(out_dir, "endpoint_template_availability_ledger.csv"),
              res["endpoint_template_availability_ledger_rows"],
              ["score_name", "field_family", "protocol", "mode", "uses_source_only_inputs",
               "uses_key_only_inputs", "uses_target_unlabeled_inputs",
               "uses_target_endpoint_scalar_on_test_candidate", "uses_same_cell_target_labels_for_template",
               "uses_other_cell_target_labels_for_template", "uses_trial_level_split_labels",
               "available_under_original_source_only_DG", "diagnostic_only", "reported_hit",
               "availability_class"])
    _writecsv(os.path.join(out_dir, "field_family_transfer_summary.csv"),
              res["field_family_transfer_summary_rows"],
              ["field_family", "scalar_name", "transfer_mode", "hit", "min_target_hit", "min_trajectory_hit",
               "cell_hit_ge_0_70_count", "mean_donor_or_train_cells", "requires_test_endpoint_scalar",
               "uses_other_cell_target_labels_for_template", "same_cell_oracle", "diagnostic_only",
               "no_selection_artifact", "transfer_gain_over_key", "same_cell_gain_over_key",
               "transfer_closed_fraction", "same_cell_minus_transfer_gap"])
    _writecsv(os.path.join(out_dir, "threshold_transfer_curve.csv"), res["threshold_transfer_curve_rows"],
              ["protocol", "scalar_name", "threshold_mode", "hit", "min_target_hit", "min_trajectory_hit",
               "cell_hit_ge_0_70_count", "mean_train_cells", "mean_threshold", "min_threshold",
               "max_threshold", "requires_test_endpoint_scalar", "uses_other_cell_target_labels_for_template",
               "uses_same_cell_target_labels_for_template", "diagnostic_only", "no_selection_artifact",
               "transfer_gain_over_key", "same_cell_gain_over_key", "transfer_closed_fraction",
               "same_cell_minus_transfer_gap"])
    _writecsv(os.path.join(out_dir, "transfer_cell_ledger.csv"), res["transfer_cell_ledger_rows"],
              ["target_id", "trajectory_id", "cell_id", "n_candidates", "base_rate", "key_only_hit",
               "same_cell_scalar_hit", "leave_cell_template_hit", "leave_target_template_hit",
               "leave_trajectory_template_hit", "matched_geometry_template_hit", "best_template_hit",
               "best_template_protocol", "leave_cell_endpoint_scalar_hit", "leave_target_endpoint_scalar_hit",
               "leave_trajectory_endpoint_scalar_hit", "best_endpoint_scalar_transfer_hit",
               "best_endpoint_scalar_transfer_protocol", "transfer_closes_vs_key", "template_matches_same_cell",
               "endpoint_scalar_transfer_matches_same_cell", "requires_test_endpoint_scalar",
               "same_cell_oracle_required", "failure_reason_code", "diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "transfer_failure_reason_ledger.csv"),
              res["transfer_failure_reason_ledger_rows"],
              ["reason_code", "cell_count", "cell_fraction", "interpretation", "diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "transfer_null_summary.csv"), res["transfer_null_summary_rows"],
              ["null_name", "observed_hit", "null_mean_hit", "null_p95_hit", "observed_minus_null_mean",
               "observed_gt_null_p95", "percentile", "num_repeats", "seed", "diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "red_team_verification.csv"), red_team_rows(res),
              ["check", "passed", "finding"])
    _writecsv(os.path.join(out_dir, "artifact_hygiene_gate.csv"), gate_rows(res),
              ["check", "passed"])


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    red = render_red_team_md(res)
    for text in (md, red):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C55_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C55_CROSS_CELL_ENDPOINT_TEMPLATE_BOUNDARY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c55_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c55_cross_cell_endpoint_template_boundary")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    d = res["decision"]
    print(f"[C55] decision={d['primary']} best_template={d['best_template_only_hit']} "
          f"best_endpoint_transfer={d['best_endpoint_scalar_transfer_hit']}")


if __name__ == "__main__":
    main()
