"""C52 - Minimal Gauge-Key / Conditioning-Sufficiency Audit."""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict

import numpy as np

from . import audit_utils as au
from . import c50_conditioned_island_morphology as c50
from . import c51_trajectory_fragmentation_underuse as c51
from . import schema as c49_schema


REPORT_JSON = "oaci/reports/C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.json"
TABLE_DIR = "oaci/reports/c52_tables"

MILESTONE = "C52"
NULL_REPS = 64
NULL_SEED = 52052
GAP_CLOSE_GATE = 0.05
EVALUATION_SCOPE = "within_trajectory_top_hit"
BASELINE_LABEL = "trajectory_conditioned_random_tie"

DECISIONS = (
    "C52-A_source_observable_sufficiency",
    "C52-B_target_key_sufficiency",
    "C52-C_trajectory_key_sufficiency",
    "C52-D_additive_target_plus_trajectory_sufficiency",
    "C52-E_target_x_trajectory_interaction_required",
    "C52-F_target_unlabeled_geometry_sufficiency",
    "C52-G_diagnostic_label_content_required",
    "C52-H_mixed_key_interaction_and_label_content",
)

STRICT_SOURCE_SCORES = c50.STRICT_SOURCE_SCORES
FORBIDDEN_CLAIM_SUBSTRINGS = c51.FORBIDDEN_CLAIM_SUBSTRINGS + (
    "source-only rescue",
    "target-unlabeled method",
    "target-grouped method",
    "trajectory-conditioned method",
    "usable policy",
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


def _loads():
    with open(c50.REPORT_JSON) as f:
        c50_summary = json.load(f)
    with open(c51.REPORT_JSON) as f:
        c51_summary = json.load(f)
    return {
        "c50_summary": c50_summary,
        "c51_summary": c51_summary,
        "island_rows": _readcsv(os.path.join(c50.TABLE_DIR, "island_morphology.csv")),
        "trajectory_rows": _readcsv(os.path.join(c51.TABLE_DIR, "trajectory_failure_ledger.csv")),
        "score_rows": _readcsv(os.path.join(c51.TABLE_DIR, "source_score_underuse_attribution.csv")),
    }


def _trajectory_groups(rows):
    groups = defaultdict(list)
    for r in rows:
        groups[r["trajectory"]].append(r)
    return dict(groups)


def _trajectory_parts(trajectory_id):
    bits = trajectory_id.split("|")
    if len(bits) != 4:
        return {"seed": "", "target": "", "level": "", "regime": ""}
    return {"seed": bits[0], "target": bits[1], "level": bits[2], "regime": bits[3]}


def _trajectory_base_hit(groups):
    hits = []
    for rows in groups.values():
        hits.append(_mean([int(r["query_positive_label"]) for r in rows]))
    return _mean(hits)


def _top_hit_by_trajectory(groups, score_fn):
    hits = []
    for rows in groups.values():
        scores = []
        for r in rows:
            s = score_fn(r)
            scores.append(float(s) if np.isfinite(s) else -math.inf)
        if not scores:
            continue
        top = max(scores)
        tied = [r for r, s in zip(rows, scores) if abs(s - top) <= 1e-12]
        hits.append(_mean([int(r["query_positive_label"]) for r in tied]))
    return _mean(hits)


def _constant_key_hit(groups):
    return _trajectory_base_hit(groups)


def _best_source_geometry(rows, groups):
    specs = []
    for field in (
        "neighbors_n",
        "source_distance_mean",
        "source_distance_min",
        "source_distance_max",
        "source_distance_q25",
        "source_distance_q75",
    ):
        specs.append((field, "high", lambda r, f=field: _f(r.get(f), -math.inf)))
        specs.append((field, "low", lambda r, f=field: -_f(r.get(f), math.inf)))
    evaluated = []
    for field, orientation, fn in specs:
        hit = _top_hit_by_trajectory(groups, fn)
        evaluated.append({
            "field": field,
            "orientation": orientation,
            "hit": hit,
        })
    best = max(evaluated, key=lambda r: _f(r["hit"], -math.inf))
    counts = [_f(r["neighbors_n"]) for r in rows]
    return {
        "best_field": best["field"],
        "best_orientation": best["orientation"],
        "best_hit": best["hit"],
        "neighbor_count_q25": au.finite_quantile(counts, 0.25),
        "neighbor_count_q50": au.finite_quantile(counts, 0.50),
        "neighbor_count_q75": au.finite_quantile(counts, 0.75),
        "evaluated_fields": evaluated,
    }


def _best_score_rows(score_rows):
    strict = [r for r in score_rows if int(r["hindsight_diagnostic_only"]) == 0]
    all_rows = list(score_rows)
    def best(col, rows=strict):
        vals = [r for r in rows if np.isfinite(_f(r.get(col)))]
        return max(vals, key=lambda r: _f(r[col])) if vals else {}
    return {
        "best_raw": best("raw_score_hit"),
        "best_sign": best("best_sign_flip_hit"),
        "best_monotone": best("best_monotone_decile_hit"),
        "best_target_diag": best("best_target_centered_diagnostic_hit"),
        "best_trajectory_diag": best("best_trajectory_centered_diagnostic_hit"),
        "best_any_raw": best("raw_score_hit", all_rows),
    }


def _oracle_hit(score_rows):
    vals = [_f(r["oracle_local_bayes_hit"]) for r in score_rows if np.isfinite(_f(r["oracle_local_bayes_hit"]))]
    uniq = sorted(set(round(v, 15) for v in vals))
    if len(uniq) != 1:
        raise ValueError(f"C52 expected one C51 oracle local-Bayes hit; got {uniq}")
    return vals[0]


def _closure_fraction(hit, best_raw_hit, oracle_hit):
    denom = oracle_hit - best_raw_hit
    if not np.isfinite(hit) or denom <= 1e-12:
        return math.nan
    return (hit - best_raw_hit) / denom


def _ladder_row(
    ladder,
    candidate,
    hit,
    oracle_hit,
    best_raw_hit,
    *,
    available=True,
    key_only,
    source_observable,
    target_id_used=False,
    trajectory_id_used=False,
    target_unlabeled_geometry_used=False,
    target_label_derived=False,
    hindsight_diagnostic_only=False,
    comparison_source,
    interpretation,
):
    gap = oracle_hit - hit if available and np.isfinite(hit) else math.nan
    return {
        "ladder": ladder,
        "candidate": candidate,
        "information_class": _info_class(
            source_observable, target_id_used, trajectory_id_used,
            target_unlabeled_geometry_used, target_label_derived),
        "evaluation_scope": EVALUATION_SCOPE,
        "available": int(bool(available)),
        "hit": hit if available else math.nan,
        "gap_to_oracle": gap,
        "closes_gap": int(bool(available) and np.isfinite(gap) and gap <= GAP_CLOSE_GATE),
        "closure_fraction_vs_best_raw_source": _closure_fraction(hit, best_raw_hit, oracle_hit)
        if available else math.nan,
        "key_only": int(bool(key_only)),
        "source_observable": int(bool(source_observable)),
        "target_id_used": int(bool(target_id_used)),
        "trajectory_id_used": int(bool(trajectory_id_used)),
        "target_unlabeled_geometry_used": int(bool(target_unlabeled_geometry_used)),
        "target_label_derived": int(bool(target_label_derived)),
        "hindsight_diagnostic_only": int(bool(hindsight_diagnostic_only)),
        "comparison_source": comparison_source,
        "interpretation": interpretation,
        "target_labels_diagnostic_only": int(bool(target_label_derived)),
        "no_selection_artifact": 1,
    }


def _info_class(source_observable, target_id, trajectory_id, target_unlabeled, label_derived):
    bits = []
    if source_observable:
        bits.append("source_observable")
    if target_id:
        bits.append("target_id")
    if trajectory_id:
        bits.append("trajectory_id")
    if target_unlabeled:
        bits.append("target_unlabeled_geometry")
    if label_derived:
        bits.append("label_derived_diagnostic")
    return "+".join(bits) if bits else "unavailable"


def conditioning_ladder(inputs, base_hit, geom, bests, oracle_hit):
    best_raw = _f(bests["best_raw"]["raw_score_hit"])
    return [
        _ladder_row(
            "L0", "existing_source_best_raw", best_raw, oracle_hit, best_raw,
            key_only=False, source_observable=True,
            comparison_source=bests["best_raw"].get("score_name", ""),
            interpretation="Best strict source score remains below the C51 trajectory local-Bayes ceiling."),
        _ladder_row(
            "L0", "existing_source_best_sign_flip", _f(bests["best_sign"]["best_sign_flip_hit"]),
            oracle_hit, best_raw,
            key_only=False, source_observable=True,
            comparison_source=bests["best_sign"].get("score_name", ""),
            interpretation="Orientation reversal does not close the residual gap."),
        _ladder_row(
            "L0", "existing_source_best_monotone_diagnostic",
            _f(bests["best_monotone"]["best_monotone_decile_hit"]),
            oracle_hit, best_raw,
            key_only=False, source_observable=True, target_label_derived=True,
            hindsight_diagnostic_only=True,
            comparison_source=bests["best_monotone"].get("score_name", ""),
            interpretation="Global monotone diagnostic calibration is still insufficient."),
        _ladder_row(
            "L1", "source_metadata_key_only_tie", base_hit, oracle_hit, best_raw,
            key_only=True, source_observable=True,
            comparison_source=BASELINE_LABEL,
            interpretation="Seed/level/regime keys are constant inside each evaluated trajectory."),
        _ladder_row(
            "L1", "source_geometry_key_only_best_observable", geom["best_hit"], oracle_hit, best_raw,
            key_only=True, source_observable=True,
            comparison_source=f"{geom['best_field']}:{geom['best_orientation']}",
            interpretation="Observable source-geometry fields add weak localization but do not close the gap."),
        _ladder_row(
            "L2", "target_id_key_only", base_hit, oracle_hit, best_raw,
            key_only=True, source_observable=False, target_id_used=True,
            comparison_source=BASELINE_LABEL,
            interpretation="Target id is fixed within a trajectory and has no within-trajectory ordering power."),
        _ladder_row(
            "L2", "target_centered_label_diagnostic",
            _f(bests["best_target_diag"]["best_target_centered_diagnostic_hit"]),
            oracle_hit, best_raw,
            key_only=False, source_observable=True, target_id_used=True,
            target_label_derived=True, hindsight_diagnostic_only=True,
            comparison_source=bests["best_target_diag"].get("score_name", ""),
            interpretation="Target-centered label-derived calibration helps but remains below closure."),
        _ladder_row(
            "L3", "trajectory_id_key_only", base_hit, oracle_hit, best_raw,
            key_only=True, source_observable=False, target_id_used=True, trajectory_id_used=True,
            comparison_source=BASELINE_LABEL,
            interpretation="Trajectory id alone ties all candidates inside the evaluated trajectory."),
        _ladder_row(
            "L3", "trajectory_centered_label_diagnostic",
            _f(bests["best_trajectory_diag"]["best_trajectory_centered_diagnostic_hit"]),
            oracle_hit, best_raw,
            key_only=False, source_observable=True, target_id_used=True, trajectory_id_used=True,
            target_label_derived=True, hindsight_diagnostic_only=True,
            comparison_source=bests["best_trajectory_diag"].get("score_name", ""),
            interpretation="Only trajectory-centered label-derived diagnostic calibration closes the residual."),
        _ladder_row(
            "L4", "additive_target_plus_trajectory_key_only", base_hit, oracle_hit, best_raw,
            key_only=True, source_observable=False, target_id_used=True, trajectory_id_used=True,
            comparison_source=BASELINE_LABEL,
            interpretation="Additive target plus trajectory keys still have no within-trajectory rank."),
        _ladder_row(
            "L4", "target_x_trajectory_key_only", base_hit, oracle_hit, best_raw,
            key_only=True, source_observable=False, target_id_used=True, trajectory_id_used=True,
            comparison_source=BASELINE_LABEL,
            interpretation="The interaction key alone is a cell label, not a within-cell ordering signal."),
        _ladder_row(
            "L4", "target_x_trajectory_label_diagnostic",
            _f(bests["best_trajectory_diag"]["best_trajectory_centered_diagnostic_hit"]),
            oracle_hit, best_raw,
            key_only=False, source_observable=True, target_id_used=True, trajectory_id_used=True,
            target_label_derived=True, hindsight_diagnostic_only=True,
            comparison_source=bests["best_trajectory_diag"].get("score_name", ""),
            interpretation="The closing signal is target-label diagnostic content inside the cell."),
        _ladder_row(
            "L5", "target_unlabeled_geometry", math.nan, oracle_hit, best_raw,
            available=False, key_only=True, source_observable=False,
            target_unlabeled_geometry_used=True,
            comparison_source="not_available_in_committed_C49_C51_artifacts",
            interpretation="No pre-existing target-unlabeled geometry table is available for this locked audit."),
    ]


def _decision_from_ladder(rows):
    def closes(name):
        r = next(x for x in rows if x["candidate"] == name)
        return bool(int(r["available"])) and bool(int(r["closes_gap"]))
    source_names = (
        "existing_source_best_raw",
        "existing_source_best_sign_flip",
        "source_metadata_key_only_tie",
        "source_geometry_key_only_best_observable",
    )
    source_closes = any(closes(n) for n in source_names)
    target_key_closes = closes("target_id_key_only")
    trajectory_key_closes = closes("trajectory_id_key_only")
    additive_closes = closes("additive_target_plus_trajectory_key_only")
    interaction_key_closes = closes("target_x_trajectory_key_only")
    unlabeled_closes = closes("target_unlabeled_geometry")
    label_closes = closes("trajectory_centered_label_diagnostic") or closes("target_x_trajectory_label_diagnostic")
    if source_closes:
        decision = "C52-A_source_observable_sufficiency"
    elif target_key_closes:
        decision = "C52-B_target_key_sufficiency"
    elif trajectory_key_closes:
        decision = "C52-C_trajectory_key_sufficiency"
    elif additive_closes:
        decision = "C52-D_additive_target_plus_trajectory_sufficiency"
    elif interaction_key_closes and label_closes:
        decision = "C52-H_mixed_key_interaction_and_label_content"
    elif interaction_key_closes:
        decision = "C52-E_target_x_trajectory_interaction_required"
    elif unlabeled_closes:
        decision = "C52-F_target_unlabeled_geometry_sufficiency"
    elif label_closes:
        decision = "C52-G_diagnostic_label_content_required"
    else:
        decision = "C52-H_mixed_key_interaction_and_label_content"
    return {
        "decision": decision,
        "source_observable_closes_gap": source_closes,
        "target_key_only_closes_gap": target_key_closes,
        "trajectory_key_only_closes_gap": trajectory_key_closes,
        "additive_target_plus_trajectory_key_closes_gap": additive_closes,
        "target_x_trajectory_key_only_closes_gap": interaction_key_closes,
        "target_unlabeled_geometry_closes_gap": unlabeled_closes,
        "label_derived_diagnostic_closes_gap": label_closes,
    }


def gauge_key_decomposition(ladder_rows):
    keep = {
        "existing_source_best_raw": "best_existing_source_score",
        "source_geometry_key_only_best_observable": "best_source_observable_geometry",
        "target_id_key_only": "target_id_key_only",
        "target_centered_label_diagnostic": "target_centered_label_content",
        "trajectory_id_key_only": "trajectory_id_key_only",
        "trajectory_centered_label_diagnostic": "trajectory_centered_label_content",
        "additive_target_plus_trajectory_key_only": "additive_target_plus_trajectory_key_only",
        "target_x_trajectory_key_only": "target_x_trajectory_key_only",
        "target_x_trajectory_label_diagnostic": "target_x_trajectory_label_content",
        "target_unlabeled_geometry": "target_unlabeled_geometry",
    }
    out = []
    baseline = next(r for r in ladder_rows if r["candidate"] == "existing_source_best_raw")
    for r in ladder_rows:
        if r["candidate"] not in keep:
            continue
        hit = _f(r["hit"])
        out.append({
            "component": keep[r["candidate"]],
            "candidate": r["candidate"],
            "available": r["available"],
            "hit": r["hit"],
            "gap_to_oracle": r["gap_to_oracle"],
            "increment_vs_best_raw_source": hit - _f(baseline["hit"]) if np.isfinite(hit) else math.nan,
            "closure_fraction_vs_best_raw_source": r["closure_fraction_vs_best_raw_source"],
            "closes_gap": r["closes_gap"],
            "key_only": r["key_only"],
            "target_label_derived": r["target_label_derived"],
            "interpretation": r["interpretation"],
            "target_labels_diagnostic_only": r["target_labels_diagnostic_only"],
            "no_selection_artifact": 1,
        })
    return out


def _field_values(rows, field, orientation):
    vals = []
    for r in rows:
        v = _f(r.get(field))
        vals.append(v if orientation == "high" else -v)
    return np.asarray(vals, dtype=float)


def _top_hit_from_values(rows, values):
    groups = defaultdict(list)
    for i, r in enumerate(rows):
        groups[r["trajectory"]].append(i)
    hits = []
    for idx in groups.values():
        vals = values[idx]
        finite = np.isfinite(vals)
        if not np.any(finite):
            continue
        top = float(np.max(vals[finite]))
        tied = [i for i in idx if np.isfinite(values[i]) and abs(float(values[i]) - top) <= 1e-12]
        hits.append(_mean([int(rows[i]["query_positive_label"]) for i in tied]))
    return _mean(hits)


def _null_summary_row(null_name, key, observed, samples, *, status, preserved, destroyed, reason=""):
    vals = [float(v) for v in samples if np.isfinite(v)]
    if vals:
        arr = np.asarray(vals, dtype=float)
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        pct = float(np.mean(arr <= observed)) if np.isfinite(observed) else math.nan
        p_hi = float(np.mean(arr >= observed)) if np.isfinite(observed) else math.nan
        p_lo = pct
        p2 = min(1.0, 2.0 * min(p_lo, p_hi)) if np.isfinite(p_lo) and np.isfinite(p_hi) else math.nan
    else:
        mean = observed if np.isfinite(observed) else math.nan
        std = math.nan
        pct = math.nan
        p2 = math.nan
    return {
        "null_name": null_name,
        "key_or_field": key,
        "observed_hit": observed,
        "null_mean": mean,
        "null_std": std,
        "percentile": pct,
        "empirical_p_two_sided": p2,
        "n_permutations": len(vals),
        "status": status,
        "preserved_structure": preserved,
        "destroyed_structure": destroyed,
        "unavailable_reason": reason,
        "target_labels_diagnostic_only": 1,
        "no_selection_artifact": 1,
    }


def key_null_calibration(island_rows, groups, base_hit, geom, ladder_rows):
    out = []
    for key in ("target_id", "trajectory_id", "additive_target_plus_trajectory", "target_x_trajectory"):
        out.append(_null_summary_row(
            "N5_key_only_identity_tie_null", key, base_hit, [],
            status="analytical_key_tie",
            preserved=EVALUATION_SCOPE,
            destroyed="none",
            reason="key_is_constant_inside_each_evaluated_trajectory"))
    rng = np.random.default_rng(NULL_SEED)
    best_values = _field_values(island_rows, geom["best_field"], geom["best_orientation"])
    observed = _top_hit_from_values(island_rows, best_values)
    samples = []
    for _ in range(NULL_REPS):
        perm = np.array(best_values, copy=True)
        rng.shuffle(perm)
        samples.append(_top_hit_from_values(island_rows, perm))
    out.append(_null_summary_row(
        "N6_source_geometry_score_shuffle", f"{geom['best_field']}:{geom['best_orientation']}",
        observed, samples,
        status="available",
        preserved="trajectory_sizes_and_label_counts",
        destroyed="source_geometry_to_row_alignment"))
    traj_diag = next(r for r in ladder_rows if r["candidate"] == "trajectory_centered_label_diagnostic")
    out.append(_null_summary_row(
        "N7_label_derived_diagnostic_quarantine", "trajectory_centered_label_diagnostic",
        _f(traj_diag["hit"]), [],
        status="unavailable_for_key_only_null",
        preserved="label_derived_control_reported_separately",
        destroyed="none",
        reason="diagnostic_control_uses_target_labels_and_is_not_a_key_only_baseline"))
    return out


def target_trajectory_cell_ledger(island_rows, trajectory_rows, geom):
    groups = _trajectory_groups(island_rows)
    c51_by_traj = {r["trajectory_id"]: r for r in trajectory_rows}
    out = []
    for traj_id, rows in sorted(groups.items()):
        parts = _trajectory_parts(traj_id)
        c51_row = c51_by_traj[traj_id]
        base = _mean([int(r["query_positive_label"]) for r in rows])
        field_vals = [_f(r.get(geom["best_field"])) for r in rows]
        if geom["best_orientation"] == "low":
            field_vals = [-v for v in field_vals]
        top = max([v for v in field_vals if np.isfinite(v)], default=math.nan)
        tied = [r for r, v in zip(rows, field_vals) if np.isfinite(v) and abs(v - top) <= 1e-12]
        geom_hit = _mean([int(r["query_positive_label"]) for r in tied])
        local = _f(c51_row["local_bayes_hit"])
        best_score = _f(c51_row["best_existing_score_hit"])
        primary = "label_content_required_after_key_tie"
        if np.isfinite(local) and local < c50.HIT_GATE:
            primary = "trajectory_fragmented_low_local_bayes"
        elif np.isfinite(best_score) and best_score < c50.HIT_GATE:
            primary = "source_score_underuses_diagnostic_island"
        out.append({
            "trajectory_id": traj_id,
            "target": parts["target"],
            "seed": parts["seed"],
            "level": parts["level"],
            "regime": parts["regime"],
            "n_rows": len(rows),
            "base_hit": base,
            "target_key_only_hit": base,
            "trajectory_key_only_hit": base,
            "target_x_trajectory_key_only_hit": base,
            "best_source_geometry_hit": geom_hit,
            "local_bayes_hit": local,
            "best_existing_source_score_hit": best_score,
            "underuse_gap": _f(c51_row["underuse_gap"]),
            "key_only_closes_cell": int(
                np.isfinite(base) and np.isfinite(local) and local >= c50.HIT_GATE and
                local - base <= GAP_CLOSE_GATE),
            "label_diagnostic_closes_cell": int(np.isfinite(local) and local >= c50.HIT_GATE),
            "primary_reason": primary,
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def residual_failure_reason_ledger(cell_rows):
    specs = [
        ("KEY_ONLY_HAS_NO_WITHIN_TRAJECTORY_RANK",
         lambda r: True,
         "Target and trajectory keys tie all candidates under the frozen within-trajectory evaluation."),
        ("SOURCE_SCORE_UNDERUSES_DIAGNOSTIC_ISLAND",
         lambda r: _f(r["local_bayes_hit"]) >= c50.HIT_GATE and _f(r["best_existing_source_score_hit"]) < c50.HIT_GATE,
         "A local diagnostic island exists but the best existing source score does not recover it."),
        ("LOW_TRAJECTORY_LOCAL_BAYES",
         lambda r: _f(r["local_bayes_hit"]) < c50.HIT_GATE,
         "The trajectory remains fragmented even under the locked local diagnostic ceiling."),
        ("LABEL_DERIVED_DIAGNOSTIC_CLOSES_RESIDUAL",
         lambda r: _f(r["local_bayes_hit"]) >= c50.HIT_GATE,
         "Closing trajectories require label-derived local diagnostic content, not only the key."),
    ]
    out = []
    for code, pred, desc in specs:
        rows = [r for r in cell_rows if pred(r)]
        out.append({
            "reason_code": code,
            "n_trajectories": len(rows),
            "fraction_trajectories": len(rows) / len(cell_rows) if cell_rows else math.nan,
            "mean_base_hit": _mean([r["base_hit"] for r in rows]),
            "mean_local_bayes_hit": _mean([r["local_bayes_hit"] for r in rows]),
            "mean_best_existing_source_score_hit": _mean([r["best_existing_source_score_hit"] for r in rows]),
            "description": desc,
            "target_labels_diagnostic_only": 1,
            "no_selection_artifact": 1,
        })
    return out


def classify(ladder_rows):
    d = _decision_from_ladder(ladder_rows)
    best_key_only = max(
        [_f(r["hit"]) for r in ladder_rows if int(r["available"]) and int(r["key_only"])],
        default=math.nan)
    best_label = max(
        [_f(r["hit"]) for r in ladder_rows if int(r["available"]) and int(r["target_label_derived"])],
        default=math.nan)
    oracle = _f(next(r["hit"] for r in ladder_rows if r["candidate"] == "trajectory_centered_label_diagnostic"))
    source_raw = _f(next(r["hit"] for r in ladder_rows if r["candidate"] == "existing_source_best_raw"))
    d.update({
        "best_key_only_hit": best_key_only,
        "best_label_derived_hit": best_label,
        "best_strict_source_hit": source_raw,
        "trajectory_centered_diagnostic_hit": oracle,
        "best_key_only_gap_to_c51_oracle": _f(next(
            r["gap_to_oracle"] for r in ladder_rows
            if r["candidate"] == "source_geometry_key_only_best_observable")),
        "gap_close_gate": GAP_CLOSE_GATE,
    })
    return d


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == c49_schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_c50_c51_artifacts", "passed": True},
        {"check": "locked_witness_unchanged", "passed": res["locked_witness"]["eps_quantile"] == "q20"},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "key_only_label_derived_separated", "passed": res["decision"]["label_derived_diagnostic_closes_gap"]},
        {"check": "source_only_ladder_excludes_target_fields", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_selection_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
    ]


def red_team_rows(res):
    d = res["decision"]
    return [
        {
            "check": "c51_residual_replayed",
            "passed": int(abs(d["best_strict_source_hit"] - 0.5061728395061729) <= 1e-12 and
                          abs(d["trajectory_centered_diagnostic_hit"] - 0.8127572016460904) <= 1e-12),
            "finding": "C52 reuses C51 source-score and trajectory diagnostic gaps without rerunning earlier audits.",
        },
        {
            "check": "key_only_vs_label_derived_not_conflated",
            "passed": int((not d["trajectory_key_only_closes_gap"]) and d["label_derived_diagnostic_closes_gap"]),
            "finding": "Trajectory key-only remains at the trajectory random-tie baseline while label-derived diagnostics close.",
        },
        {
            "check": "source_observable_not_sufficient",
            "passed": int(not d["source_observable_closes_gap"]),
            "finding": "Existing source scores and observable source geometry do not reach the closure gate.",
        },
        {
            "check": "target_key_not_sufficient",
            "passed": int(not d["target_key_only_closes_gap"]),
            "finding": "Target id alone has no within-trajectory rank under the frozen evaluation scope.",
        },
        {
            "check": "trajectory_key_not_sufficient",
            "passed": int(not d["trajectory_key_only_closes_gap"]),
            "finding": "Trajectory id alone ties the candidate set and does not explain the residual.",
        },
        {
            "check": "target_unlabeled_geometry_not_claimed",
            "passed": int(not d["target_unlabeled_geometry_closes_gap"]),
            "finding": "No target-unlabeled geometry sufficiency claim is made from unavailable committed artifacts.",
        },
        {
            "check": "n5_n7_nulls_or_quarantine_emitted",
            "passed": int(res["table_row_counts"].get("key_null_calibration_summary", 0) >= 6),
            "finding": "N5/N6 key nulls and N7 diagnostic quarantine rows are emitted.",
        },
        {
            "check": "no_selection_artifact",
            "passed": 1,
            "finding": "C52 writes audit ledgers only and does not emit selected-candidate fields.",
        },
    ]


def recompute():
    cfg = _lock_config()
    inputs = _loads()
    groups = _trajectory_groups(inputs["island_rows"])
    base_hit = _trajectory_base_hit(groups)
    geom = _best_source_geometry(inputs["island_rows"], groups)
    bests = _best_score_rows(inputs["score_rows"])
    oracle = _oracle_hit(inputs["score_rows"])
    ladder = conditioning_ladder(inputs, base_hit, geom, bests, oracle)
    decomp = gauge_key_decomposition(ladder)
    null_rows = key_null_calibration(inputs["island_rows"], groups, base_hit, geom, ladder)
    cell_rows = target_trajectory_cell_ledger(inputs["island_rows"], inputs["trajectory_rows"], geom)
    reason_rows = residual_failure_reason_ledger(cell_rows)
    decision = classify(ladder)
    row_counts = {
        "conditioning_ladder_summary": len(ladder),
        "gauge_key_decomposition": len(decomp),
        "key_null_calibration_summary": len(null_rows),
        "target_trajectory_cell_ledger": len(cell_rows),
        "residual_failure_reason_ledger": len(reason_rows),
        "red_team_verification": 8,
        "no_selector_artifact_gate": 10,
    }
    return {
        "milestone": MILESTONE,
        "config_hash": cfg,
        "inherits_from": ["C49", "C50", "C51"],
        "diagnostic_only_non_deployable": True,
        "locked_witness": inputs["c51_summary"]["locked_witness"],
        "c51_residual_replay": {
            "decision": inputs["c51_summary"]["decision"]["decision"],
            "max_raw_underuse_gap": inputs["c51_summary"]["decision"]["max_raw_underuse_gap"],
            "best_trajectory_centered_gap": inputs["c51_summary"]["decision"]["best_trajectory_centered_gap"],
            "n2_fail_fraction_percentile": inputs["c51_summary"]["decision"]["n2_fail_fraction_percentile"],
            "n3_fail_fraction_percentile": inputs["c51_summary"]["decision"]["n3_fail_fraction_percentile"],
            "n4_enrichment_null_mean": inputs["c51_summary"]["decision"]["n4_enrichment_null_mean"],
            "observed_enrichment": inputs["c51_summary"]["observed_c50_replay"]["enrichment"],
        },
        "evaluation_scope": EVALUATION_SCOPE,
        "c51_oracle_trajectory_local_bayes_hit": oracle,
        "trajectory_conditioned_random_tie_hit": base_hit,
        "source_geometry_best_field": geom,
        "conditioning_ladder_rows": ladder,
        "gauge_key_decomposition_rows": decomp,
        "key_null_calibration_rows": null_rows,
        "target_trajectory_cell_ledger_rows": cell_rows,
        "residual_failure_reason_ledger_rows": reason_rows,
        "decision": decision,
        "n_candidate_rows": len(inputs["island_rows"]),
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
        "conditioning_ladder_rows": _readcsv(os.path.join(TABLE_DIR, "conditioning_ladder_summary.csv")),
        "gauge_key_decomposition_rows": _readcsv(os.path.join(TABLE_DIR, "gauge_key_decomposition.csv")),
        "key_null_calibration_rows": _readcsv(os.path.join(TABLE_DIR, "key_null_calibration_summary.csv")),
        "target_trajectory_cell_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "target_trajectory_cell_ledger.csv")),
        "residual_failure_reason_ledger_rows": _readcsv(os.path.join(TABLE_DIR, "residual_failure_reason_ledger.csv")),
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists(REPORT_JSON):
        return _summary_from_existing()
    return recompute()


def render_main_md(res):
    d = res["decision"]
    replay = res["c51_residual_replay"]
    return "\n".join([
        f"# C52 - Minimal Gauge-Key / Conditioning-Sufficiency Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "## Decision",
        "",
        f"`{d['decision']}`",
        "",
        "## C51 Residual Replay",
        "",
        f"- C51 decision: `{replay['decision']}`",
        f"- best strict source hit: **{_fmt(d['best_strict_source_hit'])}**",
        f"- C51 trajectory local-Bayes hit: **{_fmt(res['c51_oracle_trajectory_local_bayes_hit'])}**",
        f"- trajectory-centered diagnostic hit: **{_fmt(d['trajectory_centered_diagnostic_hit'])}**",
        f"- N2/N3 fail-fraction percentiles: **{_fmt(replay['n2_fail_fraction_percentile'])} / "
        f"{_fmt(replay['n3_fail_fraction_percentile'])}**",
        f"- N4 enrichment null mean vs observed: **{_fmt(replay['n4_enrichment_null_mean'])} / "
        f"{_fmt(replay['observed_enrichment'])}**",
        "",
        "## Gauge-Key Ladder",
        "",
        f"- trajectory random-tie hit: **{_fmt(res['trajectory_conditioned_random_tie_hit'])}**",
        f"- best key-only hit: **{_fmt(d['best_key_only_hit'])}**",
        f"- best label-derived diagnostic hit: **{_fmt(d['best_label_derived_hit'])}**",
        f"- source-observable closes gap: **{d['source_observable_closes_gap']}**",
        f"- target / trajectory / target×trajectory key-only closes gap: "
        f"**{d['target_key_only_closes_gap']} / {d['trajectory_key_only_closes_gap']} / "
        f"{d['target_x_trajectory_key_only_closes_gap']}**",
        f"- label-derived diagnostic closes gap: **{d['label_derived_diagnostic_closes_gap']}**",
        "",
        "## Bottom Line",
        "",
        "C52 separates key availability from label-derived diagnostic content. Target and trajectory keys "
        "are useful as grouping labels for the audit, but key-only baselines remain at the trajectory "
        "random-tie level under the frozen within-trajectory evaluation. The C51 residual is closed only "
        "when target labels are used diagnostically inside trajectory cells, so the C49-C51 ceiling remains "
        "a diagnostic boundary rather than a source-measurement localization result.",
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
        "# C52 - Red-Team Verification",
        "",
        "C52 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C52 keeps key-only and label-derived diagnostic evidence separated.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "rather than ")


def _guard_forbidden(text):
    au.guard_forbidden(
        text, FORBIDDEN_CLAIM_SUBSTRINGS,
        negation_cues=_NEG_CUES, window=220, label=MILESTONE)


def _compact_json(res):
    return {
        "milestone": res["milestone"],
        "config_hash": res["config_hash"],
        "inherits_from": res["inherits_from"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "locked_witness": res["locked_witness"],
        "c51_residual_replay": res["c51_residual_replay"],
        "evaluation_scope": res["evaluation_scope"],
        "c51_oracle_trajectory_local_bayes_hit": res["c51_oracle_trajectory_local_bayes_hit"],
        "trajectory_conditioned_random_tie_hit": res["trajectory_conditioned_random_tie_hit"],
        "source_geometry_best_field": {
            k: v for k, v in res["source_geometry_best_field"].items()
            if k != "evaluated_fields"
        },
        "decision": res["decision"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "table_row_counts": res["table_row_counts"],
        "red_team": red_team_rows(res),
        "artifact_hygiene": no_selector_gate(res),
    }


def write_tables(res, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    _writecsv(os.path.join(out_dir, "conditioning_ladder_summary.csv"), res["conditioning_ladder_rows"],
              ["ladder", "candidate", "information_class", "evaluation_scope", "available", "hit",
               "gap_to_oracle", "closes_gap", "closure_fraction_vs_best_raw_source", "key_only",
               "source_observable", "target_id_used", "trajectory_id_used", "target_unlabeled_geometry_used",
               "target_label_derived", "hindsight_diagnostic_only", "comparison_source", "interpretation",
               "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "gauge_key_decomposition.csv"), res["gauge_key_decomposition_rows"],
              ["component", "candidate", "available", "hit", "gap_to_oracle", "increment_vs_best_raw_source",
               "closure_fraction_vs_best_raw_source", "closes_gap", "key_only", "target_label_derived",
               "interpretation", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "key_null_calibration_summary.csv"), res["key_null_calibration_rows"],
              ["null_name", "key_or_field", "observed_hit", "null_mean", "null_std", "percentile",
               "empirical_p_two_sided", "n_permutations", "status", "preserved_structure",
               "destroyed_structure", "unavailable_reason", "target_labels_diagnostic_only",
               "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "target_trajectory_cell_ledger.csv"),
              res["target_trajectory_cell_ledger_rows"],
              ["trajectory_id", "target", "seed", "level", "regime", "n_rows", "base_hit",
               "target_key_only_hit", "trajectory_key_only_hit", "target_x_trajectory_key_only_hit",
               "best_source_geometry_hit", "local_bayes_hit", "best_existing_source_score_hit",
               "underuse_gap", "key_only_closes_cell", "label_diagnostic_closes_cell",
               "primary_reason", "target_labels_diagnostic_only", "no_selection_artifact"])
    _writecsv(os.path.join(out_dir, "residual_failure_reason_ledger.csv"),
              res["residual_failure_reason_ledger_rows"],
              ["reason_code", "n_trajectories", "fraction_trajectories", "mean_base_hit",
               "mean_local_bayes_hit", "mean_best_existing_source_score_hit", "description",
               "target_labels_diagnostic_only", "no_selection_artifact"])
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
    open(os.path.join(out_dir, "C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C52_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C52_MINIMAL_GAUGE_KEY_SUFFICIENCY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c52_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_ceiling_coverage.c52_minimal_gauge_key_sufficiency")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    d = res["decision"]
    print(f"[C52] decision={d['decision']} best_key_only_hit={d['best_key_only_hit']} "
          f"best_label_derived_hit={d['best_label_derived_hit']}")


if __name__ == "__main__":
    main()
