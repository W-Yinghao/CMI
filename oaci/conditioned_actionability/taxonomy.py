"""C47 deterministic taxonomy over conditioned actionability results."""
from __future__ import annotations

import math

from . import artifact_loader as al
from . import schema


def _q10(summary, scope):
    return summary["conditioning_neighbor_summary"][scope]["source_equivalent_q10_target_divergent_rate"]


def _best(best_rows, scope, label="primary_joint_good", top_k=1, kind="best_strict_source"):
    rs = [
        r for r in best_rows
        if r["group_scope"] == scope and r["label"] == label and int(r["top_k"]) == int(top_k) and
        r["best_kind"] == kind
    ]
    return rs[0] if rs else {}


def _f(row, key, default=math.nan):
    return float(row[key]) if row and al.finite(row.get(key)) else default


def _reliable(row):
    return (
        _f(row, "mean_any_hit") >= schema.RELIABLE_TOP1_HIT_GATE and
        _f(row, "mean_any_hit_enrichment") >= schema.RELIABLE_ENRICHMENT_GATE and
        _f(row, "mean_relative_regret_reduction_vs_random", 0.0) >= schema.REGRET_REDUCTION_GATE
    )


def _scope_best_gain(best_rows, scope, kind="best_strict_source"):
    r = _best(best_rows, scope, kind=kind)
    return _f(r, "mean_any_hit_gain_vs_random"), r


def _max_smoothing_gain(rows):
    vals = [
        float(r["smoothing_gain_delta"])
        for r in rows
        if r["label"] == "primary_joint_good" and int(r["top_k"]) == 1 and
        int(r["hindsight_diagnostic_only"]) == 0 and al.finite(r.get("smoothing_gain_delta"))
    ]
    return max(vals) if vals else math.nan


def _max_sign_auc(rows, scope):
    vals = [
        float(r["pairwise_auc_vs_target_utility"])
        for r in rows
        if r["group_scope"] == scope and int(r["source_only"]) == 1 and
        int(r["hindsight_diagnostic_only"]) == 0 and al.finite(r.get("pairwise_auc_vs_target_utility"))
    ]
    return max(vals) if vals else math.nan


def classify(c46_summary, action, smoothing, sign):
    best = action["best_rows"]
    wt_q10 = _q10(c46_summary, "within_target")
    wtraj_q10 = _q10(c46_summary, "within_trajectory")
    cross_q10 = _q10(c46_summary, "cross_target")
    global_gain, global_row = _scope_best_gain(best, "global")
    target_gain, target_row = _scope_best_gain(best, "within_target")
    traj_gain, traj_row = _scope_best_gain(best, "within_trajectory")
    target_seed_gain, target_seed_row = _scope_best_gain(best, "within_target_seed")
    target_level_gain, target_level_row = _scope_best_gain(best, "within_target_level")
    regime_gain, regime_row = _scope_best_gain(best, "within_regime")
    finite_conditioned = [
        v for v in (target_gain, traj_gain, target_seed_gain, target_level_gain)
        if al.finite(v)
    ]
    conditioned_gain = max(finite_conditioned) if finite_conditioned else math.nan
    conditioned_row = max(
        [r for r in (target_row, traj_row, target_seed_row, target_level_row) if r],
        key=lambda r: _f(r, "mean_any_hit_gain_vs_random"),
    )
    smoothing_gain = _max_smoothing_gain(smoothing["summary_rows"])
    global_auc = _max_sign_auc(sign["rows"], "global")
    target_auc = _max_sign_auc(sign["rows"], "within_target")
    traj_auc = _max_sign_auc(sign["rows"], "within_trajectory")
    any_conditioned_reliable = any(_reliable(r) for r in (target_row, traj_row, target_seed_row, target_level_row))
    target_reliable = _reliable(target_row)
    traj_advantage = traj_gain - target_gain if al.finite(traj_gain) and al.finite(target_gain) else math.nan
    non_traj_conditioned = [
        v for v in (target_gain, target_seed_gain, target_level_gain)
        if al.finite(v)
    ]
    best_non_traj_conditioned = max(non_traj_conditioned) if non_traj_conditioned else math.nan
    traj_unique_advantage = (
        traj_gain - best_non_traj_conditioned
        if al.finite(traj_gain) and al.finite(best_non_traj_conditioned) else math.nan
    )
    target_sufficient = target_reliable
    global_fails = (
        _f(global_row, "mean_any_hit") < schema.RELIABLE_TOP1_HIT_GATE or
        _f(global_row, "mean_any_hit_gain_vs_random") <= schema.GLOBAL_FAILURE_GAIN_GATE
    )
    conditioning_improves = (
        conditioned_gain - global_gain >= schema.IMPROVEMENT_GATE if
        al.finite(conditioned_gain) and al.finite(global_gain) else False
    )
    base_limited = (
        not any_conditioned_reliable or
        _f(conditioned_row, "mean_any_hit") < schema.BASE_RATE_LIMITED_TOP1_GATE or
        _f(conditioned_row, "mean_any_hit_enrichment") < schema.RELIABLE_ENRICHMENT_GATE
    )
    established = {
        schema.GCA1: (
            wt_q10 <= 0.05 and wtraj_q10 <= 0.20 and cross_q10 >= 0.75
        ),
        schema.GCA2: (
            conditioning_improves and not any_conditioned_reliable
        ),
        schema.GCA3: (
            al.finite(traj_unique_advantage) and traj_unique_advantage >= schema.IMPROVEMENT_GATE and
            not target_sufficient
        ),
        schema.GCA4: target_sufficient,
        schema.GCA5: base_limited,
        schema.GCA6: global_fails,
        schema.GCA7: (
            conditioning_improves or target_reliable or al.finite(smoothing_gain)
        ),
        schema.GCA8: False,
    }
    evidence = {
        schema.GCA1: (
            f"within_target_q10={wt_q10}, within_trajectory_q10={wtraj_q10}, cross_target_q10={cross_q10}"
        ),
        schema.GCA2: (
            f"best_conditioned_gain={conditioned_gain}, global_gain={global_gain}, "
            f"conditioned_top1_hit={_f(conditioned_row, 'mean_any_hit')}, "
            f"conditioned_enrichment={_f(conditioned_row, 'mean_any_hit_enrichment')}"
        ),
        schema.GCA3: (
            f"within_trajectory_gain={traj_gain}, within_target_gain={target_gain}, "
            f"within_target_seed_gain={target_seed_gain}, within_target_level_gain={target_level_gain}, "
            f"trajectory_advantage_vs_target={traj_advantage}, "
            f"trajectory_unique_advantage={traj_unique_advantage}"
        ),
        schema.GCA4: (
            f"within_target_top1_hit={_f(target_row, 'mean_any_hit')}, "
            f"within_target_enrichment={_f(target_row, 'mean_any_hit_enrichment')}, "
            f"within_target_regret_reduction={_f(target_row, 'mean_relative_regret_reduction_vs_random')}"
        ),
        schema.GCA5: (
            f"conditioned_top1_hit={_f(conditioned_row, 'mean_any_hit')}, "
            f"conditioned_enrichment={_f(conditioned_row, 'mean_any_hit_enrichment')}, "
            f"any_conditioned_reliable={any_conditioned_reliable}"
        ),
        schema.GCA6: (
            f"global_top1_hit={_f(global_row, 'mean_any_hit')}, global_gain={global_gain}, "
            f"global_sign_auc={global_auc}"
        ),
        schema.GCA7: (
            f"target_grouping_gain={target_gain}, trajectory_grouping_gain={traj_gain}, "
            f"max_smoothing_gain_delta={smoothing_gain}, "
            f"target_sign_auc={target_auc}, trajectory_sign_auc={traj_auc}"
        ),
        schema.GCA8: "required committed artifacts available",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
        "primary_metrics": {
            "global_strict_source_top1_gain": global_gain,
            "within_target_strict_source_top1_gain": target_gain,
            "within_trajectory_strict_source_top1_gain": traj_gain,
            "within_target_seed_strict_source_top1_gain": target_seed_gain,
            "within_target_level_strict_source_top1_gain": target_level_gain,
            "within_regime_strict_source_top1_gain": regime_gain,
            "best_conditioned_strict_source_top1_gain": conditioned_gain,
            "best_conditioned_strict_source_top1_hit": _f(conditioned_row, "mean_any_hit"),
            "best_conditioned_strict_source_top1_enrichment": _f(conditioned_row, "mean_any_hit_enrichment"),
            "max_primary_top1_smoothing_gain_delta": smoothing_gain,
            "global_max_strict_source_pairwise_auc": global_auc,
            "within_target_max_strict_source_pairwise_auc": target_auc,
            "within_trajectory_max_strict_source_pairwise_auc": traj_auc,
        },
    }
