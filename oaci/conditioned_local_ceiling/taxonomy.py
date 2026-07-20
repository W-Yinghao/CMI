"""C48 deterministic local-ceiling taxonomy."""
from __future__ import annotations

import math

from . import artifact_loader as al
from . import schema


CONDITIONED_SCOPES = ("within_target", "within_trajectory", "within_target_seed", "within_target_level")
MIXED_SCOPES = ("global", "within_regime")


def _rows(best_rows, scope=None, label="primary_joint_good"):
    return [
        r for r in best_rows
        if (scope is None or r["group_scope"] == scope) and r["label"] == label
    ]


def _best(rows):
    if not rows:
        return {}
    return max(
        rows,
        key=lambda r: (
            float(r["mean_local_bayes_top1_hit"]) if al.finite(r.get("mean_local_bayes_top1_hit")) else -1.0,
            float(r["mean_local_bayes_enrichment"]) if al.finite(r.get("mean_local_bayes_enrichment")) else -1.0,
            float(r["mean_gap_vs_c47_actual_top1"]) if al.finite(r.get("mean_gap_vs_c47_actual_top1")) else -1.0,
        ),
    )


def _f(row, key, default=math.nan):
    return float(row[key]) if row and al.finite(row.get(key)) else default


def _reliable(row):
    return (
        _f(row, "mean_local_bayes_top1_hit") >= schema.RELIABLE_TOP1_HIT_GATE and
        _f(row, "mean_local_bayes_enrichment") >= schema.RELIABLE_ENRICHMENT_GATE and
        _f(row, "mean_local_bayes_gap_vs_permutation") >= schema.PERMUTATION_GAP_GATE
    )


def classify(local, c47_summary):
    best_rows = local["best_rows"]
    best_conditioned = _best([r for s in CONDITIONED_SCOPES for r in _rows(best_rows, s)])
    best_global = _best(_rows(best_rows, "global"))
    best_regime = _best(_rows(best_rows, "within_regime"))
    best_mixed = _best([r for s in MIXED_SCOPES for r in _rows(best_rows, s)])
    best_any = _best(_rows(best_rows))
    c47_hit = c47_summary["taxonomy"]["primary_metrics"]["best_conditioned_strict_source_top1_hit"]
    c47_enrich = c47_summary["taxonomy"]["primary_metrics"]["best_conditioned_strict_source_top1_enrichment"]
    conditioned_high = _reliable(best_conditioned)
    conditioned_low = not conditioned_high
    gap_vs_c47 = _f(best_conditioned, "mean_gap_vs_c47_actual_top1")
    hit_gap_mixed = _f(best_conditioned, "mean_local_bayes_top1_hit") - _f(best_mixed, "mean_local_bayes_top1_hit")
    gain_gap_mixed = _f(best_conditioned, "mean_local_bayes_gain_vs_random") - _f(
        best_mixed, "mean_local_bayes_gain_vs_random")
    perm_gap = _f(best_conditioned, "mean_local_bayes_gap_vs_permutation")
    mixed_reliable = _reliable(best_mixed)
    base_rate_explains = (
        _f(best_conditioned, "mean_local_bayes_top1_hit") >=
        c47_hit + schema.MEANINGFUL_CEILING_GAP and
        (
            _f(best_conditioned, "mean_local_bayes_enrichment") < schema.RELIABLE_ENRICHMENT_GATE or
            _f(best_conditioned, "mean_local_bayes_gain_vs_random") <= schema.BASE_RATE_GAIN_GATE or
            perm_gap < schema.PERMUTATION_GAP_GATE
        )
    )
    established = {
        schema.LC1: conditioned_high,
        schema.LC2: conditioned_low,
        schema.LC3: gap_vs_c47 >= schema.MEANINGFUL_CEILING_GAP,
        schema.LC4: (
            hit_gap_mixed >= schema.STABILITY_GAP_GATE or gain_gap_mixed >= schema.MEANINGFUL_GAIN_GAP
        ) and not mixed_reliable,
        schema.LC5: base_rate_explains,
        schema.LC6: (
            conditioned_low and
            c47_hit < schema.RELIABLE_TOP1_HIT_GATE and
            c47_enrich < schema.RELIABLE_ENRICHMENT_GATE
        ),
        schema.LC7: False,
    }
    evidence = {
        schema.LC1: (
            f"best_conditioned_scope={best_conditioned.get('group_scope')}, "
            f"hit={_f(best_conditioned, 'mean_local_bayes_top1_hit')}, "
            f"enrichment={_f(best_conditioned, 'mean_local_bayes_enrichment')}, "
            f"permutation_gap={perm_gap}"
        ),
        schema.LC2: (
            f"best_conditioned_scope={best_conditioned.get('group_scope')}, "
            f"hit={_f(best_conditioned, 'mean_local_bayes_top1_hit')}, "
            f"enrichment={_f(best_conditioned, 'mean_local_bayes_enrichment')}, "
            f"permutation_gap={perm_gap}, "
            f"gates=({schema.RELIABLE_TOP1_HIT_GATE},{schema.RELIABLE_ENRICHMENT_GATE})"
        ),
        schema.LC3: (
            f"best_conditioned_gap_vs_c47={gap_vs_c47}, "
            f"c47_hit={c47_hit}, "
            f"local_hit={_f(best_conditioned, 'mean_local_bayes_top1_hit')}"
        ),
        schema.LC4: (
            f"best_conditioned_scope={best_conditioned.get('group_scope')}, "
            f"best_mixed_scope={best_mixed.get('group_scope')}, "
            f"hit_gap_mixed={hit_gap_mixed}, gain_gap_mixed={gain_gap_mixed}, "
            f"mixed_reliable={mixed_reliable}"
        ),
        schema.LC5: (
            f"conditioned_hit={_f(best_conditioned, 'mean_local_bayes_top1_hit')}, "
            f"conditioned_base={_f(best_conditioned, 'mean_random_top1_baseline')}, "
            f"conditioned_gain={_f(best_conditioned, 'mean_local_bayes_gain_vs_random')}, "
            f"conditioned_enrichment={_f(best_conditioned, 'mean_local_bayes_enrichment')}, "
            f"permutation_gap={perm_gap}"
        ),
        schema.LC6: (
            f"local_hit={_f(best_conditioned, 'mean_local_bayes_top1_hit')}, "
            f"local_enrichment={_f(best_conditioned, 'mean_local_bayes_enrichment')}, "
            f"c47_hit={c47_hit}, c47_enrichment={c47_enrich}"
        ),
        schema.LC7: "required C47/C45 artifacts available",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
        "primary_metrics": {
            "best_any_scope": best_any.get("group_scope"),
            "best_any_source_space": best_any.get("source_space"),
            "best_any_neighborhood": best_any.get("neighborhood"),
            "best_any_top1_hit": _f(best_any, "mean_local_bayes_top1_hit"),
            "best_any_enrichment": _f(best_any, "mean_local_bayes_enrichment"),
            "best_conditioned_scope": best_conditioned.get("group_scope"),
            "best_conditioned_source_space": best_conditioned.get("source_space"),
            "best_conditioned_neighborhood": best_conditioned.get("neighborhood"),
            "best_conditioned_top1_hit": _f(best_conditioned, "mean_local_bayes_top1_hit"),
            "best_conditioned_enrichment": _f(best_conditioned, "mean_local_bayes_enrichment"),
            "best_conditioned_gain_vs_random": _f(best_conditioned, "mean_local_bayes_gain_vs_random"),
            "best_conditioned_permutation_top1_hit": _f(
                best_conditioned, "mean_permutation_local_bayes_top1_hit"),
            "best_conditioned_gap_vs_permutation": perm_gap,
            "best_conditioned_gap_vs_c47": gap_vs_c47,
            "best_global_top1_hit": _f(best_global, "mean_local_bayes_top1_hit"),
            "best_global_enrichment": _f(best_global, "mean_local_bayes_enrichment"),
            "best_global_gap_vs_permutation": _f(best_global, "mean_local_bayes_gap_vs_permutation"),
            "best_within_regime_top1_hit": _f(best_regime, "mean_local_bayes_top1_hit"),
            "best_within_regime_enrichment": _f(best_regime, "mean_local_bayes_enrichment"),
            "best_within_regime_gap_vs_permutation": _f(best_regime, "mean_local_bayes_gap_vs_permutation"),
            "hit_gap_conditioned_vs_mixed": hit_gap_mixed,
            "gain_gap_conditioned_vs_mixed": gain_gap_mixed,
            "c47_best_conditioned_strict_source_top1_hit": c47_hit,
            "c47_best_conditioned_strict_source_top1_enrichment": c47_enrich,
        },
    }
