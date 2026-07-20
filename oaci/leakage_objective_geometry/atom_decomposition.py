"""Leakage atom/support decomposition boundary checks."""
from __future__ import annotations

from collections import defaultdict
from statistics import mean


ATOM_FAMILIES = (
    "probe_fold",
    "class_conditioned",
    "domain_group",
    "source_domain",
    "support_cell",
    "rare_high_variance_cell",
)


def atom_rows(ctx, ucl_rows):
    rows = []
    for fam in ATOM_FAMILIES:
        rows.append({
            "atom_family": fam,
            "atom_key": "not_persisted",
            "n_pairs": len(ucl_rows),
            "atom_available": 0,
            "selected_advantage_sum": "",
            "selected_advantage_fraction": "",
            "concentration_rank": "",
            "concentration_share": "",
            "interpretation": "atom-level leakage contribution unavailable; no class/domain/support atom claim",
        })
    return {
        "rows": rows,
        "summary": {
            "atom_decomposition_available": False,
            "n_atom_families_requested": len(ATOM_FAMILIES),
            "n_atom_families_available": 0,
            "cell_concentration_claim_supported": False,
            "broad_cell_claim_supported": False,
        },
    }


def support_audit(ctx, ucl_rows):
    by_regime = defaultdict(list)
    by_pair_key = defaultdict(list)
    for r in ucl_rows:
        by_regime[r["regime"]].append(r)
        by_pair_key[r["pair_key"]].append(r)
    rows = []
    for regime, rs in sorted(by_regime.items()):
        rows.append({
            "scope": "regime",
            "key": regime,
            "n_pairs": len(rs),
            "ucl_prefers_selected_count": sum(1 for r in rs if r["ucl_prefers"] == "selected"),
            "mean_ucl_delta_better_minus_selected": mean(
                [r["ucl_delta_better_minus_selected"] for r in rs]),
            "mean_point_delta_better_minus_selected": mean(
                [r["point_delta_better_minus_selected"] for r in rs]),
            "support_edge_driver": 0,
            "note": "regime-level exact UCL preference is selected for every row",
        })
    invariant = 0
    for key, rs in sorted(by_pair_key.items()):
        deltas = {round(float(r["ucl_delta_better_minus_selected"]), 15) for r in rs}
        prefs = {r["ucl_prefers"] for r in rs}
        inv = int(len(deltas) == 1 and prefs == {"selected"} and len(rs) == 3)
        invariant += inv
        rows.append({
            "scope": "pair_key_across_regimes",
            "key": key,
            "n_pairs": len(rs),
            "ucl_prefers_selected_count": sum(1 for r in rs if r["ucl_prefers"] == "selected"),
            "mean_ucl_delta_better_minus_selected": mean(
                [r["ucl_delta_better_minus_selected"] for r in rs]),
            "mean_point_delta_better_minus_selected": mean(
                [r["point_delta_better_minus_selected"] for r in rs]),
            "support_edge_driver": 0 if inv else "",
            "note": "same selected/better pair repeated across S0/S2/S3 regimes" if inv else
                    "regime invariance not established",
        })
    summary = {
        "n_regimes": len(by_regime),
        "n_pair_keys": len(by_pair_key),
        "regime_counts": {k: len(v) for k, v in sorted(by_regime.items())},
        "regime_invariant_pair_keys": invariant,
        "support_or_estimability_artifact_supported": False,
    }
    return {"rows": rows, "summary": summary}

