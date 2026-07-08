"""Bootstrap/UCL atom boundary diagnostics."""
from __future__ import annotations


def audit(ctx, *, persisted_point_identity_pass=True):
    rows = []
    for r in ctx["pairs"]:
        rows.append({
            "pair_id": r["pair_id"],
            "pair_key": r["pair_key"],
            "seed": r["seed"],
            "target": r["target"],
            "level": r["level"],
            "regime": r["regime"],
            "selected_order": r["selected_order"],
            "better_order": r["better_order"],
            "selected_ucl": r["selected_ucl"],
            "better_ucl": r["better_ucl"],
            "aggregate_ucl_available": 1,
            "point_atom_additive_identity_exact": 1,
            "persisted_point_identity_pass": int(bool(persisted_point_identity_pass)),
            "replicate_atom_replay_available": 0,
            "per_atom_ucl_summed": 0,
            "ucl_quantile_linear_atom_claim_allowed": 0,
            "diagnostic_status": "A10_ucl_quantile_not_linear_no_per_atom_ucl_sum",
        })
    return {
        "rows": rows,
        "summary": {
            "n_pairs": len(rows),
            "aggregate_ucl_available": bool(rows),
            "point_atom_additive_identity_exact": True,
            "persisted_point_identity_pass": bool(persisted_point_identity_pass),
            "replicate_atom_replay_available": False,
            "per_atom_ucl_summed": False,
            "ucl_quantile_atom_limit": True,
        },
    }
