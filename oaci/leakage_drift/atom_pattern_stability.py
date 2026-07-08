"""Blocked diagnostic atom-pattern stability under observed drift."""
from __future__ import annotations

from . import artifact_loader as al
from . import schema


def evaluate(ctx, manifest_rows):
    by_job = {r["job_key"]: r for r in manifest_rows if r["split"] == "selection"}
    rows = []
    for pair in ctx["tables"]["c37"]["exact"]:
        selected_key, better_key = al.pair_candidate_keys(pair)
        s = by_job[selected_key]
        b = by_job[better_key]
        persisted_delta = al.as_float(pair["point_delta_better_minus_selected"])
        recomputed_delta = al.as_float(b["recomputed_point"]) - al.as_float(s["recomputed_point"])
        delta_drift = recomputed_delta - persisted_delta
        conc = ctx["by_pair"]["c39_concentration"][pair["pair_id"]]
        support = ctx["by_pair"]["c39_support"][pair["pair_id"]]
        gauge = ctx["by_pair"]["c39_gauge"][pair["pair_id"]]
        stable_by_tol = {}
        for tol in schema.TOLERANCE_LADDER:
            stable_by_tol[str(tol)] = int(
                persisted_delta > schema.POINT_SIGN_EPS and
                persisted_delta - abs(delta_drift) - 2.0 * tol > schema.POINT_SIGN_EPS)
        rows.append({
            "pair_id": pair["pair_id"],
            "pair_key": pair["pair_key"],
            "seed": pair["seed"],
            "target": pair["target"],
            "level": pair["level"],
            "regime": pair["regime"],
            "selected_order": pair["selected_order"],
            "better_order": pair["better_order"],
            "persisted_point_delta_better_minus_selected": persisted_delta,
            "recomputed_point_delta_better_minus_selected": recomputed_delta,
            "delta_drift": delta_drift,
            "point_sign_stable_under_observed_drift": int(al.pref_from_delta(persisted_delta) ==
                                                           al.pref_from_delta(recomputed_delta) == "selected"),
            "stable_at_1e_9": stable_by_tol["1e-09"],
            "stable_at_1e_8": stable_by_tol["1e-08"],
            "stable_at_1e_6": stable_by_tol["1e-06"],
            "stable_at_1e_4": stable_by_tol["0.0001"],
            "stable_at_1e_3": stable_by_tol["0.001"],
            "diagnostic_concentration_class": conc["concentration_class"],
            "diagnostic_top3_positive_share": conc["top3_positive_share"],
            "diagnostic_support_artifact_flag": support["support_artifact_flag"],
            "diagnostic_atom_target_gauge_conflict": gauge["atom_target_gauge_conflict"],
            "pattern_claim_elevated": 0,
        })
    summary = {
        "n_pairs": len(rows),
        "point_sign_stable_fraction": al.finite_mean([r["point_sign_stable_under_observed_drift"] for r in rows]),
        "stable_at_1e_9_fraction": al.finite_mean([r["stable_at_1e_9"] for r in rows]),
        "stable_at_1e_4_fraction": al.finite_mean([r["stable_at_1e_4"] for r in rows]),
        "stable_at_1e_3_fraction": al.finite_mean([r["stable_at_1e_3"] for r in rows]),
        "broad_diagnostic_count": sum(1 for r in rows if r["diagnostic_concentration_class"] == "broad"),
        "support_artifact_diagnostic_count": sum(int(r["diagnostic_support_artifact_flag"]) for r in rows),
        "atom_gauge_conflict_diagnostic_count": sum(int(r["diagnostic_atom_target_gauge_conflict"]) for r in rows),
        "pattern_claims_elevated": False,
    }
    return {"rows": rows, "summary": summary}

