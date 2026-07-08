"""C38 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def classify(ucl, atom, inversion, conflict, gauge, support):
    us = ucl["summary"]
    atoms = atom["summary"]
    inv = inversion["summary"]
    con = conflict["summary"]
    gau = gauge["summary"]
    sup = support["summary"]
    established = {
        schema.L1: (
            us["point_prefers_selected_count"] == us["n_pairs"] and
            us["point_dominant_fraction"] >= schema.POINT_DOMINANT_FRACTION_GATE),
        schema.L2: us["uncertainty_driven_count"] / us["n_pairs"] >= 0.5 if us["n_pairs"] else False,
        schema.L3: atoms["cell_concentration_claim_supported"],
        schema.L4: atoms["broad_cell_claim_supported"],
        schema.L5: inv["selection_ucl_to_audit_inversion_rate"] >= schema.INVERSION_RATE_GATE,
        schema.L6: con["source_rational_target_wrong_fraction"] == 1.0,
        schema.L7: gau["leakage_target_gauge_conflict_fraction"] >= schema.GAUGE_CONFLICT_RATE_GATE,
        schema.L8: (
            con["source_endpoint_majority_prefers_better_count"] > 0 and
            us["ucl_prefers_selected_count"] == us["n_pairs"]),
        schema.L9: sup["support_or_estimability_artifact_supported"],
        schema.L10: not atoms["atom_decomposition_available"],
    }
    evidence = {
        schema.L1: f"point_dominant_fraction={us['point_dominant_fraction']}",
        schema.L2: f"uncertainty_driven_count={us['uncertainty_driven_count']}",
        schema.L3: "atom-level class/domain/support contribution table unavailable",
        schema.L4: "broad cell-level claim blocked by unavailable leakage atoms",
        schema.L5: f"selection_ucl_to_audit_inversion_rate={inv['selection_ucl_to_audit_inversion_rate']}",
        schema.L6: f"source_rational_target_wrong_fraction={con['source_rational_target_wrong_fraction']}",
        schema.L7: f"leakage_target_gauge_conflict_fraction={gau['leakage_target_gauge_conflict_fraction']}",
        schema.L8: (
            "UCL prefers selected for all pairs while source endpoint majority is split "
            f"selected/better/flat={con['source_endpoint_majority_prefers_selected_count']}/"
            f"{con['source_endpoint_majority_prefers_better_count']}/"
            f"{con['source_endpoint_majority_flat_count']}"),
        schema.L9: f"regime_invariant_pair_keys={sup['regime_invariant_pair_keys']}",
        schema.L10: "exact UCL recovered but atom-level decomposition not persisted",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
    }

