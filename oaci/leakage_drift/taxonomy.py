"""C40 deterministic taxonomy."""
from __future__ import annotations

from . import schema


def classify(manifest, stagewise, numeric, stability, spec):
    ms = manifest["summary"]
    ss = stagewise["summary"]
    ns = numeric["summary"]
    ps = stability["summary"]
    fs = spec["summary"]
    exact = bool(ms["selection_identity_pass"])
    semantic_mismatch = ss["observed_semantic_mismatch_count"] > 0
    bounded = bool(ns["bounded_at_1e_3"])
    stable_blocked = (
        ps["point_sign_stable_fraction"] is not None and
        ps["point_sign_stable_fraction"] >= schema.STABILITY_FRACTION_GATE and
        not exact)
    established = {
        schema.D1: exact,
        schema.D2: (not exact and bounded and not semantic_mismatch),
        schema.D3: semantic_mismatch,
        schema.D4: ss["aggregate_vs_atom_path_divergence_count"] > 0,
        schema.D5: stable_blocked,
        schema.D6: (not exact and fs["future_instrumentation_required"]),
        schema.D7: fs["future_instrumentation_required"],
        schema.D8: exact,
    }
    evidence = {
        schema.D1: f"selection_identity_pass={ms['selection_identity_pass']}",
        schema.D2: (
            f"max_abs_drift={ns['max_abs_drift']}, bounded_at_1e_3={bounded}, "
            f"observed_semantic_mismatch_count={ss['observed_semantic_mismatch_count']}"),
        schema.D3: f"observed_semantic_mismatch_count={ss['observed_semantic_mismatch_count']}",
        schema.D4: f"first divergent stage counts={ss['selection_first_divergent_stage_counts']}",
        schema.D5: (
            f"point_sign_stable_fraction={ps['point_sign_stable_fraction']}; "
            "diagnostic atom patterns remain blocked by identity failure"),
        schema.D6: "per-fold probe/cell outputs and persisted atom traces are absent, so exact identity is not restored",
        schema.D7: f"missing_future_fields={fs['n_currently_missing']}",
        schema.D8: "requires D1 exact identity recovery without semantic changes",
    }
    rows = [{"case": c, "established": int(bool(established[c])), "evidence": evidence[c]}
            for c in schema.ALL_CASES]
    return {
        "cases": [c for c in schema.ALL_CASES if established[c]],
        "case_rows": rows,
        "established": established,
        "evidence": evidence,
    }
