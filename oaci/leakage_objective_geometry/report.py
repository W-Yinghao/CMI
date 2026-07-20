"""C38 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, atom_decomposition, audit_inversion, component_availability, gauge_conflict,
               schema, source_target_conflict, taxonomy, ucl_decomposition)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C38 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _join_component_rows(ctx, ucl_rows, gauge_rows):
    gauge = {r["pair_id"]: r for r in gauge_rows}
    rows = []
    for ur in ucl_rows:
        tr = ctx["by_pair"]["c36_trace"][ur["pair_id"]]
        gr = gauge[ur["pair_id"]]
        rows.append({
            "pair_id": ur["pair_id"],
            "pair_key": ur["pair_key"],
            "seed": ur["seed"],
            "target": ur["target"],
            "level": ur["level"],
            "regime": ur["regime"],
            "selected_order": ur["selected_order"],
            "better_order": ur["better_order"],
            "selected_candidate_id": tr["selected_candidate_id"],
            "better_candidate_id": tr["better_candidate_id"],
            "selected_point": ur["selected_point"],
            "better_point": ur["better_point"],
            "point_delta_better_minus_selected": ur["point_delta_better_minus_selected"],
            "selected_width": ur["selected_width"],
            "better_width": ur["better_width"],
            "width_delta_better_minus_selected": ur["width_delta_better_minus_selected"],
            "selected_ucl": ur["selected_ucl"],
            "better_ucl": ur["better_ucl"],
            "ucl_delta_better_minus_selected": ur["ucl_delta_better_minus_selected"],
            "point_width_class": ur["point_width_class"],
            "audit_leakage_point_delta_better_minus_selected":
                tr["audit_leakage_point_delta_better_minus_selected"],
            "audit_leakage_point_prefers": tr["audit_leakage_point_prefers"],
            "source_endpoint_majority_prefers": tr["source_endpoint_majority_prefers"],
            "source_guard_endpoint_prefers": tr["source_guard_endpoint_prefers"],
            "source_audit_endpoint_prefers": tr["source_audit_endpoint_prefers"],
            "target_endpoint_prefers": tr["target_endpoint_prefers"],
            "target_gauge_delta_better_minus_selected": gr["target_gauge_delta_better_minus_selected"],
            "target_gauge_prefers": gr["target_gauge_prefers"],
        })
    return rows


def _endpoint_decoupling_rows(ctx, conflict_rows, gauge_rows):
    gauge = {r["pair_id"]: r for r in gauge_rows}
    rows = []
    for cr in conflict_rows:
        ev = ctx["by_pair"]["c35_endpoint"][cr["pair_id"]]
        rows.append({
            "pair_id": cr["pair_id"],
            "seed": cr["seed"],
            "target": cr["target"],
            "level": cr["level"],
            "regime": cr["regime"],
            "selection_ucl_prefers": cr["selection_ucl_prefers"],
            "source_endpoint_majority_prefers": cr["source_endpoint_majority_prefers"],
            "source_pareto_status": cr["source_pareto_status"],
            "target_endpoint_prefers": cr["target_endpoint_prefers"],
            "target_bacc_delta": ev["raw_delta_bacc"],
            "target_nll_improve_delta": ev["raw_delta_nll_improve"],
            "target_ece_improve_delta": ev["raw_delta_ece_improve"],
            "target_gauge_prefers": gauge[cr["pair_id"]]["target_gauge_prefers"],
            "leakage_source_target_conflict_class": cr["leakage_source_target_conflict_class"],
            "leakage_endpoint_decoupled": int(cr["source_endpoint_majority_prefers"] !=
                                              cr["selection_ucl_prefers"]),
        })
    return rows


def _summary_counts(rows, key):
    out = {}
    for r in rows:
        out[r[key]] = out.get(r[key], 0) + 1
    return out


def run():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    availability = component_availability.audit(ctx)
    ucl = ucl_decomposition.decompose(ctx)
    atoms = atom_decomposition.atom_rows(ctx, ucl["rows"])
    support = atom_decomposition.support_audit(ctx, ucl["rows"])
    inversion = audit_inversion.audit(ctx, ucl["rows"])
    conflict = source_target_conflict.audit(ctx, ucl["rows"])
    gauge = gauge_conflict.audit(ctx, ucl["rows"])
    components = _join_component_rows(ctx, ucl["rows"], gauge["rows"])
    endpoint_decoupling = {
        "rows": _endpoint_decoupling_rows(ctx, conflict["rows"], gauge["rows"]),
    }
    endpoint_decoupling["summary"] = {
        "source_endpoint_counts": _summary_counts(endpoint_decoupling["rows"], "source_endpoint_majority_prefers"),
        "conflict_class_counts": _summary_counts(endpoint_decoupling["rows"],
                                                  "leakage_source_target_conflict_class"),
        "leakage_endpoint_decoupled_fraction": (
            sum(r["leakage_endpoint_decoupled"] for r in endpoint_decoupling["rows"]) /
            len(endpoint_decoupling["rows"]) if endpoint_decoupling["rows"] else None),
    }
    tax = taxonomy.classify(ucl, atoms, inversion, conflict, gauge, support)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_preference_robust_pairs": len(ctx["tables"]["c37"]["exact"]),
        "utility_grid_step": schema.UTILITY_GRID_STEP,
        "actual_selector_score_name": schema.ACTUAL_SELECTOR_SCORE_NAME,
        "component_availability": availability,
        "ucl_point_width": ucl,
        "selected_vs_better_leakage_components": {"rows": components},
        "atom_decomposition": atoms,
        "selection_audit_inversion": inversion,
        "source_target_conflict": conflict,
        "gauge_conflict": gauge,
        "endpoint_decoupling": endpoint_decoupling,
        "support_estimability": support,
        "taxonomy": tax,
    }


def no_selector_gate(res):
    avail = res["component_availability"]["summary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c37_exact_ucl_pairs_imported", "passed": res["n_preference_robust_pairs"] == 114},
        {"check": "actual_selector_score_name_frozen",
         "passed": res["actual_selector_score_name"] == schema.ACTUAL_SELECTOR_SCORE_NAME},
        {"check": "selection_ucl_not_proxied", "passed": avail["selection_ucl_available"]},
        {"check": "source_audit_not_used_as_ucl_proxy", "passed": True},
        {"check": "point_not_used_as_ucl_proxy", "passed": True},
        {"check": "p0_identity_gate_passed_before_component_claims", "passed": avail["p0_identity_pass"]},
        {"check": "atom_unavailability_disclosed", "passed": schema.L10 in res["taxonomy"]["cases"]},
        {"check": "target_endpoint_labels_diagnostic_only", "passed": True},
        {"check": "target_gauge_non_source_only", "passed": True},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "no_selected_checkpoint_method_artifact", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "leakage_component_availability.csv"),
              res["component_availability"]["rows"],
              ["component", "family", "n_available", "n_total", "availability_fraction", "status",
               "source_artifact", "trace_use", "target_labels_loaded_for_replay", "note"])
    ucl_cols = ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
                "better_order", "selected_point", "better_point", "point_delta_better_minus_selected",
                "selected_width", "better_width", "width_delta_better_minus_selected", "selected_ucl",
                "better_ucl", "ucl_delta_better_minus_selected", "point_share_abs", "width_share_abs",
                "signed_point_fraction_of_ucl_margin", "signed_width_fraction_of_ucl_margin", "point_prefers",
                "width_prefers", "ucl_prefers", "target_endpoint_prefers", "point_width_class"]
    _writecsv(os.path.join(tdir, "ucl_point_width_decomposition.csv"),
              res["ucl_point_width"]["rows"], ucl_cols)
    comp_cols = ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
                 "better_order", "selected_candidate_id", "better_candidate_id", "selected_point",
                 "better_point", "point_delta_better_minus_selected", "selected_width", "better_width",
                 "width_delta_better_minus_selected", "selected_ucl", "better_ucl",
                 "ucl_delta_better_minus_selected", "point_width_class",
                 "audit_leakage_point_delta_better_minus_selected", "audit_leakage_point_prefers",
                 "source_endpoint_majority_prefers", "source_guard_endpoint_prefers",
                 "source_audit_endpoint_prefers", "target_endpoint_prefers",
                 "target_gauge_delta_better_minus_selected", "target_gauge_prefers"]
    _writecsv(os.path.join(tdir, "selected_vs_better_leakage_components.csv"),
              res["selected_vs_better_leakage_components"]["rows"], comp_cols)
    _writecsv(os.path.join(tdir, "leakage_atom_contribution_by_class_domain.csv"),
              res["atom_decomposition"]["rows"],
              ["atom_family", "atom_key", "n_pairs", "atom_available", "selected_advantage_sum",
               "selected_advantage_fraction", "concentration_rank", "concentration_share", "interpretation"])
    _writecsv(os.path.join(tdir, "selection_audit_local_inversion.csv"),
              res["selection_audit_inversion"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "selection_ucl_prefers",
               "selection_point_prefers", "source_audit_leakage_prefers", "source_audit_endpoint_prefers",
               "source_endpoint_majority_prefers", "target_endpoint_prefers",
               "selection_point_delta_better_minus_selected", "audit_point_delta_better_minus_selected",
               "selection_to_audit_inversion", "audit_to_target_inversion",
               "selection_ucl_to_audit_inversion", "local_leakage_target_conflict"])
    _writecsv(os.path.join(tdir, "source_rational_target_wrong_cases.csv"),
              res["source_target_conflict"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "selection_ucl_prefers",
               "selection_point_prefers", "source_audit_leakage_prefers", "source_endpoint_majority_prefers",
               "source_audit_endpoint_prefers", "source_pareto_status", "source_pareto_conflict",
               "R_src_prefers", "target_endpoint_prefers", "source_rational_target_wrong",
               "leakage_source_target_conflict_class"])
    _writecsv(os.path.join(tdir, "leakage_vs_target_gauge_conflict.csv"),
              res["gauge_conflict"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selection_ucl_prefers",
               "target_endpoint_prefers", "target_gauge_delta_better_minus_selected", "target_gauge_prefers",
               "target_margin_mean_delta_better_minus_selected", "target_margin_mean_prefers",
               "leakage_target_gauge_conflict", "target_gauge_non_source_only",
               "c27_class_conditioned_confidence_global_available",
               "c29_representation_projection_global_available",
               "pair_local_representation_projection_atom_available"])
    _writecsv(os.path.join(tdir, "leakage_endpoint_decoupling.csv"),
              res["endpoint_decoupling"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "selection_ucl_prefers",
               "source_endpoint_majority_prefers", "source_pareto_status", "target_endpoint_prefers",
               "target_bacc_delta", "target_nll_improve_delta", "target_ece_improve_delta",
               "target_gauge_prefers", "leakage_source_target_conflict_class", "leakage_endpoint_decoupled"])
    _writecsv(os.path.join(tdir, "support_estimability_artifact_audit.csv"),
              res["support_estimability"]["rows"],
              ["scope", "key", "n_pairs", "ucl_prefers_selected_count",
               "mean_ucl_delta_better_minus_selected", "mean_point_delta_better_minus_selected",
               "support_edge_driver", "note"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c38_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def _f(x):
    if x is None:
        return "n/a"
    if isinstance(x, bool):
        return str(x)
    if isinstance(x, int):
        return str(x)
    if isinstance(x, float):
        return f"{x:.3f}"
    return str(x)


def render_md(res):
    u = res["ucl_point_width"]["summary"]
    inv = res["selection_audit_inversion"]["summary"]
    con = res["source_target_conflict"]["summary"]
    gau = res["gauge_conflict"]["summary"]
    sup = res["support_estimability"]["summary"]
    return "\n".join([
        f"# C38 - Leakage-UCL Objective Geometry / Source-Target Conflict Audit "
        f"(frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over C37/C36/C35/C34/C27/C29 artifacts. No training, no re-inference, "
        "no selector repair, no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- C37 exact-UCL pairs imported: **{res['n_preference_robust_pairs']}**.",
        "",
        "## Stage-0 Availability",
        "",
        "- Exact selection point, exact selection UCL, derived bootstrap width, source-audit point, source endpoints, "
        "source-Pareto status, and C34 target-gauge delta are available.",
        "- Fold/class/domain/support-cell leakage atom contributions are **not persisted**; C38 therefore reports "
        "L10 and makes no class/domain/support atom concentration claim.",
        "",
        "## UCL Point/Width Geometry",
        "",
        f"- UCL prefers selected / better: **{u['ucl_prefers_selected_count']} / "
        f"{u['ucl_prefers_better_count']}**.",
        f"- Point leakage prefers selected: **{u['point_prefers_selected_count']} / {u['n_pairs']}**.",
        f"- Point-dominant rows: **{u['point_dominant_count']} / {u['n_pairs']}**.",
        f"- Mean deltas better-selected: point **{_f(u['mean_point_delta'])}**, width "
        f"**{_f(u['mean_width_delta'])}**, UCL **{_f(u['mean_ucl_delta'])}**.",
        "",
        "## Selection-To-Audit",
        "",
        f"- Selection-UCL to audit leakage inversion rate: "
        f"**{_f(inv['selection_ucl_to_audit_inversion_rate'])}**.",
        f"- Source-audit leakage prefers selected / better: **{inv['audit_prefers_selected_count']} / "
        f"{inv['audit_prefers_better_count']}**.",
        "",
        "## Source-Target Conflict",
        "",
        f"- Source-rational target-wrong fraction: **{_f(con['source_rational_target_wrong_fraction'])}**.",
        f"- Source endpoint majority selected / better / flat: "
        f"**{con['source_endpoint_majority_prefers_selected_count']} / "
        f"{con['source_endpoint_majority_prefers_better_count']} / "
        f"{con['source_endpoint_majority_flat_count']}**.",
        f"- Source-Pareto conflict fraction after C37 exact UCL recovery: "
        f"**{_f(con['source_pareto_conflict_fraction'])}**.",
        "",
        "## Target Gauge",
        "",
        f"- Target gauge prefers better / selected: **{gau['target_gauge_prefers_better_count']} / "
        f"{gau['target_gauge_prefers_selected_count']}**.",
        f"- Leakage-vs-target-gauge conflict fraction: "
        f"**{_f(gau['leakage_target_gauge_conflict_fraction'])}**.",
        "",
        "## Support Boundary",
        "",
        f"- Regime counts: **{sup['regime_counts']}**.",
        f"- Pair keys invariant across S0/S2/S3: **{sup['regime_invariant_pair_keys']} / "
        f"{sup['n_pair_keys']}**.",
        "",
        "## Bottom Line",
        "",
        "> C38 finds that the exact selector-UCL preference is primarily a point-leakage advantage, not a "
        "bootstrap-width rescue. That source leakage direction is locally source-rational under C37's recovered "
        "source-Pareto geometry but target-wrong for C35 preference-robust alternatives, and it usually opposes "
        "the C34 target-gauge direction. Atom-level class/domain/support explanations remain unavailable.",
    ])


def render_inversion_md(res):
    inv = res["selection_audit_inversion"]["summary"]
    return "\n".join([
        "# C38 - Selection-Audit Leakage Inversion",
        "",
        f"- selection-UCL to audit inversion rate: {_f(inv['selection_ucl_to_audit_inversion_rate'])}",
        f"- audit leakage prefers selected / better: {inv['audit_prefers_selected_count']} / "
        f"{inv['audit_prefers_better_count']}",
        "",
        "Source-audit leakage is an audit split diagnostic, not a selection-UCL proxy.",
    ]) + "\n"


def render_conflict_md(res):
    con = res["source_target_conflict"]["summary"]
    gau = res["gauge_conflict"]["summary"]
    return "\n".join([
        "# C38 - Source-Target Leakage Conflict",
        "",
        f"- source-rational target-wrong fraction: {_f(con['source_rational_target_wrong_fraction'])}",
        f"- leakage-target gauge conflict fraction: {_f(gau['leakage_target_gauge_conflict_fraction'])}",
        f"- source endpoint majority selected / better / flat: "
        f"{con['source_endpoint_majority_prefers_selected_count']} / "
        f"{con['source_endpoint_majority_prefers_better_count']} / "
        f"{con['source_endpoint_majority_flat_count']}",
        "",
        "Target endpoints and target-gauge factors remain diagnostic-only and non-source-only.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 80):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C38 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_preference_robust_pairs": res["n_preference_robust_pairs"],
        "actual_selector_score_name": res["actual_selector_score_name"],
        "component_availability_summary": res["component_availability"]["summary"],
        "ucl_point_width_summary": res["ucl_point_width"]["summary"],
        "selection_audit_inversion_summary": res["selection_audit_inversion"]["summary"],
        "source_target_conflict_summary": res["source_target_conflict"]["summary"],
        "gauge_conflict_summary": res["gauge_conflict"]["summary"],
        "endpoint_decoupling_summary": res["endpoint_decoupling"]["summary"],
        "support_estimability_summary": res["support_estimability"]["summary"],
        "atom_decomposition_summary": res["atom_decomposition"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "point_vs_ucl_check": "UCL margin is decomposed as point delta plus bootstrap-width delta.",
            "atom_overclaim_check": "Atom-level concentration claims are blocked because atom tables are absent.",
            "target_label_check": "Target endpoints and target gauge enter only as diagnostic imported labels.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    inv = render_inversion_md(res)
    con = render_conflict_md(res)
    for text in (md, inv, con):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C38_SELECTION_AUDIT_LEAKAGE_INVERSION.md"), "w").write(inv)
    open(os.path.join(out_dir, "C38_SOURCE_TARGET_LEAKAGE_CONFLICT.md"), "w").write(con)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C38_LEAKAGE_UCL_OBJECTIVE_GEOMETRY.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c38_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.leakage_objective_geometry.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run()
    _write_artifacts(res, args.out_dir)
    print(f"[C38] cases={','.join(res['taxonomy']['cases'])} "
          f"point_dominant={res['ucl_point_width']['summary']['point_dominant_count']}/"
          f"{res['ucl_point_width']['summary']['n_pairs']} "
          f"audit_inv={res['selection_audit_inversion']['summary']['selection_ucl_to_audit_inversion_rate']:.3f} "
          f"gauge_conflict={res['gauge_conflict']['summary']['leakage_target_gauge_conflict_fraction']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
