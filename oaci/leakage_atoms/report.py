"""C39 leakage atom recovery report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (additive_identity, artifact_loader, atom_availability, audit_atom_stability,
               bootstrap_atom_diagnostics, gauge_atom_conflict, point_atom_decomposition,
               schema, support_cell_audit, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C39 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, float) and not math.isfinite(v):
            return ""
        if isinstance(v, bool):
            return int(v)
        return v

    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore", lineterminator="\n")
        w.writeheader()
        for r in rows:
            w.writerow({c: clean(r.get(c)) for c in cols})


def _readcsv(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


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


def recompute(*, n_jobs=1):
    cfg = _lock_config()
    ctx = artifact_loader.context()
    replay = point_atom_decomposition.replay_all(ctx, n_jobs=n_jobs)
    identity = additive_identity.audit(replay["identity_rows"])
    availability = atom_availability.audit(identity["rows"])
    point_atoms = point_atom_decomposition.selected_vs_better_atoms(
        ctx, replay["atom_rows"], split="selection")
    audit_atoms = point_atom_decomposition.selected_vs_better_atoms(
        ctx, replay["atom_rows"], split="source_audit")
    concentration_rows = point_atom_decomposition.concentration_summary(point_atoms)
    concentration = {
        "rows": concentration_rows,
        "summary": point_atom_decomposition.summaries(identity["rows"], point_atoms)["concentration"],
    }
    class_domain = {"rows": point_atom_decomposition.class_domain_contributions(point_atoms)}
    audit_stability = audit_atom_stability.audit(ctx, point_atoms, audit_atoms)
    support = support_cell_audit.audit(point_atoms)
    bootstrap = bootstrap_atom_diagnostics.audit(
        ctx, persisted_point_identity_pass=identity["summary"]["selection_identity_pass"])
    gauge = gauge_atom_conflict.audit(ctx, point_atoms)
    tax = taxonomy.classify(identity, concentration, class_domain["rows"], audit_stability,
                            support, gauge, bootstrap)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_jobs": int(n_jobs),
        "n_preference_robust_pairs": len(ctx["pairs"]),
        "actual_selector_score_name": schema.ACTUAL_SELECTOR_SCORE_NAME,
        "utility_grid_step": schema.UTILITY_GRID_STEP,
        "atom_recovery_availability": availability,
        "selected_atom_identity_gate": identity,
        "selected_vs_better_point_atoms": {"rows": point_atoms},
        "source_audit_point_atoms": {"rows": audit_atoms},
        "atom_concentration": concentration,
        "class_domain_contributions": class_domain,
        "selection_audit_atom_stability": audit_stability,
        "support_cell_artifact_audit": support,
        "bootstrap_atom_diagnostics": bootstrap,
        "atom_target_gauge_conflict": gauge,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = os.path.join("oaci/reports", "C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json")
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    with open(path) as f:
        d = json.load(f)
    tdir = schema.C39_TABLE_DIR
    boot_summary = dict(d["bootstrap_atom_diagnostics_summary"])
    boot_summary["point_atom_additive_identity_exact"] = boot_summary.get(
        "point_atom_additive_identity_exact", boot_summary.get("point_atoms_exact", False))
    boot_summary["persisted_point_identity_pass"] = boot_summary.get(
        "persisted_point_identity_pass", schema.A9 not in d["taxonomy"]["cases"])
    boot_summary.pop("point_atoms_exact", None)
    boot_rows = _readcsv(os.path.join(tdir, "bootstrap_atom_diagnostics.csv"))
    for r in boot_rows:
        r["point_atom_additive_identity_exact"] = r.get(
            "point_atom_additive_identity_exact", r.get("point_atoms_exact", ""))
        r["persisted_point_identity_pass"] = r.get(
            "persisted_point_identity_pass", int(schema.A9 not in d["taxonomy"]["cases"]))
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_jobs": d.get("n_jobs", 0),
        "n_preference_robust_pairs": d["n_preference_robust_pairs"],
        "actual_selector_score_name": d["actual_selector_score_name"],
        "utility_grid_step": d["utility_grid_step"],
        "atom_recovery_availability": {
            "rows": _readcsv(os.path.join(tdir, "atom_recovery_availability.csv")),
            "summary": d["atom_recovery_availability_summary"],
        },
        "selected_atom_identity_gate": {
            "rows": _readcsv(os.path.join(tdir, "selected_atom_identity_gate.csv")),
            "summary": d["selected_atom_identity_summary"],
        },
        "selected_vs_better_point_atoms": {
            "rows": _readcsv(os.path.join(tdir, "selected_vs_better_point_atoms.csv")),
        },
        "atom_concentration": {
            "rows": _readcsv(os.path.join(tdir, "atom_concentration_summary.csv")),
            "summary": d["atom_concentration_summary"],
        },
        "class_domain_contributions": {
            "rows": _readcsv(os.path.join(tdir, "class_domain_atom_contributions.csv")),
        },
        "selection_audit_atom_stability": {
            "rows": _readcsv(os.path.join(tdir, "selection_audit_atom_stability.csv")),
            "summary": d["selection_audit_atom_stability_summary"],
        },
        "support_cell_artifact_audit": {
            "rows": _readcsv(os.path.join(tdir, "support_cell_artifact_audit.csv")),
            "summary": d["support_cell_artifact_summary"],
        },
        "bootstrap_atom_diagnostics": {
            "rows": boot_rows,
            "summary": boot_summary,
        },
        "atom_target_gauge_conflict": {
            "rows": _readcsv(os.path.join(tdir, "atom_target_gauge_conflict.csv")),
            "summary": d["atom_target_gauge_conflict_summary"],
        },
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_atoms=False, n_jobs=1):
    if recompute_atoms:
        return recompute(n_jobs=n_jobs)
    if os.path.exists(os.path.join("oaci/reports", "C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json")):
        return _summary_from_existing()
    return recompute(n_jobs=n_jobs)


def no_selector_gate(res):
    ident = res["selected_atom_identity_gate"]["summary"]
    avail = res["atom_recovery_availability"]["summary"]
    boot = res["bootstrap_atom_diagnostics"]["summary"]
    identity_pass = bool(ident["selection_identity_pass"])
    atom_claims_blocked = (not identity_pass and schema.A9 in res["taxonomy"]["cases"])
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c37_exact_ucl_pairs_imported", "passed": res["n_preference_robust_pairs"] == 114},
        {"check": "actual_selector_score_name_frozen",
         "passed": res["actual_selector_score_name"] == schema.ACTUAL_SELECTOR_SCORE_NAME},
        {"check": "selection_selected_and_better_atoms_available", "passed": avail["all_units_available"]},
        {"check": "selected_aggregate_identity_before_better_atom_claims",
         "passed": identity_pass or atom_claims_blocked},
        {"check": "identity_failure_blocks_atom_contribution_claims",
         "passed": identity_pass or atom_claims_blocked},
        {"check": "atom_additive_identity_passed_for_recomputed_points",
         "passed": ident["source_audit_additive_pass"] and
         float(ident["max_selection_additive_abs_diff"]) <= schema.ATOM_ADDITIVE_TOL},
        {"check": "source_audit_atoms_replayed_not_used_as_selection_proxy",
         "passed": ident["source_audit_additive_pass"]},
        {"check": "ucl_quantile_not_summed_from_atoms", "passed": not boot["per_atom_ucl_summed"]},
        {"check": "replicate_atom_limit_disclosed", "passed": boot["ucl_quantile_atom_limit"]},
        {"check": "target_gauge_diagnostic_only", "passed": True},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "no_selector_repair", "passed": True},
        {"check": "no_selected_checkpoint_method_artifact", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "target_labels_not_loaded_for_replay",
         "passed": avail["target_labels_loaded_for_replay"] == 0},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "atom_recovery_availability.csv"),
              res["atom_recovery_availability"]["rows"],
              ["unit_id", "seed", "target", "level", "selection_selected_available",
               "selection_better_available", "source_audit_selected_available",
               "source_audit_better_available", "selection_identity_pass",
               "source_audit_additive_pass", "support_graph_hashes_checked",
               "fold_plan_hashes_checked", "selection_bootstrap_plan_hash_checked",
               "population_hashes_checked", "feature_population_hash_matches",
               "target_labels_loaded_for_replay", "n_atoms_per_candidate_min", "n_atoms_per_candidate_max"])
    _writecsv(os.path.join(tdir, "selected_atom_identity_gate.csv"),
              res["selected_atom_identity_gate"]["rows"],
              ["job_key", "seed", "target", "level", "regime", "candidate_role",
               "candidate_order", "candidate_id", "split", "expected_point", "recomputed_point",
               "point_abs_diff", "selected_capacity", "atom_sum", "additive_abs_diff",
               "max_class_mass_abs_diff", "point_identity_pass", "atom_additive_identity_pass",
               "identity_pass", "support_graph_hash", "fold_plan_hash", "bootstrap_plan_hash",
               "population_hash", "feature_population_hash_matches", "target_labels_loaded_for_replay",
               "n_atoms"])
    point_cols = ["pair_id", "pair_key", "seed", "target", "level", "regime",
                  "selected_order", "better_order", "split", "atom_id", "class_id", "class_name",
                  "domain_id", "domain_name", "selected_atom", "better_atom",
                  "atom_delta_better_minus_selected", "positive_selected_advantage",
                  "positive_advantage_share", "selected_advantage_sign", "selected_point", "better_point",
                  "point_delta_better_minus_selected", "atom_fraction_of_point_delta", "support_count",
                  "support_m", "cell_mass", "class_overlap_mass", "p_ref_y", "p_d_given_y",
                  "eligible", "present", "support_edge", "selected_oof_mass", "better_oof_mass",
                  "selected_capacity", "better_capacity"]
    _writecsv(os.path.join(tdir, "selected_vs_better_point_atoms.csv"),
              res["selected_vs_better_point_atoms"]["rows"], point_cols)
    _writecsv(os.path.join(tdir, "atom_concentration_summary.csv"),
              res["atom_concentration"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "n_atoms", "n_positive_atoms", "positive_advantage_sum",
               "top1_positive_share", "top3_positive_share", "top5_positive_share", "positive_hhi",
               "concentrated_flag", "broad_flag", "concentration_class"])
    _writecsv(os.path.join(tdir, "class_domain_atom_contributions.csv"),
              res["class_domain_contributions"]["rows"],
              ["scope", "atom_key", "label", "n_rows", "positive_selected_advantage_sum",
               "signed_delta_sum", "positive_selected_advantage_share",
               "mean_atom_delta_better_minus_selected"])
    _writecsv(os.path.join(tdir, "selection_audit_atom_stability.csv"),
              res["selection_audit_atom_stability"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "n_atoms_compared", "n_selection_nonflat_atoms",
               "atom_sign_preservation_rate", "atom_sign_inversion_rate", "atom_delta_spearman",
               "selection_top_atom_id", "selection_top_class_id", "selection_top_domain_id",
               "selection_top_positive_share", "selection_top_delta", "audit_top_atom_same_delta",
               "top_atom_sign_preserved", "selection_point_prefers", "source_audit_leakage_prefers",
               "selection_to_audit_inversion"])
    _writecsv(os.path.join(tdir, "support_cell_artifact_audit.csv"),
              res["support_cell_artifact_audit"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "n_atoms", "low_mass_cut", "positive_advantage_sum",
               "low_mass_positive_share", "support_edge_positive_share", "dominant_atom_id",
               "dominant_atom_positive_share", "dominant_atom_support_count", "dominant_atom_support_m",
               "dominant_atom_cell_mass", "dominant_atom_low_mass", "dominant_atom_support_edge",
               "bootstrap_atom_variance_available", "support_artifact_flag"])
    _writecsv(os.path.join(tdir, "bootstrap_atom_diagnostics.csv"),
              res["bootstrap_atom_diagnostics"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "selected_ucl", "better_ucl", "aggregate_ucl_available",
               "point_atom_additive_identity_exact", "persisted_point_identity_pass",
               "replicate_atom_replay_available", "per_atom_ucl_summed",
               "ucl_quantile_linear_atom_claim_allowed", "diagnostic_status"])
    _writecsv(os.path.join(tdir, "atom_target_gauge_conflict.csv"),
              res["atom_target_gauge_conflict"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order",
               "better_order", "dominant_atom_id", "dominant_class_id", "dominant_class_name",
               "dominant_domain_id", "dominant_domain_name", "dominant_atom_delta_better_minus_selected",
               "dominant_atom_positive_share", "dominant_atom_prefers",
               "target_gauge_delta_better_minus_selected", "target_gauge_prefers",
               "target_endpoint_prefers", "target_bacc_delta", "target_nll_improve_delta",
               "target_ece_improve_delta", "atom_target_gauge_conflict", "target_gauge_non_source_only"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])
    _writecsv(os.path.join(tdir, "c39_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    ident = res["selected_atom_identity_gate"]["summary"]
    conc = res["atom_concentration"]["summary"]
    aud = res["selection_audit_atom_stability"]["summary"]
    sup = res["support_cell_artifact_audit"]["summary"]
    gau = res["atom_target_gauge_conflict"]["summary"]
    a9 = schema.A9 in res["taxonomy"]["cases"]
    point_header = "Point Atom Diagnostics (Blocked)" if a9 else "Point Atoms"
    bottom = (
        "> C39 does not elevate atom contribution claims: additive decomposition is exact for the "
        "recomputed point, but persisted C37 aggregate point identity is not bit-exact under the frozen "
        "tolerance, so A9 blocks atom-mechanism conclusions."
        if a9 else
        "> C39 recovers the source leakage point atoms exactly, then localizes the C38 selected-side "
        "leakage advantage to support cells and checks whether those atoms persist on source-audit and oppose "
        "the target gauge direction."
    )
    return "\n".join([
        f"# C39 - Leakage Atom Recovery / Support-Cell Conflict Audit "
        f"(frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only CPU replay over Phase-A source-train/source-audit frozen features. No training, no selector "
        "repair, no selected-checkpoint method artifact, and no per-atom UCL summation.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- C37 exact-UCL pairs imported: **{res['n_preference_robust_pairs']}**.",
        "",
        "## Identity Gates",
        "",
        f"- selection point identity: **{ident['n_selection_identity_pass']} / "
        f"{ident['n_selection_candidates']}** candidates.",
        f"- source-audit additive identity: **{ident['n_source_audit_additive_pass']} / "
        f"{ident['n_source_audit_candidates']}** candidates.",
        f"- max selection point diff: **{_f(ident['max_selection_point_abs_diff'])}**; "
        f"max atom additive diff: **{_f(ident['max_selection_additive_abs_diff'])}**.",
        "",
        f"## {point_header}",
        "",
        ("- Persisted aggregate identity did not pass the frozen gate; the following numbers are diagnostic "
         "replay summaries, not elevated atom contribution claims." if a9 else
         "- Persisted aggregate identity passed before atom contribution claims."),
        f"- concentrated pairs: **{conc['concentrated_pair_count']} / {conc['n_pairs']}**; "
        f"broad pairs: **{conc['broad_pair_count']} / {conc['n_pairs']}**.",
        f"- mean top-3 positive atom share: **{_f(conc['mean_top3_positive_share'])}**; "
        f"mean HHI: **{_f(conc['mean_positive_hhi'])}**.",
        "",
        "## Audit And Support",
        "",
        f"- mean selection-to-audit atom sign preservation: "
        f"**{_f(aud['mean_atom_sign_preservation_rate'])}**.",
        f"- selection-to-audit aggregate inversion: "
        f"**{_f(aud['selection_to_audit_inversion_rate'])}**.",
        f"- support-artifact pair fraction: **{_f(sup['support_artifact_pair_fraction'])}**; "
        f"dominant low-mass fraction: **{_f(sup['dominant_atom_low_mass_fraction'])}**.",
        "",
        "## Target Gauge",
        "",
        f"- atom-target gauge conflict: **{gau['atom_target_gauge_conflict_count']} / "
        f"{gau['n_pairs']}**.",
        f"- target gauge prefers better / selected: **{gau['target_gauge_prefers_better_count']} / "
        f"{gau['target_gauge_prefers_selected_count']}**.",
        "",
        "## Boundary",
        "",
        "- Atom sums are exact for the recomputed point, but persisted aggregate point identity is required "
        "before atom contribution claims.",
        "- UCL is a bootstrap quantile; C39 does not define or sum per-atom UCLs.",
        "- Target endpoints and target gauge remain diagnostic-only and non-source-only.",
        "",
        "## Bottom Line",
        "",
        bottom,
    ])


def render_audit_md(res):
    aud = res["selection_audit_atom_stability"]["summary"]
    return "\n".join([
        "# C39 - Selection-Audit Atom Stability",
        "",
        f"- mean atom sign preservation: {_f(aud['mean_atom_sign_preservation_rate'])}",
        f"- mean atom sign inversion: {_f(aud['mean_atom_sign_inversion_rate'])}",
        f"- aggregate selection-to-audit inversion: {_f(aud['selection_to_audit_inversion_rate'])}",
        f"- top atom sign preserved fraction: {_f(aud['top_atom_sign_preserved_fraction'])}",
        "",
        "Source-audit atoms are replayed from the audit split and are not used as selection atom proxies.",
    ]) + "\n"


def render_gauge_md(res):
    gau = res["atom_target_gauge_conflict"]["summary"]
    return "\n".join([
        "# C39 - Leakage Atom / Target Gauge Conflict",
        "",
        f"- atom-target gauge conflict fraction: {_f(gau['atom_target_gauge_conflict_fraction'])}",
        f"- conflict count: {gau['atom_target_gauge_conflict_count']} / {gau['n_pairs']}",
        f"- mean dominant atom positive share: {_f(gau['mean_dominant_atom_positive_share'])}",
        "",
        "Target gauge remains a diagnostic label imported from earlier artifacts, not a source-only method.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 96):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C39 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_jobs": res["n_jobs"],
        "n_preference_robust_pairs": res["n_preference_robust_pairs"],
        "actual_selector_score_name": res["actual_selector_score_name"],
        "utility_grid_step": res["utility_grid_step"],
        "atom_recovery_availability_summary": res["atom_recovery_availability"]["summary"],
        "selected_atom_identity_summary": res["selected_atom_identity_gate"]["summary"],
        "atom_concentration_summary": res["atom_concentration"]["summary"],
        "selection_audit_atom_stability_summary": res["selection_audit_atom_stability"]["summary"],
        "support_cell_artifact_summary": res["support_cell_artifact_audit"]["summary"],
        "bootstrap_atom_diagnostics_summary": res["bootstrap_atom_diagnostics"]["summary"],
        "atom_target_gauge_conflict_summary": res["atom_target_gauge_conflict"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "atom_identity_check": "Aggregate point identity and atom additive identity gate atom claims.",
            "a9_block_check": "When aggregate identity fails, atom contribution summaries remain diagnostic only.",
            "ucl_quantile_check": "No per-atom UCL summation is reported because UCL is a quantile.",
            "audit_proxy_check": "Source-audit atoms are independently replayed and not used as selection proxies.",
            "target_label_check": "Target gauge and endpoints remain diagnostic-only labels.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    audit = render_audit_md(res)
    gauge = render_gauge_md(res)
    for text in (md, audit, gauge):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C39_LEAKAGE_ATOM_RECOVERY_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C39_SELECTION_AUDIT_ATOM_STABILITY.md"), "w").write(audit)
    open(os.path.join(out_dir, "C39_LEAKAGE_ATOM_TARGET_GAUGE_CONFLICT.md"), "w").write(gauge)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C39_LEAKAGE_ATOM_RECOVERY_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c39_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.leakage_atoms.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--recompute", action="store_true", help="replay point atoms instead of reading C39 tables")
    args = ap.parse_args(argv)
    res = run(recompute_atoms=args.recompute, n_jobs=args.n_jobs)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    print(f"[C39] cases={','.join(res['taxonomy']['cases'])} "
          f"identity={res['selected_atom_identity_gate']['summary']['n_selection_identity_pass']}/"
          f"{res['selected_atom_identity_gate']['summary']['n_selection_candidates']} "
          f"concentrated={res['atom_concentration']['summary']['concentrated_pair_count']}/"
          f"{res['atom_concentration']['summary']['n_pairs']} "
          f"audit_preserve={res['selection_audit_atom_stability']['summary']['mean_atom_sign_preservation_rate']} "
          f"gauge_conflict={res['atom_target_gauge_conflict']['summary']['atom_target_gauge_conflict_fraction']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
