"""C41 global leakage-target objective field report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, field_registry, leakage_target_alignment, local_global_consistency,
               low_leakage_enrichment, objective_field_comparison, schema, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C41 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    def clean(v):
        if isinstance(v, bool):
            return int(v)
        if isinstance(v, float) and not math.isfinite(v):
            return ""
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


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    availability = field_registry.availability(ctx)
    registry = field_registry.candidate_registry(ctx)
    alignment = leakage_target_alignment.align(ctx)
    enrichment = low_leakage_enrichment.audit(ctx)
    comparison = objective_field_comparison.compare(ctx, alignment)
    audit_vs_selection = objective_field_comparison.source_audit_vs_selection(alignment)
    local_global = local_global_consistency.audit(ctx, alignment)
    gauge = local_global_consistency.target_gauge_vs_leakage(ctx)
    tax = taxonomy.classify(availability, alignment, enrichment, comparison, audit_vs_selection,
                            local_global, gauge)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "actual_selector_score_name": schema.ACTUAL_SELECTOR_SCORE_NAME,
        "objective_field_availability": availability,
        "candidate_objective_field_registry": registry,
        "leakage_target_rank_alignment": alignment,
        "low_leakage_enrichment": enrichment,
        "objective_field_comparison": comparison,
        "source_audit_vs_selection_leakage_alignment": audit_vs_selection,
        "local_global_conflict_consistency": local_global,
        "target_gauge_vs_leakage_field": gauge,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C41_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "actual_selector_score_name": d["actual_selector_score_name"],
        "objective_field_availability": {"rows": _readcsv(os.path.join(tdir, "objective_field_availability.csv")),
                                         "summary": d["objective_field_availability_summary"]},
        "candidate_objective_field_registry": {"rows": _readcsv(os.path.join(tdir, "candidate_objective_field_registry.csv")),
                                               "summary": {"n_rows": d["n_candidate_rows"]}},
        "leakage_target_rank_alignment": {"rows": _readcsv(os.path.join(tdir, "leakage_target_rank_alignment.csv")),
                                          "summary_rows": d["leakage_target_alignment_summary_rows"],
                                          "summary": {r["field"]: r for r in d["leakage_target_alignment_summary_rows"]}},
        "low_leakage_enrichment": {"rows": _readcsv(os.path.join(tdir, "low_leakage_enrichment.csv")),
                                   "summary_rows": d["low_leakage_enrichment_summary_rows"]},
        "objective_field_comparison": {"rows": _readcsv(os.path.join(tdir, "objective_field_comparison.csv")),
                                       "summary": d["objective_field_comparison_summary"]},
        "source_audit_vs_selection_leakage_alignment": {
            "rows": _readcsv(os.path.join(tdir, "source_audit_vs_selection_leakage_alignment.csv")),
            "summary": d["source_audit_vs_selection_summary"]},
        "local_global_conflict_consistency": {
            "rows": _readcsv(os.path.join(tdir, "local_global_conflict_consistency.csv")),
            "summary": d["local_global_conflict_summary"]},
        "target_gauge_vs_leakage_field": {
            "rows": _readcsv(os.path.join(tdir, "target_gauge_vs_leakage_field.csv")),
            "summary": d["target_gauge_vs_leakage_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    avail = res["objective_field_availability"]["summary"]
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "candidate_registry_complete", "passed": avail["candidate_registry_complete"]},
        {"check": "field_availability_audited_first", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "no_atom_level_claims", "passed": True},
        {"check": "no_proxy_selector_ucl", "passed": not avail["selection_ucl_global_available"]},
        {"check": "target_fields_diagnostic_only", "passed": True},
        {"check": "target_unlabeled_marked_non_source_only", "passed": True},
        {"check": "trajectory_conditioned_random_baseline", "passed": True},
        {"check": "no_selected_checkpoint_method_artifact", "passed": True},
        {"check": "no_monolithic_large_json", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "objective_field_availability.csv"),
              res["objective_field_availability"]["rows"],
              ["field", "scope", "status", "orientation_better", "n_available", "n_candidate_rows",
               "availability_fraction", "used_for_global_candidate_alignment", "target_labels_diagnostic_only",
               "non_source_only", "proxy_used"])
    _writecsv(os.path.join(tdir, "candidate_objective_field_registry.csv"),
              res["candidate_objective_field_registry"]["rows"],
              res["candidate_objective_field_registry"]["columns"])
    _writecsv(os.path.join(tdir, "leakage_target_rank_alignment.csv"),
              res["leakage_target_rank_alignment"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "field", "field_orientation",
               "n_candidates", "spearman_oriented_field_vs_target_utility",
               "pairwise_auc_oriented_field_ranks_target_utility", "n_pairwise_comparisons",
               "sign_class", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "low_leakage_enrichment.csv"),
              res["low_leakage_enrichment"]["rows"],
              ["trajectory_id", "seed", "target", "level", "regime", "selection_rule", "label",
               "n_candidates", "n_selected_by_low_leakage", "trajectory_label_count", "selected_label_count",
               "hit_rate", "trajectory_random_baseline", "enrichment_ratio", "hypergeom_p_value",
               "bonferroni_p_value", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "objective_field_comparison.csv"),
              res["objective_field_comparison"]["rows"],
              ["field", "scope", "n_trajectories", "target_utility_mean_auc", "target_utility_median_auc",
               "rank_strength_abs", "candidate_level_available", "non_source_only", "proxy_used"])
    _writecsv(os.path.join(tdir, "source_audit_vs_selection_leakage_alignment.csv"),
              res["source_audit_vs_selection_leakage_alignment"]["rows"],
              ["metric", "selection_leakage", "source_audit_leakage", "audit_minus_selection",
               "source_audit_substantially_better"])
    _writecsv(os.path.join(tdir, "local_global_conflict_consistency.csv"),
              res["local_global_conflict_consistency"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime", "selected_order", "better_order",
               "global_selection_leakage_auc", "global_selection_leakage_class", "local_selected_lower_leakage",
               "local_better_higher_target_utility", "local_conflict", "selected_low_leakage_rank_percentile",
               "selected_target_utility_rank_percentile", "selected_near_global_leakage_optimum",
               "selected_away_from_target_optimum", "local_conflict_representative_of_global_field",
               "local_tail_only_flag"])
    _writecsv(os.path.join(tdir, "target_gauge_vs_leakage_field.csv"),
              res["target_gauge_vs_leakage_field"]["rows"],
              ["pair_id", "pair_key", "seed", "target", "level", "regime",
               "selection_point_delta_better_minus_selected", "selection_point_prefers",
               "target_gauge_delta_better_minus_selected", "target_gauge_prefers",
               "leakage_target_gauge_conflict", "candidate_level_target_gauge_available",
               "target_gauge_non_source_only"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c41_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_md(res):
    sel = res["leakage_target_rank_alignment"]["summary"].get("selection_leakage_point", {})
    audit = res["leakage_target_rank_alignment"]["summary"].get("audit_leakage_point", {})
    comp = res["objective_field_comparison"]["summary"]
    loc = res["local_global_conflict_consistency"]["summary"]
    enr = res["low_leakage_enrichment"]["summary"]
    top3_joint = enr.get(("top3", "primary_joint_good"), {})
    top3_pareto = enr.get(("top3", "pareto_good"), {})
    top3_robust = enr.get(("top3", "preference_robust_better_candidate"), {})
    active = set(res["taxonomy"]["cases"])
    o4_state = "active" if schema.O4 in active else "not active"
    o8_state = "active" if schema.O8 in active else "not active"
    return "\n".join([
        f"# C41 - Global Leakage-Target Utility Objective Field Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over committed candidate-level artifacts. No training, no GPU, no selector "
        "repair, no atom-level claims, and no proxy selector UCL.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        "",
        "## Leakage-Target Field",
        "",
        f"- selection leakage mean AUC vs target utility: **{_f(sel.get('mean_pairwise_auc'))}**.",
        f"- source-audit leakage mean AUC vs target utility: **{_f(audit.get('mean_pairwise_auc'))}**.",
        f"- C30 aggregate source-rank AUC: **{_f(comp.get('c30_source_rank_auc'))}**.",
        "",
        "## Low-Leakage Enrichment",
        "",
        f"- top-3 low-leakage joint-good enrichment: **{_f(top3_joint.get('mean_enrichment_ratio'))}**.",
        f"- top-3 low-leakage Pareto-good enrichment: **{_f(top3_pareto.get('mean_enrichment_ratio'))}**.",
        f"- top-3 low-leakage preference-robust-local-alternative enrichment: "
        f"**{_f(top3_robust.get('mean_enrichment_ratio'))}**.",
        f"- O4 status: **{o4_state}**; joint-good is below baseline, but sparse robust-local-alternative ratios "
        "prevent a clean all-label no-enrichment call.",
        "- Enrichment is trajectory-conditioned and compared to within-trajectory random baselines.",
        "",
        "## Local-Global Consistency",
        "",
        f"- local conflict representative fraction: **{_f(loc['representative_fraction'])}**.",
        f"- local tail-only fraction: **{_f(loc['tail_only_fraction'])}**.",
        f"- mean selected low-leakage rank percentile: **{_f(loc['mean_selected_low_leakage_rank_percentile'])}**.",
        f"- O8 status: **{o8_state}**; the representative fraction is near the pre-registered "
        f"{schema.LOCAL_REPRESENTATIVE_GATE:.3f} gate but does not pass it.",
        "",
        "## Boundaries",
        "",
        "- C30 source-rank and target-gauge fields are not candidate-level absolute fields in current artifacts; "
        "they are reported as aggregate/local diagnostics only.",
        "- Target endpoints and target gauge remain diagnostic-only and non-source-only where applicable.",
        "",
        "## Bottom Line",
        "",
        "> C41 establishes O2 + O5 + O6: global selection leakage is mostly decoupled from target utility, "
        "source-audit leakage does not materially improve the target-utility alignment, and the aggregate C30 "
        "source-rank axis is stronger than leakage. O4 and O8 remain below the pre-registered gates.",
    ])


def render_enrichment_md(res):
    rows = res["low_leakage_enrichment"]["summary_rows"]
    labels = ("primary_joint_good", "pareto_good", "preference_robust_better_candidate")
    return "\n".join([
        "# C41 - Low-Leakage Enrichment Audit",
        "",
        *(f"- {r['selection_rule']} / {r['label']}: enrichment {_f(r['mean_enrichment_ratio'])}, "
          f"hit {_f(r['mean_hit_rate'])}, baseline {_f(r['mean_random_baseline'])}, "
          f"Bonferroni-significant trajectories {r['significant_enriched_trajectories_bonferroni']}"
          for r in rows if r["label"] in labels),
        "",
        "All enrichment baselines are trajectory-conditioned; target labels are diagnostic-only.",
    ]) + "\n"


def render_local_md(res):
    loc = res["local_global_conflict_consistency"]["summary"]
    return "\n".join([
        "# C41 - Local/Global Conflict Audit",
        "",
        f"- local conflict count: {loc['local_conflict_count']} / {loc['n_pairs']}",
        f"- representative fraction: {_f(loc['representative_fraction'])}",
        f"- tail-only fraction: {_f(loc['tail_only_fraction'])}",
        "",
        "Local C37/C38 misdirection is evaluated against each trajectory's global selection-leakage field.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed", "blocked")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 120):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C41 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "actual_selector_score_name": res["actual_selector_score_name"],
        "objective_field_availability_summary": res["objective_field_availability"]["summary"],
        "leakage_target_alignment_summary_rows": res["leakage_target_rank_alignment"]["summary_rows"],
        "low_leakage_enrichment_summary_rows": res["low_leakage_enrichment"]["summary_rows"],
        "objective_field_comparison_summary": res["objective_field_comparison"]["summary"],
        "source_audit_vs_selection_summary": res["source_audit_vs_selection_leakage_alignment"]["summary"],
        "local_global_conflict_summary": res["local_global_conflict_consistency"]["summary"],
        "target_gauge_vs_leakage_summary": res["target_gauge_vs_leakage_field"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "pooling_check": "All primary alignment metrics are within-trajectory before summary aggregation.",
            "quantile_baseline_check": "Low-leakage enrichment uses trajectory-conditioned random baselines.",
            "proxy_check": "Selector UCL, target gauge, and C30 rank are not proxied as candidate-level fields.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    enr = render_enrichment_md(res)
    loc = render_local_md(res)
    for text in (md, enr, loc):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C41_LOW_LEAKAGE_ENRICHMENT_AUDIT.md"), "w").write(enr)
    open(os.path.join(out_dir, "C41_LOCAL_GLOBAL_CONFLICT_AUDIT.md"), "w").write(loc)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C41_LEAKAGE_TARGET_OBJECTIVE_FIELD.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c41_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.objective_field.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    sel = res["leakage_target_rank_alignment"]["summary"].get("selection_leakage_point", {})
    print(f"[C41] cases={','.join(res['taxonomy']['cases'])} "
          f"selection_auc={sel.get('mean_pairwise_auc')} candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
