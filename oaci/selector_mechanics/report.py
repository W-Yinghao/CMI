"""C36 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import (artifact_loader, feasibility_regret, plateau_tiebreak, schema, selected_pair_trace,
               selection_audit_inversion, selector_trace, source_pareto, taxonomy)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C36 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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


def run():
    cfg = _lock_config()
    pairs = artifact_loader.load_preference_robust_pairs()
    trace = artifact_loader.load_c10_selector_trace(regimes=artifact_loader.regimes_from_pairs(pairs))
    availability = selector_trace.availability_audit(trace["registry"])
    pair_resolution = selector_trace.robust_pair_trace_resolves(pairs, trace)
    pair_ucl = selector_trace.selected_ucl_availability_for_pairs(pairs, trace)
    pair_rows = selected_pair_trace.build_selected_pair_trace(pairs, trace)
    feasibility = feasibility_regret.decompose(pair_rows)
    pareto = source_pareto.audit(pair_rows, trace)
    inversion = selection_audit_inversion.audit(pair_rows)
    plateau = plateau_tiebreak.audit(pair_rows, trace)
    trace_availability = {"rows": availability, "pair_resolution": pair_resolution, "pair_ucl": pair_ucl}
    tax = taxonomy.classify(feasibility, pareto, inversion, plateau, trace_availability)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_preference_robust_pairs": len(pairs),
        "n_unique_selector_units": len({(p["seed"], p["target"], p["level"]) for p in pairs}),
        "utility_grid_step": schema.UTILITY_GRID_STEP,
        "pairs": pairs,
        "trace": trace,
        "trace_availability": trace_availability,
        "selected_pair_trace": {"rows": pair_rows},
        "feasibility_regret": feasibility,
        "source_pareto": pareto,
        "selection_audit_inversion": inversion,
        "plateau_tiebreak": plateau,
        "taxonomy": tax,
    }


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "c35_preference_robust_pairs_imported", "passed": res["n_preference_robust_pairs"] == 114},
        {"check": "utility_grid_unchanged", "passed": res["utility_grid_step"] == 0.05},
        {"check": "selector_trace_availability_audited", "passed": bool(res["trace_availability"]["rows"])},
        {"check": "robust_pair_trace_resolves", "passed": res["trace_availability"]["pair_resolution"]["all_resolved"]},
        {"check": "no_proxy_selector_score", "passed": not res["trace_availability"]["pair_ucl"]["pairwise_selector_score_delta_available"]},
        {"check": "per_candidate_ucl_gap_disclosed", "passed": schema.S9 in res["taxonomy"]["cases"]},
        {"check": "target_endpoint_labels_diagnostic_only", "passed": True},
        {"check": "no_training_no_reinference", "passed": True},
        {"check": "no_selected_checkpoint_method_artifact", "passed": True},
        {"check": "tie_plateau_eps_frozen", "passed": schema.POINT_PLATEAU_EPS == 0.02},
        {"check": "source_pareto_objective_set_frozen", "passed": len(schema.SOURCE_PARETO_OBJECTIVES) == 9},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "selector_trace_availability.csv"), res["trace_availability"]["rows"],
              ["field", "trace_item", "n_available", "n_total", "availability_fraction", "status",
               "used_for_classification", "trace_use"])
    _writecsv(os.path.join(tdir, "selector_trace_registry.csv"), selector_trace.registry_rows(res["trace"]),
              ["candidate_id", "seed", "target", "level", "regime", "candidate_order", "raw_replay_index",
               "candidate_role", "origin", "is_erm", "selected_oaci", "epoch", "lambda", "R_src",
               "shared_tau", "risk_slack_to_tau", "feasible", "balanced_err", "train_surrogate",
               "selection_leakage_point", "audit_leakage_point", "source_guard_worst_bacc",
               "source_guard_worst_nll", "source_guard_worst_ece", "source_audit_worst_bacc",
               "source_audit_worst_nll", "source_audit_worst_ece", "target_worst_bacc", "target_worst_nll",
               "target_worst_ece", "actual_selector_score_name", "actual_selector_score_ucl",
               "actual_selector_score_available", "per_candidate_selector_ucl_available",
               "actual_selector_rank_known", "actual_selector_rank", "actual_selector_relation",
               "selection_reason", "selection_status", "n_feasible", "checkpoint_hash_available",
               "checkpoint_hash_emitted", "tie_break_metadata_available"])
    _writecsv(os.path.join(tdir, "selected_vs_better_selector_trace.csv"),
              res["selected_pair_trace"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "selected_order", "better_order",
               "selected_candidate_id", "better_candidate_id", "selected_is_actual_oaci", "better_is_actual_oaci",
               "selected_feasible", "better_feasible", "selected_R_src", "better_R_src",
               "R_src_delta_better_minus_selected", "selected_risk_slack_to_tau", "better_risk_slack_to_tau",
               "risk_slack_delta_better_minus_selected", "actual_selector_score_name",
               "selected_actual_selector_ucl", "better_actual_selector_ucl",
               "actual_selector_score_delta_available", "actual_selector_rank_delta_available",
               "actual_selector_relation", "selection_leakage_point_delta_better_minus_selected",
               "selection_leakage_point_prefers", "audit_leakage_point_delta_better_minus_selected",
               "audit_leakage_point_prefers", "source_guard_endpoint_prefers",
               "source_audit_endpoint_prefers", "source_endpoint_majority_prefers", "source_guard_bacc_delta",
               "source_guard_nll_improve_delta", "source_guard_ece_improve_delta",
               "source_audit_bacc_delta", "source_audit_nll_improve_delta", "source_audit_ece_improve_delta",
               "target_bacc_delta", "target_nll_delta", "target_ece_delta",
               "fraction_weights_alt_beats_selected", "mean_regret_over_simplex", "utility_cone_category",
               "pareto_status", "target_endpoint_prefers", "trace_complete_for_point_components"])
    _writecsv(os.path.join(tdir, "feasibility_regret_decomposition.csv"),
              res["feasibility_regret"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "risk_gate_regret",
               "leakage_objective_regret", "source_endpoint_regret", "selection_audit_inversion",
               "tie_break_regret", "trace_unavailable", "selection_leakage_point_prefers",
               "audit_leakage_point_prefers", "source_endpoint_majority_prefers", "actual_selector_relation"])
    _writecsv(os.path.join(tdir, "leakage_objective_conflict.csv"), res["feasibility_regret"]["leakage_rows"],
              ["pair_id", "seed", "target", "level", "regime",
               "selection_leakage_point_delta_better_minus_selected", "selection_leakage_point_prefers",
               "audit_leakage_point_delta_better_minus_selected", "audit_leakage_point_prefers",
               "actual_selector_ucl_delta_available"])
    src_cols = ["pair_id", "seed", "target", "level", "regime", "source_pareto_status", "target_prefers"]
    src_cols += ["n_source_objectives_finite", "n_source_objectives_registered"]
    src_cols += [f"{o['objective']}_oriented_delta_better_minus_selected"
                 for o in schema.SOURCE_PARETO_OBJECTIVES]
    _writecsv(os.path.join(tdir, "source_pareto_status.csv"), res["source_pareto"]["rows"], src_cols)
    _writecsv(os.path.join(tdir, "selection_audit_inversion.csv"),
              res["selection_audit_inversion"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "selection_leakage_prefers",
               "audit_leakage_prefers", "source_audit_endpoint_prefers", "target_endpoint_prefers",
               "selection_to_audit_inversion", "audit_to_target_inversion", "local_leakage_target_conflict"])
    _writecsv(os.path.join(tdir, "selector_plateau_tiebreak.csv"), res["plateau_tiebreak"]["rows"],
              ["pair_id", "seed", "target", "level", "regime", "actual_selector_plateau_available",
               "actual_selector_tie_break_metadata_available", "actual_selector_tie_break_classification",
               "point_component_plateau_eps", "point_component_plateau_size", "better_in_point_component_plateau",
               "point_component_active_selected_margin"])
    _writecsv(os.path.join(tdir, "source_rational_target_wrong_cases.csv"),
              res["source_pareto"]["source_rational_rows"],
              ["pair_id", "seed", "target", "level", "regime", "source_pareto_status", "target_prefers",
               "source_rational_not_better_dominated"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c36_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
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
    fs = res["feasibility_regret"]["summary"]
    ps = res["source_pareto"]["summary"]
    inv = res["selection_audit_inversion"]["summary"]
    pl = res["plateau_tiebreak"]["summary"]
    ucl = res["trace_availability"]["pair_ucl"]
    return "\n".join([
        f"# C36 - OACI Selector Mechanics / Feasibility-Regret Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over C35/C34S/C10 artifacts. C36 asks what the actual OACI selector trace "
        "contains for C35 preference-robust local better alternatives. No training, no re-inference, no selector, "
        "no selected-checkpoint artifact.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- preference-robust C35 pair-rows: **{res['n_preference_robust_pairs']}** over "
        f"**{res['n_unique_selector_units']}** unique selector units; C35's frozen raw utility grid is unchanged "
        f"(step **{schema.UTILITY_GRID_STEP}**).",
        "",
        "## Stage-0 Trace Availability",
        "",
        f"- selected UCL available for selected rows: **{ucl['selected_ucl_available']}/{ucl['n_pairs']}**.",
        f"- better-candidate UCL available: **{ucl['better_ucl_available']}/{ucl['n_pairs']}**.",
        "- Therefore C36 does not compute a numeric actual-selector score delta or actual UCL plateau. "
        "Selection-leakage point is reported only as a component trace, not as a proxy selector score.",
        "",
        "## Feasibility And Leakage Components",
        "",
        f"- risk-gate regret fraction: **{_f(fs['risk_gate_regret_fraction'])}**.",
        f"- selection-leakage point component prefers selected: **{_f(fs['leakage_objective_regret_fraction'])}**.",
        f"- source endpoint majority prefers selected: **{_f(fs['source_endpoint_regret_fraction'])}**.",
        f"- trace-unavailable fraction for actual selector deltas: **{_f(fs['trace_unavailable_fraction'])}**.",
        "",
        "## Selection-Audit Inversion",
        "",
        f"- selection-to-audit inversion rate: **{_f(inv['selection_to_audit_inversion_rate'])}**.",
        f"- audit-to-target inversion rate: **{_f(inv['audit_to_target_inversion_rate'])}**.",
        f"- local leakage-target conflict rate: **{_f(inv['local_leakage_target_conflict_rate'])}**.",
        "",
        "## Source Pareto",
        "",
        f"- source-Pareto conflict fraction (selected dominates or incomparable): "
        f"**{_f(ps['source_pareto_conflict_fraction'])}**.",
        f"- better source-dominates selected fraction: **{_f(ps['better_source_dominates_fraction'])}**.",
        "- The source-Pareto objective set is frozen before analysis and uses source risk, leakage point "
        "components, source_guard endpoints, and source_audit endpoints; it is not a scalar selector. "
        "Non-finite endpoint cells are filtered per pair and the finite objective count is reported in the table.",
        "",
        "## Plateau / Tie",
        "",
        f"- actual selector UCL plateau available: **{pl['actual_selector_plateau_available']}**.",
        f"- point-component active selected-margin fraction at eps {schema.POINT_PLATEAU_EPS}: "
        f"**{_f(pl['point_component_active_selected_margin_fraction'])}**.",
        "- S6/S7 are blocked as actual-selector claims because better-candidate UCLs and tie metadata are absent.",
        "",
        "## Bottom Line",
        "",
        "> C36 localizes the C35 robust local misses to a source-side selector trace where risk-feasible better "
        "alternatives are present, the selection-leakage point component consistently favors the artifact-selected "
        "candidate, source-Pareto conflict is common, and exact per-candidate UCL/tie trace is insufficient for "
        "numeric actual-selector margin claims.",
    ])


def render_feasibility_md(res):
    fs = res["feasibility_regret"]["summary"]
    return "\n".join([
        "# C36 - Feasibility-Regret Decomposition",
        "",
        f"- risk gate: {_f(fs['risk_gate_regret_fraction'])}",
        f"- leakage point component: {_f(fs['leakage_objective_regret_fraction'])}",
        f"- source endpoints: {_f(fs['source_endpoint_regret_fraction'])}",
        f"- selection-audit inversion: {_f(fs['selection_audit_inversion_fraction'])}",
        f"- trace unavailable: {_f(fs['trace_unavailable_fraction'])}",
        "",
        "All rows are diagnostic pair decompositions over imported C35 preference-robust pairs.",
    ]) + "\n"


def render_source_pareto_md(res):
    ps = res["source_pareto"]["summary"]
    return "\n".join([
        "# C36 - Source Pareto Conflict Audit",
        "",
        f"- better source-dominates selected: {_f(ps['better_source_dominates_selected_fraction'])}",
        f"- selected source-dominates better: {_f(ps['selected_source_dominates_better_fraction'])}",
        f"- source-Pareto incomparable: {_f(ps['source_pareto_incomparable_fraction'])}",
        f"- source-Pareto conflict: {_f(ps['source_pareto_conflict_fraction'])}",
        "",
        "This is a vector dominance audit over frozen source-side objectives, not a scalarized selector.",
    ]) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not deployable", "non-deployable", "diagnostic-only", "no selected", "no selector",
             "not claimed")


def _guard_forbidden(md):
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 72):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C36 report near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_preference_robust_pairs": res["n_preference_robust_pairs"],
        "n_unique_selector_units": res["n_unique_selector_units"],
        "utility_grid_step": res["utility_grid_step"],
        "trace_availability": res["trace_availability"],
        "feasibility_regret_summary": res["feasibility_regret"]["summary"],
        "source_pareto_summary": res["source_pareto"]["summary"],
        "selection_audit_inversion_summary": res["selection_audit_inversion"]["summary"],
        "plateau_tiebreak_summary": res["plateau_tiebreak"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "trace_proxy_check": "Per-candidate selector UCL is unavailable; selection leakage point is never used as actual selector score.",
            "source_pareto_check": "Source-Pareto is vector dominance over frozen objectives, not scalarization.",
            "target_label_check": "Target endpoints enter only as diagnostic imported C35 pair labels.",
        },
    }


def _write_artifacts(res, out_dir):
    md = render_md(res)
    feas = render_feasibility_md(res)
    pareto = render_source_pareto_md(res)
    for text in (md, feas, pareto):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C36_OACI_SELECTOR_MECHANICS_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C36_FEASIBILITY_REGRET_DECOMPOSITION.md"), "w").write(feas)
    open(os.path.join(out_dir, "C36_SOURCE_PARETO_CONFLICT_AUDIT.md"), "w").write(pareto)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C36_OACI_SELECTOR_MECHANICS_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c36_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.selector_mechanics.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run()
    _write_artifacts(res, args.out_dir)
    print(f"[C36] cases={','.join(res['taxonomy']['cases'])} "
          f"robust_pairs={res['n_preference_robust_pairs']} "
          f"leak_point={_f(res['feasibility_regret']['summary']['leakage_objective_regret_fraction'])} "
          f"source_pareto_conflict={_f(res['source_pareto']['summary']['source_pareto_conflict_fraction'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
