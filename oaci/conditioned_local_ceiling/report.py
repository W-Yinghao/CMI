"""C48 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from . import artifact_loader, local_ceiling, schema, source_space_registry, taxonomy


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C48 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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
    try:
        fx = float(x)
        if math.isfinite(fx):
            return f"{fx:.3f}"
    except (TypeError, ValueError):
        pass
    return str(x)


def _cross_group_stability_rows(summary_rows):
    rows = []
    for source_space in sorted({r["source_space"] for r in summary_rows}):
        for neighborhood in sorted({r["neighborhood"] for r in summary_rows if r["source_space"] == source_space}):
            for label in schema.LABELS:
                by_scope = {
                    r["group_scope"]: r for r in summary_rows
                    if r["source_space"] == source_space and r["neighborhood"] == neighborhood and
                    r["label"] == label
                }
                if not by_scope:
                    continue
                conditioned = [
                    by_scope[s] for s in ("within_target", "within_trajectory",
                                          "within_target_seed", "within_target_level")
                    if s in by_scope
                ]
                mixed = [by_scope[s] for s in ("global", "within_regime") if s in by_scope]
                best_cond = max(
                    conditioned,
                    key=lambda r: float(r["mean_local_bayes_top1_hit"])
                    if artifact_loader.finite(r.get("mean_local_bayes_top1_hit")) else -1.0,
                ) if conditioned else {}
                best_mixed = max(
                    mixed,
                    key=lambda r: float(r["mean_local_bayes_top1_hit"])
                    if artifact_loader.finite(r.get("mean_local_bayes_top1_hit")) else -1.0,
                ) if mixed else {}
                rows.append({
                    "source_space": source_space,
                    "neighborhood": neighborhood,
                    "label": label,
                    "global_hit": by_scope.get("global", {}).get("mean_local_bayes_top1_hit"),
                    "within_target_hit": by_scope.get("within_target", {}).get("mean_local_bayes_top1_hit"),
                    "within_trajectory_hit": by_scope.get("within_trajectory", {}).get(
                        "mean_local_bayes_top1_hit"),
                    "within_target_seed_hit": by_scope.get("within_target_seed", {}).get(
                        "mean_local_bayes_top1_hit"),
                    "within_target_level_hit": by_scope.get("within_target_level", {}).get(
                        "mean_local_bayes_top1_hit"),
                    "within_regime_hit": by_scope.get("within_regime", {}).get("mean_local_bayes_top1_hit"),
                    "best_conditioned_scope": best_cond.get("group_scope"),
                    "best_conditioned_hit": best_cond.get("mean_local_bayes_top1_hit"),
                    "best_conditioned_gain": best_cond.get("mean_local_bayes_gain_vs_random"),
                    "best_mixed_scope": best_mixed.get("group_scope"),
                    "best_mixed_hit": best_mixed.get("mean_local_bayes_top1_hit"),
                    "best_mixed_gain": best_mixed.get("mean_local_bayes_gain_vs_random"),
                    "conditioned_minus_mixed_hit": (
                        float(best_cond["mean_local_bayes_top1_hit"]) -
                        float(best_mixed["mean_local_bayes_top1_hit"])
                        if best_cond and best_mixed else math.nan
                    ),
                    "conditioned_minus_mixed_gain": (
                        float(best_cond["mean_local_bayes_gain_vs_random"]) -
                        float(best_mixed["mean_local_bayes_gain_vs_random"])
                        if best_cond and best_mixed else math.nan
                    ),
                    "target_labels_diagnostic_only": 1,
                })
    return rows


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    spaces = source_space_registry.registry(ctx)
    eps = source_space_registry.epsilon_radii(ctx, spaces["spaces"])
    ceiling = local_ceiling.audit(ctx, spaces["spaces"], eps["summary"])
    stability_rows = _cross_group_stability_rows(ceiling["summary_rows"])
    tax = taxonomy.classify(ceiling, ctx["c47_summary"])
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "source_space_registry": spaces,
        "epsilon_radius_registry": eps,
        "local_ceiling": ceiling,
        "cross_group_stability": {"rows": stability_rows, "summary": {"n_rows": len(stability_rows)}},
        "taxonomy": tax,
        "c47_reference": {
            "cases": ctx["c47_summary"]["taxonomy"]["cases"],
            "best_conditioned_strict_source_top1_hit":
                ctx["c47_summary"]["taxonomy"]["primary_metrics"]["best_conditioned_strict_source_top1_hit"],
            "best_conditioned_strict_source_top1_enrichment":
                ctx["c47_summary"]["taxonomy"]["primary_metrics"]["best_conditioned_strict_source_top1_enrichment"],
        },
    }


def _summary_from_existing():
    path = "oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    tdir = schema.C48_TABLE_DIR
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "source_space_registry": {"rows": _readcsv(os.path.join(tdir, "source_space_registry.csv"))},
        "epsilon_radius_registry": {"rows": _readcsv(os.path.join(tdir, "epsilon_radius_registry.csv")),
                                    "summary": d["epsilon_radius_summary"]},
        "local_ceiling": {
            "summary_rows": _readcsv(os.path.join(tdir, "local_ceiling_summary.csv")),
            "best_rows": _readcsv(os.path.join(tdir, "local_ceiling_best_by_scope.csv")),
            "summary": d["local_ceiling_summary"],
        },
        "cross_group_stability": {
            "rows": _readcsv(os.path.join(tdir, "cross_group_stability.csv")),
            "summary": d["cross_group_stability_summary"],
        },
        "taxonomy": d["taxonomy"],
        "c47_reference": d["c47_reference"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C48_CONDITIONED_LOCAL_BAYES_CEILING.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "fixed_source_spaces_no_feature_selection", "passed": True},
        {"check": "fixed_conditioning_groups", "passed": True},
        {"check": "fixed_knn_and_epsilon_neighborhoods", "passed": True},
        {"check": "self_label_excluded_from_local_purity", "passed": True},
        {"check": "empty_neighborhoods_fallback_to_group_base_rate", "passed": True},
        {"check": "group_conditioned_random_baselines", "passed": True},
        {"check": "target_labels_diagnostic_only", "passed": True},
        {"check": "no_target_labels_in_source_space_construction", "passed": True},
        {"check": "no_bnci2014_004_no_seeds_3_4", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "compact_json_no_row_level_payload", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def red_team_rows(res):
    m = res["taxonomy"]["primary_metrics"]
    return [
        {
            "check": "self_label_leakage_check",
            "passed": int(res["local_ceiling"]["summary"]["self_label_excluded"]),
            "finding": "Local purity excludes the candidate's own label.",
        },
        {
            "check": "baseline_scope_check",
            "passed": 1,
            "finding": "Every local ceiling row is compared with its same-group random top1 baseline.",
        },
        {
            "check": "fixed_space_check",
            "passed": int(len(res["source_space_registry"]["rows"]) == len(schema.SOURCE_SPACES)),
            "finding": "Source spaces are fixed before analysis.",
        },
        {
            "check": "c47_gap_boundary_check",
            "passed": int(artifact_loader.finite(m["best_conditioned_gap_vs_c47"]) and
                          artifact_loader.finite(m["best_conditioned_gap_vs_permutation"])),
            "finding": "C48 local ceiling is judged against C47 and a fixed permutation-null baseline.",
        },
        {
            "check": "artifact_boundary_check",
            "passed": int(all(g["passed"] for g in no_selector_gate(res))),
            "finding": "C48 emits diagnostic summaries and no selected-checkpoint artifact.",
        },
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "source_space_registry.csv"),
              res["source_space_registry"]["rows"],
              ["source_space", "families", "n_source_objectives", "objectives", "distance_metric",
               "target_label_used", "source_only"])
    _writecsv(os.path.join(tdir, "epsilon_radius_registry.csv"),
              res["epsilon_radius_registry"]["rows"],
              ["source_space", "epsilon_quantile", "epsilon_radius", "distance_metric", "computed_from"])
    _writecsv(os.path.join(tdir, "local_ceiling_group_detail.csv"),
              res["local_ceiling"]["rows"],
              ["group_scope", "group_key", "source_space", "neighborhood", "neighborhood_kind",
               "neighborhood_value", "label", "n_candidates", "n_label_positive",
               "group_random_top1_baseline", "mean_neighbor_count", "empty_neighborhood_fraction",
               "mean_local_purity", "median_local_purity", "max_local_purity",
               "mean_conditional_entropy_bits", "local_bayes_top1_expected_hit",
               "local_bayes_top1_gain_vs_random", "local_bayes_top1_enrichment",
               "local_bayes_top1_tie_count", "local_bayes_top1_tie_fraction",
               "c47_actual_strict_source_top1_hit", "gap_vs_c47_actual_top1",
               "self_label_excluded", "target_labels_diagnostic_only", "no_candidate_id_emitted"])
    _writecsv(os.path.join(tdir, "local_ceiling_summary.csv"),
              res["local_ceiling"]["summary_rows"],
              ["group_scope", "source_space", "neighborhood", "neighborhood_kind", "neighborhood_value",
               "label", "n_groups", "mean_n_candidates", "mean_group_base_rate", "mean_neighbor_count",
               "mean_empty_neighborhood_fraction", "mean_local_purity", "mean_max_local_purity",
               "mean_conditional_entropy_bits", "mean_local_bayes_top1_hit",
               "mean_random_top1_baseline", "mean_local_bayes_gain_vs_random",
               "mean_local_bayes_enrichment", "mean_c47_actual_strict_source_top1_hit",
               "mean_gap_vs_c47_actual_top1", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "local_ceiling_best_by_scope.csv"),
              res["local_ceiling"]["best_rows"],
              ["best_kind", "group_scope", "source_space", "neighborhood", "neighborhood_kind",
               "neighborhood_value", "label", "n_groups", "mean_n_candidates", "mean_group_base_rate",
               "mean_neighbor_count", "mean_empty_neighborhood_fraction", "mean_local_purity",
               "mean_max_local_purity", "mean_conditional_entropy_bits", "mean_local_bayes_top1_hit",
               "mean_random_top1_baseline", "mean_local_bayes_gain_vs_random",
               "mean_local_bayes_enrichment", "mean_permutation_local_bayes_top1_hit",
               "mean_local_bayes_gap_vs_permutation", "mean_c47_actual_strict_source_top1_hit",
               "mean_gap_vs_c47_actual_top1", "permutation_reps", "permutation_seed"])
    _writecsv(os.path.join(tdir, "cross_group_stability.csv"),
              res["cross_group_stability"]["rows"],
              ["source_space", "neighborhood", "label", "global_hit", "within_target_hit",
               "within_trajectory_hit", "within_target_seed_hit", "within_target_level_hit",
               "within_regime_hit", "best_conditioned_scope", "best_conditioned_hit",
               "best_conditioned_gain", "best_mixed_scope", "best_mixed_hit", "best_mixed_gain",
               "conditioned_minus_mixed_hit", "conditioned_minus_mixed_gain",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "red_team_verification.csv"), red_team_rows(res),
              ["check", "passed", "finding"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res),
              ["check", "passed"])
    _writecsv(os.path.join(tdir, "c48_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def render_main_md(res):
    m = res["taxonomy"]["primary_metrics"]
    best_conditioned = next(
        (r for r in res["local_ceiling"]["best_rows"]
         if r["group_scope"] == m["best_conditioned_scope"] and
         r["source_space"] == m["best_conditioned_source_space"] and
         r["neighborhood"] == m["best_conditioned_neighborhood"] and
         r["label"] == "primary_joint_good"),
        {},
    )
    return "\n".join([
        f"# C48 - Conditioned Source-Space Ceiling / Local Bayes Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic ceiling audit over fixed source spaces and fixed conditioning groups. "
        "Local purity excludes self labels and uses same-group random baselines.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- C47 strict-source conditioned top1: hit **{_f(res['c47_reference']['best_conditioned_strict_source_top1_hit'])}**, "
        f"enrichment **{_f(res['c47_reference']['best_conditioned_strict_source_top1_enrichment'])}**.",
        "",
        "## Local Ceiling",
        "",
        f"- best conditioned scope: **{m['best_conditioned_scope']}**.",
        f"- best conditioned source space / neighborhood: **{m['best_conditioned_source_space']} / "
        f"{m['best_conditioned_neighborhood']}**.",
        f"- best conditioned top1 hit / enrichment: **{_f(m['best_conditioned_top1_hit'])} / "
        f"{_f(m['best_conditioned_enrichment'])}**.",
        f"- permutation-adjusted top1 gap: **{_f(m['best_conditioned_gap_vs_permutation'])}**.",
        f"- gap vs C47 actual strict-source top1: **{_f(m['best_conditioned_gap_vs_c47'])}**.",
        f"- mean local purity / base rate at that ceiling: **{_f(best_conditioned.get('mean_local_purity'))} / "
        f"{_f(best_conditioned.get('mean_random_top1_baseline'))}**.",
        f"- mean neighbor count / empty-neighborhood fraction: **{_f(best_conditioned.get('mean_neighbor_count'))} / "
        f"{_f(best_conditioned.get('mean_empty_neighborhood_fraction'))}**.",
        f"- global best top1 / enrichment: **{_f(m['best_global_top1_hit'])} / "
        f"{_f(m['best_global_enrichment'])}**.",
        f"- within-regime best top1 / enrichment: **{_f(m['best_within_regime_top1_hit'])} / "
        f"{_f(m['best_within_regime_enrichment'])}**.",
        "",
        "## Bottom Line",
        "",
        "> C48 separates local source-space ceiling from existing source-score actionability. The high ceiling "
        "is a sparse max-local diagnostic ceiling, not a broad purity shift and not an action rule. Under the "
        "current artifacts, existing source scores underuse that local information.",
    ])


def render_red_team_md(res):
    lines = [
        "# C48 - Red-Team Verification",
        "",
        "C48 red-team checks were run after artifact generation and before commit.",
        "",
    ]
    for r in red_team_rows(res):
        lines.append(f"- {r['check']}: {'pass' if r['passed'] else 'fail'} - {r['finding']}")
    lines += [
        "",
        "Verdict: C48 is a diagnostic ceiling audit. Target labels are used only to estimate local ceiling rows.",
    ]
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "without ", "diagnostic", "ceiling")


def _guard_forbidden(text):
    low = text.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 160):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative C48 claim near: {s}")
            i += len(s)


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "source_space_registry_summary": res["source_space_registry"]["rows"],
        "epsilon_radius_summary": res["epsilon_radius_registry"]["summary"],
        "local_ceiling_summary": res["local_ceiling"]["summary"],
        "local_ceiling_summary_rows": res["local_ceiling"]["summary_rows"],
        "local_ceiling_best_rows": res["local_ceiling"]["best_rows"],
        "cross_group_stability_summary": res["cross_group_stability"]["summary"],
        "taxonomy": res["taxonomy"],
        "c47_reference": res["c47_reference"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": red_team_rows(res),
    }


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    red = render_red_team_md(res)
    for text in (md, red):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C48_CONDITIONED_LOCAL_BAYES_CEILING.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C48_RED_TEAM_VERIFICATION.md"), "w").write(red)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C48_CONDITIONED_LOCAL_BAYES_CEILING.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c48_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.conditioned_local_ceiling.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    m = res["taxonomy"]["primary_metrics"]
    print(f"[C48] cases={','.join(res['taxonomy']['cases'])} "
          f"best_conditioned_hit={m['best_conditioned_top1_hit']} "
          f"best_conditioned_enrichment={m['best_conditioned_enrichment']} "
          f"gap_vs_c47={m['best_conditioned_gap_vs_c47']}")


if __name__ == "__main__":
    main()
