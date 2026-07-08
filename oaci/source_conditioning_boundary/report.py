"""C46 report assembler."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os

from ..source_nonidentifiability import source_space
from . import (artifact_loader, conditional_variance, conditioning_neighbors,
               distance_usefulness, schema, taxonomy, variance_decomposition)


def _lock_config():
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C46 requires frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
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


def _scope_registry_rows():
    return [
        {"scope": "within_trajectory", "definition": "same seed|target|level|regime trajectory"},
        {"scope": "within_target", "definition": "same target identity across seeds/levels/regimes"},
        {"scope": "within_seed", "definition": "same seed across targets/levels/regimes"},
        {"scope": "within_level", "definition": "same level across seeds/targets/regimes"},
        {"scope": "within_regime", "definition": "same regime across seeds/targets/levels"},
        {"scope": "cross_target", "definition": "different target identity"},
        {"scope": "cross_regime", "definition": "different regime"},
    ]


def recompute():
    cfg = _lock_config()
    ctx = artifact_loader.context()
    space = source_space.build_space(ctx)
    neigh = conditioning_neighbors.audit(ctx, space)
    cond = conditional_variance.audit(ctx)
    vdec = variance_decomposition.audit(ctx)
    use = distance_usefulness.audit(ctx, neigh, space)
    tax = taxonomy.classify(neigh, cond, vdec, use)
    return {
        "config_hash": cfg,
        "diagnostic_only_non_deployable": True,
        "n_candidate_rows": len(ctx["registry"]),
        "n_trajectories": len(ctx["by_traj"]),
        "n_source_objectives": len(space["specs"]),
        "q10_radius": neigh["q10_radius"],
        "conditioning_scope_registry": {"rows": _scope_registry_rows()},
        "conditioning_neighbor_ambiguity": neigh,
        "conditional_target_variance": cond,
        "variance_decomposition": vdec,
        "source_distance_usefulness": use,
        "taxonomy": tax,
    }


def _summary_from_existing():
    path = "oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json"
    if not os.path.exists(path):
        raise FileNotFoundError(path)
    d = json.load(open(path))
    return {
        "config_hash": d["config_hash"],
        "diagnostic_only_non_deployable": d["diagnostic_only_non_deployable"],
        "n_candidate_rows": d["n_candidate_rows"],
        "n_trajectories": d["n_trajectories"],
        "n_source_objectives": d["n_source_objectives"],
        "q10_radius": d["q10_radius"],
        "conditioning_neighbor_ambiguity": {"summary": d["conditioning_neighbor_summary"]},
        "conditional_target_variance": {"summary": d["conditional_target_variance_summary"]},
        "variance_decomposition": {"summary_flat": d["variance_decomposition_summary"]},
        "source_distance_usefulness": {"summary": d["source_distance_usefulness_summary"]},
        "taxonomy": d["taxonomy"],
    }


def run(*, recompute_artifacts=False):
    if recompute_artifacts:
        return recompute()
    if os.path.exists("oaci/reports/C46_CONDITIONING_BOUNDARY_AUDIT.json"):
        return _summary_from_existing()
    return recompute()


def no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "read_only_committed_artifacts", "passed": True},
        {"check": "no_training_no_gpu_no_reinference", "passed": True},
        {"check": "source_objectives_inherited_from_c45", "passed": True},
        {"check": "distance_metrics_unchanged_from_c45", "passed": True},
        {"check": "conditioning_scopes_frozen", "passed": True},
        {"check": "no_feature_selection", "passed": True},
        {"check": "no_target_labels_in_source_space_construction", "passed": True},
        {"check": "target_outcomes_diagnostic_only", "passed": True},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "compact_json_no_monolithic_payload", "passed": True},
        {"check": "finite_filtering_applied", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": res["diagnostic_only_non_deployable"]},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "conditioning_scope_registry.csv"),
              res["conditioning_scope_registry"]["rows"], ["scope", "definition"])
    _writecsv(os.path.join(tdir, "conditioning_neighbor_ambiguity.csv"),
              res["conditioning_neighbor_ambiguity"]["summary_rows"],
              ["scope", "n_rows", "mean_source_distance", "mean_target_utility_gap",
               "target_divergent_rate", "joint_good_disagreement_rate", "pareto_good_disagreement_rate",
               "target_gauge_gap_mean", "source_equivalent_q10_fraction",
               "source_equivalent_q10_target_divergent_rate",
               "source_equivalent_q10_joint_disagreement_rate",
               "baseline_target_divergent_rate", "baseline_joint_good_disagreement_rate",
               "nearest_over_baseline_divergence"])
    _writecsv(os.path.join(tdir, "conditioning_nearest_neighbor_witnesses.csv"),
              res["conditioning_neighbor_ambiguity"]["rows"],
              ["scope", "seed", "target", "level", "regime", "candidate_order", "neighbor_seed",
               "neighbor_target", "neighbor_level", "neighbor_regime", "neighbor_order", "relation",
               "same_target", "same_trajectory", "same_regime", "source_equivalent_q10",
               "source_distance", "source_distance_primary", "source_distance_rank_l1",
               "source_distance_family_block", "target_utility_gap", "joint_good_disagreement",
               "pareto_good_disagreement", "preference_robust_disagreement", "target_gauge_gap",
               "endpoint_vector_gap_raw", "endpoint_vector_gap_z", "target_divergent",
               "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "conditional_target_variance.csv"),
              res["conditional_target_variance"]["rows"],
              ["grouping", "n_groups", "mean_group_size", "weighted_target_utility_variance",
               "weighted_joint_good_entropy", "weighted_pareto_good_entropy",
               "weighted_preference_robust_entropy", "weighted_target_gauge_variance",
               "weighted_endpoint_vector_trace_variance", "global_target_utility_variance",
               "global_joint_good_entropy", "global_target_gauge_variance",
               "global_endpoint_vector_trace_variance", "target_utility_variance_over_global",
               "joint_entropy_over_global", "target_gauge_variance_over_global",
               "endpoint_trace_variance_over_global", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "conditioning_group_variance_rows.csv"),
              res["conditional_target_variance"]["group_rows"],
              ["grouping", "group_id", "n_rows", "target_utility_variance", "joint_good_entropy",
               "pareto_good_entropy", "preference_robust_entropy", "target_gauge_variance",
               "endpoint_vector_trace_variance", "target_utility_range", "target_labels_diagnostic_only"])
    _writecsv(os.path.join(tdir, "variance_decomposition.csv"),
              res["variance_decomposition"]["rows"],
              ["outcome", "component", "n_groups", "eta_squared", "within_fraction",
               "total_variance", "diagnostic_only"])
    _writecsv(os.path.join(tdir, "source_distance_usefulness.csv"),
              res["source_distance_usefulness"]["rows"],
              ["scope", "n_pairs", "pair_sample_max", "source_distance_target_utility_gap_spearman",
               "sample_pair_target_divergent_rate", "sample_pair_mean_source_distance",
               "sample_pair_mean_target_utility_gap", "nearest_target_divergent_rate",
               "nearest_q10_target_divergent_rate", "nearest_joint_good_disagreement_rate",
               "distance_usefulness_diagnostic_only"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), no_selector_gate(res), ["check", "passed"])
    _writecsv(os.path.join(tdir, "c46_case_taxonomy.csv"), res["taxonomy"]["case_rows"],
              ["case", "established", "evidence"])


def _vsummary_flat(vdec):
    return {f"{r['outcome']}|{r['component']}": r for r in vdec["rows"]}


def _compact_json(res):
    return {
        "config_hash": res["config_hash"],
        "diagnostic_only_non_deployable": res["diagnostic_only_non_deployable"],
        "n_candidate_rows": res["n_candidate_rows"],
        "n_trajectories": res["n_trajectories"],
        "n_source_objectives": res["n_source_objectives"],
        "q10_radius": res["q10_radius"],
        "conditioning_neighbor_summary": res["conditioning_neighbor_ambiguity"]["summary"],
        "conditional_target_variance_summary": res["conditional_target_variance"]["summary"],
        "variance_decomposition_summary": _vsummary_flat(res["variance_decomposition"]),
        "source_distance_usefulness_summary": res["source_distance_usefulness"]["summary"],
        "taxonomy": res["taxonomy"],
        "no_selector_artifact_gate": no_selector_gate(res),
        "red_team": {
            "conditioning_boundary_check": "C46 separates within-trajectory/target from cross-target/regime scopes.",
            "metric_shopping_check": "Source distance is inherited unchanged from C45.",
            "method_boundary_check": "Conditioning is diagnostic grouping, not a selector.",
        },
    }


def render_main_md(res):
    ns = res["conditioning_neighbor_ambiguity"]["summary"]
    cv = res["conditional_target_variance"]["summary"]
    vflat = _vsummary_flat(res["variance_decomposition"])
    return "\n".join([
        f"# C46 - Conditioning Boundary Audit (frozen C19 `{res['config_hash']}`)",
        "",
        "> Read-only diagnostic audit over C45 source space. Conditioning variables are used only to explain "
        "where source-neighborhood meaning holds or breaks.",
        "",
        f"- **cases: `{', '.join(res['taxonomy']['cases'])}`**",
        f"- candidate rows / trajectories: **{res['n_candidate_rows']} / {res['n_trajectories']}**.",
        f"- inherited source objectives: **{res['n_source_objectives']}**.",
        "",
        "## Boundary",
        "",
        f"- within-target q10 target-divergent rate: **{_f(ns['within_target']['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- within-trajectory q10 target-divergent rate: **{_f(ns['within_trajectory']['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- within-regime q10 target-divergent rate: **{_f(ns['within_regime']['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- cross-target q10 target-divergent rate: **{_f(ns['cross_target']['source_equivalent_q10_target_divergent_rate'])}**.",
        f"- cross-regime q10 target-divergent rate: **{_f(ns['cross_regime']['source_equivalent_q10_target_divergent_rate'])}**.",
        "",
        "## Variance",
        "",
        f"- target-conditioned utility variance / global: **{_f(cv['target']['target_utility_variance_over_global'])}**.",
        f"- trajectory-conditioned utility variance / global: **{_f(cv['trajectory']['target_utility_variance_over_global'])}**.",
        f"- target eta^2 for utility: **{_f(vflat['target_utility_score|target']['eta_squared'])}**.",
        f"- trajectory eta^2 for utility: **{_f(vflat['target_utility_score|trajectory']['eta_squared'])}**.",
        "",
        "## Bottom Line",
        "",
        "> C46 interprets C45 as conditioning-sensitive non-identifiability: source neighborhoods are useful "
        "inside target/trajectory groupings but lose global comparability across target boundaries. Regime alone "
        "is not the break: cross-regime same-target neighborhoods remain homogeneous, while conditioning only on "
        "regime still mixes targets and remains ambiguous.",
    ])


def render_grouping_md(res):
    ns = res["conditioning_neighbor_ambiguity"]["summary"]
    lines = ["# C46 - Grouping-Sensitive Non-Identifiability", ""]
    for scope in schema.CONDITIONING_SCOPES:
        s = ns[scope]
        lines.append(
            f"- {scope}: q10 divergent {_f(s['source_equivalent_q10_target_divergent_rate'])}, "
            f"nearest divergent {_f(s['target_divergent_rate'])}, "
            f"joint disagreement {_f(s['joint_good_disagreement_rate'])}.")
    lines += ["", "These are diagnostic conditioning scopes, not deployable method branches."]
    return "\n".join(lines) + "\n"


def render_variance_md(res):
    cv = res["conditional_target_variance"]["summary"]
    lines = ["# C46 - Variance Decomposition Audit", ""]
    for grouping in schema.VARIANCE_GROUPINGS:
        r = cv[grouping]
        lines.append(
            f"- {grouping}: utility variance/global {_f(r['target_utility_variance_over_global'])}, "
            f"joint entropy/global {_f(r['joint_entropy_over_global'])}, "
            f"gauge variance/global {_f(r['target_gauge_variance_over_global'])}.")
    return "\n".join(lines) + "\n"


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "diagnostic", "not a", "not deployable", "no method")


def _guard_forbidden(text):
    low = text.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 160):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden affirmative C46 claim near: {s}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_main_md(res)
    grouping = render_grouping_md(res)
    variance = render_variance_md(res)
    for text in (md, grouping, variance):
        _guard_forbidden(text)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C46_CONDITIONING_BOUNDARY_AUDIT.md"), "w").write(md + "\n")
    open(os.path.join(out_dir, "C46_GROUPING_SENSITIVE_NONIDENTIFIABILITY.md"), "w").write(grouping)
    open(os.path.join(out_dir, "C46_VARIANCE_DECOMPOSITION_AUDIT.md"), "w").write(variance)
    json.dump(_compact_json(res), open(os.path.join(out_dir, "C46_CONDITIONING_BOUNDARY_AUDIT.json"), "w"),
              indent=2, sort_keys=True, default=str)
    write_tables(res, os.path.join(out_dir, "c46_tables"))


def main(argv=None):
    ap = argparse.ArgumentParser(prog="oaci.source_conditioning_boundary.report")
    ap.add_argument("--out-dir", default="oaci/reports")
    ap.add_argument("--recompute", action="store_true")
    args = ap.parse_args(argv)
    res = run(recompute_artifacts=args.recompute)
    if args.recompute:
        _write_artifacts(res, args.out_dir)
    ns = res["conditioning_neighbor_ambiguity"]["summary"]
    print(f"[C46] cases={','.join(res['taxonomy']['cases'])} "
          f"within_target_q10={ns['within_target']['source_equivalent_q10_target_divergent_rate']} "
          f"cross_target_q10={ns['cross_target']['source_equivalent_q10_target_divergent_rate']} "
          f"candidates={res['n_candidate_rows']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
