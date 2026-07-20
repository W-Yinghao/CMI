"""C25 — assemble the Target-Unlabeled Gauge Mechanism + Grouping Boundary Audit. Read-only: locks the C19
config hash, reads the C22 score sidecar + C24 target-unlabeled sidecar, decomposes the weak R3 recovery across
the FROZEN families (Q1), audits target-identity signature vs marginal geometry (Q2), diagnoses the R4 source-
interference collapse (Q3), and lays the grouping problem-class ladder (Q4). No re-inference, no probe tuning,
no feature selection, no selector. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

from ..information_ladder import few_label_calibration
from ..information_ladder import target_unlabeled_features as tuf
from ..score_gauge import gauge_feature_registry as gfr
from ..score_gauge import identity_leakage_audit as ila
from ..score_gauge import offset_model
from ..score_gauge.ceiling_ladder import _pooled_auc, _within_target_mean
from . import (artifact_loader, family_registry, grouping_boundary, identity_signature, r3_family_decomposition,
               r4_interference, schema, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C25 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(scores_sidecar=None, target_unlabeled_sidecar=None) -> dict:
    cfg = _lock_config()
    rows = artifact_loader.load_scores(scores_sidecar)
    reinf = artifact_loader.load_target_unlabeled(target_unlabeled_sidecar)
    mode = "in_regime"
    mr = [r for r in rows if r["mode"] == mode]
    r3_gt, r3_names = artifact_loader.r3_gauge(rows, reinf, mode)
    family_registry.assert_partition(r3_names)
    source_gt = gfr.build_gauge_table(rows, mode)
    raw = _pooled_auc(mr)
    tmean = {t: float(sum(c["score"] for c in mr if c["target"] == t) / max(1, sum(1 for c in mr if c["target"] == t)))
             for t in {r["target"] for r in mr}}
    oracle = _pooled_auc(mr, subtract=lambda r: tmean[r["target"]])
    within_ceiling = _within_target_mean(mr)
    # R1 source-only gap
    sfit = offset_model.fit_offsets(source_gt)
    r1_auc = _pooled_auc(mr, subtract=lambda r: sfit["offset_hat_loto"].get(r["target"], 0.0))
    r1_gap = ((r1_auc - raw) / (oracle - raw)) if (oracle - raw) > 1e-6 else None
    # Full R3 recovery + permutation (the identity control)
    full = tuf.r3_loto_permutation(rows, r3_gt, r3_names, mode, raw, oracle)
    r3_gap = full["gap_closed"]; survives = full["survives_permutation"]
    # Q1 family decomposition
    fam_only = r3_family_decomposition.family_only(rows, r3_gt, mode, raw, oracle)
    lofo = r3_family_decomposition.leave_one_family_out(rows, r3_gt, mode, raw, oracle)
    shap = r3_family_decomposition.shapley(rows, r3_gt, mode, raw, oracle)
    regime_stab = r3_family_decomposition.per_regime_stability(rows, r3_gt, mode, r3_names)
    # Q2 identity signature
    joined = artifact_loader.per_candidate_join(rows, reinf, mode)
    src_id = ila.identity_leakage_audit(rows, mode)["target_id_accuracy_from_source_features"]
    identity = identity_signature.identity_signature_audit(joined, fam_only, survives, src_id)
    # Q3 R4 interference
    r4 = r4_interference.interference_audit(rows, source_gt, gfr.gauge_feature_names(), r3_gt, r3_names, mode, raw, oracle)
    # Q4 grouping boundary
    r5_refine = few_label_calibration.few_label_curve(rows, mode)["max_gap_closed"]
    grouping = grouping_boundary.grouping_boundary(r1_gap, r3_gap, 1.0, r5_refine, within_ceiling)
    tax = taxonomy.gauge_taxonomy(shap, identity, r4, grouping)
    return {"config_hash": cfg, "mode": mode, "n_candidates": len(joined),
            "raw_pooled": raw, "target_centered_oracle": oracle, "within_target_ceiling": within_ceiling,
            "r1_source_gap": r1_gap, "r3_full": full, "family_only": fam_only, "leave_one_family_out": lofo,
            "shapley": shap, "per_regime_stability": regime_stab, "identity_signature": identity,
            "r4_interference": r4, "grouping_boundary": grouping, "taxonomy": tax,
            "not_computed_families": list(schema.NOT_COMPUTED_FAMILIES), "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "r3_feature_family_registry.csv"), family_registry.feature_family_rows(),
              ["feature", "family", "computed_in_recovering_r3"])
    _writecsv(os.path.join(tdir, "r3_family_only_gap_closure.csv"), res["family_only"],
              ["family", "n_features", "gap_closed", "auc_improve", "perm_p", "survives_permutation", "loto_r2"])
    _writecsv(os.path.join(tdir, "r3_leave_one_family_out.csv"), res["leave_one_family_out"],
              ["dropped_family", "remaining_features", "gap_closed", "perm_p", "survives_permutation"])
    sh = res["shapley"]
    _writecsv(os.path.join(tdir, "r3_family_orthogonal_contributions.csv"),
              [{"family": f, "shapley_gap_closure": sh["shapley"][f], "positive_share": sh["positive_share"][f]}
               for f in sh["shapley"]], ["family", "shapley_gap_closure", "positive_share"])
    idn = res["identity_signature"]
    _writecsv(os.path.join(tdir, "r3_target_identity_signature.csv"), idn["per_family"] +
              [{"family": "ALL_R3", "target_id_accuracy": idn["r3_target_id_accuracy"], "gap_closed": res["r3_full"]["gap_closed"]},
               {"family": "source(reference)", "target_id_accuracy": idn["source_target_id_accuracy"], "gap_closed": res["r1_source_gap"]}],
              ["family", "target_id_accuracy", "gap_closed", "survives_permutation"])
    _writecsv(os.path.join(tdir, "r3_identity_controlled_recovery.csv"),
              [{"metric": "r3_target_id_accuracy", "value": idn["r3_target_id_accuracy"]},
               {"metric": "chance", "value": idn["chance"]},
               {"metric": "r3_features_identity_separable", "value": idn["r3_features_identity_separable"]},
               {"metric": "recovery_survives_loto_permutation", "value": idn["recovery_survives_loto_permutation"]},
               {"metric": "recovery_identity_dissociated", "value": idn["recovery_identity_dissociated"]},
               {"metric": "identity_signature_dominates", "value": idn["identity_signature_dominates"]}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "r3_per_target_regime_stability.csv"), res["per_regime_stability"],
              ["regime", "raw_pooled", "gauge_pooled", "oracle", "gap_closed"])
    r4 = res["r4_interference"]
    _writecsv(os.path.join(tdir, "r4_source_interference_diagnostics.csv"),
              [{"metric": k, "value": r4[k]} for k in ("r3_gap", "r4_gap", "n_source_features", "n_target_features",
               "source_coef_norm_share", "source_hijacks_ridge", "condition_number_r3", "condition_number_r4",
               "random_dim_control_mean_gap", "random_dims_also_collapse", "mechanism")], ["metric", "value"])
    _writecsv(os.path.join(tdir, "source_vs_target_family_alignment.csv"),
              [{"metric": "source_coef_norm", "value": r4["source_coef_norm"]},
               {"metric": "target_coef_norm", "value": r4["target_coef_norm"]},
               {"metric": "source_coef_norm_share", "value": r4["source_coef_norm_share"]},
               {"metric": "source_target_offset_alignment", "value": r4["source_target_offset_alignment"]}],
              ["metric", "value"])
    gb = res["grouping_boundary"]
    _writecsv(os.path.join(tdir, "grouping_information_boundary.csv"),
              [{"metric": "grouping_value_over_marginal", "value": gb["grouping_value_over_marginal"]},
               {"metric": "within_target_ceiling", "value": gb["within_target_ceiling"]},
               {"metric": "grouping_is_separate_problem_class", "value": gb["grouping_is_separate_problem_class"]}],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "problem_class_ladder.csv"), gb["ladder"],
              ["rung", "target_inputs", "target_grouping", "target_labels", "uses_held_out_target_scores",
               "deployable_transductively", "recovers_gap_closed", "recovers"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c25_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "established": ";".join(t["established"]),
                "dominant_family": t["dominant_family"], "dominant_share": t["dominant_share"],
                "r4_mechanism": t["r4_mechanism"], "interpretation": t["interpretation"]}],
              ["primary_case", "established", "dominant_family", "dominant_share", "r4_mechanism", "interpretation"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; sh = res["shapley"]; idn = res["identity_signature"]; r4 = res["r4_interference"]
    gb = res["grouping_boundary"]
    L = [f"# C25 — Target-Unlabeled Gauge Mechanism + Grouping Boundary Audit (frozen C19 `{res['config_hash']}`)", "",
         "> Read-only MECHANISM audit of the C24 result (R1 source HURTS, R3 target-unlabeled PARTIALLY recovers, "
         "R4 source+target-unlabeled COLLAPSES, R6 target-grouping FULLY recovers). No re-inference, no probe "
         "tuning, no feature selection, no selector. DIAGNOSTIC-ONLY.", "",
         f"- **PRIMARY: `{t['primary_case']}`** — {t['interpretation']}",
         f"- established: **{', '.join(t['established'])}**", "",
         "## HARD GATE — target-identity signature (reported before any positive mechanism claim)", "",
         f"- R3-feature 9-way target-id accuracy **{_f(idn['r3_target_id_accuracy'])}** vs chance "
         f"**{_f(idn['chance'])}** (source ref {_f(idn['source_target_id_accuracy'])}) → identity-separable: "
         f"**{idn['r3_features_identity_separable']}** (expected for per-target moments).",
         f"- recovery SURVIVES the LOTO offset-permutation control: **{idn['recovery_survives_loto_permutation']}** "
         f"→ identity signature dominates: **{idn['identity_signature_dominates']}**. {idn['note']}",
         f"- **DISCLOSED entanglement**: the carrying family (**{idn['recovering_family']}**) is ALSO the most "
         f"identity-predictive family (**{idn['most_identity_family']}**) → dissociated: "
         f"**{idn['recovery_identity_dissociated']}**. The recovery is entangled with target-identity structure; "
         f"the permutation control is what distinguishes a transferable marginal relationship (survives) from a "
         f"pure fingerprint (would not). Not over-claimed as identity-free.", "",
         "## Q1 — which target-unlabeled family carries the weak R3 recovery?", "",
         f"- full R3 gap closed **{_f(res['r3_full']['gap_closed'])}** (perm p {_f(res['r3_full']['auc_improve_perm_p'])}).",
         "", "| family | family-only gap | perm p | survives | Shapley gap | share |", "|---|---:|---:|:--:|---:|---:|"]
    fo = {f["family"]: f for f in res["family_only"]}
    for fam in schema.FAMILIES:
        L.append(f"| {fam} | {_f(fo[fam]['gap_closed'])} | {_f(fo[fam]['perm_p'])} | {fo[fam]['survives_permutation']} | "
                 f"{_f(sh['shapley'][fam])} | {_f(sh['positive_share'][fam])} |")
    L += ["", f"- Shapley dominant family **{sh['dominant_family']}** (share {_f(sh['dominant_share'])}); single "
          f"family dominates (≥{schema.DOMINANT_FAMILY_SHARE}): **{sh['single_family_dominates']}**.",
          f"- leave-one-family-out: " + ", ".join(f"−{r['dropped_family']}→{_f(r['gap_closed'])}" for r in res["leave_one_family_out"]),
          f"- NOT-computed families (out of scope; would need target-Z re-inference): {', '.join(res['not_computed_families'])}.", "",
          "## Q3 — why do source features DESTROY R3 in R4?", "",
          f"- R3 gap **{_f(r4['r3_gap'])}** → R4 (source+target-unlabeled) gap **{_f(r4['r4_gap'])}** (collapse).",
          f"- source coef-norm share **{_f(r4['source_coef_norm_share'])}** (hijacks ridge: {r4['source_hijacks_ridge']}); "
          f"cond# R3 {_f(r4['condition_number_r3'])} → R4 {_f(r4['condition_number_r4'])}; source↔target offset "
          f"alignment {_f(r4['source_target_offset_alignment'])}.",
          f"- **RANDOM-DIM control**: adding {r4['n_source_features']} random noise dims to R3 → mean gap "
          f"**{_f(r4['random_dim_control_mean_gap'])}** → random dims also collapse: "
          f"**{r4['random_dims_also_collapse']}** → mechanism: **`{r4['mechanism']}`**. {r4['note']}", "",
          "## Q4 — grouping problem-class boundary", "",
          "| problem class | tgt inputs | grouping | labels | uses held-out scores | gap | deployable-transductive |",
          "|---|:--:|:--:|:--:|:--:|---:|:--:|"]
    for row in gb["ladder"]:
        L.append(f"| {row['rung']} | {row['target_inputs']} | {row['target_grouping']} | {row['target_labels']} | "
                 f"{row['uses_held_out_target_scores']} | {_f(row['recovers_gap_closed'])} | {row['deployable_transductively']} |")
    L += ["", f"- value of GROUPING beyond target-unlabeled marginal geometry (R6−R3): "
          f"**{_f(gb['grouping_value_over_marginal'])}**; within-target ceiling {_f(gb['within_target_ceiling'])}.",
          f"- {gb['boundary']}", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY mechanism audit. Families are FROZEN (no feature selection). Target grouping is NOT "
          "source-only DG and the target-centered oracle is NOT a deployable selector. Target labels never "
          "entered the R3 feature construction."]
    return "\n".join(L)


def render_identity_md(res) -> str:
    idn = res["identity_signature"]
    lines = [f"# C25 Q2 — R3 target-identity signature audit\n\n> {idn['note']}\n",
             f"- R3-feature target-id accuracy: **{_f(idn['r3_target_id_accuracy'])}** (chance {_f(idn['chance'])}, "
             f"source ref {_f(idn['source_target_id_accuracy'])}); identity-separable: {idn['r3_features_identity_separable']}",
             f"- recovery survives LOTO offset-permutation: **{idn['recovery_survives_loto_permutation']}**; "
             f"identity signature dominates: **{idn['identity_signature_dominates']}**",
             f"- recovering family **{idn['recovering_family']}** vs most-identity family **{idn['most_identity_family']}** "
             f"→ dissociated: **{idn['recovery_identity_dissociated']}**\n",
             "| family | target-id acc | gap closed | survives perm |", "|---|---:|---:|:--:|"]
    for r in idn["per_family"]:
        lines.append(f"| {r['family']} | {_f(r['target_id_accuracy'])} | {_f(r['gap_closed'])} | {r['survives_permutation']} |")
    return "\n".join(lines)


def render_r4_md(res) -> str:
    r4 = res["r4_interference"]
    return (f"# C25 Q3 — R4 source-interference audit\n\n> {r4['note']}\n\n"
            f"- R3 gap {_f(r4['r3_gap'])} → R4 gap {_f(r4['r4_gap'])} (source+target-unlabeled collapses)\n"
            f"- source features {r4['n_source_features']} vs target-unlabeled {r4['n_target_features']}; source "
            f"coef-norm share {_f(r4['source_coef_norm_share'])} (hijacks ridge: {r4['source_hijacks_ridge']})\n"
            f"- condition number R3 {_f(r4['condition_number_r3'])} → R4 {_f(r4['condition_number_r4'])}\n"
            f"- source↔target offset-prediction alignment: {_f(r4['source_target_offset_alignment'])}\n"
            f"- RANDOM-DIM control ({r4['n_source_features']} noise dims): mean gap {_f(r4['random_dim_control_mean_gap'])}; "
            f"random dims also collapse: {r4['random_dims_also_collapse']}\n\n"
            f"**Mechanism: `{r4['mechanism']}`**\n")


def render_grouping_md(res) -> str:
    gb = res["grouping_boundary"]
    lines = [f"# C25 Q4 — grouping problem-class boundary\n\n> {gb['boundary']}\n\n{gb['note']}\n",
             f"- value of grouping over marginal (R6−R3): {_f(gb['grouping_value_over_marginal'])}; within-target "
             f"ceiling {_f(gb['within_target_ceiling'])}\n",
             "| problem class | target inputs | grouping | labels | uses held-out scores | gap closed |",
             "|---|:--:|:--:|:--:|:--:|---:|"]
    for row in gb["ladder"]:
        lines.append(f"| {row['rung']} | {row['target_inputs']} | {row['target_grouping']} | {row['target_labels']} | "
                     f"{row['uses_held_out_target_scores']} | {_f(row['recovers_gap_closed'])} |")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ",
             "not a", "not established", "fails", "barred", "instead of", "not source-only", "not deployable",
             "not dg", "never")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 30):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C25 report near: ...{low[max(0, i - 30):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    idm = render_identity_md(res); _guard_forbidden(idm)
    r4m = render_r4_md(res); _guard_forbidden(r4m)
    gbm = render_grouping_md(res); _guard_forbidden(gbm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C25_TARGET_UNLABELED_GAUGE_MECHANISM.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C25_TARGET_UNLABELED_GAUGE_MECHANISM.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C25_R3_IDENTITY_SIGNATURE_AUDIT.md"), "w").write(idm)
    open(os.path.join(out_dir, "C25_R4_SOURCE_INTERFERENCE_AUDIT.md"), "w").write(r4m)
    open(os.path.join(out_dir, "C25_GROUPING_BOUNDARY_AUDIT.md"), "w").write(gbm)
    write_tables(res, os.path.join(out_dir, "c25_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.unlabeled_gauge.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--target-unlabeled-sidecar", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.target_unlabeled_sidecar)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; sh = res["shapley"]; r4 = res["r4_interference"]; idn = res["identity_signature"]
    print(f"[C25] primary={t['primary_case']} | dominant_family={sh['dominant_family']}({_f(sh['dominant_share'])}) "
          f"single_dominates={sh['single_family_dominates']} | id_acc={_f(idn['r3_target_id_accuracy'])} "
          f"id_dominates={idn['identity_signature_dominates']} | r4_mechanism={r4['mechanism']} "
          f"random_collapse={r4['random_dims_also_collapse']} | established={','.join(t['established'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
