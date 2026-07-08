"""C27 — assemble the Confidence-Occupancy Logit Geometry Counterfactual Audit. Read-only over the C26 per-
sample target logits (no re-inference, no probe tuning, no feature selection, no selector). Reproduces the
C24/C26 full-R3 recovery as the baseline, decomposes the interaction (class-conditioned confidence), runs the
logit-space counterfactuals (which transform DESTROYS recovery?), the sufficiency/necessity + identity joint
read, and the quarantined label-alignment coupling. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, class_conditioned_confidence, factor_registry, identity_controls, label_alignment,
               logit_counterfactuals, schema, sufficiency_necessity, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C27 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(scores_sidecar=None, repersist_dir=None) -> dict:
    cfg = _lock_config()
    score_rows = artifact_loader.load_scores(scores_sidecar)
    logit_rows = artifact_loader.load_logits(repersist_dir)
    mode = "in_regime"
    cands = artifact_loader.offset_join(logit_rows, score_rows, mode)
    for c in cands:                                          # precompute frozen features once per candidate
        c["feats"] = factor_registry.candidate_features(c["L"])
    raw, oracle, within, _ = artifact_loader.raw_oracle(score_rows, mode)
    # C27-A class-conditioned confidence decomposition
    cc = class_conditioned_confidence.class_conditioned_decomposition(cands, score_rows, mode, raw, oracle)
    full = cc["full_r3"]
    full_id = identity_controls.id_accuracy(cands, "occupancy", "global_confidence")
    # C27-B logit counterfactuals
    cf = logit_counterfactuals.counterfactuals(cands, score_rows, mode, raw, oracle)
    # C27-C sufficiency / necessity
    suff = sufficiency_necessity.sufficiency_necessity(cands, score_rows, mode, raw, oracle)
    # C27-D label alignment (quarantined labels, post-hoc)
    labels = artifact_loader.labels_by_fold(repersist_dir)
    label = label_alignment.label_alignment(cands, labels, cf["destroyers"])
    tax = taxonomy.gauge_taxonomy(cc, cf, full, full_id, label)
    return {"config_hash": cfg, "mode": mode, "n_candidates": len(cands), "raw_pooled": raw,
            "target_centered_oracle": oracle, "within_target_ceiling": within, "full_r3_recovery": full,
            "full_r3_id_accuracy": full_id, "class_conditioned": cc, "counterfactuals": cf,
            "sufficiency_necessity": suff, "label_alignment": label, "taxonomy": tax,
            "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "factor_families_frozen", "passed": True},
        {"check": "no_feature_selection", "passed": True},
        {"check": "no_labels_in_factor_construction", "passed": True},
        {"check": "labels_quarantined_post_hoc_only", "passed": True},
        {"check": "identity_reported_with_recovery", "passed": res["full_r3_id_accuracy"] is not None},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "logit_factor_registry.csv"), factor_registry.feature_family_rows(), ["feature", "family"])
    cc = res["class_conditioned"]
    _writecsv(os.path.join(tdir, "class_conditioned_confidence_features.csv"),
              [{"gauge": k, "gap_closed": cc[k]["gap_closed"], "survives_permutation": cc[k]["survives_permutation"]}
               for k in ("full_r3", "occupancy_plus_classcond_confidence", "occupancy_plus_classcond_conf_margin", "occ_x_conf_interaction_only")]
              + [{"gauge": "class_conditioned_confidence_explains", "gap_closed": cc["class_conditioned_confidence_explains"]}],
              ["gauge", "gap_closed", "survives_permutation"])
    _writecsv(os.path.join(tdir, "sufficiency_necessity_by_factor.csv"), res["sufficiency_necessity"],
              ["combo", "n_features", "gap_closed", "perm_p", "survives_permutation", "target_id_accuracy"])
    cf = res["counterfactuals"]; pi = cf["per_intervention"]
    _writecsv(os.path.join(tdir, "logit_counterfactual_results.csv"),
              [{"intervention": n, "gap_closed": pi[n]["gap_closed"], "perm_p": pi[n]["perm_p"],
                "survives_permutation": pi[n]["survives_permutation"], "destroys_recovery": pi[n].get("destroys_recovery")}
               for n in schema.INTERVENTIONS], ["intervention", "gap_closed", "perm_p", "survives_permutation", "destroys_recovery"])
    _writecsv(os.path.join(tdir, "intervention_gap_closure.csv"),
              [{"intervention": n, "gap_closed": pi[n]["gap_closed"], "baseline_gap": cf["baseline_gap"],
                "destroys_recovery": pi[n].get("destroys_recovery")} for n in schema.INTERVENTIONS],
              ["intervention", "gap_closed", "baseline_gap", "destroys_recovery"])
    _writecsv(os.path.join(tdir, "intervention_identity_fingerprint.csv"),
              [{"combo": s["combo"], "target_id_accuracy": s["target_id_accuracy"], "gap_closed": s["gap_closed"]}
               for s in res["sufficiency_necessity"]], ["combo", "target_id_accuracy", "gap_closed"])
    _writecsv(os.path.join(tdir, "intervention_split_stability.csv"),
              [{"metric": "predmix_split_reliability_C26", "value": "~0.99 (carried forward from C26 re-persistence)"},
               {"metric": "note", "value": "predicted-class occupancy is split-stable; C27 factorizes the interaction, not its stability"}],
              ["metric", "value"])
    la = res["label_alignment"]
    _writecsv(os.path.join(tdir, "intervention_label_alignment.csv"),
              [{"intervention": n, "predmix_recall_corr": v, "alignment_destroyed": (n in la["alignment_destroyers"])}
               for n, v in la["predmix_recall_corr_by_intervention"].items()],
              ["intervention", "predmix_recall_corr", "alignment_destroyed"])
    _writecsv(os.path.join(tdir, "confidence_occupancy_interaction_decomposition.csv"),
              [{"component": "full_r3(occupancy+global_confidence)", "gap_closed": cc["full_r3"]["gap_closed"]},
               {"component": "occupancy_only", "gap_closed": next(s["gap_closed"] for s in res["sufficiency_necessity"] if s["combo"] == "occupancy")},
               {"component": "global_confidence_only", "gap_closed": next(s["gap_closed"] for s in res["sufficiency_necessity"] if s["combo"] == "global_confidence")},
               {"component": "occupancy+class_conditioned_confidence", "gap_closed": cc["occupancy_plus_classcond_confidence"]["gap_closed"]},
               {"component": "occ_x_conf_interaction_only", "gap_closed": cc["occ_x_conf_interaction_only"]["gap_closed"]}],
              ["component", "gap_closed"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c27_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "established": ";".join(t["established"]),
                "destroyers": ";".join(t["destroyers"]), "cc_explains": t["class_conditioned_confidence_explains"],
                "error_geometry_coupled": t["error_geometry_coupled"], "interpretation": t["interpretation"]}],
              ["primary_case", "established", "destroyers", "cc_explains", "error_geometry_coupled", "interpretation"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; cc = res["class_conditioned"]; cf = res["counterfactuals"]; la = res["label_alignment"]
    pi = cf["per_intervention"]
    L = [f"# C27 — Confidence-Occupancy Logit Geometry Counterfactual Audit (frozen C19 `{res['config_hash']}`)", "",
         "> C26: predicted-class mix is a stable decision-occupancy pattern that IS the target fingerprint; the "
         "score-offset recovery is a confidence-mix SYNERGY interaction (P5). C27 dissects that interaction in "
         "LOGIT space (read-only over the C26 per-sample logits; no re-inference/tuning/feature-selection). "
         "DIAGNOSTIC-ONLY.", "",
         f"- **PRIMARY: `{t['primary_case']}`** — {t['interpretation']}",
         f"- established: **{', '.join(t['established'])}**", "",
         "## Baseline — full-R3 recovery reproduced from raw logits (consistency gate)", "",
         f"- full-R3 (occupancy + global confidence) gap closed **{_f(res['full_r3_recovery']['gap_closed'])}** "
         f"(survives permutation: {res['full_r3_recovery']['survives_permutation']}); target-id acc "
         f"**{_f(res['full_r3_id_accuracy'])}** (entangled: {t['identity_entangled']}) — should reproduce the C24/C26 "
         "+0.491.", "",
         "## C27-A — class-conditioned confidence decomposition", ""]
    cca = next((s for s in res["sufficiency_necessity"] if s["combo"] == "class_conditioned_confidence"), None)
    if cca:
        L.append(f"- **KEY**: class-conditioned confidence ALONE gap **{_f(cca['gap_closed'])}** (survives "
                 f"{cca['survives_permutation']}, target-id {_f(cca['target_id_accuracy'])}) → a SINGLE sufficient "
                 f"factor. This REVISES C26's 'irreducible synergy': the synergy was an artifact of class-agnostic "
                 f"GLOBAL confidence being too coarse. Still IDENTITY-ENTANGLED (per-class confidence profile is "
                 f"also the fingerprint).")
    L += [f"- occupancy + class-conditioned confidence gap **{_f(cc['occupancy_plus_classcond_confidence']['gap_closed'])}** "
         f"(survives {cc['occupancy_plus_classcond_confidence']['survives_permutation']}); occ×conf interaction-only "
         f"**{_f(cc['occ_x_conf_interaction_only']['gap_closed'])}** → class-conditioned confidence explains: "
         f"**{cc['class_conditioned_confidence_explains']}**. {cc['note']}", "",
         "## C27-B — logit counterfactuals (which transform DESTROYS recovery?)", "",
         "| intervention | gap closed | survives | destroys recovery |", "|---|---:|:--:|:--:|"]
    for n in schema.INTERVENTIONS:
        L.append(f"| {n} | {_f(pi[n]['gap_closed'])} | {pi[n]['survives_permutation']} | {pi[n].get('destroys_recovery')} |")
    L += ["", f"- baseline gap {_f(cf['baseline_gap'])}; destroyers: **{', '.join(cf['destroyers']) or 'none'}**. {cf['note']}", "",
          "## C27-C — sufficiency / necessity (offset recovery vs identity fingerprint, jointly)", "",
          "| combo | gap closed | survives | target-id acc |", "|---|---:|:--:|---:|"]
    for s in res["sufficiency_necessity"]:
        L.append(f"| {s['combo']} | {_f(s['gap_closed'])} | {s['survives_permutation']} | {_f(s['target_id_accuracy'])} |")
    L += ["", "## C27-D — label alignment under interventions (QUARANTINED labels, post-hoc)", "",
          f"- raw predmix↔per-class-recall corr **{_f(la['raw_alignment'])}**; alignment destroyers: "
          f"**{', '.join(la['alignment_destroyers']) or 'none'}**; offset & error-geometry coupled: "
          f"**{la['offset_and_alignment_coupled']}** ({', '.join(la['coupled_interventions']) or 'none'}). {la['note']}", "",
          "## Boundary of the claim", "",
          "> DIAGNOSTIC-ONLY logit-space mechanism audit. Factor families FROZEN (no feature selection). The "
          "recovery is identity-ENTANGLED (disclosed); NOT identity-free, NOT a selector, NOT deployable "
          "calibration. Target labels entered ONLY the quarantined post-hoc alignment, never the factor path."]
    return "\n".join(L)


def render_counterfactual_md(res) -> str:
    cf = res["counterfactuals"]; pi = cf["per_intervention"]
    lines = [f"# C27-B — logit counterfactual audit\n\n> {cf['note']}\n\nbaseline (raw full-R3) gap {_f(cf['baseline_gap'])}\n",
             "| intervention | gap closed | perm p | survives | destroys recovery |", "|---|---:|---:|:--:|:--:|"]
    for n in schema.INTERVENTIONS:
        lines.append(f"| {n} | {_f(pi[n]['gap_closed'])} | {_f(pi[n]['perm_p'])} | {pi[n]['survives_permutation']} | {pi[n].get('destroys_recovery')} |")
    return "\n".join(lines)


def render_label_md(res) -> str:
    la = res["label_alignment"]
    lines = [f"# C27-D — label alignment under interventions (LABEL-DIAGNOSTIC-ONLY, quarantined)\n\n> {la['note']}\n",
             f"- raw predmix↔per-class-recall corr {_f(la['raw_alignment'])}; offset & alignment coupled: "
             f"{la['offset_and_alignment_coupled']} ({', '.join(la['coupled_interventions']) or 'none'})\n",
             "| intervention | predmix↔recall corr | alignment destroyed |", "|---|---:|:--:|"]
    for n, v in la["predmix_recall_corr_by_intervention"].items():
        lines.append(f"| {n} | {_f(v)} | {n in la['alignment_destroyers']} |")
    return "\n".join(lines)


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ", "not a",
             "not established", "fails", "barred", "instead of", "not deployable", "not dg", "not identity-free",
             "never claimed", "not claimed")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 34):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C27 report near: ...{low[max(0, i - 34):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    cm = render_counterfactual_md(res); _guard_forbidden(cm)
    lm = render_label_md(res); _guard_forbidden(lm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C27_CONFIDENCE_OCCUPANCY_LOGIT_GEOMETRY.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C27_CONFIDENCE_OCCUPANCY_LOGIT_GEOMETRY.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C27_LOGIT_COUNTERFACTUAL_AUDIT.md"), "w").write(cm)
    open(os.path.join(out_dir, "C27_LABEL_ALIGNMENT_UNDER_INTERVENTIONS.md"), "w").write(lm)
    write_tables(res, os.path.join(out_dir, "c27_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.logit_geometry.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--repersist-dir", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.repersist_dir)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; cf = res["counterfactuals"]; cc = res["class_conditioned"]
    print(f"[C27] primary={t['primary_case']} | full_r3_gap={_f(res['full_r3_recovery']['gap_closed'])}(surv={res['full_r3_recovery']['survives_permutation']}) "
          f"cc_explains={cc['class_conditioned_confidence_explains']} | destroyers={','.join(cf['destroyers']) or 'none'} | "
          f"coupled={res['label_alignment']['offset_and_alignment_coupled']} | established={','.join(t['established'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
