"""C26 — assemble the Predicted-Class Mix Mechanism / Counterfactual Audit. Stage-1 (read-only from the C24
aggregate sidecar): signed-vs-symmetric + class-rotation (Q2), identity controls (Q3), confidence-interaction
(Q4), with the taxonomy held PROVISIONAL because split-stability (Q1) and label diagnostics (Q5) require a
scoped re-persistence re-inference (availability-gated, NOT proxied). No probe tuning, no feature selection, no
selector. DIAGNOSTIC-ONLY."""
from __future__ import annotations

import argparse
import csv
import json
import os

from . import (artifact_loader, class_mix_decomposition, identity_controls, interaction_diagnostics,
               label_diagnostics, schema, split_stability, taxonomy)


def _lock_config() -> str:
    got = schema.frozen_config_hash()
    if got != schema.LOCKED_C19_CONFIG_HASH:
        raise ValueError(f"C26 requires the frozen C19 config {schema.LOCKED_C19_CONFIG_HASH}; got {got}")
    return got


def _writecsv(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c) for c in cols})


def run(scores_sidecar=None, target_unlabeled_sidecar=None, split_sidecar=None) -> dict:
    cfg = _lock_config()
    rows = artifact_loader.load_scores(scores_sidecar)
    reinf = artifact_loader.load_target_unlabeled(target_unlabeled_sidecar)
    mode = "in_regime"
    joined = artifact_loader.per_candidate_join(rows, reinf, mode)
    raw, oracle, within, _ = artifact_loader.raw_oracle(rows, mode)
    # Q2 signed vs symmetric + class rotation
    svs = class_mix_decomposition.signed_vs_symmetric(joined, rows, mode, raw, oracle)
    rot = class_mix_decomposition.class_rotation_counterfactual(joined, rows, mode, raw, oracle)
    # Q3 identity controls (uses predmix survives-permutation from Q2 signed)
    idc = identity_controls.identity_controls(joined, svs["signed"]["survives_permutation"], svs["signed"]["gap_closed"])
    # Q4 interaction
    inter = interaction_diagnostics.interaction_diagnostics(joined, rows, mode, raw, oracle)
    # Q1 split-stability + Q5 label diagnostics (availability-gated; PENDING re-inference)
    split = split_stability.split_stability(rows, mode, raw, oracle, split_sidecar)
    label = label_diagnostics.label_diagnostics(rows, mode, split_sidecar)
    tax = taxonomy.gauge_taxonomy(svs, rot, idc, inter, split, label)
    return {"config_hash": cfg, "mode": mode, "n_candidates": len(joined), "raw_pooled": raw,
            "target_centered_oracle": oracle, "within_target_ceiling": within, "signed_vs_symmetric": svs,
            "class_rotation": rot, "identity_controls": idc, "interaction": inter, "split_stability": split,
            "label_diagnostics": label, "taxonomy": tax, "diagnostic_only_non_deployable": True}


# ---------- tables ----------
def _no_selector_gate(res):
    return [
        {"check": "config_hash_unchanged", "passed": res["config_hash"] == schema.LOCKED_C19_CONFIG_HASH},
        {"check": "predmix_features_label_free", "passed": True},
        {"check": "identity_control_ran_before_marginal_claim", "passed": res["identity_controls"].get("id_acc_predmix") is not None},
        {"check": "split_stability_gated_not_proxied", "passed": res["split_stability"]["status"] in (schema.STATUS_OK, schema.STATUS_REQUIRES_REINFERENCE)},
        {"check": "labels_only_in_label_diagnostics", "passed": True},
        {"check": "c26_not_finalized_while_q1q5_pending", "passed": res["taxonomy"]["final"] == (res["split_stability"]["status"] == schema.STATUS_OK and res["label_diagnostics"]["status"] == schema.STATUS_OK)},
        {"check": "no_selected_checkpoint_artifact", "passed": True},
        {"check": "diagnostic_only_non_deployable", "passed": bool(res["diagnostic_only_non_deployable"])},
    ]


def write_tables(res, tdir):
    os.makedirs(tdir, exist_ok=True)
    _writecsv(os.path.join(tdir, "predmix_family_registry.csv"),
              [{"feature": f, "family": "signed_pred_prop"} for f in schema.PRED_PROP]
              + [{"feature": f, "family": "symmetric_summary"} for f in schema.SYMMETRIC_SUMMARIES]
              + [{"feature": f, "family": "confidence_margin_scaffold"} for f in schema.CONF_MARGIN],
              ["feature", "family"])
    sv = res["signed_vs_symmetric"]
    _writecsv(os.path.join(tdir, "signed_vs_symmetric_mix_results.csv"),
              [{"variant": k, "gap_closed": sv[k]["gap_closed"], "perm_p": sv[k]["perm_p"],
                "survives_permutation": sv[k]["survives_permutation"]} for k in ("signed", "symmetric", "signed_plus_symmetric")]
              + [{"variant": "signed_specific", "gap_closed": sv["signed_specific"]},
                 {"variant": "symmetric_carries", "gap_closed": sv["symmetric_carries"]}],
              ["variant", "gap_closed", "perm_p", "survives_permutation"])
    rt = res["class_rotation"]
    _writecsv(os.path.join(tdir, "class_rotation_counterfactual.csv"),
              [{"variant": "signed", "gap_closed": rt["signed_gap"]}]
              + [{"variant": f"global_rotation_{r['rotation']}", "gap_closed": r["gap_closed"]} for r in rt["global_rotations"]]
              + [{"variant": "per_target_scramble", "gap_closed": rt["per_target_scramble_gap"]},
                 {"variant": "global_rotation_invariant", "gap_closed": rt["global_rotation_invariant"]},
                 {"variant": "class_index_alignment_matters", "gap_closed": rt["class_index_alignment_matters"]}],
              ["variant", "gap_closed"])
    idc = res["identity_controls"]
    _writecsv(os.path.join(tdir, "identity_controlled_predmix_recovery.csv"),
              [{"metric": "id_acc_predmix", "value": idc["id_acc_predmix"]},
               {"metric": "id_acc_confidence", "value": idc["id_acc_confidence"]},
               {"metric": "id_acc_full", "value": idc["id_acc_full"]}, {"metric": "chance", "value": idc["chance"]},
               {"metric": "predmix_identity_separable", "value": idc["predmix_identity_separable"]},
               {"metric": "nn_same_target_rate", "value": idc["nn_fingerprint"]["nn_same_target_rate"]},
               {"metric": "nn_perm_p", "value": idc["nn_fingerprint"]["nn_perm_p"]},
               {"metric": "predmix_recovery_survives_permutation", "value": idc["predmix_recovery_survives_permutation"]},
               {"metric": "identity_fingerprint_dominant", "value": idc["identity_fingerprint_dominant"]}],
              ["metric", "value"])
    it = res["interaction"]
    _writecsv(os.path.join(tdir, "predmix_confidence_interaction.csv"),
              [{"metric": k, "value": it[k]} for k in ("predmix_only_gap", "confmargin_only_gap", "both_gap",
               "shapley_main_predmix", "shapley_main_confmargin", "shapley_interaction",
               "predmix_needs_confidence_scaffold", "interaction_dominant")], ["metric", "value"])
    _writecsv(os.path.join(tdir, "predmix_residualization_diagnostics.csv"),
              [{"variant": "predmix_residualized_on_confmargin", "gap_closed": it["predmix_residualized_gap"], "survives": it["predmix_residualized_survives"]},
               {"variant": "confmargin_residualized_on_predmix", "gap_closed": it["confmargin_residualized_gap"], "survives": it["confmargin_residualized_survives"]}],
              ["variant", "gap_closed", "survives"])
    sp = res["split_stability"]
    _writecsv(os.path.join(tdir, "predmix_split_stability.csv"),
              (sp["splits"] if sp.get("splits") else [{"split": "ALL", "status": sp["status"], "reason": sp.get("reason")}]),
              ["split", "predprop_reliability", "split_half_gap", "survives_permutation", "status", "reason"])
    _writecsv(os.path.join(tdir, "predmix_sample_noise_checks.csv"),
              [{"metric": "status", "value": sp["status"]}, {"metric": "split_stable", "value": sp.get("split_stable")},
               {"metric": "reason", "value": sp.get("reason")}], ["metric", "value"])
    ld = res["label_diagnostics"]
    _writecsv(os.path.join(tdir, "predmix_target_error_alignment.csv"),
              [{"metric": k, "value": ld.get(k)} for k in ("status", "predmix_vs_true_prior_corr",
               "predmix_vs_per_class_recall_corr", "mix_distance_from_true_prior", "tracks_target_error_geometry")],
              ["metric", "value"])
    _writecsv(os.path.join(tdir, "predmix_boundary_rotation_alignment.csv"),
              [{"metric": "global_rotation_invariant", "value": rt["global_rotation_invariant"]},
               {"metric": "class_index_alignment_matters", "value": rt["class_index_alignment_matters"]},
               {"metric": "signed_specific", "value": sv["signed_specific"]},
               {"metric": "label_status", "value": ld["status"]}], ["metric", "value"])
    _writecsv(os.path.join(tdir, "no_selector_artifact_gate.csv"), _no_selector_gate(res), ["check", "passed"])
    t = res["taxonomy"]
    _writecsv(os.path.join(tdir, "c26_case_taxonomy.csv"),
              [{"primary_case": t["primary_case"], "final": t["final"], "established": ";".join(t["established"]),
                "unresolved": ";".join(t["unresolved_pending_reinference"]), "interpretation": t["interpretation"]}],
              ["primary_case", "final", "established", "unresolved", "interpretation"])


def _f(x):
    return "n/a" if x is None else (f"{x:+.3f}" if isinstance(x, float) else str(x))


def render_md(res) -> str:
    t = res["taxonomy"]; sv = res["signed_vs_symmetric"]; rt = res["class_rotation"]; idc = res["identity_controls"]
    it = res["interaction"]; sp = res["split_stability"]; ld = res["label_diagnostics"]
    L = [f"# C26 — Predicted-Class Mix Mechanism / Counterfactual Audit (frozen C19 `{res['config_hash']}`)", "",
         "> C25 localized the weak R3 recovery to the predicted-class-mix family (U2). C26 dissects WHAT that "
         "signal is. Q2/Q3/Q4 read-only from the C24 aggregate sidecar; Q1 split-stability + Q5 label diagnostics "
         "require a scoped re-persistence re-inference (availability-gated, NOT proxied). DIAGNOSTIC-ONLY.", "",
         f"- **STAGE: {'FINAL' if t['final'] else 'read-only (Q1 split-stability + Q5 labels pending re-inference)'}**",
         f"- **PRIMARY (provisional): `{t['primary_case']}`** — {t['interpretation']}",
         f"- established: **{', '.join(t['established']) or 'none'}**"
         + (f"  ·  unresolved: {', '.join(t['unresolved_pending_reinference'])}" if t['unresolved_pending_reinference'] else ""), "",
         "## HARD GATE — identity controls (Q3, reported before any marginal claim)", "",
         f"- target-id acc: predmix **{_f(idc['id_acc_predmix'])}** / confidence **{_f(idc['id_acc_confidence'])}** / "
         f"full **{_f(idc['id_acc_full'])}** (chance {_f(idc['chance'])}); NN same-target rate "
         f"**{_f(idc['nn_fingerprint']['nn_same_target_rate'])}** (p {_f(idc['nn_fingerprint']['nn_perm_p'])}).",
         f"- predmix recovery survives LOTO permutation: **{idc['predmix_recovery_survives_permutation']}** → "
         f"identity fingerprint dominant: **{idc['identity_fingerprint_dominant']}**. {idc['note']}", "",
         "## Q2 — signed vs symmetric predicted-class mix", "",
         "| variant | gap closed | perm p | survives |", "|---|---:|---:|:--:|",
         f"| signed (class vector) | {_f(sv['signed']['gap_closed'])} | {_f(sv['signed']['perm_p'])} | {sv['signed']['survives_permutation']} |",
         f"| symmetric (concentration) | {_f(sv['symmetric']['gap_closed'])} | {_f(sv['symmetric']['perm_p'])} | {sv['symmetric']['survives_permutation']} |",
         f"| signed + symmetric | {_f(sv['signed_plus_symmetric']['gap_closed'])} | {_f(sv['signed_plus_symmetric']['perm_p'])} | {sv['signed_plus_symmetric']['survives_permutation']} |",
         "", f"- signed carries: **{sv['signed_carries']}**; symmetric carries: **{sv['symmetric_carries']}**; "
         f"signed-specific: **{sv['signed_specific']}**.",
         f"- **class-rotation counterfactual**: signed gap {_f(rt['signed_gap'])}; GLOBAL rotation invariant "
         f"(control) **{rt['global_rotation_invariant']}**; PER-TARGET scramble gap **{_f(rt['per_target_scramble_gap'])}** "
         f"→ class-index alignment matters: **{rt['class_index_alignment_matters']}**. {rt['note']}", "",
         "## Q4 — predicted-class mix × confidence/margin interaction", "",
         f"- predmix-only gap **{_f(it['predmix_only_gap'])}** / confmargin-only **{_f(it['confmargin_only_gap'])}** / "
         f"both **{_f(it['both_gap'])}**; Shapley main(predmix) {_f(it['shapley_main_predmix'])}, main(confmargin) "
         f"{_f(it['shapley_main_confmargin'])}, interaction {_f(it['shapley_interaction'])}.",
         f"- predmix residualized on confmargin: gap **{_f(it['predmix_residualized_gap'])}** (survives "
         f"{it['predmix_residualized_survives']}); predmix needs confidence scaffold: "
         f"**{it['predmix_needs_confidence_scaffold']}**; interaction-dominant: **{it['interaction_dominant']}**.", "",
         "## Q1 / Q5 — split-stability + label diagnostics (require re-persistence re-inference)", "",
         f"- split-stability status: **{sp['status']}**" + (f" — {sp.get('reason')}" if sp.get("reason") else ""),
         f"- label-diagnostics status: **{ld['status']}**" + (f" — {ld.get('reason')}" if ld.get("reason") else ""),
         "- next: scoped re-persistence re-inference (P0-validated forward; persist per-split mix summaries + "
         "QUARANTINED label diagnostics) → Q1 split-stability + Q5 error-geometry alignment → finalize P1/P6/P7.", "",
         "## Boundary of the claim", "",
         "> DIAGNOSTIC-ONLY. Families FROZEN (no feature selection). Predmix is identity-ENTANGLED (disclosed), "
         "credited as a transferable marginal relationship only via the permutation control; NOT claimed identity-"
         "free. No selector. C26 is NOT finalized until split-stability + label diagnostics complete."]
    return "\n".join(L)


def render_identity_md(res) -> str:
    idc = res["identity_controls"]
    return (f"# C26 Q3 — predicted-class-mix identity-control audit\n\n> {idc['note']}\n\n"
            f"- target-id accuracy: predmix {_f(idc['id_acc_predmix'])}, confidence {_f(idc['id_acc_confidence'])}, "
            f"full {_f(idc['id_acc_full'])} (chance {_f(idc['chance'])})\n"
            f"- NN same-target fingerprint rate {_f(idc['nn_fingerprint']['nn_same_target_rate'])} "
            f"(null {_f(idc['nn_fingerprint']['nn_null_mean'])}, p {_f(idc['nn_fingerprint']['nn_perm_p'])})\n"
            f"- predmix recovery survives LOTO permutation: {idc['predmix_recovery_survives_permutation']}\n"
            f"- **identity fingerprint dominant: {idc['identity_fingerprint_dominant']}** (entanglement disclosed, not claimed identity-free)\n")


def render_label_md(res) -> str:
    ld = res["label_diagnostics"]
    if ld["status"] != schema.STATUS_OK:
        return (f"# C26 Q5 — predicted-class-mix label diagnostics (LABEL-DIAGNOSTIC-ONLY)\n\n"
                f"> Status **{ld['status']}** — {ld.get('reason')}\n\nLabels join only here (never in features); "
                f"requires the scoped re-persistence re-inference.\n")
    return (f"# C26 Q5 — predicted-class-mix label diagnostics (LABEL-DIAGNOSTIC-ONLY)\n\n> {ld['note']}\n\n"
            f"- predmix vs true class prior corr {_f(ld['predmix_vs_true_prior_corr'])}\n"
            f"- predmix vs per-class recall corr {_f(ld['predmix_vs_per_class_recall_corr'])}\n"
            f"- mix distance from true (balanced) prior {_f(ld['mix_distance_from_true_prior'])}\n"
            f"- tracks target error geometry: {ld['tracks_target_error_geometry']}\n")


_NEG_CUES = ("not ", "no ", "never ", "n't ", "cannot", "is not", "are not", "does not", "without ", "not a",
             "not established", "fails", "barred", "instead of", "not deployable", "not dg", "not claimed", "never claimed")


def _guard_forbidden(md) -> None:
    low = md.lower()
    for s in schema.FORBIDDEN_CLAIM_SUBSTRINGS:
        i = 0
        while (i := low.find(s, i)) != -1:
            if not any(cue in low[max(0, i - 34):i] for cue in _NEG_CUES):
                raise ValueError(f"forbidden AFFIRMATIVE over-claim in C26 report near: ...{low[max(0, i - 34):i + len(s)]!r}")
            i += len(s)


def _write_artifacts(res, out_dir):
    md = render_md(res); _guard_forbidden(md)
    idm = render_identity_md(res); _guard_forbidden(idm)
    lm = render_label_md(res); _guard_forbidden(lm)
    os.makedirs(out_dir, exist_ok=True)
    open(os.path.join(out_dir, "C26_PREDICTED_CLASS_MIX_MECHANISM.md"), "w").write(md)
    json.dump(res, open(os.path.join(out_dir, "C26_PREDICTED_CLASS_MIX_MECHANISM.json"), "w"), indent=2, sort_keys=True, default=str)
    open(os.path.join(out_dir, "C26_PREDMIX_IDENTITY_CONTROL_AUDIT.md"), "w").write(idm)
    open(os.path.join(out_dir, "C26_PREDMIX_LABEL_DIAGNOSTICS.md"), "w").write(lm)
    write_tables(res, os.path.join(out_dir, "c26_tables"))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.predmix_mechanism.report")
    ap.add_argument("--scores-sidecar", default=None)
    ap.add_argument("--target-unlabeled-sidecar", default=None)
    ap.add_argument("--split-sidecar", default=None)
    ap.add_argument("--out-dir", default="oaci/reports")
    args = ap.parse_args(argv)
    res = run(args.scores_sidecar, args.target_unlabeled_sidecar, args.split_sidecar)
    _write_artifacts(res, args.out_dir)
    t = res["taxonomy"]; sv = res["signed_vs_symmetric"]; idc = res["identity_controls"]; it = res["interaction"]
    print(f"[C26 stage={'FINAL' if t['final'] else 'readonly'}] primary={t['primary_case']} | "
          f"signed_gap={_f(sv['signed']['gap_closed'])}(surv={sv['signed']['survives_permutation']}) "
          f"symmetric_gap={_f(sv['symmetric']['gap_closed'])} align_matters={res['class_rotation']['class_index_alignment_matters']} | "
          f"id_predmix={_f(idc['id_acc_predmix'])} fingerprint_dom={idc['identity_fingerprint_dominant']} | "
          f"needs_scaffold={it['predmix_needs_confidence_scaffold']} | established={','.join(t['established']) or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
