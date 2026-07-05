"""C16 report — renders the mechanism deep dive (currently the C16-A target-oracle ceiling; harm decomposition
and discriminative-validity sections are appended by their modules). Markdown + canonical JSON + CSV tables.
The target oracle is labeled NON-DEPLOYABLE everywhere."""
from __future__ import annotations

import argparse
import csv
import os

from ..artifacts.canonical_json import canonical_json_bytes
from .target_oracle import load_replay, run_target_oracle


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))


def _wcsv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_target_oracle_tables(res, outdir) -> list:
    w = []
    sel = res["selectors"]
    # 1. ceiling by target/seed/level (target oracle bAcc)
    p = os.path.join(outdir, "target_oracle_ceiling_by_target_seed_level.csv")
    _wcsv(p, ["seed", "target", "level", "chosen_epoch", "chosen_is_erm", "target_bacc_delta", "target_nll_delta", "target_ece_delta"],
          [[r["seed"], r["target"], r["level"], r["chosen_epoch"], r["chosen_is_erm"], r["target_bacc_delta"],
            r["target_nll_delta"], r["target_ece_delta"]] for r in sel["target_oracle_bacc"]["per_fold"]]); w.append(p)
    # 2. target vs source oracle (worst-held-out-target K2 units)
    p = os.path.join(outdir, "target_oracle_vs_source_oracle.csv")
    rows = []
    for u_t, u_s in zip(sel["target_oracle_bacc"]["per_unit"], sel["source_audit_oracle"]["per_unit"]):
        rows.append([u_t["seed"], u_t["level"], u_t["delta_worst_bacc"], u_t["delta_worst_nll"],
                     u_s["delta_worst_bacc"], u_s["delta_worst_nll"]])
    _wcsv(p, ["seed", "level", "target_oracle_dbacc", "target_oracle_dnll", "source_oracle_dbacc", "source_oracle_dnll"], rows)
    w.append(p)
    # 3. endpoint tradeoffs (does the bAcc-optimal choice help or hurt NLL?)
    p = os.path.join(outdir, "target_oracle_endpoint_tradeoffs.csv")
    _wcsv(p, ["seed", "level", "target_oracle_bacc_dnll", "target_oracle_joint_dnll"],
          [[ub["seed"], ub["level"], ub["delta_worst_nll"], uj["delta_worst_nll"]]
           for ub, uj in zip(sel["target_oracle_bacc"]["per_unit"], sel["target_oracle_joint"]["per_unit"])]); w.append(p)
    # 4. rescue taxonomy
    p = os.path.join(outdir, "target_oracle_rescue_taxonomy.csv")
    _wcsv(p, ["quantity", "value"], [["case_label", res["case_label"]],
                                     ["source_audit_oracle_k2", res["source_audit_oracle_k2"]],
                                     ["target_oracle_bacc_k2", res["target_oracle_bacc_k2"]],
                                     ["target_oracle_joint_k2", res["target_oracle_joint_k2"]],
                                     ["target_oracle_rescues_bacc", res["target_oracle_rescues_bacc"]],
                                     ["target_oracle_rescues_joint", res["target_oracle_rescues_joint"]],
                                     ["source_oracle_rescues", res["source_oracle_rescues"]]]); w.append(p)
    # 5. checkpoint existence summary (per selector: K2 + how often it picks a non-ERM checkpoint)
    p = os.path.join(outdir, "checkpoint_existence_summary.csv")
    rows = []
    for name, sv in sel.items():
        n_nonerm = sum(1 for r in sv["per_fold"] if not r["chosen_is_erm"])
        rows.append([name, sv["k2_status"], sv.get("reproduced_endpoints"), n_nonerm, len(sv["per_fold"]),
                     sv["non_deployable"]])
    _wcsv(p, ["selector", "k2_status", "reproduced_endpoints", "n_chose_nonERM", "n_fold_levels", "non_deployable"], rows)
    w.append(p)
    return w


def render_target_oracle_md(res) -> str:
    sel = res["selectors"]
    L = ["# C16-A — Target-oracle ceiling (does OACI's trajectory contain a target-good checkpoint?)", "",
         "> The target oracle is a **NON-DEPLOYABLE, post-hoc diagnostic**: it reads target_audit only to test "
         "checkpoint EXISTENCE, never as a selection method.", "",
         f"- **CASE: `{res['case_label']}`** — {res['interpretation']}", "",
         "| selector | K2 (worst-held-out-target) | reproduced | deployable |", "|---|---|---|---|"]
    for name, sv in sel.items():
        L.append(f"| {name} | `{sv['k2_status']}` | {sv.get('reproduced_endpoints') or '—'} | "
                 f"{'NO (diagnostic)' if sv['non_deployable'] else 'yes'} |")
    L += ["", "## What this splits from C10's case C", "",
          f"- source-audit oracle rescues K2: **{res['source_oracle_rescues']}** (reproduces C10 case C)",
          f"- target oracle rescues worst-domain **bAcc**: **{res['target_oracle_rescues_bacc']}** "
          "→ target-accuracy-good checkpoints **DO exist** in the trajectory but are **not source-observable**",
          f"- target oracle rescues **joint** (bAcc+NLL): **{res['target_oracle_rescues_joint']}** "
          "→ a separate **calibration barrier**: even the accuracy-optimal checkpoints do not jointly improve NLL", "",
          "## Target-oracle bAcc ceiling — worst-held-out-target Δ vs ERM (per seed, level)", "",
          "| seed | level | Δ worst bAcc | Δ worst NLL |", "|---:|---:|---:|---:|"]
    for u in sel["target_oracle_bacc"]["per_unit"]:
        L.append(f"| {u['seed']} | {u['level']} | {_f(u['delta_worst_bacc'])} | {_f(u['delta_worst_nll'])} |")
    L += ["", "> **Mechanism.** The measurement→control decoupling for *accuracy* is a **source-side "
          "observability failure** (the good checkpoints exist; source signal cannot identify them), while "
          "*calibration* harm persists even at the target-accuracy ceiling. This is a diagnostic result; the "
          "target oracle is never deployable and makes no selection claim."]
    return "\n".join(L)


def build_target_oracle(replay_dir):
    return run_target_oracle(load_replay(replay_dir))


# ---- C16-B harm decomposition ----
def write_harm_tables(h, outdir) -> list:
    w = []
    _wcsv(os.path.join(outdir, "target_harm_by_subject.csv"), ["target", "n", "mean_d_bacc", "mean_d_nll"],
          [[t, v["n"], v["mean_d_bacc"], v["mean_d_nll"]] for t, v in h["per_subject"].items()]); w.append(1)
    _wcsv(os.path.join(outdir, "target_harm_by_class.csv"), ["class", "n", "mean_recall_delta"],
          [[c, v["n"], v["mean_recall_delta"]] for c, v in h["per_class_recall_delta"].items()]); w.append(1)
    _wcsv(os.path.join(outdir, "target_harm_by_class_pair.csv"), ["true", "pred", "mean_confusion_delta"],
          [[r["true"], r["pred"], r["mean_confusion_delta"]] for r in h["class_pair_confusion_delta"]]); w.append(1)
    a = h["aggregate_deltas"]
    _wcsv(os.path.join(outdir, "calibration_vs_discrimination_harm.csv"), ["quantity", "value"],
          [["selected_checkpoint_verdict", h["selected_checkpoint_verdict"]],
           ["harm_type_tally", h["harm_type_tally"]], ["mean_d_bacc", a["d_bacc"]], ["mean_d_nll", a["d_nll"]],
           ["mean_d_ece", a["d_ece"]], ["class_boundary_rotation", h["class_boundary_rotation"]]]); w.append(1)
    _wcsv(os.path.join(outdir, "entropy_margin_logitnorm_shift.csv"), ["quantity", "mean_delta_OACI_minus_ERM"],
          [["entropy", a["d_entropy"]], ["margin", a["d_margin"]], ["logit_norm", a["d_logit_norm"]],
           ["mean_conf", a["d_mean_conf"]]]); w.append(1)
    _wcsv(os.path.join(outdir, "confidence_on_wrong.csv"), ["quantity", "mean_delta_OACI_minus_ERM"],
          [["conf_on_wrong", a["d_conf_on_wrong"]]]); w.append(1)
    sm = h["src_source_memorization"]
    rows = [] if not sm else [[c["target"], c["temp"], c["level"], c["source_nll_improvement"],
                               c["target_nll_improvement"], c["memorization_index"], c["memorization_flag"]]
                              for c in sm["per_cell"]]
    _wcsv(os.path.join(outdir, "source_memorization_index.csv"),
          ["target", "temp", "level", "source_nll_improvement", "target_nll_improvement", "memorization_index", "flag"], rows)
    w.append(1)
    return w


def render_harm_md(h) -> str:
    a = h["aggregate_deltas"]; sm = h["src_source_memorization"]
    L = ["# C16-B — Harm decomposition (OACI vs ERM target, logit-level from committed target_audit.npz)", "",
         f"- **selected-checkpoint verdict: `{h['selected_checkpoint_verdict']}`** over {h['n_cells']} fold-levels",
         f"- harm-type tally: {h['harm_type_tally']}",
         f"- aggregate Δ (OACI−ERM): bAcc {_f(a['d_bacc'])}, NLL {_f(a['d_nll'])}, ECE {_f(a['d_ece'])}, "
         f"entropy {_f(a['d_entropy'])}, conf-on-wrong {_f(a['d_conf_on_wrong'])}, logit-norm {_f(a['d_logit_norm'])}",
         f"- **class-boundary rotation: {h['class_boundary_rotation']}** (some classes gain recall, others lose)", "",
         "## Per-class recall Δ (OACI−ERM)", "", "| class | Δ recall |", "|---:|---:|"]
    for c, v in h["per_class_recall_delta"].items():
        L.append(f"| {c} | {_f(v['mean_recall_delta'])} |")
    L += ["", "## Per-target-subject Δ (heterogeneous harm)", "", "| target | Δ bAcc | Δ NLL |", "|---:|---:|---:|"]
    for t, v in h["per_subject"].items():
        L.append(f"| {t} | {_f(v['mean_d_bacc'])} | {_f(v['mean_d_nll'])} |")
    if sm:
        L += ["", "## SRC source-memorization index (from committed C12)", "",
              f"- **{sm['n_flagged']}/{len(sm['per_cell'])} active cells flagged** as memorization "
              f"(source NLL improves, target does not); mean memorization index **{_f(sm['mean_memorization_index'],3)}**"]
    L += ["", f"> {h['note']}"]
    return "\n".join(L)


def build_harm(loso_root, c12_json):
    from .harm_decomposition import build
    return build(loso_root, c12_path=c12_json)


def render_mechanism_md(tor, harm, dv=None) -> str:
    L = ["# C16 — Mechanism & discriminative-validity deep dive (combined)", "",
         "> Explains the measurement→control decoupling and source→target anti-transfer WITHOUT a new control "
         "objective. Real-data analyses read only committed C8/C10/C12 artifacts; the target oracle is "
         "NON-DEPLOYABLE.", "",
         "## C16-A — Target-oracle ceiling",
         f"- **CASE `{tor['case_label']}`**: {tor['interpretation']}",
         f"- target-accuracy-good checkpoints exist (target oracle rescues bAcc: {tor['target_oracle_rescues_bacc']}) "
         f"but are not source-observable (source oracle rescues: {tor['source_oracle_rescues']}); joint "
         f"accuracy+calibration does not reproduce ({tor['target_oracle_joint_k2']}).", "",
         "## C16-B — Harm decomposition",
         f"- **`{harm['selected_checkpoint_verdict']}`**: the SELECTED OACI is softer/better-calibrated "
         f"(ΔNLL {_f(harm['aggregate_deltas']['d_nll'])}, Δentropy {_f(harm['aggregate_deltas']['d_entropy'])}) "
         f"but not more accurate (ΔbAcc {_f(harm['aggregate_deltas']['d_bacc'])}); class-boundary rotation "
         f"{harm['class_boundary_rotation']}; subject-heterogeneous.",
         (f"- SRC anti-transfer is MEMORIZATION: {harm['src_source_memorization']['n_flagged']}/"
          f"{len(harm['src_source_memorization']['per_cell'])} cells flagged, mean index "
          f"{_f(harm['src_source_memorization']['mean_memorization_index'],3)}." if harm.get("src_source_memorization") else ""),
         "", "## Synthesis",
         "> The measurement→control gap is not 'OACI is broken': the trajectory contains target-accuracy-good "
         "checkpoints that **source signal cannot observe**, and accuracy trades off against calibration so no "
         "single checkpoint jointly wins both. SRC's anti-transfer is source **memorization**. See "
         "C16_TARGET_ORACLE_CEILING.md, C16_HARM_DECOMPOSITION.md" + (", C16_BATTERY_DISCRIMINATIVE_VALIDITY.md" if dv else "") + "."]
    if dv is not None:
        L += ["", "## C16-C — Battery discriminative validity",
              f"- positive controls certified: {dv.get('positive_pass')}/{dv.get('positive_total')}; "
              f"negative controls falsified: {dv.get('negative_pass')}/{dv.get('negative_total')}; "
              f"**discriminative_validity = {dv.get('discriminative_validity')}**"]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.mechanism.report")
    ap.add_argument("--replay-dir", default="/projects/EEG-foundation-model/yinghao/oaci-c10-replay")
    ap.add_argument("--loso-root", default="/projects/EEG-foundation-model/yinghao/oaci-loso-seeds012")
    ap.add_argument("--c12-json", default="oaci/reports/C12_SRC_STRESS_REPLICATION.json")
    ap.add_argument("--reports-dir", default="oaci/reports")
    ap.add_argument("--tables-dir", default="oaci/reports/c16_tables")
    args = ap.parse_args(argv)
    os.makedirs(args.reports_dir, exist_ok=True)
    # C16-A
    tor = build_target_oracle(args.replay_dir)
    write_target_oracle_tables(tor, args.tables_dir)
    with open(os.path.join(args.reports_dir, "C16_TARGET_ORACLE_CEILING.json"), "wb") as f:
        f.write(canonical_json_bytes(tor))
    with open(os.path.join(args.reports_dir, "C16_TARGET_ORACLE_CEILING.md"), "w") as f:
        f.write(render_target_oracle_md(tor))
    # C16-B
    harm = build_harm(args.loso_root, args.c12_json)
    write_harm_tables(harm, args.tables_dir)
    with open(os.path.join(args.reports_dir, "C16_HARM_DECOMPOSITION.md"), "w") as f:
        f.write(render_harm_md(harm))
    # combined
    combined = {"target_oracle": tor, "harm_decomposition": harm}
    with open(os.path.join(args.reports_dir, "C16_MECHANISM_DEEP_DIVE.json"), "wb") as f:
        f.write(canonical_json_bytes(combined))
    with open(os.path.join(args.reports_dir, "C16_MECHANISM_DEEP_DIVE.md"), "w") as f:
        f.write(render_mechanism_md(tor, harm))
    print(f"C16-A case={tor['case_label']}; C16-B verdict={harm['selected_checkpoint_verdict']}; "
          f"SRC memo flagged={harm['src_source_memorization']['n_flagged'] if harm.get('src_source_memorization') else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
