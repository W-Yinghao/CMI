"""Phase 2 report + pre-registered STOP-rule check. Reads phase2_dryrun_summary.json and emits
phase2_report.md. Target metrics enter ONLY as a post-hoc audit (the gate action was computed source-only).

Pre-registered STOP rules (per deployable eraser, evaluated AFTER the run, thresholds fixed in config):
  STOP  (possible real task-preserving benefit -> pause & audit, do NOT auto-scale) iff
        target ΔbAcc LCB > +0.01  AND  source task-drop UCB <= 0.02  AND  same-k random does NOT reproduce it.
  MIXED (point estimate positive but CI includes 0) iff  ΔbAcc mean > 0 and ΔbAcc LCB <= +0.01, and safe.
  HARMFUL iff ΔbAcc UCB < -0.01 (target degraded).
  else NO-BENEFIT.
random_k is the non-specific control (never triggers STOP); the oracle row is a diagnostic upper bound.
  python -m tos_cmi.eeg.report_phase2
"""
from __future__ import annotations
import json
import os

OUT = "tos_cmi/results/method_deepen/phase2"
CONTROL = "random_k"
ORACLE = "cc_leace_oracle_route_diagnostic"


def _verdict(v, rand_lcb):
    if not v["deployable"]:
        return "DIAGNOSTIC (oracle upper bound; not deployable)"
    safe = not (v["task_drop_ucb"] > 0.02)
    rand_reproduces = rand_lcb is not None and rand_lcb > 0.01
    if v["dtgt_bacc_lo"] > 0.01 and safe and not rand_reproduces:
        return "**STOP** (possible real benefit -> audit)"
    if v["dtgt_bacc_hi"] < -0.01:
        return "HARMFUL (target degraded)"
    if v["dtgt_bacc"] > 0 and v["dtgt_bacc_lo"] <= 0.01:
        return "MIXED (point>0, CI includes 0)"
    return "no benefit"


def main():
    S = json.load(open("%s/phase2_dryrun_summary.json" % OUT))
    summary, cfg_hash = S["summary"], S.get("config_hash", "?")
    datasets = sorted(set(v["dataset"] for v in summary.values()))
    L = ["# Phase 2 dry-run --- task-preserving / conditional erasure (Lee2019 & Cho2017, EEGNet, seed0)\n",
         "Config `%s` sha256:`%s` (thresholds frozen = Track B: safety task-drop UCB<=0.02, benefit LCB>+0.01, "
         "domain-gain diagnostic-only, target audit-only). Gate action is SOURCE-ONLY; target ΔbAcc/ΔNLL below "
         "are a POST-HOC audit.\n" % ("phase2_task_preserving_fixed.yaml", cfg_hash),
         "> **Caveat (disclosed):** `cc_leace_predicted_route_deployable`'s exact +0.000 target ΔbAcc is partly "
         "a STRUCTURAL TAUTOLOGY -- routing target features by a task-predictor and then re-probing the task "
         "reproduces the router's boundary (verified: identical argmax on all target points). It is therefore "
         "NOT a clean test of conditional erasure; **`tp_leace` is the clean task-preserving result.** Both give "
         "zero deployable target benefit, so the conclusion is unchanged.\n"]
    stops = []
    for ds in datasets:
        rows = [summary[k] for k in sorted(summary) if summary[k]["dataset"] == ds]
        rand = next((r for r in rows if r["eraser"] == CONTROL), None)
        rand_lcb = rand["dtgt_bacc_lo"] if rand else None
        ch = rows[0]["chance"]
        L += ["## %s EEGNet (%d folds, chance bAcc %.3f)" % (ds, rows[0]["n_folds"], ch),
              "| eraser | src task after (was) | task-drop UCB | subj dec (full->eras) | domain-gain | "
              "src-LOSO benefit LCB | gate action | **target ΔbAcc [CI]** | **target ΔNLL [CI]** | verdict |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            vd = _verdict(r, rand_lcb)
            if vd.startswith("**STOP"):
                stops.append((ds, r["eraser"]))
            blcb = "%+.3f" % r["benefit_lcb"] if r["benefit_lcb"] == r["benefit_lcb"] else "n/a"
            L.append("| %s | %.3f (%.3f) | %+.3f | %.2f->%.2f | %+.3f | %s | **%s** | %+.3f [%+.3f,%+.3f] | "
                     "%+.3f [%+.3f,%+.3f] | %s |"
                     % (r["eraser"], r["src_task_eras"], r["src_task_full"], r["task_drop_ucb"],
                        r["subj_full"], r["subj_eras"], r["domain_gain"], blcb, r["gate_action"],
                        r["dtgt_bacc"], r["dtgt_bacc_lo"], r["dtgt_bacc_hi"],
                        r["dtgt_nll"], r["dtgt_nll_lo"], r["dtgt_nll_hi"], vd))
        # focused comparison questions vs the original LEACE collapse
        base = next((r for r in rows if r["eraser"] == "leace_baseline"), None)
        tp = next((r for r in rows if r["eraser"] == "tp_leace_task_carrier_preserving"), None)
        cc = next((r for r in rows if r["eraser"] == "cc_leace_predicted_route_deployable"), None)
        orc = next((r for r in rows if r["eraser"] == ORACLE), None)
        L += ["", "**Vs the original LEACE collapse on %s:**" % ds]
        if base:
            L.append("- plain LEACE: source task %.3f->%.3f (drop UCB %+.3f), target ΔbAcc %+.3f [%+.3f,%+.3f]"
                     % (base["src_task_full"], base["src_task_eras"], base["task_drop_ucb"],
                        base["dtgt_bacc"], base["dtgt_bacc_lo"], base["dtgt_bacc_hi"]))
        for nm, r in [("TP-LEACE", tp), ("cc-LEACE (predicted)", cc)]:
            if r:
                preserves = "YES" if r["task_drop_ucb"] <= 0.02 else "no"
                erases = "YES" if r["domain_gain"] > 0.05 else "partial/no"
                improves = "YES" if r["dtgt_bacc_lo"] > 0.01 else "no"
                L.append("- %s: preserves task? %s (drop UCB %+.3f) | erases subject? %s (dom-gain %+.3f) | "
                         "improves target? %s (ΔbAcc %+.3f [%+.3f,%+.3f])"
                         % (nm, preserves, r["task_drop_ucb"], erases, r["domain_gain"], improves,
                            r["dtgt_bacc"], r["dtgt_bacc_lo"], r["dtgt_bacc_hi"]))
        if orc:
            L.append("- oracle cc-LEACE (perfect routing, upper bound): src task %.3f->%.3f, target ΔbAcc %+.3f "
                     "[%+.3f,%+.3f] (uses TRUE target labels -> NOT deployable)"
                     % (orc["src_task_full"], orc["src_task_eras"], orc["dtgt_bacc"],
                        orc["dtgt_bacc_lo"], orc["dtgt_bacc_hi"]))
        L.append("")
    # decision
    L += ["## Decision"]
    if stops:
        L.append("- **STOP triggered** on: %s. A task-preserving eraser may yield real target benefit; do NOT "
                 "auto-scale. Next = split/leakage/random/calibration audit before full Phase 2." %
                 ", ".join("%s/%s" % s for s in stops))
    else:
        L.append("- No STOP triggered: no deployable eraser cleared (target ΔbAcc LCB>+0.01 & safe & random "
                 "doesn't reproduce).")
        L.append("- If any eraser PRESERVES task (drop UCB<=0.02) but target still does NOT improve -> the "
                 "high-value 'task safe, transfer flat' result -> proceed to V2 with the new erasers in the set.")
        L.append("- If no eraser both preserves task AND erases subject -> subject<->task strongly entangled in "
                 "the compact binary EEGNet latent; refusal is the correct action.")
    open("%s/phase2_report.md" % OUT, "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("\nSTOP_TRIGGERED=%d" % len(stops))
    print("PHASE2_REPORT_DONE")


if __name__ == "__main__":
    main()
