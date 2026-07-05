"""C14 report — render the falsification battery to Markdown + canonical JSON + CSV tables. The interpretation
wording is fixed to what the evidence supports (localized falsification of the CONTROL hypothesis), and
explicitly avoids over-claims ('all DG impossible', 'OACI mathematically wrong')."""
from __future__ import annotations

import argparse
import csv
import os

from ..artifacts.canonical_json import canonical_json_bytes
from .battery import build_from_reports

_CORE = ("The experiments do not merely show that OACI underperforms ERM; they LOCALIZE the failure of the "
         "control hypothesis. Selection-time leakage reductions do not reliably survive audit; nominal audit "
         "leakage reductions do not produce endpoint gains; a source-audit oracle cannot rescue OACI "
         "trajectories; and a separate source-endpoint objective produces anti-transfer. Therefore the "
         "support-aware machinery should be retained as a FALSIFICATION and MEASUREMENT instrument, not as a "
         "control objective under this protocol.")
_SAY = ("Under BNCI2014-001 LOSO with strict source/target isolation, the tested source-side control "
        "mechanisms do not transfer to target worst-domain endpoints. The measurement framework is useful "
        "precisely because it makes that failure visible.")
_DONT = ["All DG is impossible.", "EEG DG cannot work.", "Support-aware invariance is useless.",
         "OACI is mathematically wrong."]


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) and not isinstance(x, bool) else str(x))


def _wcsv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_tables(bat, outdir) -> list:
    os.makedirs(outdir, exist_ok=True)
    g = bat["gates"]; written = []
    p = os.path.join(outdir, "gate_summary.csv")
    _wcsv(p, ["gate", "status"], [[gid, gg["status"]] for gid, gg in
                                  [(o, g[o]) for o in bat["gate_order"]]]); written.append(p)
    g1 = g["G1_selection_optimism"]
    p = os.path.join(outdir, "selection_audit_optimism.csv")
    _wcsv(p, ["metric", "value"], [["delta_selection_leakage_mean", g1["delta_selection_leakage_mean"]],
                                   ["delta_audit_leakage_mean", g1["delta_audit_leakage_mean"]],
                                   ["selection_to_audit_transfer_ratio", g1["selection_to_audit_transfer_ratio"]],
                                   ["optimism_gap", g1["optimism_gap"]],
                                   ["corr_selection_vs_audit", g1["corr_selection_vs_audit"]],
                                   ["fraction_sign_preserved", g1["fraction_sign_preserved"]]]); written.append(p)
    g2 = g["G2_heldout_leakage"]
    p = os.path.join(outdir, "leakage_target_transfer.csv")
    tc = bat["diagnostics"]["transfer_correlations"]
    _wcsv(p, ["metric", "value"], [["k1_sweep_status", g2["sweep_status"]], ["n_nominal_detected", g2["n_nominal_detected"]],
                                   ["n_bh_survivors", g2["n_bh_survivors"]], ["weak_nominal_signal", g2["weak_nominal_signal"]],
                                   ["audit_leakage_to_target_bacc_pearson",
                                    (tc["audit_leakage_to_target_worst_bacc"] or {}).get("pearson", {}).get("r")],
                                   ["audit_leakage_to_target_nll_pearson",
                                    (tc["audit_leakage_to_target_worst_nll"] or {}).get("pearson", {}).get("r")]]); written.append(p)
    st = tc["source_nll_to_target_nll"]
    p = os.path.join(outdir, "source_target_transfer.csv")
    im = bat["diagnostics"]["instability"]
    _wcsv(p, ["metric", "value"], [["source_nll_to_target_nll_pearson", st["pearson"].get("r")],
                                   ["source_nll_to_target_nll_spearman", st["spearman"].get("rho")],
                                   ["sign_agreement", st["sign_agreement"]], ["ATI_NLL", im["ATI_NLL"]],
                                   ["ATI_severity", im["ATI_severity_mean_target_nll_harm"]],
                                   ["source_target_instability_score", im["source_target_instability_score"]]]); written.append(p)
    g4 = g["G4_oracle_rescue"]
    p = os.path.join(outdir, "oracle_rescue.csv")
    _wcsv(p, ["selector", "k2_status"], list(g4["per_selector_k2"].items())); written.append(p)
    p = os.path.join(outdir, "antitransfer_flags.csv")
    _wcsv(p, ["target", "temp", "level", "target_nll_blowup", "target_bacc_harmed", "target_nll_harmed", "fallback_erm"],
          [[c["target"], c["temp"], c["level"], c["target_nll_blowup"], c["target_bacc_harmed"], c["target_nll_harmed"],
            c["fallback_erm"]] for c in bat["diagnostics"]["harm_localization"]["per_cell"]]); written.append(p)
    p = os.path.join(outdir, "method_closure_table.csv")
    _wcsv(p, ["method_hypothesis", "evidence", "status", "next_allowed_action"],
          [[r["method_hypothesis"], r["evidence"], r["status"], r["next_allowed_action"]]
           for r in bat["method_closure_table"]]); written.append(p)
    return written


def render_md(bat) -> str:
    g = bat["gates"]; v = bat["verdict"]; im = bat["diagnostics"]["instability"]
    L = ["# C14 — EEG-DG Falsification Battery", "",
         "> Support-aware leakage, selector-oracle replay, and source→target instability diagnostics as a "
         "reusable MEASUREMENT / FALSIFICATION instrument. OACI / SRC are NOT control methods.", "",
         f"- **CONTROL-HYPOTHESIS STATUS: `{v['control_hypothesis_status']}`**",
         f"- falsification reasons: {v['falsification_reasons'] or 'none'}", "",
         "## Gates", "", "| gate | status |", "|---|---|"]
    for gid in bat["gate_order"]:
        L.append(f"| {gid} | `{g[gid]['status']}` |")
    g1, g2, g4, g5 = g["G1_selection_optimism"], g["G2_heldout_leakage"], g["G4_oracle_rescue"], g["G5_source_target_transfer"]
    L += ["", "## Evidence highlights", "",
          f"- **G0 integrity**: deep-verified={g['G0_integrity']['c8_deep_verified']}, target_fit_empty="
          f"{g['G0_integrity']['c8_target_fit_empty']}, replay identity all-pass="
          f"{g['G0_integrity']['replay_identity_all_pass']} (argmax flips {g['G0_integrity']['replay_argmax_flips']})",
          f"- **G1 selection optimism**: Δsel {_f(g1['delta_selection_leakage_mean'])} vs Δaudit "
          f"{_f(g1['delta_audit_leakage_mean'])}, transfer ratio {_f(g1['selection_to_audit_transfer_ratio'],3)}, "
          f"corr {_f(g1['corr_selection_vs_audit'],3)}",
          f"- **G2 held-out leakage (K1)**: `{g2['status']}` — {g2['n_nominal_detected']} nominal, "
          f"{g2['n_bh_survivors']} BH survivors of {g2['n_tests']}",
          f"- **G3 endpoint (K2)**: `{g['G3_endpoint_transfer']['status']}` ({g['G3_endpoint_transfer']['k2_status']})",
          f"- **G4 oracle rescue**: `{g4['status']}` (S5 oracle K2 = {g4['oracle_k2_status']}, source-only "
          f"reproducing = {g4['source_only_selectors_reproducing'] or 'none'}, S0=C8 check {g4['s0_current_k2']})",
          f"- **G5 source→target**: `{g5['status']}` — ATI {_f(im['ATI_NLL'],3)}, instability score "
          f"{_f(im['source_target_instability_score'],3)}, anti-transfer {im['n_anti_transfer']}/{im['n_active']} "
          f"active cells, blowup {g5['n_target_nll_blowup']}", "",
          "## Source→target instability", "",
          f"- source_nll→target_nll pearson {_f(g5['source_nll_to_target_nll_pearson'],3)} "
          "(near-zero/negative ⇒ source improvement does NOT reduce target loss)",
          f"- **anti-transfer index (ATI_NLL) {_f(im['ATI_NLL'],3)}**, severity (mean target-NLL harm) "
          f"{_f(im['ATI_severity_mean_target_nll_harm'])}, STI {_f(im['source_target_instability_score'],3)}", "",
          "## Method closure table", "", "| hypothesis | status | next allowed action |", "|---|---|---|"]
    for r in bat["method_closure_table"]:
        L.append(f"| {r['method_hypothesis']} | `{r['status']}` | {r['next_allowed_action']} |")
    L += ["", "## Interpretation", "", f"> {_CORE}", "", f"**Say:** {_SAY}", "",
          "**Do not over-claim:** " + " / ".join(f"~~{d}~~" for d in _DONT), ""]
    if v["falsification_reasons"]:
        L += [f"> **Verdict: the control hypothesis is FALSIFIED** ({', '.join(v['falsification_reasons'])}). "
              "Retain support-aware leakage + K1/K2 + oracle replay + anti-transfer diagnostics as the "
              "falsification instrument; do NOT build another DG control penalty under this protocol."]
    return "\n".join(L)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="oaci.falsification.report")
    ap.add_argument("--report-dir", default="oaci/reports")
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tables-dir", required=True)
    args = ap.parse_args(argv)
    bat = build_from_reports(args.report_dir)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tables = write_tables(bat, args.tables_dir)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(bat))
    with open(args.out_md, "w") as f:
        f.write(render_md(bat))
    v = bat["verdict"]
    print(f"wrote {args.out_json} + {args.out_md} + {len(tables)} tables; status="
          f"{v['control_hypothesis_status']} reasons={v['falsification_reasons']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
