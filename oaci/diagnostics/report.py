"""C10 report — renders the artifact-only Part 1 diagnostics (and, when C10b supplies it, the epoch-level
selector-replay Part 2) to Markdown + canonical JSON, and writes human-readable CSV(.gz) tables. The JSON is
canonical-serializable (no int mapping keys — the recurring aggregate-write trap). Part 2 fields stay null
until the GPU replay lands, but the case A/B/C decision scaffold is always present."""
from __future__ import annotations

import csv
import gzip
import io
import os

from ..artifacts.canonical_json import canonical_json_bytes
from .c8_loader import flat_records, load_all
from .transfer import run_all_transfer

_LABEL = ("BNCI2014-001 minimum-seed diagnostics (seeds [0,1,2]). Artifact-only Part 1 on SELECTED "
          "checkpoints; epoch-level counterfactual replay is Part 2 (C10b).")


def _f(x, nd=4):
    return "n/a" if x is None else (f"{x:+.{nd}f}" if isinstance(x, (int, float)) else str(x))


def _corr_str(c):
    p, rho, n = c["pearson"].get("r"), c["spearman"].get("rho"), c["pearson"].get("n")
    return (f"pearson {p:+.3f}, spearman {rho:+.3f} (n={n})" if p is not None else f"n/a (n={n})")


def _lean_case(transfer) -> dict:
    """Part-1 LEAN only (never the final call — that needs Part 2's oracle). Case C is indicated when audit
    leakage is orthogonal to every target metric AND selection→audit transfer is ~0."""
    a = transfer["audit_to_target_transfer"]
    corrs = [a[f"corr_audit_vs_target_{k}"]["pearson"].get("r") for k in ("worst_bacc", "worst_nll", "worst_ece")]
    corrs = [abs(c) for c in corrs if c is not None]
    sel = transfer["selection_to_audit_optimism"]["corr_selection_vs_audit_delta"]["pearson"].get("r")
    orthogonal = bool(corrs) and max(corrs) < 0.25
    optimism = sel is not None and abs(sel) < 0.20
    lean = "case_C_candidate" if (orthogonal and optimism) else "ambiguous_needs_part2"
    return {"part1_lean": lean, "audit_target_orthogonal": orthogonal, "selection_optimism": optimism,
            "max_abs_audit_target_pearson": (max(corrs) if corrs else None),
            "note": "FINAL case A/B/C is decided in C10b with the epoch-level source-only + oracle replay; "
                    "Part 1 alone cannot distinguish 'no good checkpoint exists' from 'selector picked badly'."}


def to_json(folds, transfer, *, part2=None) -> dict:
    n_fl = sum(len(f["levels"]) for f in folds)
    return {"label": _LABEL, "run": {"dataset": "BNCI2014-001", "seeds": sorted({f["seed"] for f in folds}),
                                     "targets": sorted({f["target"] for f in folds}), "n_folds": len(folds),
                                     "n_fold_levels": n_fl, "methods": ["ERM", "OACI", "global_lpc", "uniform"]},
            "part1_transfer": transfer, "part1_case_lean": _lean_case(transfer),
            "part2_selector_replay": part2,
            "case_determination": {"final_case": (part2.get("final_case") if part2 else None),
                                   "decision_rule": {
                                       "A_source_only_selector_works": "S1-S4 improve K2 without losing K1 "
                                                                       "-> build OACI-v2 selector",
                                       "B_only_source_audit_oracle_works": "only S5 improves K2 -> investigate "
                                                                           "source-audit/selection-guard protocol",
                                       "C_oracle_also_fails": "S5 also fails -> stop treating leakage control "
                                                              "as a downstream mechanism; keep it as measurement"}}}


# ---- CSV tables ----
def _write_csv_gz(path, header, rows):
    buf = io.StringIO(); w = csv.writer(buf); w.writerow(header)
    for r in rows:
        w.writerow(r)
    with gzip.open(path, "wt", newline="") as f:
        f.write(buf.getvalue())


def _write_csv(path, header, rows):
    with open(path, "w", newline="") as f:
        w = csv.writer(f); w.writerow(header)
        for r in rows:
            w.writerow(r)


def write_tables(folds, transfer, outdir) -> list:
    os.makedirs(outdir, exist_ok=True)
    written = []
    # candidate_scores (Part 1 = selected checkpoints; C10b appends per-candidate rows)
    recs = flat_records(folds)
    hdr = sorted(recs[0].keys()) if recs else []
    p = os.path.join(outdir, "selected_checkpoint_scores.csv.gz")
    _write_csv_gz(p, hdr, [[r.get(k) for k in hdr] for r in recs]); written.append(p)
    # transfer_correlations
    tc = []
    a = transfer["audit_to_target_transfer"]
    for k in ("worst_bacc", "worst_nll", "worst_ece"):
        c = a[f"corr_audit_vs_target_{k}"]
        tc.append(["audit_leakage", f"target_{k}", c["pearson"].get("r"), c["spearman"].get("rho"), c["pearson"]["n"]])
    o = transfer["selection_to_audit_optimism"]["corr_selection_vs_audit_delta"]
    tc.append(["selection_leakage", "audit_leakage", o["pearson"].get("r"), o["spearman"].get("rho"), o["pearson"]["n"]])
    rt = transfer["risk_tradeoff"]
    for name in ("corr_lambda_vs_delta_target_worst_bacc", "corr_lambda_vs_delta_target_worst_nll",
                 "corr_R_src_gap_vs_delta_target_worst_bacc", "corr_epoch_vs_delta_target_worst_bacc"):
        c = rt[name]; tc.append([name, "", c["pearson"].get("r"), c["spearman"].get("rho"), c["pearson"]["n"]])
    p = os.path.join(outdir, "transfer_correlations.csv")
    _write_csv(p, ["x", "y", "pearson", "spearman", "n"], tc); written.append(p)
    return written


def render_report_md(folds, transfer, *, part2=None) -> str:
    o = transfer["selection_to_audit_optimism"]; a = transfer["audit_to_target_transfer"]
    rt = transfer["risk_tradeoff"]; mc = transfer["method_comparison"]; h = transfer["harm_localization"]
    lean = _lean_case(transfer)
    L = [f"# C10 — OACI failure-mode diagnostics (BNCI2014-001 seeds {sorted({f['seed'] for f in folds})})", "",
         f"> **{_LABEL}**", "",
         "## Q1 — selection → audit optimism (does the selection-time leakage win transfer?)", "",
         f"- Δ selection leakage (OACI−ERM): mean **{_f(o['delta_selection_leakage']['mean'])}**, "
         f"reduced **{o['n_selection_reduced']}/{o['n_fold_levels']}**",
         f"- Δ audit leakage (OACI−ERM): mean **{_f(o['delta_audit_leakage']['mean'])}**, "
         f"reduced **{o['n_audit_reduced']}/{o['n_fold_levels']}**",
         f"- corr(Δselection, Δaudit): {_corr_str(o['corr_selection_vs_audit_delta'])} — "
         "**near-zero ⇒ selection-optimism / criterion overfit**", "",
         "## Q2 — audit leakage → target transfer (is held-out leakage predictive of DG?)", ""]
    for k in ("worst_bacc", "worst_nll", "worst_ece"):
        L.append(f"- Δ target {k} mean {_f(a['delta_target_'+k]['mean'])} · "
                 f"corr(Δaudit, Δtarget {k}): {_corr_str(a['corr_audit_vs_target_'+k])}")
    L += [f"- {a['interpretation_hint']}", "",
          "## Risk tradeoff & level effect", "",
          f"- λ mean {_f(rt['lambda']['mean'],3)}; corr(λ, Δtarget worst bAcc): "
          f"{_corr_str(rt['corr_lambda_vs_delta_target_worst_bacc'])} — negative ⇒ heavier penalty costs accuracy"]
    for Lk, v in transfer["level_effect"].items():
        L.append(f"- level {v['level']}: Δtarget worst bAcc {_f(v['delta_target_worst_bacc']['mean'])} · "
                 f"Δaudit leakage {_f(v['delta_audit_leakage']['mean'])}")
    L += ["", "## Method comparison (OACI − baseline, target worst bAcc)", ""]
    for b, v in mc.items():
        L.append(f"- vs {b}: mean {_f(v['delta_target_worst_bacc']['mean'])} "
                 f"(improved {v['n_bacc_improved']}, harmed {v['n_bacc_harmed']})")
    L += ["", "## Harm localization", "",
          f"- target worst bAcc harmed in **{h['n_harmed_bacc']}/{h['n_fold_levels']}** fold-levels; total loss "
          f"{_f(h['total_bacc_loss'],3)}; top-5 folds = {_f(h['top5_harm_share_of_total_loss'],2)} of the loss "
          "(diffuse if ≈ 5/28, concentrated if near 1)",
          "- worst fold-levels (seed,target,level,Δbacc): "
          + ", ".join(f"({p['seed']},{p['target']},{p['level']},{_f(p['delta_worst_bacc'],3)})"
                      for p in h["worst5_fold_levels"]), "",
          "## Part-1 lean & case A/B/C", "",
          f"- **part1_lean: `{lean['part1_lean']}`** (audit⊥target={lean['audit_target_orthogonal']}, "
          f"selection_optimism={lean['selection_optimism']}, max|audit-target r|="
          f"{_f(lean['max_abs_audit_target_pearson'],3)})",
          f"- {lean['note']}", ""]
    if part2 is None:
        L += ["> **Part 2 (epoch-level counterfactual selector replay) — PENDING C10b GPU replay.** "
              "The final case A/B/C call requires the source-only (S1–S4) and source-audit-oracle (S5) replay "
              "over OACI's own risk-feasible trajectory."]
    else:
        L += [f"> **FINAL CASE: `{part2.get('final_case')}`** — see Part 2 selector-replay section."]
    return "\n".join(L)


def build_report(loso_root, *, part2=None):
    folds = load_all(loso_root)
    transfer = run_all_transfer(folds)
    return folds, transfer, render_report_md(folds, transfer, part2=part2), to_json(folds, transfer, part2=part2)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="oaci.diagnostics.report")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tables-dir", required=True)
    args = ap.parse_args(argv)
    folds, transfer, md, js = build_report(args.loso_root)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tables = write_tables(folds, transfer, args.tables_dir)
    with open(args.out_md, "w") as f:
        f.write(md)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(js))
    print(f"wrote {args.out_md} + {args.out_json} + {len(tables)} tables; part1_lean="
          f"{js['part1_case_lean']['part1_lean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
