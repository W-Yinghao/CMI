"""C10 report — renders the artifact-only Part 1 diagnostics (and, when C10b supplies it, the epoch-level
selector-replay Part 2) to Markdown + canonical JSON, and writes human-readable CSV(.gz) tables. The JSON is
canonical-serializable (no int mapping keys — the recurring aggregate-write trap). Part 2 fields stay null
until the GPU replay lands, but the case A/B/C decision scaffold is always present."""
from __future__ import annotations

import csv
import glob
import gzip
import io
import json
import os

from ..artifacts.canonical_json import canonical_json_bytes
from ..decision.k2_decision import k2_decision
from .c8_loader import flat_records, load_all
from .selectors import SELECTORS, _ORACLE, run_selectors_on_level
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


# ---- Part 2: epoch-level counterfactual selector replay aggregation ----
def load_replay_dir(replay_dir) -> list:
    folds = [json.load(open(p)) for p in sorted(glob.glob(os.path.join(replay_dir, "seed-*-target-*.json")))]
    if not folds:
        raise ValueError(f"no replay fold JSONs in {replay_dir}")
    return folds


def _erm_row(rows):
    for r in rows:
        if r.get("is_erm"):
            return r
    return None


def aggregate_selector_replay(replay_folds, *, margins=None, k2_min_seeds=3) -> dict:
    """Run S0–S5 per (seed,target,level), evaluate each choice on target (eval-only), and aggregate the same
    worst-held-out-target K2 as C8 per selector (Δ vs the ERM selector). Then decide case A/B/C."""
    seeds = sorted({f["seed"] for f in replay_folds})
    targets = sorted({f["target"] for f in replay_folds})
    levels = sorted({int(L) for f in replay_folds for L in f["levels"]})
    idx = {(f["seed"], f["target"], int(L)): lv for f in replay_folds for L, lv in f["levels"].items()}
    n_expected = len(seeds) * len(targets) * len(levels)
    if len(idx) != n_expected:
        raise ValueError(f"replay coverage {len(idx)} != expected {n_expected} (seed×target×level)")

    # identity summary (numeric gate: argmax parity + tiny logit tol vs stored .npz; byte-hash where node matched)
    ich = [c for f in replay_folds for c in f.get("identity", [])]
    diffs = [c["max_logit_diff"] for c in ich if c.get("max_logit_diff") is not None]
    identity = {"n_checks": len(ich), "n_all_match": sum(1 for c in ich if c["match"]),
                "n_byte_hash_match": sum(1 for c in ich if c.get("hash_match")),
                "n_numeric_only": sum(1 for c in ich if c["match"] and not c.get("hash_match")),
                "total_argmax_flips": sum(c.get("argmax_flips") or 0 for c in ich),
                "max_logit_diff": (max(diffs) if diffs else None), "all_pass": all(c["match"] for c in ich)}

    choices, access = {}, {n: {"target_read": False, "read_source_audit": False, "forbidden": []} for n in SELECTORS}
    for (s, t, L), lv in idx.items():
        res = run_selectors_on_level(lv["candidates"], selected_oaci_hash=lv["selected"]["OACI"], margins=margins)
        for name, c in res.items():
            choices.setdefault(name, {})[(s, t, L)] = c
            a = c["access"]
            access[name]["target_read"] = access[name]["target_read"] or a["target_read"]
            access[name]["read_source_audit"] = access[name]["read_source_audit"] or ("source_audit" in a["roles_actually_read"])
            access[name]["forbidden"] += a["forbidden_fields"]

    def _erm_worst(s, L):
        bb = [_erm_row(idx[(s, t, L)]["candidates"])["target_worst_bacc"] for t in targets]
        nn = [_erm_row(idx[(s, t, L)]["candidates"])["target_worst_nll"] for t in targets]
        bb = [x for x in bb if x is not None]; nn = [x for x in nn if x is not None]
        return (min(bb) if bb else None, max(nn) if nn else None)

    per_selector = {}
    for name in SELECTORS:
        units, per_unit = [], []
        for s in seeds:
            for L in levels:
                sb = [choices[name][(s, t, L)]["target_worst_bacc"] for t in targets]
                sn = [choices[name][(s, t, L)]["target_worst_nll"] for t in targets]
                sb = [x for x in sb if x is not None]; sn = [x for x in sn if x is not None]
                wb, wn = (min(sb) if sb else None), (max(sn) if sn else None)
                eb, en = _erm_worst(s, L)
                db = None if (wb is None or eb is None) else wb - eb
                dn = None if (wn is None or en is None) else wn - en
                units.append({"seed": s, "level": L, "deltas": {"worst_domain_bacc": db, "worst_domain_nll": dn}})
                per_unit.append({"seed": s, "level": L, "worst_bacc": wb, "worst_nll": wn,
                                 "delta_worst_bacc": db, "delta_worst_nll": dn})
        k2 = k2_decision(units, endpoints=["worst_domain_bacc", "worst_domain_nll"], min_seeds=int(k2_min_seeds),
                         level_policy="both_levels", margins={"worst_domain_bacc": 0.0, "worst_domain_nll": 0.0})
        n_erm = sum(1 for c in choices[name].values() if c["is_erm"])
        per_selector[name] = {"k2_status": k2["k2_status"], "reproduced_endpoints": k2.get("reproduced_endpoints"),
                              "per_unit": per_unit, "n_choose_erm": n_erm, "n_fold_levels": len(choices[name]),
                              "access": access[name], "is_oracle": name in _ORACLE}
    source_only = [n for n in SELECTORS if n not in _ORACLE and n != "S0_current"]
    src_repro = [n for n in source_only if per_selector[n]["k2_status"] == "reproducible_gain"]
    oracle_repro = per_selector["S5_source_audit_oracle"]["k2_status"] == "reproducible_gain"
    case = ("A_source_only_selector_works" if src_repro
            else "B_only_source_audit_oracle_works" if oracle_repro else "C_oracle_also_fails")
    # access invariants (machine-checkable): NO selector reads target; S1-S4 never read source_audit
    access_ok = (all(not a["target_read"] and not a["forbidden"] for a in access.values())
                 and all(not access[n]["read_source_audit"] for n in source_only + ["S0_current"]))
    return {"seeds": seeds, "targets": targets, "levels": levels, "selectors": per_selector,
            "source_only_reproducible": src_repro, "oracle_reproducible": oracle_repro,
            "s0_current_k2": per_selector["S0_current"]["k2_status"], "access_invariants_ok": access_ok,
            "identity": identity, "final_case": case}


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


def write_tables(folds, transfer, outdir, *, part2=None) -> list:
    os.makedirs(outdir, exist_ok=True)
    written = []
    if part2 is not None:
        rows = [[n, sv["k2_status"], (sv.get("reproduced_endpoints") or ""), sv["n_choose_erm"],
                 sv["n_fold_levels"], sv["access"]["read_source_audit"], sv["access"]["target_read"], sv["is_oracle"]]
                for n, sv in part2["selectors"].items()]
        p = os.path.join(outdir, "selector_replay_summary.csv")
        _write_csv(p, ["selector", "k2_status", "reproduced", "n_choose_erm", "n_fold_levels",
                       "reads_source_audit", "target_read", "is_oracle"], rows); written.append(p)
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
        return "\n".join(L)
    idn = part2["identity"]
    L += ["", "## Part 2 — epoch-level counterfactual selector replay (K2 vs ERM per selector)", "",
          f"- **replay identity: {idn['n_all_match']}/{idn['n_checks']} selected-checkpoint checks pass** "
          f"({idn['n_byte_hash_match']} byte-hash exact, {idn['n_numeric_only']} numeric-only) · "
          f"total argmax flips **{idn['total_argmax_flips']}** · max|Δlogit| **{_f(idn['max_logit_diff'], 2) if idn['max_logit_diff'] is not None else 'n/a'}** "
          "(cross-node FP; worst-domain bAcc is argmax-based ⇒ exact)",
          f"- access invariants OK (no selector reads target; S0–S4 never read source_audit): "
          f"**{part2['access_invariants_ok']}**",
          f"- S0_current K2 = `{part2['s0_current_k2']}` (must equal the C8 OACI verdict — consistency check)", "",
          "| selector | K2 | reproduced | chooses ERM | reads source_audit | oracle |", "|---|---|---|---:|---|---|"]
    for name, sv in part2["selectors"].items():
        L.append(f"| {name} | `{sv['k2_status']}` | {sv.get('reproduced_endpoints') or '—'} | "
                 f"{sv['n_choose_erm']}/{sv['n_fold_levels']} | {sv['access']['read_source_audit']} | {sv['is_oracle']} |")
    case = part2["final_case"]
    L += ["", "## FINAL case A/B/C", "",
          f"- source-only selectors reproducing K2 gain: {part2['source_only_reproducible'] or 'none'}",
          f"- source-audit oracle (S5) reproduces K2 gain: {part2['oracle_reproducible']}",
          f"- **FINAL CASE: `{case}`**", ""]
    if case == "A_source_only_selector_works":
        L += ["> A source-only guard selector recovers reproducible K2 gain without target/oracle info. "
              "**Next: build OACI-v2 = that constrained selector.**"]
    elif case == "B_only_source_audit_oracle_works":
        L += ["> Only the source-audit ORACLE recovers gain — a better OACI checkpoint EXISTS but the "
              "source-only selection split can't find it. **Next: fix the selection/validation split protocol, "
              "not the objective.**"]
    else:
        L += ["> Even the source-audit oracle cannot recover reproducible K2 gain: better OACI checkpoints do "
              "NOT exist in the trajectory as judged by held-out source signal. **Leakage control is not a "
              "downstream-benefit mechanism under this protocol — keep support-aware leakage as a measurement/"
              "falsification tool, stop investing in it as a control objective.**"]
    return "\n".join(L)


def build_report(loso_root, *, replay_dir=None, margins=None):
    folds = load_all(loso_root)
    transfer = run_all_transfer(folds)
    part2 = aggregate_selector_replay(load_replay_dir(replay_dir), margins=margins) if replay_dir else None
    return folds, transfer, render_report_md(folds, transfer, part2=part2), to_json(folds, transfer, part2=part2)


def main(argv=None) -> int:
    import argparse
    ap = argparse.ArgumentParser(prog="oaci.diagnostics.report")
    ap.add_argument("--loso-root", required=True)
    ap.add_argument("--out-md", required=True)
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--tables-dir", required=True)
    ap.add_argument("--replay-dir", default=None, help="C10b replay JSON dir; enables Part 2 + final case")
    args = ap.parse_args(argv)
    folds, transfer, md, js = build_report(args.loso_root, replay_dir=args.replay_dir)
    for p in (args.out_md, args.out_json):
        os.makedirs(os.path.dirname(os.path.abspath(p)), exist_ok=True)
    tables = write_tables(folds, transfer, args.tables_dir, part2=js.get("part2_selector_replay"))
    with open(args.out_md, "w") as f:
        f.write(md)
    with open(args.out_json, "wb") as f:
        f.write(canonical_json_bytes(js))
    print(f"wrote {args.out_md} + {args.out_json} + {len(tables)} tables; part1_lean="
          f"{js['part1_case_lean']['part1_lean']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
