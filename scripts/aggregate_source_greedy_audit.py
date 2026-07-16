#!/usr/bin/env python
"""Aggregate the GREEDY source-only identifiability audit -> per (dataset, family) cluster CIs + verdict.

Identifiability holds for a family iff the source-greedy deletion, applied to the TRUE target, gives a target
gain with LCB>0 that BEATS matched-rank random, AND its subspace aligns with the greedy target ticket.
Otherwise the greedy ticket is confirmed-existent (oracle) but NOT source-identifiable = TARGET_HINDSIGHT_ONLY.
"""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import cluster_bootstrap_ci

OUT = REPO / "results" / "cmi_trace_dg_identifiability"


def _ci(rr, key):
    by = defaultdict(list)
    for r in rr:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by[r["heldout_subject"]].append(v)
    vals = [np.mean(v) for v in by.values()]
    if not vals:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    m, lo, hi, n = cluster_bootstrap_ci(vals)
    return dict(mean=m, lo=lo, hi=hi, n=n)


def main():
    rows = []
    for fp in glob.glob(str(OUT / "*/audit_rows_seed*.jsonl")):
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        sys.exit("[src-audit-agg] no rows")
    table = []
    for ds in sorted({r["dataset"] for r in rows}):
        dr = [r for r in rows if r["dataset"] == ds]
        print(f"\n== {ds} EEGNet  (greedy source-only identifiability audit; subject/fold-cluster 95% CI) ==")
        print(f"   {'family':5s} {'src->tgt Δ':>20s} {'src random':>12s} {'align':>7s} {'k_src':>6s} | {'oracle(exist)':>14s}")
        for fam in ["marg", "cond", "rule", "grad"]:
            rr = [r for r in dr if r["family"] == fam]
            if not rr:
                continue
            sd = _ci(rr, "src_delta_target"); rnd = _ci(rr, "src_delta_target_random")
            al = _ci(rr, "src_tgt_alignment"); orc = _ci(rr, "oracle_delta_query")
            ksrc = float(np.mean([r["k_src"] for r in rr]))
            frac_sel = float(np.mean([r["k_src"] > 0 for r in rr]))
            identifiable = (sd["lo"] > 0) and (sd["mean"] > rnd["mean"] + 1e-6)
            table.append({"dataset": ds, "family": fam, "src_delta_mean": sd["mean"], "src_delta_lo": sd["lo"],
                          "src_delta_hi": sd["hi"], "src_random_mean": rnd["mean"], "alignment_mean": al["mean"],
                          "k_src_mean": ksrc, "frac_selected": frac_sel, "oracle_exist_mean": orc["mean"],
                          "oracle_exist_lo": orc["lo"], "identifiable": identifiable})
            flag = "  <-- IDENTIFIABLE" if identifiable else ""
            print(f"   {fam:5s} {sd['mean']:+.4f}[{sd['lo']:+.4f},{sd['hi']:+.4f}] {rnd['mean']:+.4f}    "
                  f"{al['mean']:.2f}   {ksrc:.1f}(sel {frac_sel:.0%}) | {orc['mean']:+.4f}[{orc['lo']:+.4f}]{flag}")
    with open(OUT / "source_greedy_audit_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); [w.writerow(r) for r in table]
    any_id = [t for t in table if t["identifiable"]]
    print(f"\n[src-audit-agg] {len(rows)} rows -> identifiable families: "
          f"{[(t['dataset'], t['family']) for t in any_id] or 'NONE (greedy ticket exists but is NOT source-identifiable)'}")
    json.dump(table, open(OUT / "source_greedy_audit_verdict.json", "w"), indent=2, default=float)


if __name__ == "__main__":
    main()
