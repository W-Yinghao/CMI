#!/usr/bin/env python
"""Aggregate P0.3 ticket CMI certification -> per (dataset, MLP-size) cluster CIs of the leakage removed
(delta_I = excess_over_null(full) - excess_over_null(deleted)). Certified iff ticket delta_I LCB>0 AND
ticket mean > random mean."""
from __future__ import annotations
import glob, json, sys
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
    m, lo, hi, n = cluster_bootstrap_ci(vals); return dict(mean=m, lo=lo, hi=hi, n=n)


def main():
    rows = []
    for fp in glob.glob(str(OUT / "*/cmi_cert_seed*.jsonl")):
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        sys.exit("[cmi-cert-agg] no rows")
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}):
        dr = [r for r in rows if r["dataset"] == ds]
        print(f"\n== {ds} EEGNet  posterior-KL leakage-removed ΔÎ (excess-over-null; subject-cluster 95% CI, n={len(dr)} rows) ==")
        print(f"   {'MLP':6s} {'subset':14s} {'ΔÎ (leakage removed)':>26s} {'perm_p':>7s}")
        for tag in ["small", "large"]:
            for name in ["ticket", "source_greedy", "random_k"]:
                ci = _ci(dr, f"dI_{tag}_{name}"); pp = _ci(dr, f"permp_{tag}_{name}")
                summary[f"{ds}|{tag}|{name}"] = ci
                cert = ci["lo"] > 0
                print(f"   {tag:6s} {name:14s} {ci['mean']:+.4f} [{ci['lo']:+.4f},{ci['hi']:+.4f}]"
                      f"   {pp['mean']:.3f}  {'<-- reduces leakage (LCB>0)' if cert else ''}")
            # certified verdict: ticket LCB>0 AND ticket>random
            tk = summary[f"{ds}|{tag}|ticket"]; rn = summary[f"{ds}|{tag}|random_k"]
            verdict = ("CERTIFIED" if (tk["lo"] > 0 and tk["mean"] > rn["mean"] + 1e-6)
                       else "NOT-certified (ticket not > random and/or LCB<=0)")
            print(f"   -> {tag}: ticket vs random => {verdict}")
    json.dump({k: v for k, v in summary.items()}, open(OUT / "cmi_cert_summary.json", "w"), indent=2, default=float)
    print(f"\n[cmi-cert-agg] wrote -> {OUT/'cmi_cert_summary.json'}")


if __name__ == "__main__":
    main()
