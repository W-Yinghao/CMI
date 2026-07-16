#!/usr/bin/env python
"""Aggregate the CORRECTED (F2.0b) ticket CMI certification -> paired dI_specific cluster CIs + verdict.

Certified iff cluster LCB95(dI_specific) > 0 at the PRIMARY (large) capacity, where
  dI_specific = mean_i kl(random_i) - kl(ticket)   (paired; positive => ticket removes MORE leakage than a
matched-rank random deletion). Robustness across {linear, small, large} capacities is reported."""
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
    for fp in glob.glob(str(OUT / "*/cmi_cert2_seed*.jsonl")):
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        sys.exit("[cmi-cert2-agg] no rows (run the corrected certification first)")
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}):
        dr = [r for r in rows if r["dataset"] == ds]
        print(f"\n== {ds} EEGNet  CORRECTED paired CMI certification (dI_specific = mean kl(random) - kl(ticket);"
              f" subject-cluster 95% CI, n={len(dr)} rows) ==")
        print(f"   {'capacity':9s} {'dI_specific (ticket beyond random)':>36s} {'dI_ticket':>11s} {'dI_random':>11s}")
        for tag in ["linear", "small", "large"]:
            spec = _ci(dr, f"dI_specific_{tag}"); dit = _ci(dr, f"dI_ticket_{tag}"); dir_ = _ci(dr, f"dI_random_{tag}")
            summary[f"{ds}|{tag}"] = spec
            cert = spec["lo"] > 0
            print(f"   {tag:9s} {spec['mean']:+.4f} [{spec['lo']:+.4f},{spec['hi']:+.4f}]     "
                  f"{dit['mean']:+.4f}   {dir_['mean']:+.4f}   {'<-- LCB>0' if cert else ''}")
        mx = _ci(dr, "dI_specific_max")
        prim = summary[f"{ds}|large"]
        verdict = ("CERTIFIED (primary large-capacity LCB>0)" if prim["lo"] > 0
                   else "NOT-certified (ticket does not remove more validated leakage than matched-rank random)")
        print(f"   max-over-capacity dI_specific {mx['mean']:+.4f} [{mx['lo']:+.4f},{mx['hi']:+.4f}] (optimistic; selection-inflated)")
        print(f"   -> {ds}: {verdict}")
        summary[f"{ds}|verdict"] = verdict
    json.dump(summary, open(OUT / "cmi_cert2_summary.json", "w"), indent=2, default=float)
    print(f"\n[cmi-cert2-agg] wrote -> {OUT/'cmi_cert2_summary.json'}")


if __name__ == "__main__":
    main()
