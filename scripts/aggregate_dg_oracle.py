#!/usr/bin/env python
"""Aggregate the DG-erasure oracle rows -> go/no-go verdict (A/B/C) with subject/fold-cluster CIs."""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import cluster_bootstrap_ci

OUT = REPO / "results" / "cmi_trace_dg_oracle"


def _ci(vals):
    return cluster_bootstrap_ci([v for v in vals if np.isfinite(v)])


def verdict(ub, meta_lo, meta_hi, meta_mean, rand_mean):
    """A: source-identifiable (meta CI>0 & beats random). B: target-only (UB>0 but meta not). C: none."""
    if meta_lo > 0 and meta_mean > rand_mean + 1e-6:
        return "A_SOURCE_IDENTIFIABLE_DG_TICKET"
    if ub > 0.01:
        return "B_TARGET_ONLY_NOT_SOURCE_IDENTIFIABLE"
    return "C_NO_BENEFICIAL_SUBSET"


def main():
    rows = []
    for fp in glob.glob(str(OUT / "*/raw_rows.jsonl")):
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        sys.exit("[dg-agg] no rows")
    summ, verdicts = [], {}
    for (ds, bb) in sorted({(r["dataset"], r["backbone"]) for r in rows}):
        rr = [r for r in rows if r["dataset"] == ds and r["backbone"] == bb]
        by = defaultdict(lambda: defaultdict(list))
        for r in rr:
            for k in ("d_target_upper_bound", "d_target_source_meta", "d_target_random", "d_target_cmi_only"):
                by[r["heldout_subject"]][k].append(r.get(k, np.nan))
        cell = {k: {s: np.mean(v[k]) for s, v in by.items()} for k in
                ("d_target_upper_bound", "d_target_source_meta", "d_target_random", "d_target_cmi_only")}
        m = {}
        for k, cl in cell.items():
            mean, lo, hi, n = _ci(list(cl.values()))
            m[k] = dict(mean=mean, lo=lo, hi=hi, n=n)
        v = verdict(m["d_target_upper_bound"]["mean"], m["d_target_source_meta"]["lo"],
                    m["d_target_source_meta"]["hi"], m["d_target_source_meta"]["mean"], m["d_target_random"]["mean"])
        verdicts[f"{ds}|{bb}"] = {"verdict": v, **{k: m[k] for k in m}}
        for k in m:
            summ.append({"dataset": ds, "backbone": bb, "metric": k, "mean": m[k]["mean"],
                         "ci_lo": m[k]["lo"], "ci_hi": m[k]["hi"], "n_folds": m[k]["n"]})
    with open(OUT / "dg_oracle_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "backbone", "metric", "mean", "ci_lo", "ci_hi", "n_folds"])
        w.writeheader(); [w.writerow(r) for r in summ]
    json.dump(verdicts, open(OUT / "dg_oracle_verdict.json", "w"), indent=2, default=float)
    print(f"[dg-agg] {len(rows)} rows -> {OUT}")
    for k, v in verdicts.items():
        print(f"  {k}: {v['verdict']}")
        for mk in ("d_target_upper_bound", "d_target_source_meta", "d_target_random", "d_target_cmi_only"):
            print(f"     {mk:22s} {v[mk]['mean']:+.3f} [{v[mk]['lo']:+.3f},{v[mk]['hi']:+.3f}] n={v[mk]['n']}")


if __name__ == "__main__":
    main()
