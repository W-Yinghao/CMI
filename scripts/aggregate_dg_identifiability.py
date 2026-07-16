#!/usr/bin/env python
"""Aggregate the HARDENED DG-identifiability rows -> per-config RecoveryRatio + 4-state verdict with
subject/fold-cluster CIs. Reports the full (family x contested x objective) table, the best config per
dataset, and an overall verdict WITH an explicit multiplicity caveat (16 configs are searched, so any
SOURCE_IDENTIFIABLE_PRACTICAL hit requires the pre-registered confirmatory dataset before belief)."""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import cluster_bootstrap_ci
from tos_cmi.eval.dg_identifiability import recovery_verdict

OUT = REPO / "results" / "cmi_trace_dg_identifiability"


def _fold_means(rr, key):
    """Per-heldout-subject mean of `key` (subject = cluster unit), then cluster-bootstrap CI."""
    by = defaultdict(list)
    for r in rr:
        v = r.get(key)
        if v is not None and np.isfinite(v):
            by[r["heldout_subject"]].append(v)
    vals = [np.mean(v) for v in by.values()]
    if not vals:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    mean, lo, hi, n = cluster_bootstrap_ci(vals)
    return dict(mean=mean, lo=lo, hi=hi, n=n)


def main():
    rows = []
    for fp in glob.glob(str(OUT / "*/raw_rows.jsonl")):
        rows += [json.loads(l) for l in open(fp) if l.strip()]
    if not rows:
        sys.exit("[dg-id-agg] no rows")
    table, verdicts = [], {}
    for ds in sorted({r["dataset"] for r in rows}):
        dr = [r for r in rows if r["dataset"] == ds]
        configs = sorted({(r["family"], r["contested"], r["objective"]) for r in dr})
        best = None
        for (fam, cont, obj) in configs:
            rr = [r for r in dr if r["family"] == fam and r["contested"] == cont and r["objective"] == obj]
            orc = _fold_means(rr, "oracle_delta_query"); orc_r = _fold_means(rr, "oracle_delta_query_random")
            meta = _fold_means(rr, "meta_delta_query"); meta_r = _fold_means(rr, "meta_delta_query_random")
            kbar = float(np.mean([r["meta_k_star"] for r in rr]))
            v = recovery_verdict(orc["mean"], orc["lo"], meta["mean"], meta["lo"], meta_r["mean"])
            rec = {"dataset": ds, "family": fam, "contested": cont, "objective": obj,
                   "oracle_dq_mean": orc["mean"], "oracle_dq_lo": orc["lo"], "oracle_dq_rand": orc_r["mean"],
                   "meta_dq_mean": meta["mean"], "meta_dq_lo": meta["lo"], "meta_dq_hi": meta["hi"],
                   "meta_dq_rand": meta_r["mean"], "meta_kbar": kbar,
                   "recovery_ratio": v["recovery_ratio"], "state": v["state"], "n_folds": meta["n"]}
            table.append(rec)
            # rank configs by (source-identifiable first, then meta lower-CI)
            keyv = (v["state"] == "SOURCE_IDENTIFIABLE_PRACTICAL", meta["lo"])
            if best is None or keyv > best[0]:
                best = (keyv, rec)
        verdicts[ds] = {"best_config": best[1] if best else None,
                        "n_configs_searched": len(configs),
                        "any_source_identifiable": any(t["state"] == "SOURCE_IDENTIFIABLE_PRACTICAL"
                                                       for t in table if t["dataset"] == ds)}
    with open(OUT / "dg_identifiability_table.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(table[0].keys())); w.writeheader(); [w.writerow(r) for r in table]
    json.dump({"verdicts": verdicts, "table": table}, open(OUT / "dg_identifiability_verdict.json", "w"),
              indent=2, default=float)
    print(f"[dg-id-agg] {len(rows)} rows, {len(table)} config-cells -> {OUT}")
    for ds, v in verdicts.items():
        b = v["best_config"]
        print(f"\n== {ds}: {v['n_configs_searched']} configs; any_source_identifiable={v['any_source_identifiable']} ==")
        if b:
            print(f"   BEST: {b['family']}/cont={int(b['contested'])}/{b['objective']} -> {b['state']}")
            print(f"     oracle dq={b['oracle_dq_mean']:+.3f}[lo{b['oracle_dq_lo']:+.3f}] rand={b['oracle_dq_rand']:+.3f}")
            print(f"     meta   dq={b['meta_dq_mean']:+.3f}[{b['meta_dq_lo']:+.3f},{b['meta_dq_hi']:+.3f}] rand={b['meta_dq_rand']:+.3f} kbar={b['meta_kbar']:.1f} recovery={b['recovery_ratio']}")
        # show the strongest source-meta config regardless of state
        dst = sorted([t for t in table if t["dataset"] == ds], key=lambda t: -t["meta_dq_lo"])[:3]
        print("   top source-meta configs (by meta LCB):")
        for t in dst:
            print(f"     {t['family']:4s} cont={int(t['contested'])} {t['objective']:8s} "
                  f"meta dq={t['meta_dq_mean']:+.3f}[{t['meta_dq_lo']:+.3f},{t['meta_dq_hi']:+.3f}] "
                  f"oracle dq={t['oracle_dq_mean']:+.3f} -> {t['state']}")


if __name__ == "__main__":
    main()
