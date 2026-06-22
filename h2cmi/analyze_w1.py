"""W1-A analysis: unseen-subject (LOSO) MI benchmark, same-backbone controlled panel. Per dataset +
overall: mean bAcc, mean paired delta vs identity (subject-bootstrap CI), macro-F1, harm rate,
worst-quartile delta. Unit = target subject. (W1-B native BTTA-DG is a SEPARATE external panel and is
NOT ranked against these numbers.)"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

METHODS = ["identity", "euclidean_alignment", "always_pooled", "always_canonical_CC",
           "current_joint", "spdim"]


def _rows(paths):
    out = []
    for p in paths:
        out += [json.loads(l) for l in open(p) if l.strip()]
    return out


def _wq(d):
    d = np.sort(np.asarray(d)); k = max(1, len(d) // 4)
    return float(d[:k].mean()) if len(d) else float("nan")


def _boot(vals, n=10000, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(n)]
    return float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def _stats(rows, m):
    d = [r["delta"] for r in rows if r["method"] == m]
    bacc = [r["bacc"] for r in rows if r["method"] == m]
    f1 = [r["macro_f1"] for r in rows if r["method"] == m]
    harm = [r["harm"] for r in rows if r["method"] == m]
    lo, hi = _boot(d)
    return dict(n=len(d), bacc=float(np.mean(bacc)) if bacc else float("nan"),
                mean_delta=float(np.mean(d)) if d else float("nan"), delta_ci=[lo, hi],
                macro_f1=float(np.mean(f1)) if f1 else float("nan"),
                harm_rate=float(np.mean(harm)) if harm else float("nan"), worst_quartile=_wq(d))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True, nargs="+")
    ap.add_argument("--out", default="results/h2cmi/w1.report.json")
    args = ap.parse_args()
    rows = _rows(args.inp)
    datasets = sorted(set(r["dataset"] for r in rows))
    rep = {"panel": "W1A", "n_rows": len(rows), "per_dataset": {}, "overall": {}}
    for ds in datasets:
        dr = [r for r in rows if r["dataset"] == ds]
        rep["per_dataset"][ds] = {m: _stats(dr, m) for m in METHODS}
    rep["overall"] = {m: _stats(rows, m) for m in METHODS}
    json.dump(rep, open(args.out, "w"), indent=2)
    print(f"=== W1-A unseen-subject (LOSO) MI benchmark (same-backbone; n_rows={len(rows)}) ===")
    for ds in datasets:
        n = rep["per_dataset"][ds]["identity"]["n"]
        print(f"\n[{ds}] (n={n} LOSO subjects)")
        print("  method                 bAcc   Δ vs id [ci95]            macroF1  harm   wq")
        for m in METHODS:
            s = rep["per_dataset"][ds][m]
            print(f"  {m:22s} {s['bacc']:.3f}  {s['mean_delta']:+.3f} [{s['delta_ci'][0]:+.3f},{s['delta_ci'][1]:+.3f}]  "
                  f"{s['macro_f1']:.3f}  {s['harm_rate']:.2f}  {s['worst_quartile']:+.3f}")
    print(f"\n[OVERALL] (n={rep['overall']['identity']['n']})")
    print("  method                 bAcc   Δ vs id [ci95]            macroF1  harm   wq")
    for m in METHODS:
        s = rep["overall"][m]
        print(f"  {m:22s} {s['bacc']:.3f}  {s['mean_delta']:+.3f} [{s['delta_ci'][0]:+.3f},{s['delta_ci'][1]:+.3f}]  "
              f"{s['macro_f1']:.3f}  {s['harm_rate']:.2f}  {s['worst_quartile']:+.3f}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
