#!/usr/bin/env python
"""CMI-Trace P0.2 aggregation — read the immutable raw_rows.jsonl (one row per dataset/fold/seed/method) and
produce the objective->effect summary tables with PAIRED fold/subject-cluster 95% CIs. This script only READS
the raw artifacts; it never rewrites them.

Outputs (under results/cmi_trace_p0p1/objective_comparison/):
  objective_effect_summary.csv  per (dataset, method, metric): raw mean, cluster CI, n_folds, seed SD (descr.)
  paired_deltas.csv             per (dataset, method) delta vs ERM with cluster CI, for key endpoints
  cluster_intervals.csv         long-form (metric, method, dataset) cluster CI table
  selected_hparams.csv          per (dataset, fold, seed, select-method): selected lambda
  completeness_matrix.csv       expected vs present cells per (dataset, method)

Usage:
  python scripts/aggregate_cmi_trace_objective.py \
      --raw results/cmi_trace_p0p1/objective_comparison/*/raw_rows.jsonl \
      --out results/cmi_trace_p0p1/objective_comparison
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import summarize_metric, paired_delta_vs_baseline   # noqa: E402

METRICS = ["target_bacc", "source_bacc", "graph_kl", "node_kl", "graph_perm_p", "node_perm_p",
           "R_rel_k2", "R_rel_k2_random_control", "marginal_moment_gap", "class_conditional_moment_gap",
           "per_domain_risk_variance", "irmv1_diagnostic", "feature_norm", "top_singular_value",
           "effective_rank"]
DELTA_ENDPOINTS = ["target_bacc", "graph_kl", "node_kl", "R_rel_k2"]
EXPECTED = {"BNCI2014_001": 9, "BNCI2015_001": 12, "synthetic": 4}


def _load(paths):
    rows = []
    for p in paths:
        for gp in glob.glob(p):
            with open(gp) as fh:
                for line in fh:
                    line = line.strip()
                    if line:
                        rows.append(json.loads(line))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="+",
                    default=["results/cmi_trace_p0p1/objective_comparison/*/raw_rows.jsonl"])
    ap.add_argument("--out", default="results/cmi_trace_p0p1/objective_comparison")
    ap.add_argument("--n_boot", type=int, default=10000)
    args = ap.parse_args()
    rows = _load(args.raw)
    if not rows:
        raise SystemExit(f"[aggregate] no rows found under {args.raw}")
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    datasets = sorted({r["dataset"] for r in rows})
    methods = sorted({r["method"] for r in rows})
    print(f"[aggregate] {len(rows)} rows; datasets={datasets}; methods={methods}", flush=True)

    # --- objective_effect_summary.csv + cluster_intervals.csv
    summ, longf = [], []
    for ds in datasets:
        drows = [r for r in rows if r["dataset"] == ds]
        for m in sorted({r["method"] for r in drows}):
            mrows = [r for r in drows if r["method"] == m]
            for metric in METRICS:
                if not any(metric in r for r in mrows):
                    continue
                s = summarize_metric(mrows, metric, n_boot=args.n_boot)
                s.update(dataset=ds, method=m)
                summ.append(s)
                longf.append({"dataset": ds, "method": m, "metric": metric,
                              "mean": s["raw_mean"], "ci_lo": s["cluster_ci_lo"], "ci_hi": s["cluster_ci_hi"],
                              "n_folds": s["n_folds"], "seed_sd": s["seed_sd_descriptive"]})
    _write_csv(out / "objective_effect_summary.csv", summ,
               ["dataset", "method", "metric", "raw_mean", "cluster_ci_lo", "cluster_ci_hi",
                "n_folds", "n_clusters", "n_rows", "seed_sd_descriptive"])
    _write_csv(out / "cluster_intervals.csv", longf,
               ["dataset", "method", "metric", "mean", "ci_lo", "ci_hi", "n_folds", "seed_sd"])

    # --- paired_deltas.csv (vs ERM), cluster CI
    deltas = []
    for ds in datasets:
        drows = [r for r in rows if r["dataset"] == ds]
        if not any(r["method"] == "erm" for r in drows):
            continue
        for metric in DELTA_ENDPOINTS:
            pd = paired_delta_vs_baseline(drows, metric, baseline_method="erm", n_boot=args.n_boot)
            for method, d in pd.items():
                deltas.append({"dataset": ds, "method": method, "metric": metric,
                               "delta_mean": d["delta_mean"], "cluster_ci_lo": d["cluster_ci_lo"],
                               "cluster_ci_hi": d["cluster_ci_hi"], "n_clusters": d["n_clusters"]})
    _write_csv(out / "paired_deltas.csv", deltas,
               ["dataset", "method", "metric", "delta_mean", "cluster_ci_lo", "cluster_ci_hi", "n_clusters"])

    # --- selected_hparams.csv
    sel = [{"dataset": r["dataset"], "fold": r["fold"], "seed": r["seed"], "method": r["method"],
            "selected_lambda": r.get("selected_lambda")} for r in rows if r.get("select_row")]
    _write_csv(out / "selected_hparams.csv", sel, ["dataset", "fold", "seed", "method", "selected_lambda"])

    # --- completeness_matrix.csv (expected folds*seeds vs present)
    comp = []
    seeds = sorted({r["seed"] for r in rows})
    for ds in datasets:
        exp_folds = EXPECTED.get(ds, len({r["fold"] for r in rows if r["dataset"] == ds}))
        for m in methods:
            present = {(r["fold"], r["seed"]) for r in rows if r["dataset"] == ds and r["method"] == m}
            comp.append({"dataset": ds, "method": m, "expected_cells": exp_folds * len(seeds),
                         "present_cells": len(present),
                         "complete": len(present) >= exp_folds * len(seeds)})
    _write_csv(out / "completeness_matrix.csv", comp,
               ["dataset", "method", "expected_cells", "present_cells", "complete"])

    print(f"[aggregate] wrote 5 tables -> {out}", flush=True)
    # quick console read
    for ds in datasets:
        print(f"\n== {ds} target_bAcc delta vs ERM (cluster 95% CI) ==")
        for d in [x for x in deltas if x["dataset"] == ds and x["metric"] == "target_bacc"]:
            print(f"  {d['method']:20s} {d['delta_mean']:+.3f} [{d['cluster_ci_lo']:+.3f}, {d['cluster_ci_hi']:+.3f}] "
                  f"n={d['n_clusters']}")
    return 0


def _write_csv(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    sys.exit(main())
