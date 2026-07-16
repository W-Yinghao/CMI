#!/usr/bin/env python
"""Aggregate the erasure-oracle raw rows -> committable summary CSVs with subject/fold-cluster CIs.
Reads immutable raw_rows; never rewrites them. Fails loudly if inputs absent."""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from cmi.eval.objective_effect_report import cluster_bootstrap_ci

OUT = REPO / "results" / "cmi_trace_erasure_oracle"


def _ci(vals):
    m, lo, hi, n = cluster_bootstrap_ci(vals); return m, lo, hi, n


def main():
    # exact-head-null oracle
    ehn = []
    for fp in glob.glob(str(OUT / "dgcnn_*/raw_rows.jsonl")):
        rows = [json.loads(l) for l in open(fp) if l.strip()]
        for tm in sorted({r["training_method"] for r in rows}):
            rr = [r for r in rows if r["training_method"] == tm]
            bys = defaultdict(lambda: defaultdict(list))
            for r in rr:
                bys[r["heldout_subject"]]["hn"].append(r["delta_D_headnull"])
                bys[r["heldout_subject"]]["rn"].append(r["delta_D_randomnull"])
                bys[r["heldout_subject"]]["spec"].append(r["delta_D_headnull"] - r["delta_D_randomnull"])
            m_hn, lo_hn, hi_hn, n = _ci([np.mean(v["hn"]) for v in bys.values()])
            m_sp, lo_sp, hi_sp, _ = _ci([np.mean(v["spec"]) for v in bys.values()])
            fullkl = float(np.mean([r["cmi_full_kl"] for r in rr]))
            ehn.append({"dataset": rr[0]["dataset"], "training_method": tm, "n_folds": n,
                        "delta_D_headnull": m_hn, "ci_lo": lo_hn, "ci_hi": hi_hn,
                        "frac_of_CMI": (m_hn / fullkl if fullkl > 0 else float("nan")),
                        "specific_vs_random": m_sp, "spec_ci_lo": lo_sp, "spec_ci_hi": hi_sp,
                        "max_softmax_replay_err": max(r["softmax_replay_err_headnull"] for r in rr),
                        "task_all_unchanged": all(r["task_bacc_unchanged"] for r in rr)})
    _w(OUT / "exact_head_null_summary.csv", ehn,
       ["dataset", "training_method", "n_folds", "delta_D_headnull", "ci_lo", "ci_hi", "frac_of_CMI",
        "specific_vs_random", "spec_ci_lo", "spec_ci_hi", "max_softmax_replay_err", "task_all_unchanged"])

    # EEGNet subset oracle
    sub = []
    for fp in glob.glob(str(OUT / "subset_oracle_tos_*/raw_rows.jsonl")):
        rows = [json.loads(l) for l in open(fp) if l.strip() and "delta_best_subset" in l]
        if not rows:
            continue
        bys = defaultdict(lambda: defaultdict(list))
        for r in rows:
            for kk in ("delta_full_basis", "delta_best_prefix", "delta_best_subset", "delta_same_rank_random"):
                bys[r["heldout_subject"]][kk].append(r[kk])
        row = {"dataset": rows[0]["dataset"], "backbone": rows[0]["backbone"], "n_folds": len(bys)}
        for kk in ("delta_full_basis", "delta_best_prefix", "delta_best_subset", "delta_same_rank_random"):
            m, lo, hi, _ = _ci([np.mean(v[kk]) for v in bys.values()])
            row[kk] = m; row[kk + "_lo"] = lo; row[kk + "_hi"] = hi
        row["best_subset_beats_random_frac"] = float(np.mean([r["best_subset_beats_random"] for r in rows]))
        sub.append(row)
    fields = ["dataset", "backbone", "n_folds"]
    for kk in ("delta_full_basis", "delta_best_prefix", "delta_best_subset", "delta_same_rank_random"):
        fields += [kk, kk + "_lo", kk + "_hi"]
    fields += ["best_subset_beats_random_frac"]
    _w(OUT / "eegnet_subset_oracle_summary.csv", sub, fields)
    print(f"[oracle-agg] exact-head-null {len(ehn)} strata; subset {len(sub)} strata -> {OUT}")


def _w(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore"); w.writeheader()
        for r in rows:
            w.writerow(r)


if __name__ == "__main__":
    main()
