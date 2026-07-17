#!/usr/bin/env python
"""B0: Track B B1a WITHIN-FOLD geometry characterization aggregator (characterization-only closeout). Reports
subject-cluster CIs of task-head overlap and basis-pair overlap, plus completeness. NO cross-fold stability
(B1b STOPPED). The effective-rank fields in the raw rows are EXCLUDED: they were computed on already-
orthonormalized bases, so they equal the basis rank and carry no information (a valid effective rank would come
from the pre-orthogonalization singular spectrum). Duplicate (dataset,backbone,subject,seed) cells fail-closed.

  python scripts/aggregate_trackb_geometry.py
"""
from __future__ import annotations
import csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
G = REPO / "results" / "cmi_trace_trackb_geometry"
EXCLUDED = ["union_effective_rank", "cond_eff_rank", "rule_eff_rank", "grad_eff_rank"]   # orthonormalized-basis artifacts


def _ci(vals, seed=7):
    v = np.asarray([x for x in vals if x is not None and np.isfinite(x)], float)
    if not v.size:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed); b = [np.mean(v[rng.integers(0, v.size, v.size)]) for _ in range(10000)]
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(v.size))


def _subject_means(rows, key):
    by = defaultdict(list)
    for r in rows:
        if r.get(key) is not None and np.isfinite(r.get(key, float("nan"))):
            by[r["heldout_subject"]].append(r[key])
    return [float(np.mean(v)) for v in by.values()]


def main():
    rows = []
    for fp in sorted(glob.glob(str(G / "geometry_rows_*_0-1-2.jsonl"))):     # full-run files only (avoid smoke dups)
        rows += [json.loads(l) for l in open(fp)]
    if not rows:
        sys.exit("[trackb-agg] no full-run geometry rows")
    # fail-closed on duplicate cells
    seen = set(); dups = []
    for r in rows:
        key = (r["dataset"], r["backbone"], r["heldout_subject"], r["seed"])
        if key in seen:
            dups.append(key)
        seen.add(key)
    if dups:
        sys.exit(f"[trackb-agg] FAIL-CLOSED: {len(dups)} duplicate (dataset,backbone,subject,seed) cells: {dups[:5]}")
    ho_rows, bp_rows, comp_rows, summ = [], [], [], []
    for bb in sorted({r["backbone"] for r in rows}):
        for ds in sorted({r["dataset"] for r in rows if r["backbone"] == bb}):
            R = [r for r in rows if r["backbone"] == bb and r["dataset"] == ds]
            n_subj = len({r["heldout_subject"] for r in R})
            comp_rows.append(dict(backbone=bb, dataset=ds, n_cells=len(R), n_subjects=n_subj,
                                  seeds=sorted({r["seed"] for r in R}), latent_dim=R[0]["latent_dim"]))
            for name in ("cond", "rule", "grad"):
                ci = _ci(_subject_means(R, f"{name}_head_overlap"))
                enr = _ci(_subject_means(R, f"{name}_head_overlap_enrichment"))   # vs isotropic rank(Wc)/d
                nrank = _ci(_subject_means(R, f"{name}_numerical_rank"))
                ho_rows.append(dict(backbone=bb, dataset=ds, basis=name, head_overlap_mean=ci["mean"],
                                    lo=ci["lo"], hi=ci["hi"], head_overlap_enrichment_mean=enr["mean"],
                                    enrichment_lo=enr["lo"], enrichment_hi=enr["hi"],
                                    numerical_rank_mean=nrank["mean"], n_subjects=ci["n"]))
            for pair in ("cond_rule", "cond_grad", "rule_grad"):
                a_ci = _ci(_subject_means(R, f"angle_{pair}_meancos"))
                o_ci = _ci(_subject_means(R, f"overlap_{pair}"))
                bp_rows.append(dict(backbone=bb, dataset=ds, pair=pair, mean_cos_principal_angle=a_ci["mean"],
                                    angle_lo=a_ci["lo"], angle_hi=a_ci["hi"], projector_overlap=o_ci["mean"],
                                    overlap_lo=o_ci["lo"], overlap_hi=o_ci["hi"]))
            cr = _ci(_subject_means(R, "cond_contested_rank"))
            summ.append(dict(backbone=bb, dataset=ds, n_cells=len(R),
                             cond_head_overlap=next(h["head_overlap_mean"] for h in ho_rows if h["backbone"] == bb and h["dataset"] == ds and h["basis"] == "cond"),
                             cond_contested_rank_mean=cr["mean"],
                             note=("cond head-overlap ACCEPTED; B_grad numerical-rank FIXED (algebraic sanity: grad in "
                                   "row(Wc), rank<=C-1) but the class-CONDITIONAL g_{d,y} estimator is NOT YET "
                                   "implemented; B_rule is PROVISIONAL (free per-subject head, NOT the shared+residual "
                                   "ridge W_d=W_0+dW_d of M0.2); effective_rank fields excluded; B1b stability not run")))

    def _w(fp, r, keys):
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: x.get(k) for k in keys}) for x in r]
    _w(G / "trackb_head_overlap_cluster_ci.csv", ho_rows, ["backbone", "dataset", "basis", "head_overlap_mean", "lo", "hi",
        "head_overlap_enrichment_mean", "enrichment_lo", "enrichment_hi", "numerical_rank_mean", "n_subjects"])
    _w(G / "trackb_basis_pair_overlap_cluster_ci.csv", bp_rows, ["backbone", "dataset", "pair", "mean_cos_principal_angle", "angle_lo", "angle_hi", "projector_overlap", "overlap_lo", "overlap_hi"])
    _w(G / "trackb_geometry_completeness.csv", comp_rows, ["backbone", "dataset", "n_cells", "n_subjects", "seeds", "latent_dim"])
    _w(G / "trackb_within_fold_geometry_summary.csv", summ, list(summ[0].keys()))
    print(f"[trackb-agg] {len(rows)} cells, no duplicates. EXCLUDED effective-rank fields: {EXCLUDED}")
    for h in ho_rows:
        if h["basis"] == "cond":
            print(f"  {h['backbone']}/{h['dataset']}: cond head_overlap={h['head_overlap_mean']:.3f} [{h['lo']:.3f},{h['hi']:.3f}] (n={h['n_subjects']})")


if __name__ == "__main__":
    main()
