#!/usr/bin/env python
"""Paired λ=1.0 vs λ=0.25 comparison for the MCC lever (CPU, env c84c). Reads the two rounds' 3-arm dumps and
computes, PAIRED per (dataset, subject, seed):
  - mechanism amplification:      Δamp_BC = (dir_B−dir_C)|_{λ1} − (dir_B−dir_C)|_{λ0.25}   (did raising λ amplify
                                  the true-vs-shuffle geometry separation?)
  - geometry-vs-ERM amplification:Δamp_BA = (dir_B−dir_A)|_{λ1} − (dir_B−dir_A)|_{λ0.25}
  - mechanism-specific DG (λ1):   dU_spec1 = bAcc(B1) − bAcc(C1)
  - DG amplification:             dU_amp   = (bAcc_B−bAcc_C)|_{λ1} − (bAcc_B−bAcc_C)|_{λ0.25}
Inference unit = target subject (3 seeds paired within subject); subject-cluster bootstrap + EXACT sign-flip.
Routes PM A/B/C/D with the corrected labels (strict E is already falsified -> mechanism-specific-but-DG-inert).
REFUSES until both dirs have 63 bundles done. Manuscript FROZEN; only the project owner stops a scientific line.

  python scripts/aggregate_mcc_lambda_compare.py --lam1-dir results/cmi_trace_mcc_lambda1 --lam025-dir results/cmi_trace_mcc
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _geometry, _bacc_from_dump, _cluster_ci

DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _per_bundle(d):
    """{(ds,subj,seed): dict(dir_BA, dir_BC, dU_erm, dU_shuf, warmup_hash)} for one round dir."""
    out = {}
    for mf in sorted(glob.glob(str(Path(d) / "*.manifest.json"))):
        m = json.loads(open(mf).read()); ds, subj, seed = m["dataset"], m["subject"], m["seed"]; ar = m["arms"]
        if not all(k in ar for k in ("A_erm_continue", "B_mcc_true", "C_mcc_shuffle")):
            continue
        A, B, C = (str(Path(d) / ar[k]["dump"]) for k in ("A_erm_continue", "B_mcc_true", "C_mcc_shuffle"))
        gA, gB, gC = _geometry(A), _geometry(B), _geometry(C)
        if any("fail" in g for g in (gA, gB, gC)):
            continue
        out[(ds, subj, seed)] = dict(
            dir_BA=gB["dir_consistency"] - gA["dir_consistency"], dir_BC=gB["dir_consistency"] - gC["dir_consistency"],
            ratio_BC=gB["gdis_gshared_ratio"] - gC["gdis_gshared_ratio"],
            dU_erm=ar["B_mcc_true"]["target_bacc"] - ar["A_erm_continue"]["target_bacc"],
            dU_shuf=ar["B_mcc_true"]["target_bacc"] - ar["C_mcc_shuffle"]["target_bacc"],
            src_drop=ar["A_erm_continue"]["source_val_bacc"] - ar["B_mcc_true"]["source_val_bacc"],
            grad_ratio=ar["B_mcc_true"].get("mean_grad_ratio"), warmup_hash=m["warmup_hash"])
    return out


def _route(amp_bc_lcbs, ba_lam1_lcb, dU_spec1_lcbs, worst_src_drop):
    # amplification must be STATISTICALLY significant (a dataset's amp_BC 95% LCB > 0), not merely mean-positive:
    # a 4x stronger update that leaves the true-vs-shuffle separation unchanged has NOT amplified the geometry.
    geom_amplified = any(l > 0 for l in amp_bc_lcbs)
    geom_moves_vs_erm = ba_lam1_lcb > 0
    dg_beats_shuffle = any(l > 0 for l in dU_spec1_lcbs)
    damaged = worst_src_drop > 0.02
    if geom_amplified and geom_moves_vs_erm and dg_beats_shuffle and not damaged:
        return dict(verdict="A_global_consistency_has_DG_value", next="seeds_3_4 + ccoral_same_protocol + dgcnn_replication + CMI_posterior_audit (NOT TTE yet)")
    if geom_amplified and not dg_beats_shuffle and not damaged:
        return dict(verdict="GLOBAL_MCC_MECHANISM_SPECIFIC_BUT_DG_INERT",
                    next="risk_weighted_MCC (weight consistency by source-only predictive instability: LOSO risk / class-margin / subject-head residual)")
    if not geom_amplified:
        # amp_BC=NS rules OUT "lambda amplifies geometry" (kills GLOBAL) but does NOT positively establish an
        # estimator cause -- the noise-domination premise for EMA is unmeasured (first-batch-only grad log) and
        # grad_cos>0 in all bundles argues against it; geometry is also decoupled from DG (corr~0). So report the
        # descriptive label + an explicit FORK gated behind a cheap discriminator, NOT a committed EMA round.
        return dict(verdict="SPECIFIC_BUT_LAMBDA_INERT_estimator_UNDIAGNOSED",
                    next="present FORK (EMA/prototype estimator vs risk-weighted MCC vs drop-the-geometry-axis); gate behind ONE cheap discriminator first: full-batch/large-batch low-variance MCC + batch-to-batch variance logging; EMA earns a full round only if the low-variance limit BOTH scales B-C AND that geometry then tracks DG")
    if geom_amplified and damaged:
        return dict(verdict="D_geometry_up_but_damaged", next="C_d=C_shared+R_d, constrain only residual cross-subject variance")
    return dict(verdict="INCONCLUSIVE", next="inspect per-subject sign + seed variance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lam1-dir", default="results/cmi_trace_mcc_lambda1")
    ap.add_argument("--lam025-dir", default="results/cmi_trace_mcc"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    for d in (a.lam1_dir, a.lam025_dir):
        nd = len(glob.glob(str(Path(d) / "*_sub*_seed*.done")))
        if nd < a.expect:
            print(f"[mcc-λcmp] INCOMPLETE {d}: {nd}/{a.expect} bundles done -> REFUSING to compare."); sys.exit(2)
    P1, P0 = _per_bundle(a.lam1_dir), _per_bundle(a.lam025_dir)
    keys = sorted(set(P1) & set(P0))
    # warm-up hash identity across rounds (paired warm-ups must match)
    mismatched = [k for k in keys if P1[k]["warmup_hash"] != P0[k]["warmup_hash"]]
    per_subj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # ds -> metric -> subj -> [seed vals]
    for (ds, subj, seed) in keys:
        p1, p0 = P1[(ds, subj, seed)], P0[(ds, subj, seed)]
        per_subj[ds]["amp_bc"][subj].append(p1["dir_BC"] - p0["dir_BC"])
        per_subj[ds]["amp_ba"][subj].append(p1["dir_BA"] - p0["dir_BA"])
        per_subj[ds]["ba_lam1"][subj].append(p1["dir_BA"])
        per_subj[ds]["dU_spec1"][subj].append(p1["dU_shuf"])
        per_subj[ds]["dU_amp"][subj].append(p1["dU_shuf"] - p0["dU_shuf"])
        per_subj[ds]["dU_erm1"][subj].append(p1["dU_erm"])
        per_subj[ds]["src_drop1"][subj].append(p1["src_drop"])
        per_subj[ds]["grad_ratio1"][subj].append(p1["grad_ratio"] if p1["grad_ratio"] is not None else np.nan)

    summ = []
    for ds in DATASETS:
        if ds not in per_subj:
            continue
        M = {k: _cluster_ci([np.mean(v) for v in per_subj[ds][k].values()]) for k in per_subj[ds]}
        summ.append(dict(dataset=ds, n_subjects=M["amp_bc"]["n"],
                         amp_bc_mean=M["amp_bc"]["mean"], amp_bc_lcb=M["amp_bc"]["lo"], amp_bc_signflip_p=M["amp_bc"]["signflip_p"],
                         amp_ba_mean=M["amp_ba"]["mean"], dir_BA_lam1_mean=M["ba_lam1"]["mean"], dir_BA_lam1_lcb=M["ba_lam1"]["lo"],
                         dU_spec1_mean=M["dU_spec1"]["mean"], dU_spec1_lcb=M["dU_spec1"]["lo"], dU_spec1_signflip_p=M["dU_spec1"]["signflip_p"],
                         dU_amp_mean=M["dU_amp"]["mean"], dU_erm1_mean=M["dU_erm1"]["mean"],
                         src_drop1=M["src_drop1"]["mean"], grad_ratio1=M["grad_ratio1"]["mean"]))
    ba_lcb = max([s["dir_BA_lam1_lcb"] for s in summ], default=-1.0)
    worst_drop = max([s["src_drop1"] for s in summ], default=0.0)
    route = _route([s["amp_bc_lcb"] for s in summ], ba_lcb, [s["dU_spec1_lcb"] for s in summ], worst_drop)

    out = Path(a.lam1_dir)
    json.dump(dict(per_dataset=summ, routing=route, n_paired=len(keys), warmup_hash_mismatched=mismatched,
                   discipline="paired lambda amplification; exact sign-flip; corrected labels (strict E falsified); manuscript FROZEN"),
              open(out / "mcc_lambda_compare_verdict.json", "w"), indent=2, default=float)
    print(f"[mcc-λcmp] paired {len(keys)} bundles; warmup-hash mismatches={len(mismatched)}; routing={route['verdict']}")
    for s in summ:
        print(f"  {s['dataset']}: amp_BC={s['amp_bc_mean']:+.5f}[lcb {s['amp_bc_lcb']:+.5f}] p={s['amp_bc_signflip_p']:.3f} | "
              f"dir_BA@λ1={s['dir_BA_lam1_mean']:+.5f}[lcb {s['dir_BA_lam1_lcb']:+.5f}] | "
              f"dU_spec1={s['dU_spec1_mean']:+.4f}[lcb {s['dU_spec1_lcb']:+.4f}] p={s['dU_spec1_signflip_p']:.3f} | "
              f"grad_ratio@λ1={s['grad_ratio1']:.3f} src_drop={s['src_drop1']:+.4f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
