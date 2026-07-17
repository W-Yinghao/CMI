#!/usr/bin/env python
"""Aggregate the MCC 3-arm dumps into DG utility + geometry movement + A-E routing (CPU, env c84c). Reads the
per-bundle arm dumps written by tos_cmi/train/run_mcc_arms.py (A_erm_continue / B_mcc_true / C_mcc_shuffle) and
computes, per dataset:
  - DG utility (primary):  dU_MCC-ERM     = bAcc_target(MCC_true) - bAcc_target(ERM_continue)
                           dU_MCC-shuffle = bAcc_target(MCC_true) - bAcc_target(MCC_shuffle)   (mechanism control)
    inference unit = outer target subject; 3 seeds averaged per subject then subject-cluster bootstrap + EXACT
    sign-flip p (reuse mechanism_subspace.exact_sign_flip_p).
  - Geometry movement (source-whitened frozen features): source direction-consistency (mean cos of per-subject
    unit class-contrasts to the LOSO consensus = 1 - L_MCC), tr(G_dis)/tr(G_shared), and target-to-source contrast
    alignment A_t (target labels EVAL-only). Delta = MCC_true - ERM_continue.
  - Collapse/damage: source bAcc, effective rank, contrast norm (from the arm diag).
Routes PM A-E. NO CLOSED verdict; refuses to aggregate until all bundles are .done. Manuscript FROZEN.

  python scripts/aggregate_mcc.py --from-dir results/cmi_trace_mcc --expect 63
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from sklearn.metrics import balanced_accuracy_score
from tos_cmi.eval import targetx_metric as TM
from tos_cmi.eval import mechanism_subspace as MS
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.train.mechanism_consistency import class_pairs

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
ARMS = ["A_erm_continue", "B_mcc_true", "C_mcc_shuffle"]


def _bacc_from_dump(npz):
    d = np.load(npz, allow_pickle=True)
    return float(balanced_accuracy_score(d["y_target"], d["logits_target"].argmax(1)))


def _geometry(npz):
    """Source direction-consistency + tr(G_dis)/tr(G_shared) + target-to-source alignment A_t, on whitened feats."""
    f = feat_from_tos_dump(npz)
    Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
    Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int)
    W = TM.source_whitener(Zs); Zs_w = TM.to_whitened(Zs, W); Zt_w = TM.to_whitened(Zt, W)
    cd = MS.build_contrast_disagreement(Zs_w, ys, dsc)
    if cd["fail_closed"]:
        return dict(fail=cd["reason"])
    ratio = float(np.trace(cd["G_dis"]) / (np.trace(cd["G_shared"]) + 1e-12))
    # source direction consistency = mean over pairs,subjects of cos(u_d, LOSO-consensus) = 1 - L_MCC
    classes = sorted(np.unique(ys).tolist()); subs = np.unique(dsc); pairs = class_pairs(classes)
    def _unit(mask_mean):
        v = mask_mean; return v / (np.linalg.norm(v) + 1e-12)
    cons = []
    for (a, b) in pairs:
        U = []
        ok = True
        for s in subs:
            ma = (dsc == s) & (ys == a); mb = (dsc == s) & (ys == b)
            if ma.sum() == 0 or mb.sum() == 0:
                ok = False; break
            U.append(_unit(Zs_w[ma].mean(0) - Zs_w[mb].mean(0)))
        if not ok:
            continue
        U = np.array(U); tot = U.sum(0)
        for i in range(len(subs)):
            ubar = tot - U[i]; ubar = ubar / (np.linalg.norm(ubar) + 1e-12)
            cons.append(float(U[i] @ ubar))
    # target-to-source contrast alignment A_t (target labels EVAL only)
    At = []
    for (a, b) in pairs:
        ma = yt == a; mb = yt == b
        if ma.sum() == 0 or mb.sum() == 0:
            continue
        ct = _unit(Zt_w[ma].mean(0) - Zt_w[mb].mean(0))
        cs = []
        for s in subs:
            sa = (dsc == s) & (ys == a); sb = (dsc == s) & (ys == b)
            if sa.sum() and sb.sum():
                cs.append(_unit(Zs_w[sa].mean(0) - Zs_w[sb].mean(0)))
        if cs:
            csbar = _unit(np.array(cs).sum(0)); At.append(float(ct @ csbar))
    return dict(dir_consistency=float(np.mean(cons)) if cons else float("nan"),
                gdis_gshared_ratio=ratio, At=float(np.mean(At)) if At else float("nan"))


def _cluster_ci(vals, seed=7, n_boot=10000):
    v = np.asarray([x for x in vals if np.isfinite(x)], float)
    if not v.size:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), signflip_p=float("nan"), n=0)
    rng = np.random.default_rng(seed); b = np.array([v[rng.integers(0, v.size, v.size)].mean() for _ in range(n_boot)])
    return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)),
                signflip_p=MS.exact_sign_flip_p(v), n=int(v.size))


def _route(dU_erm, dU_shuf, dsrc_consistency, source_drop, dataset_lcbs):
    """PM A-E routing. dataset_lcbs = list of per-dataset LCB(dU_MCC-ERM)."""
    geom_up = dsrc_consistency > 0.01
    dg_up = any(l > 0 for l in dataset_lcbs)
    beats_shuffle = dU_shuf > 0
    damaged = source_drop > 0.02
    if geom_up and dg_up and beats_shuffle and not damaged:
        return dict(verdict="A_mechanism_consistency_transfers", next="seeds_3_4 + ccoral_same_protocol + dgcnn_replication")
    if geom_up and not dg_up and not damaged:
        return dict(verdict="B_geometry_moves_DG_flat", next="analyze_target_signal_covariance_alignment (do NOT raise lambda)")
    if not geom_up:
        return dict(verdict="C_geometry_did_not_move", next="one_bounded_fix: lambda=1.0 OR unfreeze (one, not both)")
    if geom_up and damaged:
        return dict(verdict="D_geometry_up_but_damaged", next="C_d=C_shared+R_d penalize residual variance / capacity-limited shared subspace")
    if abs(dU_shuf) < 1e-3:
        return dict(verdict="E_mcc_equals_shuffle_generic_reg", next="analyze true/shuffle grad cosine, margin, eff rank")
    return dict(verdict="INCONCLUSIVE", next="inspect per-subject sign + seed variance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", required=True); ap.add_argument("--expect", type=int, default=None)
    ap.add_argument("--no-geometry", action="store_true")
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(d.glob("*.done"))
    if a.expect is not None and len(done) < a.expect:
        print(f"[mcc-agg] INCOMPLETE {len(done)}/{a.expect} bundles done -> REFUSING to aggregate (no partial-result routing).")
        sys.exit(2)
    manifests = sorted(d.glob("*.manifest.json"))
    # per (dataset, subject): dU averaged over seeds
    dU_erm = defaultdict(lambda: defaultdict(list)); dU_shuf = defaultdict(lambda: defaultdict(list))
    src_bacc = defaultdict(lambda: defaultdict(list)); geom = defaultdict(lambda: defaultdict(list))
    per_bundle = []
    for mf in manifests:
        m = json.loads(mf.read_text()); ds = m["dataset"]; subj = m["subject"]
        arms = m["arms"]
        if not all(x in arms for x in ARMS):
            continue
        bA = arms["A_erm_continue"]["target_bacc"]; bB = arms["B_mcc_true"]["target_bacc"]; bC = arms["C_mcc_shuffle"]["target_bacc"]
        dU_erm[ds][subj].append(bB - bA); dU_shuf[ds][subj].append(bB - bC)
        src_bacc[ds][subj].append((arms["B_mcc_true"]["source_val_bacc"], arms["A_erm_continue"]["source_val_bacc"]))
        per_bundle.append(dict(dataset=ds, subject=subj, seed=m["seed"], warmup_hash=m["warmup_hash"],
                               bAcc=dict(A=bA, B=bB, C=bC), dU_MCC_ERM=bB - bA, dU_MCC_shuffle=bB - bC,
                               eff_rank=arms["B_mcc_true"]["eff_rank"], contrast_norm=arms["B_mcc_true"]["contrast_norm"]))
        if not a.no_geometry:
            gA = _geometry(str(d / arms["A_erm_continue"]["dump"])); gB = _geometry(str(d / arms["B_mcc_true"]["dump"]))
            if "fail" not in gA and "fail" not in gB:
                geom[ds][subj].append((gB["dir_consistency"] - gA["dir_consistency"], gB["At"] - gA["At"],
                                       gB["gdis_gshared_ratio"] - gA["gdis_gshared_ratio"]))

    summ = []; lcbs_by_ds = {}
    for ds in DATASETS:
        if ds not in dU_erm:
            continue
        erm = _cluster_ci([np.mean(v) for v in dU_erm[ds].values()])
        shuf = _cluster_ci([np.mean(v) for v in dU_shuf[ds].values()])
        # source damage = mean per-subject (ERM source-val bAcc - MCC source-val bAcc); routing needs <= 0.02
        src_drop = float(np.mean([np.mean([erm_s - mcc_s for (mcc_s, erm_s) in v]) for v in src_bacc[ds].values()])) if src_bacc[ds] else 0.0
        gvals = [np.mean([g[0] for g in v]) for v in geom[ds].values()] if geom[ds] else []
        atvals = [np.mean([g[1] for g in v]) for v in geom[ds].values()] if geom[ds] else []
        ratiovals = [np.mean([g[2] for g in v]) for v in geom[ds].values()] if geom[ds] else []
        lcbs_by_ds[ds] = erm["lo"]
        summ.append(dict(dataset=ds, n_subjects=erm["n"], source_drop=src_drop,
                         dU_MCC_ERM_mean=erm["mean"], dU_MCC_ERM_lcb=erm["lo"], dU_MCC_ERM_ucb=erm["hi"], dU_MCC_ERM_signflip_p=erm["signflip_p"],
                         dU_MCC_shuffle_mean=shuf["mean"], dU_MCC_shuffle_lcb=shuf["lo"], dU_MCC_shuffle_ucb=shuf["hi"],
                         delta_dir_consistency=float(np.mean(gvals)) if gvals else float("nan"),
                         delta_At=float(np.mean(atvals)) if atvals else float("nan"),
                         delta_gdis_gshared_ratio=float(np.mean(ratiovals)) if ratiovals else float("nan")))
    # routing (uses mean delta dir-consistency across datasets + source-drop guard)
    dcons = np.nanmean([s["delta_dir_consistency"] for s in summ]) if summ else float("nan")
    dushuf = np.nanmean([s["dU_MCC_shuffle_mean"] for s in summ]) if summ else float("nan")
    duerm = np.nanmean([s["dU_MCC_ERM_mean"] for s in summ]) if summ else float("nan")
    worst_src_drop = max([s["source_drop"] for s in summ], default=0.0)
    route = _route(duerm, dushuf, dcons if np.isfinite(dcons) else 0.0, worst_src_drop, list(lcbs_by_ds.values()))

    outd = d
    json.dump(dict(per_dataset=summ, per_bundle=per_bundle, routing=route, n_bundles=len(manifests),
                   discipline="no CLOSED; graded routing; exact sign-flip DG p; manuscript FROZEN; only the project owner stops a line"),
              open(outd / "mcc_verdict.json", "w"), indent=2, default=float)
    print(f"[mcc-agg] {len(manifests)} bundles; routing={route['verdict']}")
    for s in summ:
        print(f"  {s['dataset']}: dU_MCC-ERM={s['dU_MCC_ERM_mean']:+.4f}[{s['dU_MCC_ERM_lcb']:+.4f},{s['dU_MCC_ERM_ucb']:+.4f}] "
              f"signflip_p={s['dU_MCC_ERM_signflip_p']:.3f} | dU_MCC-shuffle={s['dU_MCC_shuffle_mean']:+.4f} | "
              f"Ddir_consistency={s['delta_dir_consistency']:+.4f} DAt={s['delta_At']:+.4f} Dratio={s['delta_gdis_gshared_ratio']:+.4f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
