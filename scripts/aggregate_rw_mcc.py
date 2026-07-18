"""Aggregate the Risk-Weighted MCC 3-arm fleet (CPU, env c84c). Reads the A=ERM / B=true-RW / C=weight-permuted
dumps and reports the PM endpoints: DG utility dU_RW-ERM (B-A) and the DECISIVE dU_RW-WPerm (B-C) with subject-
cluster bootstrap + exact sign-flip; source-LOSO excess-risk change (does RW-MCC reduce the source transfer gap it
targets); unweighted geometry (dir-consistency). Routes A/B/C/D. Success MUST depend on B-C, not just B-A. Refuses
partial. Manuscript FROZEN; only the project owner stops a scientific line.

  python scripts/aggregate_rw_mcc.py --from-dir results/cmi_trace_rw_mcc --expect 63
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _geometry, _cluster_ci
from tos_cmi.eval.mechanism_subspace import exact_sign_flip_p as MS_signflip
from tos_cmi.train.risk_weighted_mcc import source_loso_excess_risk_weights
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
ARMS = ["A_erm_continue", "B_rw_true", "C_rw_wperm"]


def _mean_excess_risk(npz):
    f = feat_from_tos_dump(npz)
    try:
        out = source_loso_excess_risk_weights(np.asarray(f["Z_source"], float), np.asarray(f["y_source"]).astype(int), _dense(f["subj_source"]))
    except Exception:
        return float("nan")
    return float(np.mean(list(out["r"].values()))) if out["r"] else float("nan")


def _route(dU_erm_lcbs, dU_wperm_lcbs, dR_source_B, dR_source_specific, src_drop):
    dg_beats_erm = any(l > 0 for l in dU_erm_lcbs)
    dg_beats_wperm = any(l > 0 for l in dU_wperm_lcbs)      # DECISIVE
    # B vs C discriminator: did TRUE RW-MCC cut source risk MORE than the weight-permuted control? (specific < 0)
    source_risk_down_specifically = dR_source_specific < -1e-4
    source_risk_down_any = dR_source_B < 0
    damaged = src_drop > 0.02
    if source_risk_down_specifically and dg_beats_erm and dg_beats_wperm and not damaged:
        return dict(verdict="A_source_risk_localizes_valuable_mechanism", next="seeds_3_4 + ccoral_irm_same_protocol + dgcnn_replication + CMI_posterior_audit")
    if source_risk_down_specifically and not dg_beats_wperm and not damaged:
        return dict(verdict="B_source_meta_gen_failure_ne_target_failure", next="next round: weights from source-only cross-SESSION (early->later) instability, not more strength")
    if not source_risk_down_specifically:
        return dict(verdict="STATIC_LOSO_WEIGHTS_DO_NOT_TARGET_TRAINABLE_MECHANISM", next="true RW-MCC does NOT cut source risk more than the permuted control -> the static LOSO weights do not target a trainable mechanism; analyze weight instability / risk-contrast cell mismatch / train pairwise margins directly (NOT back to EMA)")
    if source_risk_down and damaged:
        return dict(verdict="D_source_risk_down_but_damaged", next="downweight unstable subjects, or shared+residual decomposition")
    return dict(verdict="INCONCLUSIVE", next="inspect per-subject sign + seed variance")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_rw_mcc"); ap.add_argument("--expect", type=int, default=63)
    ap.add_argument("--no-geometry", action="store_true")
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(d.glob("*_sub*_seed*.done"))
    if len(done) < a.expect:
        print(f"[rw-agg] INCOMPLETE {len(done)}/{a.expect} bundles -> REFUSING."); raise SystemExit(2)
    mfs = sorted(d.glob("*.manifest.json"))
    dU_erm = defaultdict(lambda: defaultdict(list)); dU_wp = defaultdict(lambda: defaultdict(list))
    src_drop = defaultdict(lambda: defaultdict(list)); dR = defaultdict(list); noop = defaultdict(int)
    for mf in mfs:
        m = json.loads(mf.read_text()); ds, subj = m["dataset"], m["subject"]; ar = m["arms"]
        if not all(k in ar for k in ARMS):
            continue
        if m.get("weight_status") == "NO_POSITIVE_SOURCE_TRANSFER_GAP":
            noop[ds] += 1
        bA, bB, bC = ar["A_erm_continue"]["target_bacc"], ar["B_rw_true"]["target_bacc"], ar["C_rw_wperm"]["target_bacc"]
        dU_erm[ds][subj].append(bB - bA); dU_wp[ds][subj].append(bB - bC)
        src_drop[ds][subj].append(ar["A_erm_continue"]["source_val_bacc"] - ar["B_rw_true"]["source_val_bacc"])
        if not a.no_geometry:
            rA = _mean_excess_risk(str(d / ar["A_erm_continue"]["dump"])); rB = _mean_excess_risk(str(d / ar["B_rw_true"]["dump"]))
            rC = _mean_excess_risk(str(d / ar["C_rw_wperm"]["dump"]))
            if np.isfinite(rA) and np.isfinite(rB) and np.isfinite(rC):
                dR[ds].append((rB - rA, rC - rA, rB - rC))  # (B-A, C-A, B-C): B-C<0 = TRUE cut source risk MORE than permuted

    summ = []; erm_lcbs = {}; wp_lcbs = {}
    for ds in DATASETS:
        if ds not in dU_erm:
            continue
        e = _cluster_ci([np.mean(v) for v in dU_erm[ds].values()]); w = _cluster_ci([np.mean(v) for v in dU_wp[ds].values()])
        sd = float(np.mean([np.mean(v) for v in src_drop[ds].values()]))
        arr = np.array(dR[ds]) if dR[ds] else np.zeros((0, 3))
        drB = float(np.mean(arr[:, 0])) if arr.size else float("nan")     # B-A
        drBC = float(np.mean(arr[:, 2])) if arr.size else float("nan")    # B-C (specific: TRUE vs permuted)
        drBC_p = MS_signflip(arr[:, 2]) if arr.size else float("nan")
        erm_lcbs[ds] = e["lo"]; wp_lcbs[ds] = w["lo"]
        summ.append(dict(dataset=ds, n_subjects=e["n"], noop_bundles=noop[ds],
                         dU_RW_ERM_mean=e["mean"], dU_RW_ERM_lcb=e["lo"], dU_RW_ERM_signflip_p=e["signflip_p"],
                         dU_RW_WPerm_mean=w["mean"], dU_RW_WPerm_lcb=w["lo"], dU_RW_WPerm_signflip_p=w["signflip_p"],
                         delta_source_excess_risk_B=drB, delta_source_excess_risk_specific_BminusC=drBC,
                         delta_source_excess_risk_specific_signflip_p=drBC_p, source_drop=sd))
    drB_all = np.nanmean([s["delta_source_excess_risk_B"] for s in summ]) if summ else float("nan")
    drBC_all = np.nanmean([s["delta_source_excess_risk_specific_BminusC"] for s in summ]) if summ else float("nan")
    worst_drop = max([s["source_drop"] for s in summ], default=0.0)
    route = _route(list(erm_lcbs.values()), list(wp_lcbs.values()), drB_all if np.isfinite(drB_all) else 0.0,
                   drBC_all if np.isfinite(drBC_all) else 0.0, worst_drop)
    json.dump(dict(per_dataset=summ, routing=route, n_bundles=len(mfs),
                   discipline="success needs B-C (vs weight-permuted); exact sign-flip; no CLOSED; manuscript FROZEN"),
              open(d / "rw_mcc_verdict.json", "w"), indent=2, default=float)
    print(f"[rw-agg] {len(mfs)} bundles; routing={route['verdict']}")
    for s in summ:
        print(f"  {s['dataset']}: dU_RW-ERM={s['dU_RW_ERM_mean']:+.4f}[lcb {s['dU_RW_ERM_lcb']:+.4f}] p={s['dU_RW_ERM_signflip_p']:.3f} | "
              f"dU_RW-WPerm(B-C)={s['dU_RW_WPerm_mean']:+.4f}[lcb {s['dU_RW_WPerm_lcb']:+.4f}] p={s['dU_RW_WPerm_signflip_p']:.3f} | "
              f"dR_src(B-A)={s['delta_source_excess_risk_B']:+.5f} dR_src_specific(B-C)={s['delta_source_excess_risk_specific_BminusC']:+.5f} "
              f"p={s['delta_source_excess_risk_specific_signflip_p']:.3f} src_drop={s['source_drop']:+.4f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
