"""Aggregate the Cross-Session FULL-STRENGTH oracle-ceiling fleet (Track A; CPU, env c84c). Reads the 5-arm per-epoch
manifests (lambda=1 from epoch 0, no ramp, no source-val rollback) and reports the PM endpoints on target subjects:
PRIMARY = epoch-20 decisive specificity dU_B-C^(20) and dU_D-E^(20) (does the CORRECT source-instability assignment
beat its permuted control at FULL strength?), plus dU_B-A^(20), dU_D-A^(20), dU_D-B^(20). The per-epoch trajectory is
a mechanism diagnostic. Additionally report max_{e>=5} dU(e), LABELED as a NON-DEPLOYABLE TARGET-EPOCH ORACLE UPPER
BOUND (it selects the epoch using target labels, so it is a ceiling, not a method). Inference unit = target subject
(3 seeds -> subject first), subject-cluster bootstrap + exact sign-flip. Routes FS-A / FS-B / FS-C. Refuses partial
fleets. Manuscript FROZEN; only the project owner stops or redirects a scientific line.

  python scripts/aggregate_cs_fullstrength.py --from-dir results/cmi_trace_cs_fullstrength --expect 63
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _cluster_ci

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
ARMS = ["A_erm_continue", "B_cs_rw_mcc", "C_cs_rw_perm", "D_cs_risk", "E_cs_risk_perm"]
FINAL = 20
ORACLE_EPOCHS = [5, 10, 15, 20]      # e>=5 for the target-epoch oracle upper bound


def _subj_ci(per_subj):
    return _cluster_ci([float(np.mean(v)) for v in per_subj.values()])


def _tb(arm_manifest, epoch):
    """target bAcc at a given epoch from the per_epoch dict (keys may be str after json round-trip)."""
    pe = arm_manifest["per_epoch"]
    return pe.get(str(epoch), pe.get(epoch, {})).get("target_bacc")


def _route(ds_stats):
    """FS-A: a decisive contrast (dU_B-C or dU_D-E) is stably positive at epoch 20 on BOTH datasets (LCB>0 each).
    FS-B: positive on only one dataset (or only via the oracle-epoch upper bound). FS-C: neither, at epoch 20 or in
    the trajectory. The epoch-20 endpoint is the deployable-strength (but oracle-epoch) result; a positive here does
    NOT overturn the deployable negative from run_cs_arms (which used source-only selection) -- it localizes the
    failure to source-only SELECTABILITY, not to the objective's oracle utility."""
    dsl = list(ds_stats)
    def lcb(ds, k): return ds_stats[ds][k]["lo"]
    bc_both = all(lcb(ds, "dU_B_C_20") > 0 for ds in dsl) and len(dsl) >= 2
    de_both = all(lcb(ds, "dU_D_E_20") > 0 for ds in dsl) and len(dsl) >= 2
    bc_any = any(lcb(ds, "dU_B_C_20") > 0 for ds in dsl)
    de_any = any(lcb(ds, "dU_D_E_20") > 0 for ds in dsl)
    if bc_both or de_both:
        which = "B-C" if bc_both else "D-E"
        return dict(verdict="FS-A_OBJECTIVE_HAS_ORACLE_EFFECT_BUT_IS_NOT_SOURCE_SELECTABLE",
                    next=f"at FULL strength the decisive {which} contrast is positive on BOTH datasets at epoch 20 -> the objective can help but source-only checkpoint selection cannot identify it; the info-regime ladder's focus becomes: how much TARGET information selects the right epoch/direction? (does NOT overturn the deployable negative)")
    if bc_any or de_any:
        return dict(verdict="FS-B_DATASET_OR_EPOCH_SPECIFIC",
                    next="full-strength effect is dataset-/epoch-specific (positive on one dataset only, or only at a transient epoch) -> does not support a method; carry to the ladder to check information need + heterogeneity")
    return dict(verdict="FS-C_FULL_STRENGTH_CROSS_SESSION_OBJECTIVE_DG_NULL",
                next="neither decisive contrast is positive at epoch 20 nor across the trajectory -> stop invoking ramp/early-stopping as the explanation; the information-regime ladder drives the next question")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_cs_fullstrength"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(d.glob("*_sub*_seed*.done"))
    if len(done) < a.expect:
        print(f"[cs-fs-agg] INCOMPLETE {len(done)}/{a.expect} bundles -> REFUSING."); raise SystemExit(2)
    mfs = sorted(d.glob("*.manifest.json"))
    EP20 = ["dU_B_A_20", "dU_D_A_20", "dU_B_C_20", "dU_D_E_20", "dU_D_B_20"]
    acc = {e: defaultdict(lambda: defaultdict(list)) for e in EP20}
    # oracle upper bound: per cell, max over e>=5 of the decisive contrast
    oracle = {"B_C": defaultdict(lambda: defaultdict(list)), "D_E": defaultdict(lambda: defaultdict(list))}
    traj = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # ds -> arm -> epoch -> [bacc]
    incomplete = 0; save_epochs = None
    for mf in mfs:
        m = json.loads(mf.read_text()); ds, subj = m["dataset"], m["subject"]; ar = m.get("arms", {})
        if not all(k in ar for k in ARMS):
            incomplete += 1; continue
        save_epochs = m.get("save_epochs", [0, 1, 2, 5, 10, 15, 20])
        tb = {k: {e: _tb(ar[k], e) for e in save_epochs} for k in ARMS}
        acc["dU_B_A_20"][ds][subj].append(tb["B_cs_rw_mcc"][FINAL] - tb["A_erm_continue"][FINAL])
        acc["dU_D_A_20"][ds][subj].append(tb["D_cs_risk"][FINAL] - tb["A_erm_continue"][FINAL])
        acc["dU_B_C_20"][ds][subj].append(tb["B_cs_rw_mcc"][FINAL] - tb["C_cs_rw_perm"][FINAL])
        acc["dU_D_E_20"][ds][subj].append(tb["D_cs_risk"][FINAL] - tb["E_cs_risk_perm"][FINAL])
        acc["dU_D_B_20"][ds][subj].append(tb["D_cs_risk"][FINAL] - tb["B_cs_rw_mcc"][FINAL])
        oe = [e for e in ORACLE_EPOCHS if e in save_epochs]
        oracle["B_C"][ds][subj].append(max(tb["B_cs_rw_mcc"][e] - tb["C_cs_rw_perm"][e] for e in oe))
        oracle["D_E"][ds][subj].append(max(tb["D_cs_risk"][e] - tb["E_cs_risk_perm"][e] for e in oe))
        for k in ARMS:
            for e in save_epochs:
                traj[ds][k][e].append(tb[k][e])

    ds_stats = {}; summ = []
    for ds in DATASETS:
        if ds not in acc["dU_B_A_20"] or not acc["dU_B_A_20"][ds]:
            continue
        cis = {e: _subj_ci(acc[e][ds]) for e in EP20}
        ds_stats[ds] = cis
        ob = _subj_ci(oracle["B_C"][ds]); od = _subj_ci(oracle["D_E"][ds])
        traj_mean = {k: {int(e): float(np.mean(traj[ds][k][e])) for e in sorted(traj[ds][k])} for k in ARMS}
        summ.append(dict(dataset=ds, n_subjects=cis["dU_B_A_20"]["n"],
                         **{e: dict(mean=cis[e]["mean"], lcb=cis[e]["lo"], ucb=cis[e]["hi"], signflip_p=cis[e]["signflip_p"]) for e in EP20},
                         ORACLE_UB_B_C_maxE=dict(mean=ob["mean"], lcb=ob["lo"], ucb=ob["hi"], note="TARGET-EPOCH ORACLE UPPER BOUND (non-deployable)"),
                         ORACLE_UB_D_E_maxE=dict(mean=od["mean"], lcb=od["lo"], ucb=od["hi"], note="TARGET-EPOCH ORACLE UPPER BOUND (non-deployable)"),
                         target_trajectory=traj_mean))
    route = _route(ds_stats) if len(ds_stats) >= 1 else dict(verdict="NO_DATA", next="no complete cells")
    out = dict(per_dataset=summ, routing=route, n_bundles=len(mfs), incomplete_bundles=incomplete, save_epochs=save_epochs,
               endpoints="PRIMARY = epoch-20 decisive dU_B-C^(20)/dU_D-E^(20) (+ B-A/D-A/D-B); trajectory = diagnostic; "
                         "max_{e>=5} = TARGET-EPOCH ORACLE UPPER BOUND (non-deployable ceiling)",
               discipline="inference unit=target subject (3 seeds->subj first), subject-cluster bootstrap + exact sign-flip; "
                          "a positive epoch-20 result localizes failure to source-only SELECTABILITY, does NOT overturn the "
                          "deployable negative; manuscript FROZEN")
    json.dump(out, open(d / "cs_fullstrength_verdict.json", "w"), indent=2, default=float)
    print(f"[cs-fs-agg] {len(mfs)} bundles ({incomplete} incomplete); routing={route['verdict']}")
    for s in summ:
        def f(e): return f"{s[e]['mean']:+.4f}[lcb {s[e]['lcb']:+.4f}] p={s[e]['signflip_p']:.3f}"
        print(f"  {s['dataset']} (n={s['n_subjects']}) @epoch20:")
        print(f"    vs extra-training  dU_B-A={f('dU_B_A_20')}  dU_D-A={f('dU_D_A_20')}")
        print(f"    DECISIVE specific  dU_B-C={f('dU_B_C_20')}  dU_D-E={f('dU_D_E_20')}   mediator dU_D-B={f('dU_D_B_20')}")
        print(f"    ORACLE-UB (non-deployable) B-C_maxE={s['ORACLE_UB_B_C_maxE']['mean']:+.4f}[lcb {s['ORACLE_UB_B_C_maxE']['lcb']:+.4f}]  D-E_maxE={s['ORACLE_UB_D_E_maxE']['mean']:+.4f}[lcb {s['ORACLE_UB_D_E_maxE']['lcb']:+.4f}]")
        tj = s['target_trajectory']; es = sorted(next(iter(tj.values())))
        print(f"    target traj (A/B/C/D/E): " + " | ".join(f"e{e} " + "/".join(f"{tj[k][e]:.3f}" for k in ARMS) for e in es))
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
