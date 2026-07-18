"""Aggregate the Readout Prior Lockbox round (CPU, env c84c). Primary = LOW-SHOT AULC over k in {1,2,4,8}; Full is a
posterior-washout diagnostic (H2-H1 -> 0 is EXPECTED, not a failure). Matched-tau contrasts isolate the source-CENTER
from the shrinkage-STRENGTH. Routing driven by the two LOCKBOXES (Stieger2021 primary + Shin2017A confirmatory), which
are untouched natural multi-session MI datasets; the original 4 datasets are reported as context. Inference unit =
target subject (draw->seed->subject), subject-cluster bootstrap + exact sign-flip. Refuses partial / any solver-failed
cell. Manuscript FROZEN; only the owner stops/redirects a line.

  python scripts/aggregate_readout_prior_lockbox.py --from-dir results/cmi_trace_readout_prior_lockbox --expect 525
"""
from __future__ import annotations
import argparse, glob, json
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]
import sys; sys.path.insert(0, str(REPO))
from scripts.aggregate_mcc import _cluster_ci

DEV = ["BNCI2014_001", "BNCI2015_001"]; CONTEXT_EXT = ["Lee2019_MI", "BNCI2014_004"]
PRIMARY_LOCKBOX = "Stieger2021"; CONFIRM_LOCKBOX = "Shin2017A"; LOCKBOX = [PRIMARY_LOCKBOX, CONFIRM_LOCKBOX]
DATASETS = DEV + CONTEXT_EXT + LOCKBOX
AULC_K = ["1", "2", "4", "8"]                 # low-shot area
BUDGETS = ["1", "2", "4", "8", "16", "32", "Full"]
# per-cell endpoint -> the low-shot AULC we average (mean over AULC_K)
AULC_EP = ["dU_center", "dU_center_t0", "dU_center_ts", "dU_MAP_frozen", "dU_gate_frozen"]


def _aulc(cell_ep, key):
    return float(np.mean([cell_ep[k][key] for k in AULC_K]))


def _subj_ci(per):
    return _cluster_ci([float(np.mean(v)) for v in per.values()])


def _route(stats):
    def L(ds, k): return stats[ds][k]["lo"] if ds in stats else float("-inf")
    def M(ds, k): return stats[ds][k]["mean"] if ds in stats else 0.0
    P, Cf = PRIMARY_LOCKBOX, CONFIRM_LOCKBOX
    have_lb = P in stats and Cf in stats
    if not have_lb:
        return dict(verdict="LOCKBOX_INCOMPLETE", next=f"missing {P if P not in stats else ''} {Cf if Cf not in stats else ''}")
    # matched-tau CENTER effect (both @tau0 and @taus) on the PRIMARY, same direction + no clear negative on confirmatory
    center_primary = L(P, "dU_center_t0") > 0 and L(P, "dU_center_ts") > 0
    center_confirm_ok = M(Cf, "dU_center_t0") > 0 and M(Cf, "dU_center_ts") > 0 and L(Cf, "dU_center_t0") > -0.01 and L(Cf, "dU_center_ts") > -0.01
    center_both = center_primary and center_confirm_ok
    # policy (mixed center+strength) beats zero-centered on the primary
    policy_primary = L(P, "dU_center") > 0
    # adapt-vs-frozen on a lockbox with no clear harm on the other
    adapt = [ds for ds in LOCKBOX if L(ds, "dU_MAP_frozen") > 0]
    adapt_noharm = all(M(ds, "dU_MAP_frozen") > -0.005 for ds in LOCKBOX)
    center_any = center_primary or (L(Cf, "dU_center_t0") > 0 and L(Cf, "dU_center_ts") > 0)
    # L-D: real label-efficient adaptation (strongest)
    if adapt and adapt_noharm and center_any:
        return dict(verdict="L-D_SOURCE_ANCHORED_TARGET_READOUT_IMPROVES_LABEL_EFFICIENCY", adapt_lockbox=adapt,
                    next="H2 beats DEPLOYING the frozen head (low-shot AULC LCB>0) on a lockbox with no clear harm on the other, AND the matched-tau center effect holds -> deployable label-efficient source-anchored readout; confirm on a 3rd cohort")
    # L-A: real center effect on both lockboxes, but adaptation does not clear frozen (or headroom absent)
    if center_both:
        return dict(verdict="L-A_SOURCE_HEAD_CENTER_IMPROVES_LOW_LABEL_ESTIMATION", lockboxes=[P, Cf],
                    next="matched-tau center effect (both @tau0 and @taus) LCB>0 on Stieger2021 + same-direction no-harm on Shin2017A -> the source head as a PRIOR CENTER improves low-label readout estimation (a real, tau-isolated prior value); adapt-vs-frozen is a separate (headroom-dependent) question")
    # regime split: primary center but confirmatory not
    if center_primary and not center_confirm_ok:
        return dict(verdict="LONGITUDINAL_FEEDBACK_REGIME_DEPENDENT",
                    next="the center effect holds on Stieger2021 (longitudinal/feedback regime) but not Shin2017A -> regime-dependent, not a general external replication")
    # L-C: center positive but no adaptation headroom
    if center_any and not adapt:
        return dict(verdict="L-C_SOURCE_PRIOR_PREVENTS_LOW_SHOT_OVERFIT_NO_ADAPTATION_HEADROOM",
                    next="the source prior beats a zero-centered ridge low-shot (H2>H1) but does NOT beat deploying the frozen head (H2-H0~0) -> a safe low-shot FALLBACK, not an adaptation gain (a valid result)")
    # L-B: policy works but center not isolated (matched-tau unstable)
    if policy_primary and not center_primary:
        return dict(verdict="L-B_SOURCE_CENTERED_POLICY_OUTPERFORMS_CENTER_NOT_ISOLATED",
                    next="the source-centered POLICY (H2@taus vs H1@tau0) beats zero-centered low-shot, but the matched-tau contrasts are not both LCB>0 -> benefit mixes center + shrinkage-strength selection; next round = empirical-Bayes prior precision")
    return dict(verdict="NO_EXTERNAL_CENTER_EFFECT_LOW_SHOT_ADVANTAGE_NOT_STABLE",
                next="neither lockbox shows a matched-tau center effect -> the dev low-shot policy advantage is not externally stable; pivot to prior TRANSPORTABILITY / dataset-task compatibility / hierarchical or class-conditional prior, NOT B_cond erasure")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_readout_prior_lockbox"); ap.add_argument("--expect", type=int, default=525)
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(glob.glob(str(d / "cells" / "*.done")))
    if len(done) < a.expect:
        print(f"[lb-agg] INCOMPLETE {len(done)}/{a.expect} -> REFUSING."); raise SystemExit(2)
    aulc = {e: defaultdict(lambda: defaultdict(list)) for e in AULC_EP}
    full_center = defaultdict(lambda: defaultdict(list)); nmatch = defaultdict(list); solver_fail = 0; skipped = defaultdict(int)
    hr = defaultdict(lambda: defaultdict(list))
    for c in sorted(glob.glob(str(d / "cells" / "*.json"))):
        r = json.loads(Path(c).read_text())
        if r.get("status") == "failed_solver":
            solver_fail += 1; continue
        if r.get("status") != "ok":
            skipped[r.get("reason", "?")] += 1; continue
        ds, subj = r["dataset"], r["subject"]
        for e in AULC_EP:
            aulc[e][ds][subj].append(_aulc(r["endpoints"], e))
        full_center[ds][subj].append(r["endpoints"]["Full"]["dU_center"])
        sp = r.get("specificity", {})
        if sp.get("status") == "ok":
            nmatch[ds].append(sp.get("n_matched", 0))
        for kk, vv in r.get("headroom", {}).items():
            hr[ds][kk].append(vv)
    if solver_fail:
        print(f"[lb-agg] {solver_fail} cells FAILED SOLVER -> REFUSING (fix optimiser)."); raise SystemExit(3)

    per_ds = []; stats = {}
    for ds in DATASETS:
        if ds not in aulc["dU_center"] or not aulc["dU_center"][ds]:
            continue
        ci = {e: _subj_ci(aulc[e][ds]) for e in AULC_EP}
        stats[ds] = ci
        per_ds.append(dict(dataset=ds, role=("lockbox" if ds in LOCKBOX else ("dev" if ds in DEV else "context")),
                           n_subjects=ci["dU_center"]["n"], full_center_mean=_subj_ci(full_center[ds])["mean"],
                           n_matched_random=(float(np.mean(nmatch[ds])) if nmatch[ds] else 0),
                           AULC={e: dict(mean=ci[e]["mean"], lcb=ci[e]["lo"], p=ci[e]["signflip_p"]) for e in AULC_EP},
                           headroom_mean={kk: float(np.mean(vv)) for kk, vv in hr[ds].items()}))
    route = _route(stats) if stats else dict(verdict="NO_DATA")
    out = dict(per_dataset=per_ds, routing=route, n_cells=len(done), solver_failed=solver_fail, skipped=dict(skipped),
               primary_lockbox=PRIMARY_LOCKBOX, confirmatory_lockbox=CONFIRM_LOCKBOX,
               discipline="low-shot AULC (k in 1,2,4,8) primary; Full = posterior washout (not failure); matched-tau isolates "
                          "center from shrinkage-strength; routing driven by the 2 untouched lockboxes; manuscript FROZEN")
    json.dump(out, open(d / "readout_prior_lockbox_verdict.json", "w"), indent=2, default=float)
    print(f"[lb-agg] {len(done)} cells ({solver_fail} solver-fail, {sum(skipped.values())} skipped); routing={route['verdict']}")
    for s in per_ds:
        A = s["AULC"]
        print(f"  {s['dataset']:14s} [{s['role']}] (n={s['n_subjects']}) matched={s['n_matched_random']:.0f} full_center={s['full_center_mean']:+.3f}")
        print(f"    center@tau0={A['dU_center_t0']['mean']:+.4f}[{A['dU_center_t0']['lcb']:+.4f}] center@taus={A['dU_center_ts']['mean']:+.4f}[{A['dU_center_ts']['lcb']:+.4f}] "
              f"policy={A['dU_center']['mean']:+.4f}[{A['dU_center']['lcb']:+.4f}] MAP-frozen={A['dU_MAP_frozen']['mean']:+.4f}[{A['dU_MAP_frozen']['lcb']:+.4f}]")
    print(f"  -> {route['verdict']} : {route['next']}")


if __name__ == "__main__":
    main()
