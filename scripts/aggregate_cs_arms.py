"""Aggregate the Cross-Session 5-arm fleet (CPU, env c84c). Reads A=ERM-continue / B=CS-RW-MCC / C=weight-permuted
CS-RW-MCC / D=direct-cross-session-risk / E=permuted-direct-risk manifests and reports the PM DG endpoints on
target subjects: does either cross-session objective beat extra-training (dU_B-A, dU_D-A), the DECISIVE specificity
contrasts (dU_B-C, dU_D-E: does the CORRECT source-instability assignment beat its permuted control?), and the
mediator contrast dU_D-B (direct risk vs MCC geometry). Inference unit = target subject (3 seeds -> subject first),
subject-cluster bootstrap + exact sign-flip. Plus collapse/damage guards, the frozen source late-session drift, and
the exact-gradient target-alignment CO-DIAGNOSTIC (target labels audit-only; NOT a gate). Routes:
proxy-invalid/generic / proxy-valid-but-MCC-mediator-wrong / correct-assignment-valuable. Refuses partial fleets.
Manuscript FROZEN; only the project owner stops a scientific line.

  python scripts/aggregate_cs_arms.py --from-dir results/cmi_trace_cs_arms --expect 63
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
DAMAGE_TOL = 0.02   # source-val bAcc drop that flags an arm as damaged


def _subj_ci(per_subj):
    """per_subj: {subj: [seed values]} -> aggregate per subject over seeds FIRST, then subject-cluster CI + sign-flip
    (H1 mean>0)."""
    return _cluster_ci([float(np.mean(v)) for v in per_subj.values()])


def _route(ds_stats):
    """ds_stats: {ds: dict of endpoint CIs}. A DG win requires beating extra-training (dU_B-A or dU_D-A LCB>0) AND
    the matched specificity contrast (dU_B-C or dU_D-E LCB>0) on the SAME arm, on >=1 dataset, other not reversed."""
    def lcb(ds, k):  return ds_stats[ds][k]["lo"]
    def ucb(ds, k):  return ds_stats[ds][k]["hi"]
    def undamaged(ds):  return not ds_stats[ds].get("_meta", {}).get("damaged", False)
    dsl = list(ds_stats)
    # A DG win on a dataset requires: beats extra-training (dU_?_A LCB>0) AND beats its matched permuted control
    # (decisive dU_?_C / dU_?_E LCB>0) AND the winning arm did NOT damage source val on that dataset (else a COLLAPSED
    # permuted control could fake the specificity margin -> the exact M1-P/RW-MCC false-positive we must block). The
    # OTHER dataset must not be clearly reversed on BOTH the vs-extra-training AND the decisive contrast (UCB>-0.05).
    B_wins = any(lcb(ds, "dU_B_A") > 0 and lcb(ds, "dU_B_C") > 0 and undamaged(ds) for ds in dsl) and \
        all(ucb(ds, "dU_B_A") > -0.05 and ucb(ds, "dU_B_C") > -0.05 for ds in dsl)
    D_wins = any(lcb(ds, "dU_D_A") > 0 and lcb(ds, "dU_D_E") > 0 and undamaged(ds) for ds in dsl) and \
        all(ucb(ds, "dU_D_A") > -0.05 and ucb(ds, "dU_D_E") > -0.05 for ds in dsl)
    D_beats_B = any(lcb(ds, "dU_D_B") > 0 for ds in dsl)
    # Contract routing: when BOTH candidates pass, prefer the MORE DIRECT CS-Risk (D). Only fall to CS-RW-MCC when D
    # does NOT clear the bar but B does.
    if D_wins:
        also = " (both objectives pass; preferring the more-direct CS-Risk per contract" + (", and D>B specifically)" if D_beats_B else ")") if B_wins else ""
        return dict(verdict="CORRECT_ASSIGNMENT_VALUABLE_via_DIRECT_RISK",
                    next="EXPLORATORY: direct cross-session risk beats extra-training AND its permuted control, undamaged" + also + "; confirm on a THIRD EEG dataset before any method claim (OWNER decision)")
    if B_wins:
        return dict(verdict="CORRECT_ASSIGNMENT_VALUABLE_via_CS_RW_MCC",
                    next="EXPLORATORY: CS-RW-MCC beats extra-training AND weight-permuted control, undamaged; confirm on a THIRD EEG dataset (OWNER decision)")
    if D_beats_B and any(lcb(ds, "dU_D_E") > 0 and undamaged(ds) for ds in dsl):
        return dict(verdict="PROXY_VALID_BUT_MCC_MEDIATOR_WRONG",
                    next="direct cross-session risk (D>E specific) helps where the MCC geometry mediator (B) does not; drop the cosine mediator, but D does not clear extra-training -> treat as mechanism finding not DG win")
    return dict(verdict="CROSS_SESSION_PROXY_INVALID_or_GENERIC_EXTRA_TRAINING",
                next="no cross-session objective beats extra-training with a specific (permuted-control), undamaged margin -> pivot to the information-regime ladder (source-only -> target-X -> few-shot target labels; minimal-info sample-complexity of TARGET_HINDSIGHT_ONLY)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-dir", default="results/cmi_trace_cs_arms"); ap.add_argument("--expect", type=int, default=63)
    a = ap.parse_args()
    d = Path(a.from_dir); done = sorted(d.glob("*_sub*_seed*.done"))
    if len(done) < a.expect:
        print(f"[cs-agg] INCOMPLETE {len(done)}/{a.expect} bundles -> REFUSING."); raise SystemExit(2)
    mfs = sorted(d.glob("*.manifest.json"))
    # endpoint -> ds -> subj -> [seed deltas]
    EP = ["dU_B_A", "dU_D_A", "dU_B_C", "dU_D_E", "dU_D_B"]
    acc = {e: defaultdict(lambda: defaultdict(list)) for e in EP}
    damage = defaultdict(lambda: defaultdict(list)); effrank = defaultdict(list)
    late_risk = defaultdict(list); noop = defaultdict(int); align = defaultdict(lambda: defaultdict(list))
    incomplete = 0
    for mf in mfs:
        m = json.loads(mf.read_text()); ds, subj = m["dataset"], m["subject"]; ar = m.get("arms", {})
        if not all(k in ar for k in ARMS):
            incomplete += 1; continue
        bacc = {k: ar[k]["target_bacc"] for k in ARMS}
        acc["dU_B_A"][ds][subj].append(bacc["B_cs_rw_mcc"] - bacc["A_erm_continue"])
        acc["dU_D_A"][ds][subj].append(bacc["D_cs_risk"] - bacc["A_erm_continue"])
        acc["dU_B_C"][ds][subj].append(bacc["B_cs_rw_mcc"] - bacc["C_cs_rw_perm"])
        acc["dU_D_E"][ds][subj].append(bacc["D_cs_risk"] - bacc["E_cs_risk_perm"])
        acc["dU_D_B"][ds][subj].append(bacc["D_cs_risk"] - bacc["B_cs_rw_mcc"])
        base_src = ar["A_erm_continue"]["source_val_bacc"]
        for k in ARMS:
            damage[ds][k].append(base_src - ar[k]["source_val_bacc"])   # >0 = arm hurt source val
            effrank[ds].append(ar[k]["eff_rank"])
        if m.get("weight_status") == "NO_POSITIVE_SOURCE_TRANSFER_GAP":
            noop[ds] += 1
        late_risk[ds].append(m.get("mean_source_late_risk", float("nan")))
        al = m.get("gradient_alignment_diagnostic", {})
        for k in ("cs_rw", "cs_risk", "loso", "task"):
            if isinstance(al, dict) and k in al:
                align[ds][k].append(al[k])

    ds_stats = {}; summ = []
    for ds in DATASETS:
        if ds not in acc["dU_B_A"] or not acc["dU_B_A"][ds]:
            continue
        cis = {e: _subj_ci(acc[e][ds]) for e in EP}
        worst_damage = max((max(v) for v in [damage[ds][k] for k in ARMS if damage[ds][k]]), default=0.0)
        cis["_meta"] = dict(damaged=bool(worst_damage > DAMAGE_TOL), worst_damage=float(worst_damage))
        ds_stats[ds] = cis
        summ.append(dict(dataset=ds, n_subjects=cis["dU_B_A"]["n"], noop_bundles=noop[ds],
                         **{e: dict(mean=cis[e]["mean"], lcb=cis[e]["lo"], ucb=cis[e]["hi"], signflip_p=cis[e]["signflip_p"]) for e in EP},
                         worst_source_damage=float(worst_damage), damaged=bool(worst_damage > DAMAGE_TOL),
                         min_eff_rank=float(np.min(effrank[ds])) if effrank[ds] else float("nan"),
                         mean_source_late_drift=float(np.nanmean(late_risk[ds])) if late_risk[ds] else float("nan"),
                         align_codiag={k: float(np.mean(align[ds][k])) for k in ("cs_rw", "cs_risk", "loso", "task") if align[ds][k]}))
    route = _route(ds_stats) if ds_stats else dict(verdict="NO_DATA", next="no complete cells")
    out = dict(per_dataset=summ, routing=route, n_bundles=len(mfs), incomplete_bundles=incomplete,
               endpoints="dU_B-A/dU_D-A vs extra-training; DECISIVE dU_B-C/dU_D-E vs permuted control; dU_D-B mediator",
               discipline="inference unit=target subject (3 seeds->subj first), subject-cluster bootstrap + exact sign-flip; "
                          "gradient-alignment = CO-DIAGNOSTIC not a gate; no CLOSED without 3rd-dataset confirm; manuscript FROZEN")
    json.dump(out, open(d / "cs_arms_verdict.json", "w"), indent=2, default=float)
    print(f"[cs-agg] {len(mfs)} bundles ({incomplete} incomplete); routing={route['verdict']}")
    for s in summ:
        def f(e): return f"{s[e]['mean']:+.4f}[lcb {s[e]['lcb']:+.4f}] p={s[e]['signflip_p']:.3f}"
        print(f"  {s['dataset']} (n={s['n_subjects']}, noop={s['noop_bundles']}, damaged={s['damaged']} worstΔsrc={s['worst_source_damage']:+.3f}):")
        print(f"    vs extra-training  dU_B-A={f('dU_B_A')}  dU_D-A={f('dU_D_A')}")
        print(f"    DECISIVE specific  dU_B-C={f('dU_B_C')}  dU_D-E={f('dU_D_E')}   mediator dU_D-B={f('dU_D_B')}")
        print(f"    co-diag align={s['align_codiag']}  late_drift={s['mean_source_late_drift']:.4f}  min_effrank={s['min_eff_rank']:.2f}")
    print(f"  -> {route['verdict']} : next = {route['next']}")


if __name__ == "__main__":
    main()
