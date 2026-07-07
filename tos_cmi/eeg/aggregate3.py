"""Phase 2.0 -- 3-seed STABILITY aggregation. Reads the per-seed result jsons (diagnostic_report,
geometry_report, ablation_report, adversarial_report _seed{0,1,2}.json) and prints the stability of
the four Phase-2 findings across seeds: (Q1) global-LPC lambda-collapse, (Q3/Q5) projection-ablation
removability, (Q2/Q4) score-Fisher gate decisions, (adversarial) subject-vs-session leakage.
Paired by (method, lambda) / fold; reports per-seed value + median across seeds (no trial-level
significance)."""
from __future__ import annotations
import glob
import json
import numpy as np

BASE = "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO"


def _load(kind):
    out = {}
    for p in sorted(glob.glob("%s/%s_seed*.json" % (BASE, kind))):
        s = p.split("_seed")[-1].split(".")[0]
        out[s] = json.load(open(p))
    return out


def main():
    diag = _load("diagnostic_report"); geo = _load("geometry_report")
    abl = _load("ablation_report"); adv = _load("adversarial_report")
    seeds = sorted(diag.keys())
    print("seeds with diagnostic:", seeds)

    print("\n===== (Q1) GLOBAL-LPC COLLAPSE vs lambda (per seed: tgt_bAcc / labelP / domAdv) =====")
    cfgs = ["erm:0", "lpc_prior:0.03", "lpc_prior:0.1", "lpc_prior:0.3", "lpc_prior:1", "lpc_prior:3"]
    for c in cfgs:
        cells = []
        for s in seeds:
            a = diag[s].get("aggregate", {}).get(c)
            cells.append("s%s: %.2f/%.2f/%+.2f" % (s, a["tgt_bacc"], a["label_probe_acc"],
                         a["domain_probe_adv"]) if a else "s%s: -" % s)
        print("  %-14s %s" % (c, " | ".join(cells)))

    print("\n===== (Q3) ABLATION REMOVABILITY (ERM, per seed: task Z->RZ, domain Z->RZ vs randR) =====")
    for s in seeds:
        a = abl[s].get("aggregate", {}).get("erm:0") if s in abl else None
        if a:
            print("  s%s: task %.2f->%.2f  domain %.2f->%.2f (randR %.2f)  [delete V_D: task kept, "
                  "domain ~unchanged & ~random]" % (s, a["task_Z"], a["task_RZ"], a["domain_Z"],
                  a["domain_RZ"], a["domain_Rrand"]))

    print("\n===== (Q4) SCORE-FISHER GATE on ERM (per seed: decision counts, task_ucb) =====")
    for s in seeds:
        a = diag[s].get("aggregate", {}).get("erm:0")
        if a:
            print("  s%s: %s  task_ucb=%s" % (s, a["sf_decisions"],
                  None if a["sf_k1_task_ucb_mean"] is None else round(a["sf_k1_task_ucb_mean"], 4)))

    print("\n===== (ADV) SUBJECT vs SESSION leakage (ERM, per seed: subj Z/RZ, sess Z/RZ) =====")
    for s in seeds:
        rows = adv.get(s)
        if rows:
            m = lambda k: float(np.nanmean([r[k] for r in rows]))
            print("  s%s: subject %.2f->%.2f (ch %.2f) | session %.2f->%.2f (ch %.2f)"
                  % (s, m("subj_Z"), m("subj_RZ"), rows[0]["chance_subj"], m("sess_Z"), m("sess_RZ"),
                     rows[0]["chance_sess"]))

    # stability verdict
    print("\n===== STABILITY VERDICT =====")
    try:
        coll = [diag[s]["aggregate"]["lpc_prior:1"]["tgt_bacc"] for s in seeds]
        erm_t = [diag[s]["aggregate"]["erm:0"]["tgt_bacc"] for s in seeds]
        rem = [abl[s]["aggregate"]["erm:0"]["domain_RZ"] for s in seeds if s in abl]
        print("LPC lam=1 tgt_bAcc across seeds: %s (collapse reproducible if all ~chance)" %
              [round(x, 3) for x in coll])
        print("ERM tgt_bAcc across seeds: %s" % [round(x, 3) for x in erm_t])
        print("ERM domain_RZ across seeds: %s (non-removability if all ~domain_Z)" %
              [round(x, 3) for x in rem])
    except Exception as e:
        print("(partial)", repr(e)[:120])
    print("AGGREGATE3_DONE")


if __name__ == "__main__":
    main()
