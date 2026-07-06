"""WAVE 0 / W0.5 metric-switch (sleep part, analyzer-only -- reuses the prior-decomp saved recalls + rho_E;
no re-run). Central claim: the correct decision prior is METRIC-DEPENDENT.

For each decision prior pi in {Unif, rho_A, rho_E, pi_J} compute, from the identity-geometry per-stage
recalls already saved by run_prior_decomp:
  BA(pi)          = mean_y Recall_y(pi)                    (uniform-weighted; BA optimum = Unif)
  OrdAcc(pi)      = sum_y rho_E[y] * Recall_y(pi)          (prevalence-weighted natural ordinary accuracy)
The metric-switch is the RANK FLIP: argmax_pi BA(pi) should be Unif, while argmax_pi OrdAcc(pi) should be
a prevalence prior (rho_E). Same prior, opposite verdict under the two metrics.
(V2P matched-prevalence ordinary-accuracy lens comes from the W0.2 report once it lands; merged here.)
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
STAGES = ["W", "N1", "N2", "N3", "REM"]
PRIORS = [("Unif", "recall_unif"), ("rho_A", "recall_rhoA"), ("rho_E", "recall_rhoE"), ("pi_J", "recall_piJ")]


def _load(proto):
    rows = []
    for f in glob.glob(f"results/h2cmi/wave0_priordecomp/pd_{proto}_*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if r.get("marker") == "PRIORDECOMP" and r.get("recall_unif"):
                    rows.append(r)
    return rows


def _cluster_boot(subj_vals, seed=0):
    v = np.array([subj_vals[s] for s in subj_vals], float)
    if len(v) < 2:
        return dict(mean=float(v.mean()) if len(v) else float("nan"), ci=[float("nan")] * 2, n=len(v))
    rng = np.random.default_rng(seed)
    bs = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
    return dict(mean=float(v.mean()), ci=[round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)], n=len(v))


def _metric_tables(rows):
    # per-row BA + OrdAcc under each prior, then subject-mean, then cluster bootstrap
    ba = {p: defaultdict(list) for p, _ in PRIORS}
    oa = {p: defaultdict(list) for p, _ in PRIORS}
    for r in rows:
        rhoE = np.asarray(r["rho_E"], float); s = int(r["target_subject"])
        for p, key in PRIORS:
            rec = np.asarray(r[key], float)
            ba[p][s].append(float(np.nanmean(rec)))
            oa[p][s].append(float(np.nansum(rhoE * rec)))
    BA = {p: _cluster_boot({s: float(np.mean(v)) for s, v in ba[p].items()}) for p, _ in PRIORS}
    OA = {p: _cluster_boot({s: float(np.mean(v)) for s, v in oa[p].items()}) for p, _ in PRIORS}
    argmax_ba = max(BA, key=lambda p: BA[p]["mean"]); argmax_oa = max(OA, key=lambda p: OA[p]["mean"])
    # paired contrasts that define the switch: BA(Unif)-BA(rho_E) and OrdAcc(rho_E)-OrdAcc(Unif)
    def paired(tab_dict, pa, pb):
        d = {}
        src = ba if tab_dict == "ba" else oa
        subs = set(src[pa]) & set(src[pb])
        return _cluster_boot({s: float(np.mean(src[pa][s])) - float(np.mean(src[pb][s])) for s in subs})
    return dict(balanced_accuracy=BA, ordinary_accuracy=OA,
                argmax_prior_under_BA=argmax_ba, argmax_prior_under_OrdAcc=argmax_oa,
                switch_confirmed=bool(argmax_ba == "Unif" and argmax_oa in ("rho_E", "rho_A")),
                BA_Unif_minus_rhoE=paired("ba", "Unif", "rho_E"),
                OrdAcc_rhoE_minus_Unif=paired("oa", "rho_E", "Unif"))


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave0_metricswitch.report.json")
    args = ap.parse_args()
    rep = dict(marker="WAVE0_METRICSWITCH_SLEEP",
               crossnight=_metric_tables(_load("crossnight")), samesession=_metric_tables(_load("samesession")))
    # merge V2P lens if the W0.2 report exists
    try:
        v2 = json.load(open("results/h2cmi/wave0_v2p.report.json"))
        rep["v2p_ordinary_accuracy_present"] = "utility" in v2
    except Exception:
        rep["v2p_ordinary_accuracy_present"] = False
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    for proto in ("crossnight", "samesession"):
        t = rep[proto]
        print(f"[W0.5 {proto}] argmax under BA = {t['argmax_prior_under_BA']} | argmax under OrdAcc = {t['argmax_prior_under_OrdAcc']} | SWITCH={t['switch_confirmed']}")
        print(f"  BA:     " + "  ".join(f"{p}={t['balanced_accuracy'][p]['mean']:.3f}" for p, _ in PRIORS))
        print(f"  OrdAcc: " + "  ".join(f"{p}={t['ordinary_accuracy'][p]['mean']:.3f}" for p, _ in PRIORS))
        print(f"  BA(Unif)-BA(rhoE)={t['BA_Unif_minus_rhoE']['mean']:+.4f} {t['BA_Unif_minus_rhoE']['ci']} | OrdAcc(rhoE)-OrdAcc(Unif)={t['OrdAcc_rhoE_minus_Unif']['mean']:+.4f} {t['OrdAcc_rhoE_minus_Unif']['ci']}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
