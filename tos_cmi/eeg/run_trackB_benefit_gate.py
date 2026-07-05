"""Track B driver: run the source-OOD benefit gate over frozen dumps and emit its ACCEPT/REJECT/ABSTAIN
decisions (SOURCE-ONLY; no target). Reuses the committed frozen dumps. File-level joblib parallel.
  python -m tos_cmi.eeg.run_trackB_benefit_gate --datasets Lee2019_MI Cho2017 --backbones EEGNet TSMNet \
      --seeds 0 1 2 --folds 0 --n-pseudo 8
(--folds 0 = all folds; a positive N = first N folds for a dry-run.)
Writes tos_cmi/results/method_deepen/trackB/{trackB_actions.csv, trackB_summary.json}.
"""
from __future__ import annotations
import argparse
import csv
import glob
import json
import os
import re
import numpy as np
from joblib import Parallel, delayed

from tos_cmi.score_fisher import ScoreFisherConfig
from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import (gate_signals, gate_action, _boot_bound, METHODS,
                                                 SAFETY_EPS, BENEFIT_LCB)

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT = "tos_cmi/results/method_deepen/trackB"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))


def _dumps(dataset, backbone, seed, nfolds):
    d = "%s/%s_%s_LOSO" % (RESULTS, dataset, backbone)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (d, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def _one(dataset, backbone, seed, p, method, n_pseudo):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        subj = _ids(d["subject_source"])[0]
        n_cls = int(d["n_cls"])
        sig = gate_signals(Zs, ys, subj, n_cls, method, ScoreFisherConfig(), seed, n_pseudo=n_pseudo)
        return {"dataset": dataset, "backbone": backbone, "seed": seed, "fold": fold, "method": method, **sig}
    except Exception as e:
        return {"dataset": dataset, "backbone": backbone, "seed": seed, "fold": fold, "method": method,
                "fail": repr(e)[:150]}


def aggregate(rows):
    rows = [r for r in rows if not r.get("fail")]
    summary = {}
    for ds in sorted(set(r["dataset"] for r in rows)):
        for bb in sorted(set(r["backbone"] for r in rows if r["dataset"] == ds)):
            for m in METHODS:
                sub = [r for r in rows if r["dataset"] == ds and r["backbone"] == bb and r["method"] == m]
                if not sub:
                    continue
                tds = [r["task_drop"] for r in sub]; tfolds = [r["fold"] for r in sub]
                bvals, bfolds = [], []
                for r in sub:
                    for v in r.get("benefit", []):
                        bvals.append(v); bfolds.append(r["fold"])
                rng = np.random.default_rng(0)
                tucb = _boot_bound(tds, tfolds, "upper", rng=rng)
                blcb = _boot_bound(bvals, bfolds, "lower", rng=rng)
                dg = float(np.nanmean([r["domain_gain"] for r in sub]))
                summary["%s|%s|%s" % (ds, bb, m)] = {
                    "dataset": ds, "backbone": bb, "method": m,
                    "task_drop_ucb": tucb, "benefit_lcb": blcb, "domain_gain": dg,
                    "action": gate_action(tucb, blcb),
                    "task_drop_mean": float(np.nanmean(tds)), "benefit_mean": float(np.nanmean(bvals)) if bvals else float("nan"),
                    "n_folds": len(set(tfolds)), "n_benefit": len(bvals)}
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Lee2019_MI", "Cho2017"])
    ap.add_argument("--backbones", nargs="+", default=["EEGNet", "TSMNet"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--folds", type=int, default=0, help="0=all; N=first N folds (dry-run)")
    ap.add_argument("--n-pseudo", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    tasks = []
    for ds in a.datasets:
        for bb in a.backbones:
            for s in a.seeds:
                for p in _dumps(ds, bb, s, a.folds):
                    for m in METHODS:
                        tasks.append((ds, bb, s, p, m))
    print("Track B gate: %d (dump x method) tasks, n_jobs=%d, n_pseudo=%d, thresholds safety<=%.2f benefit>%.2f"
          % (len(tasks), N_JOBS, a.n_pseudo, SAFETY_EPS, BENEFIT_LCB), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_one)(ds, bb, s, p, m, a.n_pseudo) for ds, bb, s, p, m in tasks)
    nfail = sum(1 for r in rows if r.get("fail"))
    if nfail:
        print("[%d FAILED tasks]" % nfail, flush=True)
        for r in rows:
            if r.get("fail"):
                print("  [FAIL] %s %s seed%d fold%d %s: %s"
                      % (r["dataset"], r["backbone"], r["seed"], r["fold"], r["method"], r["fail"]), flush=True)
    summary = aggregate(rows)
    os.makedirs(OUT, exist_ok=True)
    suff = ("_" + a.tag) if a.tag else ""
    with open("%s/trackB_actions%s.csv" % (OUT, suff), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["dataset", "backbone", "method", "action", "task_drop_ucb",
                                           "benefit_lcb", "domain_gain", "task_drop_mean", "benefit_mean",
                                           "n_folds", "n_benefit"])
        w.writeheader()
        for v in summary.values():
            w.writerow({k: v[k] for k in w.fieldnames})
    json.dump({"summary": summary, "thresholds": {"safety_eps": SAFETY_EPS, "benefit_lcb": BENEFIT_LCB}},
              open("%s/trackB_summary%s.json" % (OUT, suff), "w"), indent=1)
    print("\n=== Track B gate actions (SOURCE-ONLY; target NOT used) ===")
    print("  %-11s %-7s %-9s %-8s | safety task-drop UCB | benefit LCB | domain-gain | action" % ("dataset", "bb", "method", ""))
    for k in sorted(summary):
        v = summary[k]
        print("  %-11s %-7s %-9s        |  %+.3f (mean %+.3f)   | %+.3f       | %+.3f      | %s"
              % (v["dataset"], v["backbone"], v["method"], v["task_drop_ucb"], v["task_drop_mean"],
                 v["benefit_lcb"], v["domain_gain"], v["action"]))
    acts = [v["action"] for v in summary.values()]
    print("\naction rates: ACCEPT %d / REJECT %d / ABSTAIN %d (of %d)"
          % (acts.count("ACCEPT"), acts.count("REJECT"), acts.count("ABSTAIN"), len(acts)))
    print("TRACKB_DONE")


if __name__ == "__main__":
    main()
