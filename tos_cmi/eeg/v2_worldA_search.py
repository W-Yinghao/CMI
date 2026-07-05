"""World-A construction search (design-lock verification). The adversarial design review PROVED the
'reversed' World-A nuisance yields net source-LOSO benefit <= 0 for all phi (unpassable). This script sweeps
candidate World-A mechanisms x params on REAL Lee/Cho EEGNet latents and reports, for the principled erasers
(LEACE + fair_conditional), the SOURCE-ONLY gate signals (safety task-drop UCB, source-LOSO benefit LCB) and
the post-hoc target ΔbAcc, to find a construction that is genuinely ACCEPT-able:
    safe (task-drop UCB <= 0.02)  AND  benefit LCB > +0.01  AND  actual target ΔbAcc LCB > +0.01.
If none clears it, the acceptance-power test is not constructible under this gate (a finding, not a bug).

  python -m tos_cmi.eeg.v2_worldA_search --datasets Lee2019_MI Cho2017 --folds 8 --seed 0
Writes tos_cmi/results/method_deepen/v2/worldA_search.csv (+ prints a ranked table).
"""
from __future__ import annotations
import argparse
import csv
import glob
import os
import re
import numpy as np
from joblib import Parallel, delayed

from tos_cmi.eeg.erasure_baselines import _ids
from tos_cmi.eeg.source_ood_benefit_gate import _boot_bound
from tos_cmi.eeg.run_v2_certificate import eval_v2
from tos_cmi.eeg.semi_synthetic_real_latent import inject
from tos_cmi.eeg.v2_worlds import FACTORIES

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
OUT = "tos_cmi/results/method_deepen/v2"
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))

VARIANTS = ["aligned_noise", "aligned_noise_flip", "reversed"]
F_GRID = [0.15, 0.25, 0.35]        # phi = fraction aligned (aligned_noise) / fraction reversed (reversed)
ALPHA_GRID = [0.5, 1.0, 2.0, 3.0]
ERASERS = ["leace_baseline", "fair_conditional_leace_disjoint_router"]


def _dumps(ds, bb, seed, nfolds):
    dd = "%s/%s_%s_LOSO" % (RESULTS, ds, bb)
    ps = sorted(glob.glob("%s/sub*_erm_lam0_seed%d.npz" % (dd, seed)),
                key=lambda p: int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1)))
    return ps[:nfolds] if nfolds else ps


def _cell(ds, p, variant, f_align, alpha, eraser, seed, n_pseudo):
    fold = int(re.search(r"sub(\d+)_", p.split("/")[-1]).group(1))
    try:
        d = np.load(p, allow_pickle=True)
        Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
        Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
        subj = _ids(d["subject_source"])[0]; n_cls = int(d["n_cls"])
        inj = inject("A", Zs, ys, subj, Zt, yt, alpha=alpha, phi=f_align, seed=seed, variantA=variant)
        sig = eval_v2(inj["Zs2"], ys, inj["z_src"], inj["grp_subj"], inj["Zt2"], yt, n_cls,
                      FACTORIES[eraser], seed, n_pseudo)
        return {"dataset": ds, "fold": fold, "variant": variant, "f_align": f_align, "alpha": alpha,
                "eraser": eraser, "task_drop": sig["task_drop"], "benefit": sig["benefit"],
                "domain_gain": sig["domain_gain"], "dtgt": sig["tgt_bacc_eras"] - sig["tgt_bacc_full"]}
    except Exception as e:
        return {"dataset": ds, "fold": fold, "variant": variant, "f_align": f_align, "alpha": alpha,
                "eraser": eraser, "fail": repr(e)[:150]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=["Lee2019_MI", "Cho2017"])
    ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--folds", type=int, default=8)
    ap.add_argument("--n-pseudo", type=int, default=8)
    a = ap.parse_args()
    tasks = []
    for ds in a.datasets:
        for p in _dumps(ds, a.backbone, a.seed, a.folds):
            for v in VARIANTS:
                for f in F_GRID:
                    for al in ALPHA_GRID:
                        for er in ERASERS:
                            tasks.append((ds, p, v, f, al, er))
    print("World-A search: %d cells (n_jobs=%d)" % (len(tasks), N_JOBS), flush=True)
    rows = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_cell)(ds, p, v, f, al, er, a.seed, a.n_pseudo) for (ds, p, v, f, al, er) in tasks)
    rows = [r for r in rows if not r.get("fail")]
    # aggregate per (dataset, variant, f, alpha, eraser)
    agg = {}
    for ds in a.datasets:
        for v in VARIANTS:
            for f in F_GRID:
                for al in ALPHA_GRID:
                    for er in ERASERS:
                        sub = [r for r in rows if (r["dataset"], r["variant"], r["f_align"], r["alpha"],
                                                   r["eraser"]) == (ds, v, f, al, er)]
                        if not sub:
                            continue
                        folds = [r["fold"] for r in sub]
                        tds = [r["task_drop"] for r in sub]
                        bvals, bfolds = [], []
                        for r in sub:
                            for x in r["benefit"]:
                                bvals.append(x); bfolds.append(r["fold"])
                        dts = [r["dtgt"] for r in sub]
                        tucb = _boot_bound(tds, folds, "upper", rng=np.random.default_rng(0))
                        blcb = _boot_bound(bvals, bfolds, "lower", rng=np.random.default_rng(0)) if bvals else float("nan")
                        dlcb = _boot_bound(dts, folds, "lower", rng=np.random.default_rng(1))
                        accept = (tucb <= 0.02) and (blcb > 0.01)
                        good = accept and (dlcb > 0.01)
                        agg[(ds, v, f, al, er)] = dict(dataset=ds, variant=v, f_align=f, alpha=al, eraser=er,
                            task_drop_ucb=tucb, benefit_lcb=blcb, dtgt_lcb=dlcb, dtgt=float(np.mean(dts)),
                            domain_gain=float(np.nanmean([r["domain_gain"] for r in sub])),
                            gate_accept=accept, accept_and_target_gain=good, n=len(sub))
    os.makedirs(OUT, exist_ok=True)
    with open("%s/worldA_search.csv" % OUT, "w", newline="") as fh:
        cols = ["dataset", "variant", "f_align", "alpha", "eraser", "task_drop_ucb", "benefit_lcb",
                "dtgt_lcb", "dtgt", "domain_gain", "gate_accept", "accept_and_target_gain", "n"]
        w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
        for v in agg.values():
            w.writerow(v)
    # print winners
    wins = [v for v in agg.values() if v["accept_and_target_gain"]]
    accs = [v for v in agg.values() if v["gate_accept"]]
    print("\n=== cells that ACCEPT (safe + benefit LCB>+0.01): %d ; of which real target gain (dtgt LCB>+0.01): %d ==="
          % (len(accs), len(wins)))
    hdr = "  %-11s %-18s f_al alpha %-8s | tdUCB  benLCB  dtgtLCB  dtgt   domGain  ACCEPT target?" % ("dataset", "variant", "eraser")
    print(hdr)
    for v in sorted(agg.values(), key=lambda v: (-int(v["accept_and_target_gain"]), -int(v["gate_accept"]), -v["benefit_lcb"]))[:30]:
        print("  %-11s %-18s %.2f  %.2f  %-8s | %+.3f %+.3f  %+.3f  %+.3f  %+.3f   %s    %s"
              % (v["dataset"], v["variant"], v["f_align"], v["alpha"], v["eraser"][:8],
                 v["task_drop_ucb"], v["benefit_lcb"], v["dtgt_lcb"], v["dtgt"], v["domain_gain"],
                 "Y" if v["gate_accept"] else ".", "Y" if v["accept_and_target_gain"] else "."))
    print("WORLDA_SEARCH_DONE")


if __name__ == "__main__":
    main()
