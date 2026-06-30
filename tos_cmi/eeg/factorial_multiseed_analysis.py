"""Track C hardening -- multi-seed factorial analysis (PM Step 2). Pools (seed x fold) per cell, gives
cluster-bootstrap 95% CIs, a matched-dimension contrast table (TSMNet vs EEGNet residual + diff CI), and
the regression  r_MLP-after-LEACE ~ log(d_z) + arch + log(d_z):arch  to test capacity- vs type-mediation.
Reads the factorial npz directly (all seeds) so it does not depend on per-seed json. Run on CPU.
  python -m tos_cmi.eeg.factorial_multiseed_analysis
Verdict rule: capacity-mediated if log(d_z) coef positive & stable, arch main effect small after
controlling d_z, interaction not large enough to reverse, and matched-dim CIs overlap.
"""
from __future__ import annotations
import glob
import json
import os
import re
import numpy as np
from tos_cmi.eeg.erasure_baselines import analyze

FACT = "tos_cmi/results/tos_cmi_eeg_frozen/factorial"
EXTRA = {"TSMNet_dim210": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO",
         "EEGNet_dim16": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO"}
RNG = np.random.default_rng(0)


def _cells():
    out = {}
    for d in sorted(glob.glob("%s/BNCI2014_001_*_dim*" % FACT)):
        m = re.search(r"_(TSMNet|EEGNet)_dim(\d+)$", d)
        if m:
            out["%s_dim%s" % (m.group(1), m.group(2))] = d
    out.update(EXTRA)
    return out


def _boot_ci(vals, units, B=2000):
    """cluster bootstrap over `units` (here the (seed,fold) id is the unit; folds are the clusters)."""
    vals = np.asarray(vals); uniq = sorted(set(units)); idx = {u: np.where(np.array(units) == u)[0] for u in uniq}
    means = []
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        s = np.concatenate([idx[u] for u in pick])
        means.append(vals[s].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def main():
    cells = {}
    for cell, d in sorted(_cells().items()):
        bb = "TSMNet" if "TSMNet" in cell else "EEGNet"; dz = int(cell.split("dim")[1])
        recs = []
        for p in sorted(glob.glob("%s/sub*_erm_lam0_seed*.npz" % d)):
            seed = p.split("seed")[-1].split(".")[0]; fold = re.search(r"sub(\d+)_", p).group(1)
            try:
                r = analyze(p, with_rlace=False)
                recs.append({"seed": seed, "fold": fold, "unit": "%s_%s" % (seed, fold),
                             "res": r["subj_LEACE_mlp"], "full": r["subj_full_mlp"],
                             "task_full": r["task_full_lin"], "task_leace": r["task_LEACE_lin"],
                             "tos": r["subj_TOS_VD_mlp"], "nd": r["nDcand"], "chance": r["chance_subj"]})
            except Exception as e:
                print("[FAIL] %s : %r" % (p.split('/')[-1], e), flush=True)
        if not recs:
            continue
        res = [x["res"] for x in recs]; m, lo, hi = _boot_ci(res, [x["unit"] for x in recs])
        cells[cell] = {"backbone": bb, "z_dim": dz, "n": len(recs), "nseed": len(set(x["seed"] for x in recs)),
                       "res_mean": m, "res_lo": lo, "res_hi": hi, "res_vals": res,
                       "full_mean": float(np.mean([x["full"] for x in recs])),
                       "task_full": float(np.mean([x["task_full"] for x in recs])),
                       "task_leace": float(np.mean([x["task_leace"] for x in recs])),
                       "tos_mean": float(np.mean([x["tos"] for x in recs])), "chance": recs[0]["chance"]}
        print("[%s] z=%d n=%d (%d seeds) | LEACE residual %.3f [%.3f,%.3f]"
              % (cell, dz, len(recs), cells[cell]["nseed"], m, lo, hi), flush=True)

    rows = sorted(cells.values(), key=lambda r: (r["backbone"], r["z_dim"]))
    print("\n=== removability vs (arch, d_z): LEACE MLP residual, mean [95% CI] ===")
    for r in rows:
        print("  %-7s z=%-3d  residual %.3f [%.3f, %.3f]  (full %.3f, task %.2f->%.2f, n=%d)"
              % (r["backbone"], r["z_dim"], r["res_mean"], r["res_lo"], r["res_hi"],
                 r["full_mean"], r["task_full"], r["task_leace"], r["n"]))

    # matched-dimension contrast (TSMNet vs EEGNet residual difference, bootstrap CI of the diff)
    print("\n=== matched-dimension contrast (TSMNet - EEGNet residual; CI overlap 0 => capacity-mediated) ===")
    pairs = [(21, 16), (36, 32), (55, 64), (105, 128), (210, 210)]
    T = {r["z_dim"]: r for r in rows if r["backbone"] == "TSMNet"}
    E = {r["z_dim"]: r for r in rows if r["backbone"] == "EEGNet"}
    for tz, ez in pairs:
        if tz in T and ez in E:
            a, b = np.array(T[tz]["res_vals"]), np.array(E[ez]["res_vals"])
            diffs = [RNG.choice(a, len(a)).mean() - RNG.choice(b, len(b)).mean() for _ in range(2000)]
            d, lo, hi = float(np.mean(a) - np.mean(b)), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
            print("  TSMNet z=%-3d vs EEGNet z=%-3d : diff=%+.3f [%+.3f, %+.3f] %s"
                  % (tz, ez, d, lo, hi, "(overlaps 0)" if lo <= 0 <= hi else "(*** excludes 0)"))

    # regression: residual ~ log(dz) + arch + log(dz):arch
    X, ylab, lab = [], [], []
    for r in rows:
        for v in r["res_vals"]:
            arch = 1.0 if r["backbone"] == "TSMNet" else 0.0
            X.append([1.0, np.log(r["z_dim"]), arch, np.log(r["z_dim"]) * arch]); ylab.append(v)
    X = np.array(X); ylab = np.array(ylab)
    beta, *_ = np.linalg.lstsq(X, ylab, rcond=None)
    names = ["intercept", "log(d_z)", "arch[TSMNet]", "log(d_z):arch"]
    print("\n=== OLS  residual ~ log(d_z) + arch + log(d_z):arch  (n=%d) ===" % len(ylab))
    for nm, b in zip(names, beta):
        print("  %-16s %+.4f" % (nm, b))
    json.dump({"cells": {k: {kk: vv for kk, vv in v.items() if kk != "res_vals"} for k, v in cells.items()},
               "ols": dict(zip(names, [float(b) for b in beta]))},
              open("%s/factorial_multiseed.json" % FACT, "w"), indent=1)
    print("\nVERDICT GUIDE: log(d_z)>0 & stable + small arch main effect + matched-dim diffs overlap 0 "
          "=> capacity-mediated. Large/!=0 arch or interaction => residual architecture effect.")
    print("FACTORIAL_MULTISEED_DONE")


if __name__ == "__main__":
    main()
