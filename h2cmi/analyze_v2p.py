"""V2P analysis (review V2P_FROZEN): prevalence SENSITIVITY, not a bAcc leaderboard.

Per (subject, method) over the three pools {1:1, 3:1, 1:3}:
  transform displacement D(r) = || theta_r - theta_{1:1} ||  (theta = concat(diag log-scale a, bias b));
  occupancy slope = d(predicted-right-occupancy)/d(pool right-log-odds)  -- the SIGNED prevalence-chase;
  fixed-eval DbAcc(r) = bAcc_r - bAcc_{1:1}.
Headline = the method x signed-log-odds INTERACTION: each method's occupancy slope with a subject-
bootstrap CI, compared across methods. Decided in advance: pooled&joint move / CC stable -> mechanism
split; CC also moves -> revised bias theorem; none move -> shrink the claim.
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict

import numpy as np

METHODS = ["identity", "always_pooled", "always_canonical_CC", "current_joint"]


def _slope(x, y):
    x = np.asarray(x, float); y = np.asarray(y, float)
    if len(x) < 2 or np.var(x) < 1e-12:
        return float("nan")
    return float(np.cov(x, y, bias=True)[0, 1] / np.var(x))


def _boot(vals, fn=np.mean, n_boot=10000, seed=0):
    v = np.asarray([x for x in vals if x == x], float)
    if len(v) < 2:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    bs = [fn(v[rng.integers(0, len(v), len(v))]) for _ in range(n_boot)]
    return float(fn(v)), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", default="results/h2cmi/v2p.report.json")
    args = ap.parse_args()
    rows = [json.loads(l) for l in open(args.inp) if l.strip()]
    # index by (dataset, subject, method) -> {ratio: row}
    units = defaultdict(dict)
    for r in rows:
        units[(r["dataset"], r["subject"], r["method"])][r["ratio"]] = r
    per_method = {m: dict(disp31=[], disp13=[], occ_slope=[], signed_occ=[], dbacc31=[], dbacc13=[],
                          tnorm_11=[], tnorm_31=[], tnorm_13=[]) for m in METHODS}
    n_units = len(set((d, s) for d, s, m in units))
    for (ds, subj, m), byr in units.items():
        if not all(k in byr for k in ("1:1", "3:1", "1:3")):
            continue
        t11 = np.array(byr["1:1"]["a"] + byr["1:1"]["b"])
        t31 = np.array(byr["3:1"]["a"] + byr["3:1"]["b"])
        t13 = np.array(byr["1:3"]["a"] + byr["1:3"]["b"])
        pm = per_method[m]
        pm["disp31"].append(float(np.linalg.norm(t31 - t11)))
        pm["disp13"].append(float(np.linalg.norm(t13 - t11)))
        lo = [byr[r]["pool_logodds"] for r in ("1:1", "3:1", "1:3")]
        occ = [byr[r]["occ"] for r in ("1:1", "3:1", "1:3")]
        pm["occ_slope"].append(_slope(lo, occ))
        pm["signed_occ"].append(byr["1:3"]["occ"] - byr["3:1"]["occ"])     # logodds(1:3)>logodds(3:1)
        pm["dbacc31"].append(byr["3:1"]["bacc"] - byr["1:1"]["bacc"])
        pm["dbacc13"].append(byr["1:3"]["bacc"] - byr["1:1"]["bacc"])
        for r, key in (("1:1", "tnorm_11"), ("3:1", "tnorm_31"), ("1:3", "tnorm_13")):
            pm[key].append(byr[r]["tnorm"])
    rep = {"audit": "V2P", "n_units": n_units, "methods": {}}
    for m in METHODS:
        pm = per_method[m]
        sl_m, sl_lo, sl_hi = _boot(pm["occ_slope"])
        so_m, so_lo, so_hi = _boot(pm["signed_occ"])
        rep["methods"][m] = dict(
            n=len([x for x in pm["occ_slope"] if x == x]),
            occ_slope=dict(mean=sl_m, ci95=[sl_lo, sl_hi], excludes_0=bool(sl_lo > 0 or sl_hi < 0)),
            signed_occ_move=dict(mean=so_m, ci95=[so_lo, so_hi], excludes_0=bool(so_lo > 0 or so_hi < 0)),
            mean_disp_3to1=float(np.nanmean(pm["disp31"])), mean_disp_1to3=float(np.nanmean(pm["disp13"])),
            mean_tnorm_11=float(np.nanmean(pm["tnorm_11"])),
            dbacc_3to1=float(np.nanmean(pm["dbacc31"])), dbacc_1to3=float(np.nanmean(pm["dbacc13"])))
    # interpretation grid (decided in advance)
    mv = {m: rep["methods"][m]["occ_slope"]["excludes_0"] for m in METHODS}
    pooled_joint_move = mv["always_pooled"] or mv["current_joint"]
    cc_moves = mv["always_canonical_CC"]
    if not (pooled_joint_move or cc_moves):
        verdict = "NO_DETECTABLE_CONTAMINATION_AT_3to1 -> shrink the empirical claim"
    elif cc_moves:
        verdict = "FIXED_PRIOR_CC_ALSO_MOVES -> supports revised bias theorem (fixed-prior != prevalence-invariant)"
    else:
        verdict = "POOLED_JOINT_MOVE_CC_STABLE -> supports the mechanism split"
    rep["verdict"] = verdict
    json.dump(rep, open(args.out, "w"), indent=2)

    print(f"=== V2P controlled prevalence intervention (n_units={n_units}) ===")
    print("  method                occ_slope[ci95]            signed_occ(1:3-3:1)   disp3:1 disp1:3  dbacc3:1 dbacc1:3")
    for m in METHODS:
        d = rep["methods"][m]
        s = d["occ_slope"]; so = d["signed_occ_move"]
        star = "*" if s["excludes_0"] else " "
        print(f"  {m:22s} {s['mean']:+.3f}[{s['ci95'][0]:+.3f},{s['ci95'][1]:+.3f}]{star}  "
              f"{so['mean']:+.3f}[{so['ci95'][0]:+.3f},{so['ci95'][1]:+.3f}]  "
              f"{d['mean_disp_3to1']:.3f}  {d['mean_disp_1to3']:.3f}   {d['dbacc_3to1']:+.3f}  {d['dbacc_1to3']:+.3f}")
    print(f"  VERDICT: {verdict}")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()
