"""Track C hardening -- multi-seed factorial analysis (PM Step 2).

For each (arch, latent-dim) cell we pool the LEACE nonlinear (MLP) subject-decode residual over all
(seed x fold) dumps and report:

  1. a FOLD-CLUSTER bootstrap 95% CI per cell (resample the 9 LOSO folds with replacement, carrying ALL
     seeds of each picked fold -- folds are the independent units; seeds within a fold are correlated);
  2. a PAIRED matched-dimension contrast (TSMNet vs EEGNet residual difference at matched d_z), bootstrapped
     over the SHARED fold clusters so the two arms are resampled jointly (paired, not two independent draws);
  3. the regression  residual ~ log(d_z) + arch + log(d_z):arch  with FOLD-CLUSTER bootstrap CIs on every
     coefficient, to test capacity- vs type-mediation.

P0 z_dim FIX: the real latent dimension is read from analyze()["z_dim"] (= Z.shape[1]), NOT parsed from the
directory name -- TSMNet "_dim{m}" is the SPD size m, whose tangent dimension is m(m+1)/2 (e.g. m=6 -> 21),
so the directory integer is NOT the latent dimension. An assertion verifies the mapping per cell.

Reads the factorial npz directly (all seeds). CPU only:
  python -m tos_cmi.eeg.factorial_multiseed_analysis
Verdict rule: capacity-mediated if log(d_z) coef CI is positive & excludes 0, the arch main-effect and
interaction CIs are small / straddle 0, and matched-dim contrast CIs overlap 0.
"""
from __future__ import annotations
import glob
import json
import os
import re
import numpy as np
from joblib import Parallel, delayed
from tos_cmi.eeg.erasure_baselines import analyze

FACT = "tos_cmi/results/tos_cmi_eeg_frozen/factorial"
EXTRA = {"TSMNet_dim210": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_TSMNet_LOSO",
         "EEGNet_dim16": "tos_cmi/results/tos_cmi_eeg_frozen/BNCI2014_001_EEGNet_LOSO"}
RNG = np.random.default_rng(0)
B = 2000
# analyze() calls are CPU-bound sklearn/numpy and independent per dump -> parallelize at the FILE level.
# The fold-cluster bootstraps stay SERIAL (they use the global RNG) so CIs remain fully reproducible.
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))


def _cells():
    out = {}
    for d in sorted(glob.glob("%s/BNCI2014_001_*_dim*" % FACT)):
        m = re.search(r"_(TSMNet|EEGNet)_dim(\d+)$", d)
        if m:
            out["%s_dim%s" % (m.group(1), m.group(2))] = d
    out.update(EXTRA)  # main-run cells (canonical TSMNet d_z=210, EEGNet d_z=16) override any factorial dup
    return out


def _expected_zdim(cell):
    """The latent dimension we EXPECT for a cell, from its name, to validate analyze()['z_dim'].
    Factorial TSMNet "_dim{m}" -> tangent m(m+1)/2; factorial EEGNet "_dim{F2}" -> F2.
    EXTRA cells are keyed directly by latent dim, so the name integer IS the expected z_dim."""
    n = int(cell.split("dim")[1])
    if cell in EXTRA:
        return n
    return n * (n + 1) // 2 if cell.startswith("TSMNet") else n


def _cluster_ci(vals, folds, B=B):
    """Fold-cluster bootstrap: resample unique folds with replacement, carry every record of a picked fold."""
    vals = np.asarray(vals)
    by = {}
    for i, f in enumerate(folds):
        by.setdefault(f, []).append(i)
    uniq = sorted(by)
    means = []
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[f] for f in pick])
        means.append(vals[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


SIGNALS = ["res", "full", "task_full", "task_leace", "tos", "rand"]


def _analyze_file(cell, p):
    """Top-level (picklable) per-dump worker for joblib. analyze() is deterministic (seed fixed inside),
    so parallelism does not affect results. Returns a rec dict, or a {'fail':...} marker on error."""
    seed = p.split("seed")[-1].split(".")[0]
    fold = re.search(r"sub(\d+)_", p).group(1)
    try:
        r = analyze(p, with_rlace=False)
    except Exception as e:
        return {"cell": cell, "fail": repr(e), "p": p.split('/')[-1]}
    return {"cell": cell, "seed": seed, "fold": fold, "z_dim": int(r["z_dim"]),
            "res": float(r["subj_LEACE_mlp"]), "full": float(r["subj_full_mlp"]),
            "task_full": float(r["task_full_lin"]), "task_leace": float(r["task_LEACE_lin"]),
            "tos": float(r["subj_TOS_VD_mlp"]), "rand": float(r["subj_random_k_mlp"]),
            "chance": float(r["chance_subj"])}


def _build_cell(cell, recs):
    """Aggregate already-analyzed recs for one cell into means + fold-cluster CIs (serial; deterministic)."""
    if not recs:
        return None
    dz_exp = _expected_zdim(cell)
    bb = "TSMNet" if cell.startswith("TSMNet") else "EEGNet"
    dz_set = set(x["z_dim"] for x in recs)
    assert len(dz_set) == 1, "%s: inconsistent z_dim %s across dumps" % (cell, dz_set)
    dz_seen = dz_set.pop()
    assert dz_seen == dz_exp, ("Z_DIM MISMATCH %s: analyze z_dim=%d but name implies %d "
                               "(TSMNet tangent = m(m+1)/2!)" % (cell, dz_seen, dz_exp))
    folds = [x["fold"] for x in recs]
    out = {"backbone": bb, "z_dim": dz_seen, "n": len(recs),
           "nseed": len(set(x["seed"] for x in recs)), "nfold": len(set(folds)),
           "recs": recs, "chance": recs[0]["chance"]}
    for name in SIGNALS:   # fold-cluster CI per panel signal (res == LEACE residual is the verdict signal)
        m, lo, hi = _cluster_ci([x[name] for x in recs], folds)
        out["%s_mean" % name], out["%s_lo" % name], out["%s_hi" % name] = m, lo, hi
    return out


def main():
    cell_dirs = sorted(_cells().items())
    tasks = [(cell, p) for cell, d in cell_dirs
             for p in sorted(glob.glob("%s/sub*_erm_lam0_seed*.npz" % d))]
    print("Analyzing %d dumps across %d cells (n_jobs=%d, file-level parallel) ..."
          % (len(tasks), len(cell_dirs), N_JOBS), flush=True)
    results = Parallel(n_jobs=N_JOBS, backend="loky")(delayed(_analyze_file)(c, p) for c, p in tasks)
    by_cell = {}
    for r in results:
        if r.get("fail"):
            print("[FAIL] %s : %s" % (r["p"], r["fail"]), flush=True)
            continue
        by_cell.setdefault(r["cell"], []).append(r)
    cells = {}
    for cell, _d in cell_dirs:
        c = _build_cell(cell, by_cell.get(cell, []))
        if c is None:
            continue
        cells[cell] = c
        print("[%s] z=%d EXP=%d n=%d (%d seeds x %d folds) | LEACE residual %.3f [%.3f,%.3f]"
              % (cell, c["z_dim"], _expected_zdim(cell), c["n"], c["nseed"], c["nfold"],
                 c["res_mean"], c["res_lo"], c["res_hi"]), flush=True)

    rows = sorted(cells.values(), key=lambda r: (r["backbone"], r["z_dim"]))
    print("\n=== removability vs (arch, d_z): LEACE MLP residual, mean [95%% fold-cluster CI] ===")
    for r in rows:
        print("  %-7s z=%-3d  residual %.3f [%.3f, %.3f]  (full %.3f, task %.2f->%.2f, n=%d, %d seeds)"
              % (r["backbone"], r["z_dim"], r["res_mean"], r["res_lo"], r["res_hi"],
                 r["full_mean"], r["task_full_mean"], r["task_leace_mean"], r["n"], r["nseed"]))

    # ---- matched-dimension contrast: PAIRED fold-cluster bootstrap of (TSMNet - EEGNet) residual ----
    # A pair is only trusted for the verdict if it has >= MIN_CLUSTERS common folds; a paired bootstrap over
    # very few clusters gives a degenerately wide CI that overlaps 0 regardless of the true effect, so an
    # UNDERPOWERED pair must NOT be counted as evidence for capacity-mediation (audit wf_084b6aab). Any
    # TSMNet fold with no matching EEGNet fold is DROPPED -- we say so explicitly rather than silently.
    MIN_CLUSTERS = 6
    print("\n=== matched-dim contrast (TSMNet - EEGNet residual; PAIRED fold-cluster CI; overlaps 0 => capacity) ===")
    pairs = [(21, 16), (36, 32), (55, 64), (105, 128), (210, 210)]
    T = {r["z_dim"]: r for r in rows if r["backbone"] == "TSMNet"}
    E = {r["z_dim"]: r for r in rows if r["backbone"] == "EEGNet"}
    contrasts = []
    for tz, ez in pairs:
        if tz not in T or ez not in E:
            continue
        pf_t, pf_e = {}, {}
        for x in T[tz]["recs"]:
            pf_t.setdefault(x["fold"], []).append(x["res"])
        for x in E[ez]["recs"]:
            pf_e.setdefault(x["fold"], []).append(x["res"])
        common = sorted(set(pf_t) & set(pf_e))
        if not common:
            print("  [SKIP] TSMNet z=%d vs EEGNet z=%d : no common folds" % (tz, ez), flush=True)
            continue
        dropped = sorted(set(pf_t) - set(pf_e), key=int) + sorted(set(pf_e) - set(pf_t), key=int)
        under = len(common) < MIN_CLUSTERS
        if dropped:
            print("  [WARN] TSMNet z=%d vs EEGNet z=%d : %d common folds; dropping unmatched folds %s"
                  % (tz, ez, len(common), dropped), flush=True)
        tall = np.concatenate([pf_t[f] for f in common]); eall = np.concatenate([pf_e[f] for f in common])
        point = float(tall.mean() - eall.mean())
        diffs = []
        for _ in range(B):
            pick = RNG.choice(common, size=len(common), replace=True)
            tv = np.concatenate([pf_t[f] for f in pick]); ev = np.concatenate([pf_e[f] for f in pick])
            diffs.append(tv.mean() - ev.mean())
        lo, hi = float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))
        excl = "(*** excludes 0)" if (lo > 0 or hi < 0) else "(overlaps 0)"
        tag = " [UNDERPOWERED n<%d -> EXCLUDED from verdict]" % MIN_CLUSTERS if under else ""
        contrasts.append({"tz": tz, "ez": ez, "diff": point, "lo": lo, "hi": hi,
                          "ncommon": len(common), "dropped": dropped, "underpowered": under})
        print("  TSMNet z=%-3d vs EEGNet z=%-3d : diff=%+.3f [%+.3f, %+.3f]  (%d common folds) %s%s"
              % (tz, ez, point, lo, hi, len(common), excl, tag))

    # ---- regression: residual ~ log(d_z) + arch + log(d_z):arch  with fold-cluster bootstrap CIs ----
    names = ["intercept", "log(d_z)", "arch[TSMNet]", "log(d_z):arch"]
    Xall, yall, fall = [], [], []
    for r in rows:
        arch = 1.0 if r["backbone"] == "TSMNet" else 0.0
        for x in r["recs"]:
            Xall.append([1.0, np.log(r["z_dim"]), arch, np.log(r["z_dim"]) * arch])
            yall.append(x["res"]); fall.append("%s|%s" % (r["backbone"], x["fold"]))
    Xall = np.array(Xall); yall = np.array(yall)
    beta, *_ = np.linalg.lstsq(Xall, yall, rcond=None)
    # cluster bootstrap on fold (shared LOSO folds are the resampling unit; arch keeps cells distinct)
    folds = sorted(set(f.split("|")[1] for f in fall))
    by = {}
    for i, f in enumerate(fall):
        by.setdefault(f.split("|")[1], []).append(i)
    bs = []
    for _ in range(B):
        pick = RNG.choice(folds, size=len(folds), replace=True)
        idx = np.concatenate([by[f] for f in pick])
        bb, *_ = np.linalg.lstsq(Xall[idx], yall[idx], rcond=None)
        bs.append(bb)
    bs = np.array(bs); los = np.percentile(bs, 2.5, axis=0); his = np.percentile(bs, 97.5, axis=0)
    print("\n=== OLS  residual ~ log(d_z) + arch + log(d_z):arch  (n=%d rows, %d folds) ===" % (len(yall), len(folds)))
    ols = {}
    for nm, b, lo, hi in zip(names, beta, los, his):
        excl = "***" if (lo > 0 or hi < 0) else "   "
        ols[nm] = {"coef": float(b), "lo": float(lo), "hi": float(hi)}
        print("  %-16s %+.4f  [%+.4f, %+.4f] %s" % (nm, b, lo, hi, excl))

    out = {"cells": {k: {kk: vv for kk, vv in v.items() if kk != "recs"} for k, v in cells.items()},
           "matched_dim_contrast": contrasts, "ols": ols, "B": B}
    json.dump(out, open("%s/factorial_multiseed.json" % FACT, "w"), indent=1)
    wp = [c for c in contrasts if not c["underpowered"]]
    up = [c for c in contrasts if c["underpowered"]]
    print("\nVERDICT GUIDE: capacity-mediated <=> log(d_z) CI > 0 (excludes 0), arch & interaction CIs small / "
          "straddle 0, AND all WELL-POWERED matched-dim contrast CIs (>=%d folds) overlap 0. Otherwise: "
          "residual architecture effect." % MIN_CLUSTERS)
    print("  well-powered matched pairs: %d/%d; underpowered (excluded): %s"
          % (len(wp), len(contrasts), [(c["tz"], c["ez"], c["ncommon"]) for c in up] or "none"))
    print("FACTORIAL_MULTISEED_DONE")


if __name__ == "__main__":
    main()
