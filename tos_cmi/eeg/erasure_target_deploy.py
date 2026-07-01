"""Step 3 -- frozen-erasure TARGET-DEPLOYMENT test (closes the measurement-to-control loop for Track G).

For each backbone x seed x outer-LOSO fold we:
  1. load SOURCE (Z_s, y_s, subj_s) and the held-out TARGET (Z_t, y_t) from the frozen dump;
  2. fit each concept eraser on SOURCE ONLY;
  3. transform BOTH source and target with the same eraser;
  4. train a linear task head on transformed SOURCE ONLY;
  5. evaluate on transformed held-out TARGET (balanced accuracy + NLL).

STRICT no-leakage contract (this is the whole point of the test):
  * target arrays (Z_t, y_t) are passed ONLY to the final scoring function `_task_metrics`;
  * no eraser, task head, hyper-parameter, or model is selected/tuned/calibrated on the target;
  * random-k removal is averaged over `--nrandom` draws to stabilise the random baseline.
Answers the reviewer question left open by Table 2 (a source-side control): does removing source subject
leakage -- by ANY eraser -- actually improve source->target transfer?

Methods: full Z | TOS V_D deletion | LEACE | RLACE | INLP | same-k random removal.
CPU (sklearn/numpy/torch); file-level joblib parallelism; submit via SLURM (scripts/tos_eeg_erasure_deploy.sbatch).
  python -m tos_cmi.eeg.erasure_target_deploy [--seed S] [--nrandom 8] [--no-rlace] [--limit N]
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, log_loss

from tos_cmi.score_fisher import (ScoreFisherConfig, _metric, _cross_fit_fisher, _SplitPlan,
                                  candidate_order, _m_proj)
from tos_cmi.eeg.erasure_baselines import _ids, leace_eraser, inlp_eraser, rlace_eraser

RESULTS = "tos_cmi/results/tos_cmi_eeg_frozen"
DIRS = {"TSMNet": "%s/BNCI2014_001_TSMNet_LOSO" % RESULTS,
        "EEGNet": "%s/BNCI2014_001_EEGNet_LOSO" % RESULTS}
OUT = "%s/erasure_target_deploy" % RESULTS
METHODS = ["full", "TOS_VD", "LEACE", "RLACE", "INLP", "random_k"]
N_JOBS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 4))
RNG = np.random.default_rng(0)
B = 2000


# ----------------------------- source-only fitting -----------------------------
def _fit_source_erasers(Zs, ys, subj_s, n_cls, ns, cfg, seed, with_rlace):
    """Fit every eraser on SOURCE ONLY. Returns (erasers {name: fn}, M metric, k=nDcand). No target seen."""
    zdim = Zs.shape[1]
    oh = np.eye(ns)[subj_s]
    erasers = {"full": (lambda X: X),
               "LEACE": leace_eraser(Zs, oh),
               "INLP": inlp_eraser(Zs, subj_s)}
    if with_rlace:
        try:
            erasers["RLACE"] = rlace_eraser(Zs, subj_s, seed=seed)
        except Exception as e:
            print("  (RLACE skipped: %r)" % e, flush=True)
    plan = _SplitPlan(len(ys), cfg.n_folds, 1)
    M = _metric(Zs, ys, n_cls, cfg)
    G_Y = _cross_fit_fisher(Zs, ys, None, n_cls, zdim, 0, cfg, plan, 0)
    G_DgY = _cross_fit_fisher(Zs, subj_s, np.eye(n_cls)[ys], ns, zdim, n_cls, cfg, plan, 100)
    V_D = candidate_order(G_DgY, G_Y, M, cfg, 0.0)[0]
    k = int(V_D.shape[1])
    erasers["TOS_VD"] = (lambda X: X) if k == 0 else (lambda X: X - X @ _m_proj(V_D, M).T)
    return erasers, M, k


def _task_metrics(Zs_e, ys, Zt_e, yt, n_cls, split_rng):
    """Task head on transformed SOURCE -> transformed TARGET. TARGET labels used ONLY for scoring here.
    Returns (tgt_bacc, tgt_nll, src_bacc, src_nll). Source metrics via an internal 50/50 source split."""
    head = LogisticRegression(max_iter=200, C=1.0).fit(Zs_e, ys)   # SOURCE ONLY
    pt = head.predict_proba(Zt_e)
    proba = np.zeros((len(Zt_e), n_cls)); proba[:, head.classes_] = pt   # align to full label set
    tgt_bacc = float(balanced_accuracy_score(yt, head.classes_[pt.argmax(1)]))
    tgt_nll = float(log_loss(yt, proba, labels=np.arange(n_cls)))
    # source-internal sanity split (confirms erasure did not destroy source task)
    n = len(ys); perm = split_rng.permutation(n); A, Bx = perm[:n // 2], perm[n // 2:]
    h2 = LogisticRegression(max_iter=200, C=1.0).fit(Zs_e[A], ys[A])
    p2 = h2.predict_proba(Zs_e[Bx]); pr2 = np.zeros((len(Bx), n_cls)); pr2[:, h2.classes_] = p2
    src_bacc = float(balanced_accuracy_score(ys[Bx], h2.classes_[p2.argmax(1)]))
    src_nll = float(log_loss(ys[Bx], pr2, labels=np.arange(n_cls)))
    return tgt_bacc, tgt_nll, src_bacc, src_nll


def _subj_after(Zs_e, subj_s, split_rng):
    """Source subject decode after erasure (sanity that the eraser removed subject info on source)."""
    n = len(subj_s); perm = split_rng.permutation(n); A, Bx = perm[:n // 2], perm[n // 2:]
    if len(np.unique(subj_s[A])) < 2:
        return float("nan")
    return float((LogisticRegression(max_iter=200).fit(Zs_e[A], subj_s[A]).predict(Zs_e[Bx]) == subj_s[Bx]).mean())


def _row(bb, seed, fold, tsub, nm, tb, tn, sb, sn, sd, k, n_cls):
    return {"backbone": bb, "seed": seed, "fold": fold, "target_subject": tsub, "method": nm,
            "tgt_bacc": tb, "tgt_nll": tn, "src_bacc": sb, "src_nll": sn, "subj_dec_after": sd,
            "nDcand": k, "chance_task": 1.0 / n_cls}


def _deploy_file(bb, p, with_rlace, n_random):
    """One dump -> one row per method. Target only enters _task_metrics scoring."""
    d = np.load(p, allow_pickle=True)
    Zs = d["Z_source"].astype(np.float64); ys = d["y_source"].astype(int)
    Zt = d["Z_target"].astype(np.float64); yt = d["y_target"].astype(int)
    subj_s, ns = _ids(d["subject_source"]); n_cls = int(d["n_cls"])
    seed = int(d["seed"]); fold = re.search(r"sub(\d+)_", p).group(1); tsub = int(d["target_subject"])
    cfg = ScoreFisherConfig()
    erasers, M, k = _fit_source_erasers(Zs, ys, subj_s, n_cls, ns, cfg, seed, with_rlace)  # SOURCE ONLY
    rows = []
    for nm, E in erasers.items():
        tb, tn, sb, sn = _task_metrics(E(Zs), ys, E(Zt), yt, n_cls, np.random.default_rng(seed))
        sd = _subj_after(E(Zs), subj_s, np.random.default_rng(seed))
        rows.append(_row(bb, seed, fold, tsub, nm, tb, tn, sb, sn, sd, k, n_cls))
    # same-k random removal, averaged over n_random draws
    if k > 0:
        rng = np.random.default_rng(seed)
        acc = {"tgt_bacc": [], "tgt_nll": [], "src_bacc": [], "src_nll": [], "subj_dec_after": []}
        for _ in range(n_random):
            Vr = rng.standard_normal((Zs.shape[1], k))
            E = (lambda X, Vr=Vr: X - X @ _m_proj(Vr, M).T)
            tb, tn, sb, sn = _task_metrics(E(Zs), ys, E(Zt), yt, n_cls, np.random.default_rng(seed))
            sd = _subj_after(E(Zs), subj_s, np.random.default_rng(seed))
            for kk, vv in zip(acc, [tb, tn, sb, sn, sd]):
                acc[kk].append(vv)
        rows.append(_row(bb, seed, fold, tsub, "random_k", *[float(np.mean(acc[k2])) for k2 in
                    ["tgt_bacc", "tgt_nll", "src_bacc", "src_nll", "subj_dec_after"]], k, n_cls))
    else:
        full = [r for r in rows if r["method"] == "full"][0]
        rows.append(_row(bb, seed, fold, tsub, "random_k", full["tgt_bacc"], full["tgt_nll"],
                    full["src_bacc"], full["src_nll"], full["subj_dec_after"], k, n_cls))
    return rows


# ----------------------------- aggregation -----------------------------
def _cluster_ci(vals, folds):
    """Fold-cluster bootstrap over paired deltas (folds are the units; seeds within a fold are carried)."""
    vals = np.asarray(vals); by = {}
    for i, f in enumerate(folds):
        by.setdefault(f, []).append(i)
    uniq = sorted(by); means = []
    for _ in range(B):
        pick = RNG.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([by[f] for f in pick])
        means.append(vals[idx].mean())
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(vals.mean()), float(lo), float(hi)


def aggregate(all_rows):
    idx = {(r["backbone"], r["seed"], r["fold"], r["method"]): r for r in all_rows}
    summary, paired = {}, []
    for bb in DIRS:
        units = sorted(set((r["seed"], r["fold"]) for r in all_rows if r["backbone"] == bb))
        for nm in METHODS:
            rs = [idx[(bb, s, f, nm)] for (s, f) in units if (bb, s, f, nm) in idx]
            if not rs:
                continue
            tb = [r["tgt_bacc"] for r in rs]
            # worst-subject target bAcc: min over folds of the seed-averaged target bAcc
            by_fold = {}
            for r in rs:
                by_fold.setdefault(r["fold"], []).append(r["tgt_bacc"])
            worst = float(min(np.mean(v) for v in by_fold.values()))
            rec = {"backbone": bb, "method": nm, "n": len(rs),
                   "tgt_bacc_mean": float(np.mean(tb)), "tgt_nll_mean": float(np.mean([r["tgt_nll"] for r in rs])),
                   "src_bacc_mean": float(np.mean([r["src_bacc"] for r in rs])),
                   "src_nll_mean": float(np.mean([r["src_nll"] for r in rs])),
                   "subj_dec_after_mean": float(np.mean([r["subj_dec_after"] for r in rs])),
                   "worst_subject_tgt_bacc": worst, "chance_task": rs[0]["chance_task"]}
            if nm != "full":
                d_bacc, d_nll, folds = [], [], []
                for (s, f) in units:
                    if (bb, s, f, nm) in idx and (bb, s, f, "full") in idx:
                        d_bacc.append(idx[(bb, s, f, nm)]["tgt_bacc"] - idx[(bb, s, f, "full")]["tgt_bacc"])
                        d_nll.append(idx[(bb, s, f, nm)]["tgt_nll"] - idx[(bb, s, f, "full")]["tgt_nll"])
                        folds.append(f)
                b_m, b_lo, b_hi = _cluster_ci(d_bacc, folds)
                n_m, n_lo, n_hi = _cluster_ci(d_nll, folds)
                # improvement = bAcc delta CI excludes 0 above, OR NLL delta CI excludes 0 below (lower=better)
                improves = bool(b_lo > 0 or n_hi < 0)
                worsens = bool(b_hi < 0 or n_lo > 0)
                rec.update({"dtgt_bacc": b_m, "dtgt_bacc_lo": b_lo, "dtgt_bacc_hi": b_hi,
                            "dtgt_nll": n_m, "dtgt_nll_lo": n_lo, "dtgt_nll_hi": n_hi,
                            "improves_target": improves, "worsens_target": worsens})
                paired.append({"backbone": bb, "method": nm,
                               "dtgt_bacc": b_m, "dtgt_bacc_lo": b_lo, "dtgt_bacc_hi": b_hi,
                               "dtgt_nll": n_m, "dtgt_nll_lo": n_lo, "dtgt_nll_hi": n_hi,
                               "improves_target": improves})
            summary[(bb, nm)] = rec
    return summary, paired


def _write(all_rows, summary, paired):
    os.makedirs(OUT, exist_ok=True)
    cols = ["backbone", "seed", "fold", "target_subject", "method", "tgt_bacc", "tgt_nll",
            "src_bacc", "src_nll", "subj_dec_after", "nDcand", "chance_task"]
    for bb in DIRS:
        for s in sorted(set(r["seed"] for r in all_rows if r["backbone"] == bb)):
            rows = [r for r in all_rows if r["backbone"] == bb and r["seed"] == s]
            with open("%s/BNCI2014_001_%s_seed%d.csv" % (OUT, bb, s), "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=cols); w.writeheader()
                for r in sorted(rows, key=lambda r: (r["fold"], r["method"])):
                    w.writerow({k: r[k] for k in cols})
    with open("%s/erasure_target_deploy_paired.csv" % OUT, "w", newline="") as fh:
        pk = ["backbone", "method", "dtgt_bacc", "dtgt_bacc_lo", "dtgt_bacc_hi",
              "dtgt_nll", "dtgt_nll_lo", "dtgt_nll_hi", "improves_target"]
        w = csv.DictWriter(fh, fieldnames=pk); w.writeheader()
        for r in paired:
            w.writerow(r)
    json.dump({"summary": {"%s|%s" % k: v for k, v in summary.items()}, "paired": paired,
               "n_rows": len(all_rows)}, open("%s/erasure_target_deploy_summary.json" % OUT, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=str, default=None)
    ap.add_argument("--nrandom", type=int, default=8)
    ap.add_argument("--no-rlace", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="dry-run: process only the first N dumps")
    a = ap.parse_args()
    tasks = []
    for bb, d in DIRS.items():
        for p in sorted(glob.glob("%s/sub*_erm_lam0_seed*.npz" % d)):
            if a.seed and not p.endswith("_seed%s.npz" % a.seed):
                continue
            tasks.append((bb, p))
    if a.limit:
        tasks = tasks[:a.limit]
    print("Deploying %d dumps (n_jobs=%d, nrandom=%d, rlace=%s) ..."
          % (len(tasks), N_JOBS, a.nrandom, not a.no_rlace), flush=True)
    res = Parallel(n_jobs=N_JOBS, backend="loky")(
        delayed(_deploy_file)(bb, p, not a.no_rlace, a.nrandom) for bb, p in tasks)
    all_rows = [r for sub in res for r in sub]
    summary, paired = aggregate(all_rows)
    _write(all_rows, summary, paired)
    for bb in DIRS:
        if not any(r["backbone"] == bb for r in all_rows):
            continue
        ch = next(v["chance_task"] for k, v in summary.items() if k[0] == bb)
        print("\n=== %s target deployment (mean over %d folds x seeds; chance bAcc %.3f) ==="
              % (bb, summary[(bb, "full")]["n"], ch))
        print("  %-9s tgt_bAcc  tgt_NLL  |  d_bAcc [95%% CI]      d_NLL [95%% CI]        src_bAcc subj_dec  verdict"
              % "method")
        for nm in METHODS:
            r = summary.get((bb, nm))
            if not r:
                continue
            if nm == "full":
                print("  %-9s %.3f     %.3f    |  (baseline)                                    %.3f    %.3f"
                      % (nm, r["tgt_bacc_mean"], r["tgt_nll_mean"], r["src_bacc_mean"], r["subj_dec_after_mean"]))
            else:
                v = "IMPROVES" if r["improves_target"] else ("WORSENS" if r["worsens_target"] else "no change")
                print("  %-9s %.3f     %.3f    |  %+.3f [%+.3f,%+.3f]  %+.3f [%+.3f,%+.3f]  %.3f    %.3f    %s"
                      % (nm, r["tgt_bacc_mean"], r["tgt_nll_mean"], r["dtgt_bacc"], r["dtgt_bacc_lo"],
                         r["dtgt_bacc_hi"], r["dtgt_nll"], r["dtgt_nll_lo"], r["dtgt_nll_hi"],
                         r["src_bacc_mean"], r["subj_dec_after_mean"], v))
    print("ERASURE_TARGET_DEPLOY_DONE")


if __name__ == "__main__":
    main()
