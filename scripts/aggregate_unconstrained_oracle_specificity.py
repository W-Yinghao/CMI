#!/usr/bin/env python
"""C0 closeout (reproducible): does the conditional-subject span (B_cond) carry SUBSPACE-SPECIFIC target
utility beyond an equal-budget RANDOM basis? Fair control: BOTH the informed oracle and each random-basis
oracle select their deletion GREEDILY on T_cal labels (identical rank + candidate budget) and score ONLY on
T_query -- the random control is NEVER allowed to pick the best projector on the query set. Statistical unit =
target subject (3 seeds pooled within subject), subject-cluster 10k bootstrap.

  dI_specific = dI_informed_oracle - E_R[dI_random_oracle_R]   (also vs Q95 of the random oracle)

Closure gate: if BOTH datasets have UCB95(dI_specific) <= 0  OR  informed subject-win-rate <= random ->
SUBJECT_SUBSPACE_SELECTOR_CLOSED. This re-grounds the target-X closure on a reproducible fair control; it does
NOT reopen method development and does NOT run GPU Gate 5.

  python scripts/aggregate_unconstrained_oracle_specificity.py --seeds 0 1 2 --n_random 100
"""
from __future__ import annotations
import argparse, csv, glob, json, subprocess, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eeg.relaxation_ladder import feat_from_tos_dump, _dense
from tos_cmi.eval import targetx_metric as M
from tos_cmi.eval.dg_identifiability import _select_subset, _bacc
from tos_cmi.eval.targetx_observability import session_split

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _query_macro_gain(Zs_w, ys, Xq_w, yq, sq, U, seed):
    """Session-macro query bAcc gain of deleting whitened dir set U (fresh head on whitened source)."""
    def gain(Zq, y):
        base = _bacc(Zs_w, ys, Zq, y, seed)
        got = _bacc(Zs_w - (Zs_w @ U.T) @ U if U.shape[0] else Zs_w,
                    ys, Zq - (Zq @ U.T) @ U if U.shape[0] else Zq, y, seed)
        return got - base
    per = [gain(Xq_w[sq == s], yq[sq == s]) for s in np.unique(sq)
           if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float(gain(Xq_w, yq))


def _oracle_gain(Zs_w, ys, B_dict, Xcal_w, ycal, Xq_w, yq, sq, max_k, seed):
    """Greedy select on T_cal labels over dictionary B_dict, score on T_query (session-macro). Non-deployable."""
    S = _select_subset(Zs_w, ys, Xcal_w, ycal, B_dict, "greedy", max_k, seed)   # select on CAL only
    U = M._orthonormal(B_dict[S]) if S else np.zeros((0, B_dict.shape[1]))
    return _query_macro_gain(Zs_w, ys, Xq_w, yq, sq, U, seed), len(S)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"]); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--n_random", type=int, default=100); ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    subj_rows = []
    for ds in DATASETS:
        cells = [p for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                 f"{ds}_{a.backbone}_LOSO" / "sub*_erm_lam0_seed*.npz")))
                 if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)]
        if a.limit:
            cells = cells[: a.limit]
        for cp in cells:
            f = feat_from_tos_dump(cp); sd = int(f["seed"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
            Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int); st = f["session_target"]
            W = M.source_whitener(Zs); Zs_w = M.to_whitened(Zs, W)
            cal, qry, _ = session_split(st, yt, sd)
            Xcal_w, ycal = M.to_whitened(Zt[cal], W), yt[cal]
            Xq_w, yq, sq = M.to_whitened(Zt[qry], W), yt[qry], np.asarray(st)[qry]
            B_cond_w = M.whitened_cond_basis(Zs_w, ys, dsc, max_rank=a.max_rank)
            r = B_cond_w.shape[0]
            if r == 0:
                continue
            g_inf, k_inf = _oracle_gain(Zs_w, ys, B_cond_w, Xcal_w, ycal, Xq_w, yq, sq, a.max_rank, sd)
            g_rand = []
            for i, Q in enumerate(M.ambient_random_projectors_whitened(Zs.shape[1], r, a.n_random, sd)):
                # Q is a random rank-r orthonormal DICTIONARY (same rank as B_cond); same greedy/rank budget
                g, _k = _oracle_gain(Zs_w, ys, Q, Xcal_w, ycal, Xq_w, yq, sq, a.max_rank, sd)
                g_rand.append(g)
            g_rand = np.array(g_rand)
            subj_rows.append(dict(dataset=ds, subject=str(f["heldout_subject"]), seed=sd, dict_rank=int(r),
                                  informed_gain=float(g_inf), informed_k=int(k_inf),
                                  random_gain_mean=float(g_rand.mean()), random_gain_q95=float(np.quantile(g_rand, 0.95)),
                                  dI_specific=float(g_inf - g_rand.mean()), dI_vs_q95=float(g_inf - np.quantile(g_rand, 0.95)),
                                  informed_beats_q95=bool(g_inf > np.quantile(g_rand, 0.95)), git_sha=sha))
            print(f"  {ds} sub{f['heldout_subject']} s{sd}: informed={g_inf:+.4f} rand_mean={g_rand.mean():+.4f} "
                  f"rand_q95={np.quantile(g_rand,0.95):+.4f} dI_spec={g_inf-g_rand.mean():+.4f}", flush=True)
    # aggregate: per subject (mean over seeds), cluster bootstrap
    def _ci(vals, stat=np.mean, seed=7):
        v = np.asarray([x for x in vals if np.isfinite(x)], float)
        if not v.size:
            return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
        rng = np.random.default_rng(seed); b = [stat(v[rng.integers(0, v.size, v.size)]) for _ in range(10000)]
        return dict(mean=float(stat(v)), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(v.size))
    verdict, summ = {}, []
    for ds in DATASETS:
        R = [r for r in subj_rows if r["dataset"] == ds]
        if not R:
            continue
        by = defaultdict(list); win = defaultdict(list)
        for r in R:
            by[r["subject"]].append(r["dI_specific"]); win[r["subject"]].append(1.0 if r["informed_beats_q95"] else 0.0)
        spec = _ci([np.mean(v) for v in by.values()]); winrate = _ci([np.mean(v) for v in win.values()])
        closed = bool(spec["hi"] <= 0 or winrate["mean"] <= 0.5)
        summ.append(dict(dataset=ds, dI_specific_mean=spec["mean"], dI_specific_lo=spec["lo"], dI_specific_hi=spec["hi"],
                         subject_win_rate=winrate["mean"], n_subjects=spec["n"], closes=closed))
        verdict[ds] = dict(dI_specific_ucb=spec["hi"], subject_win_rate=winrate["mean"], closes=closed)
    overall = "SUBJECT_SUBSPACE_SELECTOR_CLOSED" if all(v["closes"] for v in verdict.values()) else "NOT_CLOSED_BY_FAIR_CONTROL"
    with open(OUT / "unconstrained_oracle_subject_rows.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(subj_rows[0].keys())); w.writeheader(); [w.writerow(r) for r in subj_rows]
    with open(OUT / "unconstrained_oracle_specificity_full.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(summ[0].keys())); w.writeheader(); [w.writerow(r) for r in summ]
    json.dump({"per_dataset": verdict, "overall_verdict": overall, "n_random": a.n_random,
               "control": "greedy_select_on_Tcal_score_on_Tquery_equal_budget"},
              open(OUT / "unconstrained_oracle_closure_verdict.json", "w"), indent=2, default=float)
    print(f"\n[C0-closeout] {len(subj_rows)} fold-seed cells")
    for s in summ:
        print(f"  {s['dataset']}: dI_specific={s['dI_specific_mean']:+.4f} [{s['dI_specific_lo']:+.4f},{s['dI_specific_hi']:+.4f}] "
              f"(UCB={s['dI_specific_hi']:+.4f}) win_rate={s['subject_win_rate']:.2f} closes={s['closes']}")
    print(f"  OVERALL -> {overall}")


if __name__ == "__main__":
    main()
