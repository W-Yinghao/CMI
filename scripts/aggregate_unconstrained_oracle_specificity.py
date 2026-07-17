#!/usr/bin/env python
"""C0-v2 (stage evidence, NOT a closure): does the conditional-subject span B_cond carry more UTILITY-relevant
deletion directions than an equal-budget RANDOM basis? This measures deletion UTILITY enrichment (dU), NOT a
mutual-information difference and NOT a closure of subspace research. Fair control: BOTH the informed oracle and
each random-basis oracle select their deletion GREEDILY on T_cal labels (identical rank + candidate budget) and
score ONLY on T_query (session-macro) -- the random control never picks the best projector on the query set.
Every random draw is saved. Random dictionaries are seeded per (dataset, subject, model-seed, random-id).

  dU_specific = dU_informed_oracle - E_R[dU_random_oracle_R]   (also vs Q95 of the random oracle)

Graded stage verdict (per dataset; PM: no CLOSED wording):
  B_COND_ENRICHED_OVER_RANDOM     LCB95(dU_specific) > 0
  B_COND_NO_MEAN_ENRICHMENT       UCB95(dU_specific) <= 0
  B_COND_ENRICHMENT_INCONCLUSIVE  otherwise
Secondary: q95_exceedance_rate (subject-cluster) compared to the null rate 0.05 (a Q95 threshold). This does
NOT decide whether subspace research stops -- only the PM declares a scientific line stopped.

  python scripts/aggregate_unconstrained_oracle_specificity.py --seeds 0 1 2 --n_random 100
"""
from __future__ import annotations
import argparse, csv, glob, hashlib, json, subprocess, sys
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
Q95_NULL_RATE = 0.05


def _query_macro_gain(Zs_w, ys, Xq_w, yq, sq, U, seed):
    def gain(Zq, y):
        base = _bacc(Zs_w, ys, Zq, y, seed)
        got = _bacc(Zs_w - (Zs_w @ U.T) @ U if U.shape[0] else Zs_w, ys,
                    Zq - (Zq @ U.T) @ U if U.shape[0] else Zq, y, seed)
        return got - base
    per = [gain(Xq_w[sq == s], yq[sq == s]) for s in np.unique(sq)
           if (sq == s).sum() >= 4 and len(np.unique(yq[sq == s])) >= 2]
    return float(np.mean(per)) if per else float(gain(Xq_w, yq))


def _oracle(Zs_w, ys, B_dict, Xcal_w, ycal, Xq_w, yq, sq, max_k, seed):
    S = _select_subset(Zs_w, ys, Xcal_w, ycal, B_dict, "greedy", max_k, seed)   # select on CAL only
    U = M._orthonormal(B_dict[S]) if S else np.zeros((0, B_dict.shape[1]))
    cal_gain = _query_macro_gain(Zs_w, ys, Xcal_w, ycal, np.array(["cal"] * len(ycal)), U, seed)
    return _query_macro_gain(Zs_w, ys, Xq_w, yq, sq, U, seed), list(S), float(cal_gain)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", nargs="+", default=["0", "1", "2"]); ap.add_argument("--backbone", default="EEGNet")
    ap.add_argument("--n_random", type=int, default=100); ap.add_argument("--max_rank", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    cfg = REPO / "configs" / "cmi_trace_targetx_observability.yaml"
    cfg_hash = hashlib.sha256(cfg.read_bytes()).hexdigest()[:16] if cfg.exists() else "no_config"
    try:
        sha = subprocess.check_output(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"]).decode().strip()
    except Exception:
        sha = "unknown"
    subj_rows, random_rows, completeness = [], [], []
    for ds in DATASETS:
        cells = [p for p in sorted(glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                 f"{ds}_{a.backbone}_LOSO" / "sub*_erm_lam0_seed*.npz")))
                 if any(p.endswith(f"_seed{s}.npz") for s in a.seeds)]
        if a.limit:
            cells = cells[: a.limit]
        for cp in cells:
            f = feat_from_tos_dump(cp); sd = int(f["seed"]); subj = str(f["heldout_subject"])
            Zs = np.asarray(f["Z_source"], float); ys = np.asarray(f["y_source"]).astype(int); dsc = _dense(f["subj_source"])
            Zt = np.asarray(f["Z_target"], float); yt = np.asarray(f["y_target"]).astype(int); st = f["session_target"]
            W = M.source_whitener(Zs); Zs_w = M.to_whitened(Zs, W)
            cal, qry, _ = session_split(st, yt, sd)
            Xcal_w, ycal = M.to_whitened(Zt[cal], W), yt[cal]
            Xq_w, yq, sq = M.to_whitened(Zt[qry], W), yt[qry], np.asarray(st)[qry]
            B_cond_w = M.whitened_cond_basis(Zs_w, ys, dsc, max_rank=a.max_rank); r = B_cond_w.shape[0]
            ok = r > 0
            completeness.append(dict(dataset=ds, subject=subj, seed=sd, dict_rank=int(r),
                                     status=("ok" if ok else "excluded"), reason=("" if ok else "empty_cond_basis")))
            if not ok:
                continue
            g_inf, S_inf, cal_inf = _oracle(Zs_w, ys, B_cond_w, Xcal_w, ycal, Xq_w, yq, sq, a.max_rank, sd)
            # per-fold-unique random seed base (dataset x subject x model-seed)
            base = int(hashlib.sha1(f"{ds}|{subj}|{sd}".encode()).hexdigest()[:8], 16) % (2 ** 31)
            g_rand = []
            for rid in range(a.n_random):
                Q = M.ambient_random_projectors_whitened(Zs.shape[1], r, 1, (base + rid) % (2 ** 31))[0]
                gq, S_r, cal_r = _oracle(Zs_w, ys, Q, Xcal_w, ycal, Xq_w, yq, sq, a.max_rank, sd)
                g_rand.append(gq)
                random_rows.append(dict(dataset=ds, subject=subj, seed=sd, random_dictionary_id=rid,
                                        dictionary_hash=M._hash(Q), dictionary_rank=int(r), selected_subset=S_r,
                                        selected_rank=len(S_r), calibration_gain=cal_r, query_gain=float(gq),
                                        config_hash=cfg_hash, git_sha=sha))
            g_rand = np.array(g_rand); q95 = float(np.quantile(g_rand, 0.95))
            subj_rows.append(dict(dataset=ds, subject=subj, seed=sd, dict_rank=int(r),
                                  informed_query_gain=float(g_inf), informed_selected_rank=len(S_inf),
                                  informed_calibration_gain=cal_inf, random_gain_mean=float(g_rand.mean()),
                                  random_gain_q95=q95, dU_specific=float(g_inf - g_rand.mean()),
                                  dU_vs_random_q95=float(g_inf - q95), informed_beats_q95=bool(g_inf > q95),
                                  config_hash=cfg_hash, git_sha=sha))
            print(f"  {ds} sub{subj} s{sd}: informed={g_inf:+.4f} rand_mean={g_rand.mean():+.4f} q95={q95:+.4f} "
                  f"dU_spec={g_inf-g_rand.mean():+.4f}", flush=True)

    def _ci(vals, seed=7):
        v = np.asarray([x for x in vals if np.isfinite(x)], float)
        if not v.size:
            return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
        rng = np.random.default_rng(seed); b = [v[rng.integers(0, v.size, v.size)].mean() for _ in range(10000)]
        return dict(mean=float(v.mean()), lo=float(np.percentile(b, 2.5)), hi=float(np.percentile(b, 97.5)), n=int(v.size))
    summ, verdict = [], {}
    for ds in DATASETS:
        Rs = [x for x in subj_rows if x["dataset"] == ds]
        if not Rs:
            continue
        by, win = defaultdict(list), defaultdict(list)
        for x in Rs:
            by[x["subject"]].append(x["dU_specific"]); win[x["subject"]].append(1.0 if x["informed_beats_q95"] else 0.0)
        spec = _ci([np.mean(v) for v in by.values()]); wr = _ci([np.mean(v) for v in win.values()])
        stage = ("B_COND_ENRICHED_OVER_RANDOM" if spec["lo"] > 0 else
                 "B_COND_NO_MEAN_ENRICHMENT" if spec["hi"] <= 0 else "B_COND_ENRICHMENT_INCONCLUSIVE")
        summ.append(dict(dataset=ds, dU_specific_mean=spec["mean"], dU_specific_lo=spec["lo"], dU_specific_hi=spec["hi"],
                         q95_exceedance_rate=wr["mean"], q95_exceedance_rate_lo=wr["lo"], q95_exceedance_rate_hi=wr["hi"],
                         q95_null_rate=Q95_NULL_RATE, n_subjects=spec["n"], stage_verdict=stage))
        verdict[ds] = dict(stage_verdict=stage, dU_specific_lcb=spec["lo"], dU_specific_ucb=spec["hi"],
                           q95_exceedance_rate=wr["mean"], q95_null_rate=Q95_NULL_RATE)
    n_expected = len([1 for ds in DATASETS for _ in glob.glob(str(REPO / "tos_cmi/results/tos_cmi_eeg_frozen" /
                     f"{ds}_{a.backbone}_LOSO" / "sub*_erm_lam0_seed*.npz")) if any(_.endswith(f"_seed{s}.npz") for s in a.seeds)])
    complete = (sum(1 for c in completeness if c["status"] == "ok") == n_expected)

    def _w(fp, rows, keys):
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
    _w(OUT / "unconstrained_oracle_subject_rows.csv", subj_rows, list(subj_rows[0].keys()))
    _w(OUT / "unconstrained_oracle_specificity_full.csv", summ, list(summ[0].keys()))
    _w(OUT / "unconstrained_oracle_completeness.csv", completeness, ["dataset", "subject", "seed", "dict_rank", "status", "reason"])
    with open(OUT / "unconstrained_oracle_random_rows.jsonl", "w") as fh:
        [fh.write(json.dumps(r) + "\n") for r in random_rows]
    json.dump({"per_dataset": verdict, "n_random": a.n_random, "completeness_ok": bool(complete),
               "control": "greedy_select_on_Tcal_score_on_Tquery_equal_budget",
               "note": "STAGE EVIDENCE for B_cond utility enrichment vs equal-budget random; NOT a closure of subspace research (PM declares stops)"},
              open(OUT / "unconstrained_oracle_stage_verdict.json", "w"), indent=2, default=float)
    print(f"\n[C0-v2] {len(subj_rows)} fold-seed cells; completeness_ok={complete}; {len(random_rows)} random rows")
    for s in summ:
        print(f"  {s['dataset']}: dU_specific={s['dU_specific_mean']:+.4f} [{s['dU_specific_lo']:+.4f},{s['dU_specific_hi']:+.4f}] "
              f"q95_exceed={s['q95_exceedance_rate']:.3f}[{s['q95_exceedance_rate_lo']:.3f},{s['q95_exceedance_rate_hi']:.3f}] (null 0.05) -> {s['stage_verdict']}")


if __name__ == "__main__":
    main()
