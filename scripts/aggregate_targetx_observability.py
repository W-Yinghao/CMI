#!/usr/bin/env python
"""Aggregate the target-X observability audit (amendments 03/04) -> 7 artifacts + deterministic 5-gate verdict.

Inference unit = target subject; the 3 seeds of a subject are pooled AFTER a per-(subject,seed,rank) Spearman;
subject-cluster 10,000 bootstrap. BNCI2015 primary = query-session macro (utility_macro). Control comparisons
are PAIRED (Δ_TX - Δ_control per fold, then cluster CI). Observability uses TIE-AWARE ranks (rankdata average)
grouped per (subject,seed,rank) then rank-macro then seed-average. Gate 3 = constrained-hindsight paired
inequality (no ratio bootstrap). Gate 4 uses UCB for harm. Gate 5 read from the rule-level cross-fit artifact.

  python scripts/aggregate_targetx_observability.py --tag smoke
"""
from __future__ import annotations
import argparse, csv, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eval.targetx_gates import (gate1_observability, gate2_actionability, gate3_oracle_recovery,
                                        gate4_cross_dataset_safety, gate5_specificity, five_gate_verdict)

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
N_BOOT = 10000


def _spearman(x, y):
    from scipy.stats import rankdata
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.ptp(x) < 1e-12 or np.ptp(y) < 1e-12:
        return np.nan
    rx, ry = rankdata(x), rankdata(y)                             # tie-aware (average) ranks
    return float(np.corrcoef(rx, ry)[0, 1])


def _cluster_ci(values, stat=np.mean, seed=12345):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = [stat(v[rng.integers(0, v.size, v.size)]) for _ in range(N_BOOT)]
    return dict(mean=float(stat(v)), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n=int(v.size))


def _subject_means(folds, key):
    by = defaultdict(list)
    for f in folds:
        if f.get(key) is not None and np.isfinite(f[key]):
            by[f["heldout_subject"]].append(f[key])
    return {s: float(np.mean(v)) for s, v in by.items()}


def _paired_subject_means(folds, a, coef_b=1.0, b=None):
    """Per subject: mean over its folds of (a - coef_b*b)."""
    by = defaultdict(list)
    for f in folds:
        va, vb = f.get(a), (f.get(b) if b else 0.0)
        if va is not None and np.isfinite(va) and (b is None or (vb is not None and np.isfinite(vb))):
            by[f["heldout_subject"]].append(va - coef_b * (vb if b else 0.0))
    return {s: float(np.mean(v)) for s, v in by.items()}


def _observability_subject_rho(action_rows):
    """Per (subject,seed,rank): tie-aware Spearman(G1, utility_macro); rank-macro; seed-average -> per subject."""
    grp = defaultdict(list)
    for r in action_rows:
        if r["kind"] == "informed" and r["rank"] >= 1 and r.get("G1") is not None:
            grp[(r["subject"], r["seed"], r["rank"])].append((r["G1"], r["utility_macro"]))
    rho_ser = defaultdict(lambda: defaultdict(list))              # subject -> seed -> [rank rhos]
    for (subj, seed, rank), pairs in grp.items():
        if len(pairs) >= 3:
            rho = _spearman([p[0] for p in pairs], [p[1] for p in pairs])
            if np.isfinite(rho):
                rho_ser[subj][seed].append(rho)
    out = {}
    for subj, seeds in rho_ser.items():
        seed_macros = [np.mean(rr) for rr in seeds.values() if rr]
        if seed_macros:
            out[subj] = float(np.mean(seed_macros))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    fp_fold = OUT / f"targetx_fold_summary_{a.tag}.jsonl"
    fp_act = OUT / f"targetx_action_rows_{a.tag}.jsonl"
    if not fp_fold.exists():
        sys.exit(f"[targetx-agg] no fold summary for tag={a.tag}")
    folds = [json.loads(l) for l in open(fp_fold)]
    arows = [json.loads(l) for l in open(fp_act)] if fp_act.exists() else []
    g5fp = OUT / f"targetx_gate5_rule_{a.tag}.json"
    g5 = json.load(open(g5fp)) if g5fp.exists() else {}
    datasets = sorted({f["dataset"] for f in folds})
    cluster_rows, byrank_rows, negctrl_rows, per_ds = [], [], [], {}
    for ds in datasets:
        F = [f for f in folds if f["dataset"] == ds]
        A = [r for r in arows if r.get("dataset") == ds]
        dtx = _cluster_ci(list(_subject_means(F, "delta_tx").values()))
        hc = _cluster_ci(list(_subject_means(F, "delta_hindsight_constrained").values()))
        g3paired = _cluster_ci(list(_paired_subject_means(F, "delta_tx", 0.25, "delta_hindsight_constrained").values()))
        paired = {c: _cluster_ci(list(_paired_subject_means(F, "delta_tx", 1.0, c).values()))
                  for c in ("delta_random_selected_rank", "delta_source_greedy", "delta_whitening", "delta_mean_centering")}
        rho = _cluster_ci(list(_observability_subject_rho(A).values()), stat=np.median)
        per_ds[ds] = {"lo": dtx["lo"], "hi": dtx["hi"]}
        # Gate 1 applicability (amendment 05 P0.3): binary-task datasets have contested rank<=1 -> not rankable
        g1_applicable = bool(np.mean([f.get("gate1_applicable", False) for f in F]) >= 0.5)
        g1 = gate1_observability(rho["lo"]) if g1_applicable else None
        g2 = gate2_actionability(dtx["lo"], paired["delta_random_selected_rank"]["lo"], paired["delta_source_greedy"]["lo"],
                                 paired["delta_whitening"]["lo"], paired["delta_mean_centering"]["lo"])
        g3 = gate3_oracle_recovery(hc["lo"], g3paired["lo"])
        g5b = gate5_specificity(g5.get(ds))
        cluster_rows.append(dict(dataset=ds, gate1_applicable=g1_applicable,
                                 delta_tx_mean=dtx["mean"], delta_tx_lo=dtx["lo"], delta_tx_hi=dtx["hi"],
                                 hind_constrained_mean=hc["mean"], hind_constrained_lo=hc["lo"],
                                 g3_paired_lo=g3paired["lo"], rho_median=rho["mean"], rho_lo=rho["lo"],
                                 **{f"dtx_minus_{c}_lo": paired[c]["lo"] for c in paired},
                                 gate1=g1, gate2=g2, gate3=g3, gate5=g5b, di_specific_lcb=g5.get(ds), n_folds=dtx["n"]))
        for c in paired:
            negctrl_rows.append(dict(dataset=ds, control=c, dtx_minus_control_mean=paired[c]["mean"],
                                     lo=paired[c]["lo"], hi=paired[c]["hi"]))
        for k in sorted({r["rank"] for r in A if r["kind"] == "informed" and r["rank"] >= 1}):
            rho_k = _cluster_ci(list({s: np.mean([_spearman(
                [p["G1"] for p in A if p["subject"] == s and p["seed"] == e and p["rank"] == k and p["kind"] == "informed" and p.get("G1") is not None],
                [p["utility_macro"] for p in A if p["subject"] == s and p["seed"] == e and p["rank"] == k and p["kind"] == "informed" and p.get("G1") is not None])
                for e in {p["seed"] for p in A if p["subject"] == s}])
                for s in {p["subject"] for p in A}}.values()))
            byrank_rows.append(dict(dataset=ds, rank=k, rho_median=rho_k["mean"], rho_lo=rho_k["lo"], n_subjects=rho_k["n"]))
    g4 = gate4_cross_dataset_safety(per_ds)
    # CROSS-DATASET overall verdict (amendment 05 P0.3): Gate 1 on >=1 rankable dataset; Gate 2/3/5 on >=1
    # rankable dataset; Gate 4 = no clear harm on the other. Per-dataset rows kept for diagnostics.
    rankable = [c for c in cluster_rows if c["gate1_applicable"]]
    g1_overall = any(c["gate1"] for c in rankable) if rankable else False
    g2_overall = any(c["gate2"] for c in rankable) if rankable else any(c["gate2"] for c in cluster_rows)
    g3_overall = any(c["gate3"] for c in rankable) if rankable else any(c["gate3"] for c in cluster_rows)
    g5_overall = any(c["gate5"] for c in rankable) if rankable else any(c["gate5"] for c in cluster_rows)
    overall = {"OVERALL": five_gate_verdict(g1_overall, g2_overall, g3_overall, g4, g5_overall),
               **{c["dataset"]: {"gate1": c["gate1"], "gate1_applicable": c["gate1_applicable"], "gate2": c["gate2"],
                                 "gate3": c["gate3"], "gate5": c["gate5"]} for c in cluster_rows}}

    def _wcsv(fp, rows, keys):
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
    _wcsv(OUT / f"targetx_cluster_intervals_{a.tag}.csv", cluster_rows, list(cluster_rows[0].keys()))
    _wcsv(OUT / f"targetx_observability_by_rank_{a.tag}.csv", byrank_rows, ["dataset", "rank", "rho_median", "rho_lo", "n_subjects"])
    _wcsv(OUT / f"targetx_negative_controls_{a.tag}.csv", negctrl_rows, ["dataset", "control", "dtx_minus_control_mean", "lo", "hi"])
    json.dump({"per_dataset_gates": {c["dataset"]: {k: c[k] for k in ("gate1", "gate1_applicable", "gate2", "gate3", "gate5", "di_specific_lcb")} for c in cluster_rows},
               "cross_dataset_safety_gate4": g4, "verdict": overall}, open(OUT / f"targetx_gate_verdict_{a.tag}.json", "w"), indent=2, default=float)
    print(f"[targetx-agg] tag={a.tag}: {len(folds)} folds, {len(arows)} action rows")
    for ds in datasets:
        c = next(cr for cr in cluster_rows if cr["dataset"] == ds)
        print(f"  {ds}: Δtx={c['delta_tx_mean']:+.4f}[{c['delta_tx_lo']:+.4f},{c['delta_tx_hi']:+.4f}] "
              f"hind_c={c['hind_constrained_mean']:+.4f}[{c['hind_constrained_lo']:+.4f}] rho={c['rho_median']:+.3f}[{c['rho_lo']:+.3f}] "
              f"g1_applicable={c['gate1_applicable']}")
        print(f"     gates obs={c['gate1']} act={c['gate2']} rec={c['gate3']} safe={g4} spec={c['gate5']}")
    print(f"  OVERALL(cross-dataset) -> {overall['OVERALL']['outcome']}  gates={overall['OVERALL']['gates']}")


if __name__ == "__main__":
    main()
