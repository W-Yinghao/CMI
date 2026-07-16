#!/usr/bin/env python
"""Aggregate the target-X observability audit (amendment 03 C4) -> 7 artifacts + deterministic 5-gate verdict.

Inference unit = target subject / outer fold; the 3 seeds of a subject are pooled (mean) before the subject-
cluster 10,000 bootstrap. BNCI2015 primary = query-session MACRO (already in utility_macro). All control
comparisons are PAIRED (Δ_TX − Δ_control per fold, then cluster CI), never two independent CIs. G1 observability
is rank-stratified. Gate 5 (di_specific) is read from the rule-level cross-fit artifact if present, else null.

  python scripts/aggregate_targetx_observability.py --tag smoke
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from collections import defaultdict
from pathlib import Path
import numpy as np
REPO = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(REPO))
from tos_cmi.eval.targetx_gates import (gate1_observability, gate2_actionability, gate3_oracle_recovery,
                                        gate4_cross_dataset_safety, gate5_specificity, five_gate_verdict)

OUT = REPO / "results" / "cmi_trace_dg_identifiability"
N_BOOT = 10000


def _spearman(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return np.nan
    rx = np.argsort(np.argsort(x)); ry = np.argsort(np.argsort(y))
    return float(np.corrcoef(rx, ry)[0, 1])


def _cluster_ci(values, stat=np.mean, n_boot=N_BOOT, seed=12345):
    v = np.asarray([x for x in values if x is not None and np.isfinite(x)], float)
    if v.size == 0:
        return dict(mean=float("nan"), lo=float("nan"), hi=float("nan"), n=0)
    rng = np.random.default_rng(seed)
    boots = [stat(v[rng.integers(0, v.size, v.size)]) for _ in range(n_boot)]
    return dict(mean=float(stat(v)), lo=float(np.percentile(boots, 2.5)), hi=float(np.percentile(boots, 97.5)), n=int(v.size))


def _by_subject(folds, key):
    by = defaultdict(list)
    for f in folds:
        if f.get(key) is not None and np.isfinite(f[key]):
            by[f["heldout_subject"]].append(f[key])
    return {s: float(np.mean(v)) for s, v in by.items()}


def _paired_by_subject(folds, a, b):
    by = defaultdict(list)
    for f in folds:
        if f.get(a) is not None and f.get(b) is not None and np.isfinite(f[a]) and np.isfinite(f[b]):
            by[f["heldout_subject"]].append(f[a] - f[b])
    return {s: float(np.mean(v)) for s, v in by.items()}


def _rank_stratified_rho(action_rows_by_subject):
    """Per subject: within each informed rank>=1, Spearman(G1, utility_macro); average across ranks."""
    out = {}
    for subj, rows in action_rows_by_subject.items():
        by_rank = defaultdict(list)
        for r in rows:
            if r["kind"] == "informed" and r["rank"] >= 1 and r.get("G1") is not None:
                by_rank[r["rank"]].append((r["G1"], r["utility_macro"]))
        rhos = [_spearman([p[0] for p in v], [p[1] for p in v]) for v in by_rank.values() if len(v) >= 3]
        rhos = [x for x in rhos if np.isfinite(x)]
        if rhos:
            out[subj] = float(np.mean(rhos))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--tag", default="smoke")
    a = ap.parse_args()
    folds = [json.loads(l) for l in open(OUT / f"targetx_fold_summary_{a.tag}.jsonl")] \
        if (OUT / f"targetx_fold_summary_{a.tag}.jsonl").exists() else []
    arows = [json.loads(l) for l in open(OUT / f"targetx_action_rows_{a.tag}.jsonl")] \
        if (OUT / f"targetx_action_rows_{a.tag}.jsonl").exists() else []
    if not folds:
        sys.exit(f"[targetx-agg] no fold summary for tag={a.tag}")
    g5 = {}
    g5fp = OUT / f"targetx_gate5_rule_{a.tag}.json"
    if g5fp.exists():
        g5 = json.load(open(g5fp))                            # {dataset: di_specific_lcb}
    datasets = sorted({f["dataset"] for f in folds})
    cluster_rows, byrank_rows, negctrl_rows, verdicts = [], [], [], {}
    per_ds_dtx_lcb = {}
    for ds in datasets:
        F = [f for f in folds if f["dataset"] == ds]
        A = [r for r in arows if r["dataset"] == ds] if arows and "dataset" in arows[0] else \
            [r for r in arows if r.get("dataset", ds) == ds]
        dtx = _cluster_ci(list(_by_subject(F, "delta_tx").values()))
        rec = _cluster_ci(list(_by_subject(F, "oracle_recovery_ratio").values()))
        paired = {c: _cluster_ci(list(_paired_by_subject(F, "delta_tx", c).values()))
                  for c in ("delta_random_same_rank", "delta_source_greedy", "delta_whitening", "delta_mean_centering")}
        # observability (rank-stratified) — needs action rows grouped by subject
        arows_by_subj = defaultdict(list)
        for r in A:
            arows_by_subj[r["subject"]].append(r)
        rho_by_subj = _rank_stratified_rho(arows_by_subj)
        rho = _cluster_ci(list(rho_by_subj.values()), stat=np.median)
        per_ds_dtx_lcb[ds] = dtx["lo"]
        # gates
        g1 = gate1_observability(rho["lo"])
        g2 = gate2_actionability(dtx["lo"], paired["delta_random_same_rank"]["lo"], paired["delta_source_greedy"]["lo"],
                                 paired["delta_whitening"]["lo"], paired["delta_mean_centering"]["lo"])
        g3 = gate3_oracle_recovery(rec["lo"])
        g5b = gate5_specificity(g5.get(ds))
        cluster_rows.append(dict(dataset=ds, delta_tx_mean=dtx["mean"], delta_tx_lo=dtx["lo"], delta_tx_hi=dtx["hi"],
                                 recovery_mean=rec["mean"], recovery_lo=rec["lo"], rho_median=rho["mean"], rho_lo=rho["lo"],
                                 **{f"dtx_minus_{c}_lo": paired[c]["lo"] for c in paired}, n_folds=dtx["n"]))
        for c in paired:
            negctrl_rows.append(dict(dataset=ds, control=c, dtx_minus_control_mean=paired[c]["mean"],
                                     lo=paired[c]["lo"], hi=paired[c]["hi"]))
        # per-rank observability
        for k in sorted({r["rank"] for r in A if r["kind"] == "informed" and r["rank"] >= 1}):
            pairs = [(r["G1"], r["utility_macro"]) for r in A if r["kind"] == "informed" and r["rank"] == k and r.get("G1") is not None]
            byrank_rows.append(dict(dataset=ds, rank=k, n=len(pairs),
                                    spearman=_spearman([p[0] for p in pairs], [p[1] for p in pairs]) if len(pairs) >= 3 else None))
        verdicts[ds] = dict(gate1_observability=g1, gate2_actionability=g2, gate3_oracle_recovery=g3,
                            gate5_specificity=g5b, di_specific_lcb=g5.get(ds))
    g4 = gate4_cross_dataset_safety(per_ds_dtx_lcb)
    overall = {ds: five_gate_verdict(verdicts[ds]["gate1_observability"], verdicts[ds]["gate2_actionability"],
                                     verdicts[ds]["gate3_oracle_recovery"], g4, verdicts[ds]["gate5_specificity"])
               for ds in datasets}
    # ---- write 7 artifacts (action_rows + fold_summary already exist from the runner) ----
    def _wcsv(fp, rows, keys):
        with open(fp, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); [w.writerow({k: r.get(k) for k in keys}) for r in rows]
    _wcsv(OUT / f"targetx_cluster_intervals_{a.tag}.csv", cluster_rows, list(cluster_rows[0].keys()))
    _wcsv(OUT / f"targetx_observability_by_rank_{a.tag}.csv", byrank_rows, ["dataset", "rank", "n", "spearman"])
    _wcsv(OUT / f"targetx_negative_controls_{a.tag}.csv", negctrl_rows, ["dataset", "control", "dtx_minus_control_mean", "lo", "hi"])
    json.dump({"per_dataset": verdicts, "cross_dataset_safety": g4, "verdict": overall},
              open(OUT / f"targetx_gate_verdict_{a.tag}.json", "w"), indent=2, default=float)
    print(f"[targetx-agg] tag={a.tag}: {len(folds)} folds, {len(arows)} action rows")
    for ds in datasets:
        cr = next(c for c in cluster_rows if c["dataset"] == ds); v = verdicts[ds]; ov = overall[ds]
        print(f"  {ds}: Δtx={cr['delta_tx_mean']:+.4f}[{cr['delta_tx_lo']:+.4f}] recovery={cr['recovery_mean']:.3f}[{cr['recovery_lo']:.3f}] "
              f"rho={cr['rho_median']:+.3f}[{cr['rho_lo']:+.3f}]")
        print(f"     gates obs={v['gate1_observability']} act={v['gate2_actionability']} rec={v['gate3_oracle_recovery']} "
              f"safe={g4} spec={v['gate5_specificity']} -> {ov['outcome']}")


if __name__ == "__main__":
    main()
