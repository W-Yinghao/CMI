"""WAVE 0 / W0.2 analyzer -- fixed-reservoir prevalence UTILITY curves.

Per operator, over the 9-point q-grid, cluster-bootstrap (by (dataset,subject)) of:
  displacement from q=0.5: embedding / translation / log-scale;
  utility (pi_dec=Unif): BA, macro-F1, NLL, ECE, and negative-change rate vs q=0.5;
  matched-prevalence ordinary accuracy under pi_dec in {Unif, pi_J, oracle q}.
Seeds averaged WITHIN unit before bootstrap. Interpretation lives in the results doc, not here.
"""
from __future__ import annotations

import glob
import json
from collections import defaultdict

import numpy as np

NB = 10000
EST = ["pooled", "fixed_reference_oneshot", "fixed_iterative", "joint", "oracle_label_conditional"]


def _load():
    rows = []
    for f in glob.glob("results/h2cmi/wave0_v2p/*.jsonl"):
        for l in open(f):
            if l.strip():
                r = json.loads(l)
                if not r.get("provenance_fail"):
                    rows.append(r)
    return rows


def _cluster_boot(cluster_vals, seed=0):
    keys = [k for k in cluster_vals if len(cluster_vals[k])]
    if len(keys) < 2:
        return dict(mean=float("nan"), ci=[float("nan")] * 2, n=0)
    arrs = {k: np.asarray(cluster_vals[k], float) for k in keys}
    allv = np.concatenate([arrs[k] for k in keys])
    rng = np.random.default_rng(seed)
    bs = [np.concatenate([arrs[k] for k in (keys[i] for i in rng.integers(0, len(keys), len(keys)))]).mean()
          for _ in range(NB)]
    return dict(mean=float(allv.mean()), ci=[float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                excludes_0=bool(np.percentile(bs, 2.5) > 0 or np.percentile(bs, 97.5) < 0), n=int(len(allv)))


def _unit_key(r):
    return (r["dataset"], int(r["subject"]))


def analyze():
    rows = _load()
    qs = sorted({round(r["q"], 2) for r in rows if r.get("q") is not None and r.get("ratio") != "__disp__"} |
                {round(r["q"], 2) for r in rows if r.get("ratio") == "__disp__"})
    # per (est, q, metric): unit -> [per-seed values] -> mean -> cluster
    point = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))   # est->q->metric-> {unit: [seed vals]}
    disp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    pt_metrics = ["bacc", "macro_f1", "nll", "ece", "ord_acc_matched_unif", "ord_acc_matched_piJ",
                  "ord_acc_matched_oracleq"]
    dp_metrics = ["embed_disp", "translation_disp", "logscale_disp"]
    seen_units = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(list))))
    for r in rows:
        est = r.get("estimator"); q = round(r["q"], 2) if r.get("q") is not None else None
        if est is None or q is None:
            continue
        u = _unit_key(r); s = r["seed"]
        if r.get("ratio") == "__disp__":
            for m in dp_metrics:
                if m in r:
                    seen_units["disp"][est][(q, m)][(u, s)].append(r[m])
        elif "bacc" in r:
            for m in pt_metrics:
                if m in r:
                    seen_units["point"][est][(q, m)][(u, s)].append(r[m])
    def collapse(kind):
        out = {}
        for est in seen_units[kind]:
            out[est] = {}
            for (q, m), byus in seen_units[kind][est].items():
                # average seeds within unit
                per_unit = defaultdict(list)
                for (u, s), vals in byus.items():
                    per_unit[u].append(float(np.mean(vals)))
                cl = {u: [float(np.mean(v))] for u, v in per_unit.items()}
                out[est].setdefault(str(q), {})[m] = _cluster_boot(cl)
        return out
    pt = collapse("point"); dp = collapse("disp")
    # negative-change rate vs q=0.5 (per est, per metric where higher=better: bacc, macro_f1, ord_*)
    negchange = defaultdict(dict)
    for est in seen_units["point"]:
        # build per-unit metric(q) averaged over seeds
        for m in ["bacc", "macro_f1", "ord_acc_matched_unif"]:
            base = seen_units["point"][est].get((0.5, m), {})
            per_unit_base = defaultdict(list)
            for (u, s), vals in base.items():
                per_unit_base[u].append(float(np.mean(vals)))
            base_u = {u: float(np.mean(v)) for u, v in per_unit_base.items()}
            rates = {}
            for q in qs:
                if q == 0.5:
                    continue
                cur = seen_units["point"][est].get((q, m), {})
                pu = defaultdict(list)
                for (u, s), vals in cur.items():
                    pu[u].append(float(np.mean(vals)))
                deltas = [float(np.mean(v)) - base_u[u] for u, v in pu.items() if u in base_u]
                if deltas:
                    d = np.asarray(deltas)
                    rates[str(q)] = dict(neg_lt_0=float((d < 0).mean()), mean_delta=float(d.mean()), n=len(d))
            negchange[est][m] = rates
    return dict(marker="WAVE0_V2P_UTILITY", q_grid=qs,
                n_units=len({_unit_key(r) for r in rows if r.get("subject") is not None}),
                displacement=dp, utility=pt, negative_change_vs_q05=negchange)


def main():
    import argparse
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="results/h2cmi/wave0_v2p.report.json")
    args = ap.parse_args()
    rep = analyze()
    json.dump(rep, open(args.out, "w"), indent=2, default=str)
    print(f"[W0.2] units={rep['n_units']} q_grid={rep['q_grid']}")
    for est in ["pooled", "fixed_reference_oneshot", "oracle_label_conditional"]:
        d = rep["displacement"].get(est, {})
        e05 = d.get("0.9", {}).get("embed_disp", {}).get("mean")
        u = rep["utility"].get(est, {}).get("0.9", {}).get("bacc", {}).get("mean")
        print(f"  {est:28s} embed_disp@q0.9={e05}  BA@q0.9={u}")
    print(f"  -> {args.out}")


if __name__ == "__main__":
    main()
