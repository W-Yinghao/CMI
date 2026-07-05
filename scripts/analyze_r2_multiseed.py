#!/usr/bin/env python
"""Multi-seed (0/1/2) analysis of the R2 gate: Pareto mean +/- seed variability + dominance, and R3 reliance
(ERM vs CIGL) across seeds x folds with HONEST head-replay availability under both the absolute 1e-5 tol and a
relative tol. CPU, no retrain, no target-label fit.

    python scripts/analyze_r2_multiseed.py --gate_dir results/cigl/r2_seed0_gate --dataset BNCI2014_001
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
SEEDS = [0, 1, 2]
METHODS = ["erm", "cigl_graph_node", "dann", "cond_dann", "cdan"]


def _load_rows(gate_dir, dataset, seeds):
    rows = []
    for s in seeds:
        for jp in sorted(glob.glob(str(Path(gate_dir) / dataset / f"{dataset}_fold*_*_seed{s}.json"))):
            rec = json.load(open(jp)); pr = dict(rec["pareto_row"])
            pr["method_label"] = rec.get("gate_label", rec.get("config")); pr["seed"] = s
            rows.append(pr)
    return rows


def _ms(vals):
    vals = [v for v in vals if v is not None and v == v]
    return (float(np.mean(vals)), float(np.std(vals))) if vals else (float("nan"), 0.0)


def pareto_multiseed(rows, methods):
    out = {}
    for m in methods:
        per_seed = {}
        for s in SEEDS:
            r = [x for x in rows if x["method_label"] == m and x["seed"] == s]
            if r:
                per_seed[s] = dict(target=float(np.mean([x["target_bacc"] for x in r])),
                                   source=float(np.mean([x["source_bacc"] for x in r])),
                                   graph=float(np.mean([x["graph_kl_proxy"] for x in r])),
                                   node=float(np.mean([x["node_kl_proxy"] for x in r])))
        if not per_seed:
            continue
        tgt = _ms([v["target"] for v in per_seed.values()]); g = _ms([v["graph"] for v in per_seed.values()])
        n = _ms([v["node"] for v in per_seed.values()]); src = _ms([v["source"] for v in per_seed.values()])
        out[m] = dict(target_mean=tgt[0], target_std=tgt[1], source_mean=src[0], graph_mean=g[0], graph_std=g[1],
                      node_mean=n[0], node_std=n[1], n_seeds=len(per_seed), per_seed=per_seed)
    # dominance on across-seed means
    def dom(a, b, e=1e-9):
        ge = b["target_mean"] >= a["target_mean"] - e
        le = b["graph_mean"] <= a["graph_mean"] + e and b["node_mean"] <= a["node_mean"] + e
        strict = (b["target_mean"] > a["target_mean"] + e or b["graph_mean"] < a["graph_mean"] - e
                  or b["node_mean"] < a["node_mean"] - e)
        return ge and le and strict
    labels = list(out)
    for m in labels:
        out[m]["dominated_by"] = [o for o in labels if o != m and dom(out[m], out[o])]
        out[m]["on_frontier"] = not out[m]["dominated_by"]
    return out


def _relative_replay_ok(d, tol_rel=1e-4):
    """Recompute head-replay validity with a RELATIVE tol: max|logits-replay| / (max|logits|+1) <= tol_rel.
    The head is linear; large-activation folds fail the absolute 1e-5 tol purely via float32 accumulation."""
    if "task_head_weight" not in d:
        return False
    W = np.asarray(d["task_head_weight"], float); b = np.asarray(d.get("task_head_bias", 0.0), float)
    gz = np.asarray(d["graph_z"], float); lo = np.asarray(d["model_logits"], float)
    diff = np.abs(lo - (gz @ W.T + b)).max()
    return bool(diff / (np.abs(lo).max() + 1.0) <= tol_rel)


def reliance_multiseed(gate_dir, dataset, seeds, methods=("erm", "cigl_graph_node"), ks=(1, 2, 4, 8)):
    from cmi.eval.leakage_removal import evaluate_reliance
    from cmi.eval.audit_npz import load_audit_npz
    adir = Path(gate_dir) / dataset / "audit"
    rows = {m: [] for m in methods}
    avail = {m: dict(abs_ok=0, rel_ok=0, total=0) for m in methods}
    for m in methods:
        for s in seeds:
            for p in sorted(glob.glob(str(adir / f"{dataset}_fold*_{m}_seed{s}.audit.npz"))):
                d = load_audit_npz(p)
                if "target_indices" not in d or not len(np.asarray(d["target_indices"])):
                    continue
                avail[m]["total"] += 1
                avail[m]["abs_ok"] += 1 if bool(d.get("task_head_replay_ok", False)) else 0
                avail[m]["rel_ok"] += 1 if _relative_replay_ok(d) else 0
                tgt = int(np.asarray(d["d"])[np.asarray(d["target_indices"]).ravel()][0])
                for c in ("label_conditional", "marginal_domain", "random_subspace"):
                    for k in ks:
                        r = evaluate_reliance(d, target_domain=tgt, k=k, conditioning=c)
                        r["seed"] = s
                        rows[m].append(r)
    summary = {}
    for m in methods:
        summary[m] = dict(head_replay_abs=f"{avail[m]['abs_ok']}/{avail[m]['total']}",
                          head_replay_rel=f"{avail[m]['rel_ok']}/{avail[m]['total']}", curve={})
        for c in ("label_conditional", "marginal_domain", "random_subspace"):
            for k in (1, 2, 4, 8):
                sel = [r for r in rows[m] if r["conditioning"] == c and r["k"] == k]
                if sel:
                    td = [r["task_drop"] for r in sel]
                    summary[m]["curve"][f"{c}_k{k}"] = dict(
                        task_drop_mean=float(np.mean(td)), task_drop_std=float(np.std(td)),
                        firewall_pass=float(np.mean([r["firewall_passed"] for r in sel])), n=len(sel))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seeds", type=int, nargs="+", default=SEEDS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    rows = _load_rows(args.gate_dir, args.dataset, args.seeds)
    par = pareto_multiseed(rows, METHODS)
    rel = reliance_multiseed(args.gate_dir, args.dataset, args.seeds)
    report = dict(dataset=args.dataset, seeds=args.seeds, n_rows=len(rows), pareto=par, reliance=rel)
    outp = args.out or str(Path(args.gate_dir) / args.dataset / f"{args.dataset}_MULTISEED_ANALYSIS.json")
    json.dump(report, open(outp, "w"), indent=2)

    print(f"\n=== MULTI-SEED {args.dataset} (seeds {args.seeds}; {len(rows)} rows) ===")
    print(f"\n-- Pareto (mean +/- std over seeds; * frontier) --")
    print(f"  {'method':16s} {'target':>14s} {'graph_kl':>14s} {'node_kl':>14s}  frontier")
    for m, v in par.items():
        print(f"  {m:16s} {v['target_mean']:.3f}+/-{v['target_std']:.3f} {v['graph_mean']:6.3f}+/-{v['graph_std']:.3f} "
              f"{v['node_mean']:6.3f}+/-{v['node_std']:.3f}  {'*' if v['on_frontier'] else 'dom:'+','.join(v['dominated_by'])}")
    print(f"\n-- R3 reliance (target task_drop; head-replay avail abs | rel) --")
    for m in ("erm", "cigl_graph_node"):
        r = rel[m]; c = r["curve"]
        print(f"  {m:16s} head-replay {r['head_replay_abs']} (abs1e-5) | {r['head_replay_rel']} (rel1e-4)")
        for key in ("label_conditional_k2", "label_conditional_k8", "random_subspace_k2"):
            if key in c:
                print(f"      {key:24s} {c[key]['task_drop_mean']:+.3f}+/-{c[key]['task_drop_std']:.3f} (n={c[key]['n']})")
    e = rel["erm"]["curve"].get("label_conditional_k2"); cg = rel["cigl_graph_node"]["curve"].get("label_conditional_k2")
    if e and cg:
        print(f"\n  PRIMARY (label_conditional k2): ERM drop {e['task_drop_mean']:+.3f} vs CIGL {cg['task_drop_mean']:+.3f} "
              f"-> ERM>CIGL={e['task_drop_mean']>cg['task_drop_mean']}")
    print(f"\n[multiseed] wrote {outp}")


if __name__ == "__main__":
    sys.exit(main())
