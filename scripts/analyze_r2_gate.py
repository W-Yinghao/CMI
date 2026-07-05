#!/usr/bin/env python
"""Post-gate analysis for the R2 seed0 gate (Pareto + R3 reliance). CPU, consumes the gate's rows JSON +
per-(fold,method) .audit.npz. NO retraining, NO target-label use in any fit.

A) Pareto: per method aggregate source/target bAcc + graph/node KL proxy; dominance (dominated iff another has
   >= target task AND <= graph leakage AND <= node leakage, >=1 strict). Primary Q: does CIGL sit on the frontier?
B) R3 reliance (ERM vs cigl_graph_node): label_conditional subspace removal (primary k=2, curve {1,2,4,8}) +
   marginal_domain + random_subspace controls; head-replay when replay_ok. Primary screen: ERM task_drop > CIGL.

Usage: python analyze_r2_gate.py --gate_dir results/cigl/r2_seed0_gate --dataset BNCI2014_001 [--repo <worktree>]
"""
from __future__ import annotations
import argparse, glob, json, sys
from pathlib import Path
import numpy as np


def _agg(rows, methods):
    """Mean over folds per method of the Pareto quantities."""
    out = {}
    for m in methods:
        r = [x for x in rows if x["method_label"] == m]
        if not r:
            continue
        out[m] = dict(n_folds=len(r),
                      target_bacc=float(np.mean([x["target_bacc"] for x in r])),
                      source_bacc=float(np.mean([x["source_bacc"] for x in r])),
                      graph_kl=float(np.mean([x["graph_kl_proxy"] for x in r])),
                      node_kl=float(np.mean([x["node_kl_proxy"] for x in r])))
    return out


def _dominated(a, b, eps=1e-9):
    """a dominated by b iff b >= target task AND <= graph leak AND <= node leak, with >=1 strict."""
    ge = b["target_bacc"] >= a["target_bacc"] - eps
    le = b["graph_kl"] <= a["graph_kl"] + eps and b["node_kl"] <= a["node_kl"] + eps
    strict = (b["target_bacc"] > a["target_bacc"] + eps or b["graph_kl"] < a["graph_kl"] - eps
              or b["node_kl"] < a["node_kl"] - eps)
    return ge and le and strict


def pareto(agg):
    labels = list(agg)
    dominated = {m: any(_dominated(agg[m], agg[o]) for o in labels if o != m) for m in labels}
    return {m: {**agg[m], "dominated": dominated[m], "on_frontier": not dominated[m]} for m in labels}


def reliance(repo, gate_dir, dataset, methods=("erm", "cigl_graph_node"), ks=(1, 2, 4, 8)):
    sys.path.insert(0, repo)
    from cmi.eval.leakage_removal import evaluate_reliance
    from cmi.eval.audit_npz import load_audit_npz, head_replay_ok
    adir = Path(gate_dir) / dataset / "audit"
    out = {}
    for m in methods:
        rows = []
        for p in sorted(glob.glob(str(adir / f"{dataset}_fold*_{m}_seed0.audit.npz"))):
            d = load_audit_npz(p)
            if "target_indices" not in d or not len(np.asarray(d["target_indices"])):
                continue
            tgt = int(np.asarray(d["d"])[np.asarray(d["target_indices"]).ravel()][0])
            for c in ("label_conditional", "marginal_domain", "random_subspace"):
                for k in ks:
                    r = evaluate_reliance(d, target_domain=tgt, k=k, conditioning=c)
                    r["fold_file"] = Path(p).name
                    rows.append(r)
        out[m] = rows
    # aggregate: mean task_drop per (method, conditioning, k), head-replay share
    summary = {}
    for m, rows in out.items():
        summary[m] = {}
        for c in ("label_conditional", "marginal_domain", "random_subspace"):
            for k in (1, 2, 4, 8):
                sel = [r for r in rows if r["conditioning"] == c and r["k"] == k]
                if sel:
                    summary[m][f"{c}_k{k}"] = dict(
                        task_drop=float(np.mean([r["task_drop"] for r in sel])),
                        subject_leak_drop=float(np.nanmean([r["subject_leakage_drop"] for r in sel])),
                        head_replay=float(np.mean([r["removal_mode"] == "head_replay" for r in sel])),
                        firewall_pass=float(np.mean([r["firewall_passed"] for r in sel])), n=len(sel))
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    rows_path = Path(args.gate_dir) / args.dataset / f"{args.dataset}_seed0_pareto_rows.json"
    blob = json.load(open(rows_path))
    # attach a method label from the per-file config (config string carries the method); map via gate labels
    label_map = {"erm:0": "erm"}   # fallback; prefer reading gate_label from per-method JSON
    rows = []
    for jp in sorted(glob.glob(str(Path(args.gate_dir) / args.dataset / f"{args.dataset}_fold*_*_seed0.json"))):
        rec = json.load(open(jp))
        pr = rec.get("pareto_row");
        if pr is None: continue
        pr = dict(pr); pr["method_label"] = rec.get("gate_label", rec.get("config"))
        rows.append(pr)
    methods = ["erm", "cigl_graph_node", "dann", "cond_dann", "cdan"]
    agg = _agg(rows, methods)
    par = pareto(agg)
    rel = reliance(args.repo, args.gate_dir, args.dataset)

    report = dict(dataset=args.dataset, n_rows=len(rows), pareto=par, reliance=rel,
                  erm_vs_cigl_reliance=_erm_vs_cigl(rel))
    outp = args.out or str(Path(args.gate_dir) / args.dataset / f"{args.dataset}_seed0_ANALYSIS.json")
    json.dump(report, open(outp, "w"), indent=2)
    _print(report)
    print(f"\n[analyze] wrote {outp}")


def _erm_vs_cigl(rel, key="label_conditional_k2"):
    e = rel.get("erm", {}).get(key); c = rel.get("cigl_graph_node", {}).get(key)
    if not e or not c:
        return None
    return dict(primary_key=key, erm_task_drop=e["task_drop"], cigl_task_drop=c["task_drop"],
                erm_gt_cigl=bool(e["task_drop"] > c["task_drop"]),
                interpretation=("ERM relies MORE on subject subspace (task drops more when removed) than CIGL"
                                if e["task_drop"] > c["task_drop"] else
                                "no reliance-reduction signal (ERM task_drop <= CIGL)"))


def _print(r):
    print(f"\n=== R2 seed0 gate analysis: {r['dataset']} ({r['n_rows']} rows) ===")
    print("\n-- Pareto (task vs leakage; * = on frontier) --")
    print(f"  {'method':18s} {'tgt_bacc':>9s} {'src_bacc':>9s} {'graph_kl':>9s} {'node_kl':>9s}  frontier")
    for m, v in r["pareto"].items():
        print(f"  {m:18s} {v['target_bacc']:9.3f} {v['source_bacc']:9.3f} {v['graph_kl']:9.3f} {v['node_kl']:9.3f}  {'*' if v['on_frontier'] else ''}")
    ev = r.get("erm_vs_cigl_reliance")
    if ev:
        print(f"\n-- R3 reliance (primary {ev['primary_key']}) --")
        print(f"  ERM task_drop={ev['erm_task_drop']:+.3f}  CIGL task_drop={ev['cigl_task_drop']:+.3f}  "
              f"ERM>CIGL={ev['erm_gt_cigl']}")
        print(f"  {ev['interpretation']}")


if __name__ == "__main__":
    sys.exit(main())
