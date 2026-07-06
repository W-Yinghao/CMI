#!/usr/bin/env python
"""CIGL_68 direct-reliance seed0 readout — the 2 dcigl_consistency variants vs FROZEN comparators (old CIGL,
ERM, CDAN from CIGL_65 seed0; FCIGL-align eta0.01/0.05 from CIGL_67 seed0). PRIMARY metric = R3 task_drop k2
(the thing every proxy failed to move). CPU only, no retrain. Classifies strong/functional/fail vs old CIGL and
vs FCIGL-align.
"""
from __future__ import annotations
import argparse, csv, glob, json, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from scripts.analyze_functional_seed0 import _one, _audit_paths                       # noqa: E402

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
VARIANTS = ["dcigl_consistency_beta0.1", "dcigl_consistency_beta0.5"]


def _dcigl_perfold(gate_dir, workers):
    tasks = [(ds, v, p) for ds in DATASETS for v in VARIANTS for p in _audit_paths(gate_dir, ds, v, 0)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        rows = [r for r in ex.map(_one, tasks) if r]
    leak = {}
    for ds in DATASETS:
        for jp in glob.glob(str(Path(gate_dir) / ds / f"{ds}_fold*_*_seed0.json")):
            rec = json.load(open(jp)); pr = rec.get("pareto_row")
            if pr:
                leak[(ds, rec.get("gate_label"), int(pr["fold"]))] = pr
    for r in rows:
        pr = leak.get((r["dataset"], r["method"], r["fold"]), {})
        r["target_bacc"] = _f(pr.get("target_bacc")); r["graph_kl"] = _f(pr.get("graph_kl_proxy")); r["node_kl"] = _f(pr.get("node_kl_proxy"))
    return rows


def _f(x):
    return float(x) if x is not None else float("nan")


def _frozen(frozen_final, frozen_gate, func_final):
    """seed0 comparators: CIGL/ERM/CDAN (CIGL_65) + FCIGL-align (CIGL_67). Return {(ds,method): dict}."""
    out = {}
    # pareto (target/graph/node) seed0
    for r in csv.DictReader(open(Path(frozen_final) / "multiseed_pareto.csv")):
        if int(r["seed"]) == 0:
            out.setdefault((r["dataset"], r["method"]), {}).update(
                target=_f(r["target_bacc"]), graph_kl=_f(r["graph_kl"]), node_kl=_f(r["node_kl"]))
    # R3 k2 seed0 (erm+cigl)
    r3 = {}
    for r in csv.DictReader(open(Path(frozen_final) / "r3_reliance.csv")):
        if int(r["seed"]) == 0 and r["conditioning"] == "label_conditional" and int(r["k"]) == 2:
            r3.setdefault((r["dataset"], r["method"]), []).append(_f(r["task_drop"]))
    al = {}
    for r in csv.DictReader(open(Path(frozen_final) / "gap_alignment.csv")):
        if int(r["seed"]) == 0 and int(r["k"]) == 2:
            al.setdefault((r["dataset"], r["method"]), []).append(_f(r["task_head_alignment"]))
    for k, v in out.items():
        v["R3_k2"] = float(np.mean(r3.get(k, [np.nan]))); v["align_k2"] = float(np.mean(al.get(k, [np.nan])))
    # FCIGL-align seed0 from CIGL_67 multiseed (seed0 subset)
    fc = {}
    for r in csv.DictReader(open(Path(func_final) / "functional_multiseed_metrics.csv")):
        if int(r["seed"]) == 0:
            fc.setdefault((r["dataset"], r["method"]), {"t": [], "r": [], "a": [], "g": [], "n": []})
            d = fc[(r["dataset"], r["method"])]
            d["t"].append(_f(r["target_bacc"])); d["r"].append(_f(r["R3_task_drop_k2"]))
            d["a"].append(_f(r["task_head_alignment_k2"])); d["g"].append(_f(r["graph_kl"])); d["n"].append(_f(r["node_kl"]))
    for k, d in fc.items():
        out[k] = dict(target=float(np.nanmean(d["t"])), R3_k2=float(np.nanmean(d["r"])), align_k2=float(np.nanmean(d["a"])),
                      graph_kl=float(np.nanmean(d["g"])), node_kl=float(np.nanmean(d["n"])))
    return out


def _classify(v, cigl, fcigl, erm, eps=1e-9):
    dt = v["target"] - cigl["target"]; dr = v["R3_k2"] - cigl["R3_k2"]
    leak_below_erm = v["graph_kl"] <= erm["graph_kl"] and v["node_kl"] <= erm["node_kl"]
    if dt >= -0.005 and dr < -0.01 and v["R3_k2"] < fcigl["R3_k2"] and leak_below_erm:
        return "strong" if dt > eps else "strong_reliance"
    if dt >= -0.005 and dr < -0.01 and leak_below_erm:
        return "functional"
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/direct_reliance_gate")
    ap.add_argument("--frozen_final", default="results/cigl_r123/final")
    ap.add_argument("--frozen_gate", default="/home/infres/yinwang/CMI_AAAI_cigl_r123/results/cigl/r2_seed0_gate")
    ap.add_argument("--func_final", default="results/cigl_functional/final")
    ap.add_argument("--out_dir", default="results/cigl_direct/seed0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    rows = _dcigl_perfold(args.gate_dir, args.workers)
    frozen = _frozen(args.frozen_final, args.frozen_gate, args.func_final)
    _w(out / "dcigl_seed0_metrics.csv", rows,
       ["dataset", "method", "fold", "target_subject", "target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2",
        "R3_task_drop_k8", "random_subspace_task_drop_k2", "task_head_alignment_k2", "head_replay_ok"])

    comp = []
    for ds in DATASETS:
        cigl = frozen.get((ds, "cigl_graph_node")); erm = frozen.get((ds, "erm"))
        fcigl = frozen.get((ds, "fcigl_align_eta0.01"))
        for m in ("erm", "cigl_graph_node", "cdan", "fcigl_align_eta0.01", "fcigl_align_eta0.05"):
            fv = frozen.get((ds, m))
            if fv:
                comp.append(dict(dataset=ds, method=m, source="frozen", target_bacc=fv.get("target"),
                                 graph_kl=fv.get("graph_kl"), node_kl=fv.get("node_kl"),
                                 R3_task_drop_k2=fv.get("R3_k2"), task_head_alignment_k2=fv.get("align_k2"), classification=""))
        for v in VARIANTS:
            g = [r for r in rows if r["dataset"] == ds and r["method"] == v]
            if not g:
                continue
            vd = dict(target=float(np.nanmean([r["target_bacc"] for r in g])),
                      graph_kl=float(np.nanmean([r["graph_kl"] for r in g])), node_kl=float(np.nanmean([r["node_kl"] for r in g])),
                      R3_k2=float(np.nanmean([r["R3_task_drop_k2"] for r in g])),
                      align_k2=float(np.nanmean([r.get("task_head_alignment_k2", np.nan) for r in g])),
                      rand=float(np.nanmean([r["random_subspace_task_drop_k2"] for r in g])))
            comp.append(dict(dataset=ds, method=v, source="dcigl_gate", target_bacc=vd["target"], graph_kl=vd["graph_kl"],
                             node_kl=vd["node_kl"], R3_task_drop_k2=vd["R3_k2"], task_head_alignment_k2=vd["align_k2"],
                             random_ctrl_k2=vd["rand"],
                             classification=_classify(vd, cigl, fcigl, erm) if (cigl and fcigl and erm) else "?"))
    _w(out / "dcigl_seed0_vs_frozen.csv", comp,
       ["dataset", "method", "source", "target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "task_head_alignment_k2", "random_ctrl_k2", "classification"])
    print(f"\n[dcigl-seed0] wrote {out}/ ({len(rows)} fold rows)")
    print("\n=== dcigl vs frozen (PRIMARY = R3 task_drop k2; lower = less reliance) ===")
    for ds in DATASETS:
        for tag in ("cigl_graph_node", "fcigl_align_eta0.01", "erm"):
            c = [x for x in comp if x["dataset"] == ds and x["method"] == tag]
            if c:
                print(f"  {ds} [{tag:20s}] R3k2={c[0]['R3_task_drop_k2']:+.3f} tgt={c[0]['target_bacc']:.3f}")
        for c in comp:
            if c["dataset"] == ds and c["source"] == "dcigl_gate":
                print(f"    {c['method']:26s} R3k2={c['R3_task_drop_k2']:+.3f} tgt={c['target_bacc']:.3f} "
                      f"align={c['task_head_alignment_k2']:.3f} rand={c.get('random_ctrl_k2',0):+.4f} gKL={c['graph_kl']:.3f} -> {c['classification'].upper()}")


def _w(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


if __name__ == "__main__":
    sys.exit(main())
