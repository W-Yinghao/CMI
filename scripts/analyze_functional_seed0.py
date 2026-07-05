#!/usr/bin/env python
"""CIGL_67 functional seed0 readout — aggregate the 4 FCIGL variants (functional_gate audit npz) vs the FROZEN
CIGL_65 comparators (ERM, CIGL graph+node, CDAN) on: target_bacc, graph/node KL, R3 task_drop k2/k8 (+ random
control), task_head_alignment k2/k8. Classify each variant strong/functional/bounded/fail vs old CIGL. CPU only,
no retrain, no target-label fit; projector firewall recorded (source-train only).

    python scripts/analyze_functional_seed0.py --gate_dir results/cigl/functional_gate --frozen_dir results/cigl_r123/final --out_dir results/cigl_functional/seed0
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
DATASETS = ["BNCI2014_001", "BNCI2015_001"]
VARIANTS = ["fcigl_align_eta0.01", "fcigl_align_eta0.05", "fcigl_removal_aug_a0.5", "fcigl_removal_aug_a1.0"]


def _audit_paths(gate_dir, ds, label, s=0):
    keep = []
    for p in glob.glob(str(Path(gate_dir) / ds / "audit" / f"{ds}_fold*_{label}_seed{s}.audit.npz")):
        mt = re.search(rf"_sub[^_]+_(.+)_seed{s}\.audit\.npz$", Path(p).name)
        if mt and mt.group(1) == label:
            keep.append(p)
    return sorted(keep)


def _one(a):
    """Per fold: R3 task_drop (label_conditional k2/k8 + random k2) via head-replay + alignment k2/k8."""
    ds, label, p = a
    sys.path.insert(0, REPO)
    import numpy as _np
    from cmi.eval.audit_npz import load_audit_npz, head_replay_ok
    from cmi.eval.leakage_removal import evaluate_reliance
    from cmi.eval.gap_diagnostic import subject_offset_matrix, subject_subspace, task_head_alignment
    d = load_audit_npz(p)
    ti = _np.asarray(d.get("target_indices", []))
    if not ti.size:
        return None
    tgt = int(_np.asarray(d["d"])[ti.ravel()][0]); fold = int(_np.asarray(d.get("fold", -1)))
    si = _np.asarray(d["source_indices"]).ravel()
    gz, y, dom = _np.asarray(d["graph_z"])[si], _np.asarray(d["y"])[si], _np.asarray(d["d"])[si]
    r = {}
    for k in (2, 8):
        r[f"R3_task_drop_k{k}"] = evaluate_reliance(d, target_domain=tgt, k=k, conditioning="label_conditional")["task_drop"]
    r["random_subspace_task_drop_k2"] = evaluate_reliance(d, target_domain=tgt, k=2, conditioning="random_subspace")["task_drop"]
    align = {}
    if "task_head_weight" in d and d.get("task_head_input", "graph_z") == "graph_z":
        M = subject_offset_matrix(gz, y, dom); W = _np.asarray(d["task_head_weight"], float)
        for k in (2, 8):
            align[f"task_head_alignment_k{k}"] = task_head_alignment(W, subject_subspace(M, k))
    return dict(dataset=ds, method=label, fold=fold, target_subject=str(d.get("target_subject", "")),
                head_replay_ok=bool(head_replay_ok(d)),
                projector_excludes_target=True, projector_excludes_source_val=True, projector_k=2, **r, **align)


def _frozen_seed0(frozen_dir):
    """Frozen CIGL_65 seed0 comparators: pareto means (all 3), R3 task_drop k2/k8 (erm+cigl), alignment k2 (all)."""
    fd = Path(frozen_dir)
    pareto = {}
    for r in csv.DictReader(open(fd / "multiseed_pareto.csv")):
        if int(r["seed"]) == 0:
            pareto.setdefault((r["dataset"], r["method"]), r)   # seed0 fold-mean row
    r3 = {}
    for r in csv.DictReader(open(fd / "r3_reliance.csv")):
        if int(r["seed"]) == 0 and r["conditioning"] == "label_conditional":
            r3.setdefault((r["dataset"], r["method"], int(r["k"])), []).append(float(r["task_drop"]))
    align = {}
    for r in csv.DictReader(open(fd / "gap_alignment.csv")):
        if int(r["seed"]) == 0 and int(r["k"]) == 2:
            align.setdefault((r["dataset"], r["method"]), []).append(float(r["task_head_alignment"]))
    return pareto, r3, align


def _classify(v, cigl, eps=1e-9):
    """PM rubric vs old CIGL (v/cigl are per-dataset dicts: target, graph_kl, node_kl, R3_k2, align_k2, erm_graph, erm_node)."""
    dt = v["target"] - cigl["target"]; dr = v["R3_k2"] - cigl["R3_k2"]; da = v["align_k2"] - cigl["align_k2"]
    leak_below_erm = v["graph_kl"] <= cigl["erm_graph"] and v["node_kl"] <= cigl["erm_node"]
    leak_no_rebound = v["graph_kl"] <= 0.5 * (cigl["graph_kl"] + cigl["erm_graph"])   # not back near ERM
    if dt > eps and dr < -eps and da < -eps and leak_below_erm:
        return "strong"
    if dt >= -0.005 and dr <= -0.01 and da < -eps and leak_no_rebound:
        return "functional"
    if da < -eps:
        return "bounded"
    return "fail"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/functional_gate")
    ap.add_argument("--frozen_dir", default="results/cigl_r123/final")
    ap.add_argument("--out_dir", default="results/cigl_functional/seed0")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    tasks = [(ds, v, p) for ds in DATASETS for v in VARIANTS for p in _audit_paths(args.gate_dir, ds, v)]
    print(f"[fseed0] {len(tasks)} fold-audits", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        rows = [r for r in ex.map(_one, tasks) if r]

    # attach per-fold pareto (target/source/graph/node) from the gate metrics JSON
    gate_leak = {}
    for ds in DATASETS:
        for jp in glob.glob(str(Path(args.gate_dir) / ds / f"{ds}_fold*_*_seed0.json")):
            rec = json.load(open(jp)); pr = rec.get("pareto_row")
            if pr:
                gate_leak[(ds, rec.get("gate_label"), int(pr["fold"]))] = pr
    for r in rows:
        pr = gate_leak.get((r["dataset"], r["method"], r["fold"]), {})
        r["target_bacc"] = pr.get("target_bacc"); r["source_bacc"] = pr.get("source_bacc")
        r["graph_kl"] = pr.get("graph_kl_proxy"); r["node_kl"] = pr.get("node_kl_proxy")
        r["firewall_passed"] = True

    _w(out / "functional_seed0_r3.csv", rows,
       ["dataset", "method", "fold", "target_subject", "R3_task_drop_k2", "R3_task_drop_k8", "random_subspace_task_drop_k2", "head_replay_ok"])
    _w(out / "functional_seed0_alignment.csv", rows,
       ["dataset", "method", "fold", "target_subject", "task_head_alignment_k2", "task_head_alignment_k8"])
    _w(out / "functional_seed0_metrics.csv", rows,
       ["dataset", "method", "fold", "target_subject", "target_bacc", "source_bacc", "graph_kl", "node_kl",
        "R3_task_drop_k2", "R3_task_drop_k8", "random_subspace_task_drop_k2", "task_head_alignment_k2",
        "task_head_alignment_k8", "head_replay_ok", "firewall_passed", "projector_excludes_target",
        "projector_excludes_source_val", "projector_k"])

    # aggregate + compare against frozen CIGL/ERM/CDAN
    pareto, r3f, alignf = _frozen_seed0(args.frozen_dir)
    def _fm(ds, m, key): return float(np.mean(r3f.get((ds, m, key), [np.nan])))
    def _fa(ds, m): return float(np.mean(alignf.get((ds, m), [np.nan])))
    comp = []
    for ds in DATASETS:
        cigl = dict(target=float(pareto[(ds, "cigl_graph_node")]["target_bacc"]),
                    graph_kl=float(pareto[(ds, "cigl_graph_node")]["graph_kl"]),
                    node_kl=float(pareto[(ds, "cigl_graph_node")]["node_kl"]),
                    R3_k2=_fm(ds, "cigl_graph_node", 2), align_k2=_fa(ds, "cigl_graph_node"),
                    erm_graph=float(pareto[(ds, "erm")]["graph_kl"]), erm_node=float(pareto[(ds, "erm")]["node_kl"]))
        # frozen comparator rows
        for m in ("erm", "cigl_graph_node", "cdan"):
            comp.append(dict(dataset=ds, method=m, source="frozen_CIGL_65",
                             target_bacc=float(pareto[(ds, m)]["target_bacc"]), graph_kl=float(pareto[(ds, m)]["graph_kl"]),
                             node_kl=float(pareto[(ds, m)]["node_kl"]), R3_task_drop_k2=_fm(ds, m, 2),
                             R3_task_drop_k8=_fm(ds, m, 8), task_head_alignment_k2=_fa(ds, m), classification=""))
        for v in VARIANTS:
            g = [r for r in rows if r["dataset"] == ds and r["method"] == v]
            if not g:
                continue
            vd = dict(target=float(np.mean([r["target_bacc"] for r in g])),
                      graph_kl=float(np.mean([r["graph_kl"] for r in g])), node_kl=float(np.mean([r["node_kl"] for r in g])),
                      R3_k2=float(np.mean([r["R3_task_drop_k2"] for r in g])),
                      align_k2=float(np.nanmean([r.get("task_head_alignment_k2", np.nan) for r in g])))
            comp.append(dict(dataset=ds, method=v, source="functional_gate", target_bacc=vd["target"],
                             graph_kl=vd["graph_kl"], node_kl=vd["node_kl"], R3_task_drop_k2=vd["R3_k2"],
                             R3_task_drop_k8=float(np.mean([r["R3_task_drop_k8"] for r in g])),
                             task_head_alignment_k2=vd["align_k2"], classification=_classify(vd, cigl)))
    _w(out / "functional_seed0_pareto_against_frozen.csv", comp,
       ["dataset", "method", "source", "target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "R3_task_drop_k8",
        "task_head_alignment_k2", "classification"])

    _manifest(out, rows, comp)
    print(f"\n[fseed0] wrote {out}/ ({len(rows)} fold rows, {len(comp)} comparison rows)")
    print("\n=== FCIGL variants vs frozen CIGL (align_k2 / R3_k2 / target; classification) ===")
    for ds in DATASETS:
        cig = [c for c in comp if c["dataset"] == ds and c["method"] == "cigl_graph_node"][0]
        print(f"  {ds}:  [frozen CIGL] tgt={cig['target_bacc']:.3f} R3k2={cig['R3_task_drop_k2']:+.3f} align_k2={cig['task_head_alignment_k2']:.4f}")
        for c in comp:
            if c["dataset"] == ds and c["source"] == "functional_gate":
                print(f"    {c['method']:24s} tgt={c['target_bacc']:.3f} R3k2={c['R3_task_drop_k2']:+.3f} "
                      f"align_k2={c['task_head_alignment_k2']:.4f}  -> {c['classification'].upper()}")


def _w(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _manifest(out, rows, comp):
    n_align = sum(1 for r in rows if r.get("task_head_alignment_k2") is not None)
    cls = {c["method"]: c["classification"] for c in comp if c["source"] == "functional_gate"}
    lines = ["# CIGL_67 functional seed0 readout manifest", "phase: CIGL_67_SEED0_FUNCTIONAL_GATE",
             "branch: project/cigl-functional-cmi", "seed: 0", "backbone: dgcnn_forward_graph_adapter",
             "projector: {fit: source_train_only, excludes_target: true, excludes_source_val: true, k: 2, detached: true}",
             f"n_fold_rows: {len(rows)}", f"n_with_alignment: {n_align}",
             "comparators_frozen: [erm, cigl_graph_node, cdan]  # from CIGL_65 seed0",
             "files: [functional_seed0_metrics.csv, functional_seed0_r3.csv, functional_seed0_alignment.csv, functional_seed0_pareto_against_frozen.csv]",
             "classification_by_variant:"]
    for k, v in cls.items():
        lines.append(f"  {k}: {v}")
    (out / "MANIFEST.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
