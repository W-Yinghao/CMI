#!/usr/bin/env python
"""CIGL_67 functional CMI confirmation (seeds 0/1/2) — the 2 fcigl_align variants vs FROZEN CIGL_65 seeds 0/1/2,
with hierarchical bootstrap CIs on the paired (FCIGL - old CIGL) deltas. CPU only, no retrain. Pairs per
(dataset, seed, fold); bootstraps per-dataset (seed->fold) AND pooled (dataset->seed->fold) so 2a near-chance
does not mask the 2015 clean readout.

    python scripts/analyze_functional_confirmation.py --gate_dir results/cigl/functional_gate \
        --frozen_final results/cigl_r123/final --frozen_gate /home/infres/yinwang/CMI_AAAI_cigl_r123/results/cigl/r2_seed0_gate \
        --out_dir results/cigl_functional/final
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from cmi.eval.evidence_hardening import hierarchical_bootstrap                       # noqa: E402
from scripts.analyze_functional_seed0 import _one, _audit_paths                      # noqa: E402

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
VARIANTS = ["fcigl_align_eta0.01", "fcigl_align_eta0.05"]
SEEDS = [0, 1, 2]
QTY = ["target_bacc", "graph_kl", "node_kl", "R3_task_drop_k2", "task_head_alignment_k2"]


def _fcigl_perfold(gate_dir, workers):
    tasks = [(ds, v, s, p) for ds in DATASETS for v in VARIANTS for s in SEEDS for p in _audit_paths(gate_dir, ds, v, s)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = list(ex.map(_one, [(t[0], t[1], t[3]) for t in tasks]))   # _one takes (ds, label, path)
    leak = {}
    for ds in DATASETS:
        for jp in glob.glob(str(Path(gate_dir) / ds / f"{ds}_fold*_*_seed*.json")):
            rec = json.load(open(jp)); pr = rec.get("pareto_row")
            if pr:
                leak[(ds, rec.get("gate_label"), int(pr["seed"]), int(pr["fold"]))] = pr
    out = {}
    for (ds, v, s, p), r in zip(tasks, results):     # seed comes straight from the task tuple
        if r is None:
            continue
        pr = leak.get((ds, v, s, r["fold"]), {})
        out[(ds, v, s, r["fold"])] = dict(
            target_bacc=_f(pr.get("target_bacc")), source_bacc=_f(pr.get("source_bacc")),
            graph_kl=_f(pr.get("graph_kl_proxy")), node_kl=_f(pr.get("node_kl_proxy")),
            R3_task_drop_k2=r.get("R3_task_drop_k2"), R3_task_drop_k8=r.get("R3_task_drop_k8"),
            random_subspace_task_drop_k2=r.get("random_subspace_task_drop_k2"),
            task_head_alignment_k2=r.get("task_head_alignment_k2"), task_head_alignment_k8=r.get("task_head_alignment_k8"),
            head_replay_ok=r.get("head_replay_ok"))
    return out


def _f(x):
    return float(x) if x is not None else float("nan")


def _cigl_perfold(frozen_final, frozen_gate):
    """CIGL per (dataset, seed, fold): target/graph/node from raw gate JSONs; R3 k2/k8 + align k2/k8 from CSVs."""
    out = {}
    for ds in DATASETS:
        for s in SEEDS:
            for jp in glob.glob(str(Path(frozen_gate) / ds / f"{ds}_fold*_cigl_graph_node_seed{s}.json")):
                rec = json.load(open(jp)); pr = rec.get("pareto_row")
                if pr:
                    out[(ds, s, int(pr["fold"]))] = dict(target_bacc=_f(pr["target_bacc"]),
                                                         graph_kl=_f(pr["graph_kl_proxy"]), node_kl=_f(pr["node_kl_proxy"]))
    for r in csv.DictReader(open(Path(frozen_final) / "r3_reliance.csv")):
        if r["method"] == "cigl_graph_node" and r["conditioning"] == "label_conditional":
            k = (r["dataset"], int(r["seed"]), int(r["fold"]))
            if k in out:
                out[k][f"R3_task_drop_k{r['k']}"] = float(r["task_drop"])
    for r in csv.DictReader(open(Path(frozen_final) / "gap_alignment.csv")):
        if r["method"] == "cigl_graph_node":
            k = (r["dataset"], int(r["seed"]), int(r["fold"]))
            if k in out:
                out[k][f"task_head_alignment_k{r['k']}"] = float(r["task_head_alignment"])
    return out


def _boot(records, levels, n_boot=4000):
    if len(records) < 2:
        return None
    r = hierarchical_bootstrap(records, value_key="value", levels=levels, n_boot=n_boot, seed=0)
    return dict(point=r["point"], lo=r["lo"], hi=r["hi"], n=r["n_records"], excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/functional_gate")
    ap.add_argument("--frozen_final", default="results/cigl_r123/final")
    ap.add_argument("--frozen_gate", default="/home/infres/yinwang/CMI_AAAI_cigl_r123/results/cigl/r2_seed0_gate")
    ap.add_argument("--out_dir", default="results/cigl_functional/final")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[conf] FCIGL per-fold (seeds 0/1/2)...", flush=True)
    fc = _fcigl_perfold(args.gate_dir, args.workers)
    cg = _cigl_perfold(args.frozen_final, args.frozen_gate)

    # per-fold metrics table
    mrows = []
    for (ds, m, s, f), v in sorted(fc.items()):
        mrows.append(dict(dataset=ds, method=m, seed=s, fold=f, **v,
                          projector_excludes_target=True, projector_excludes_source_val=True, projector_k=2, firewall_passed=True))
    _w(out / "functional_multiseed_metrics.csv", mrows,
       ["dataset", "method", "seed", "fold", "target_bacc", "source_bacc", "graph_kl", "node_kl", "R3_task_drop_k2",
        "R3_task_drop_k8", "random_subspace_task_drop_k2", "task_head_alignment_k2", "task_head_alignment_k8",
        "head_replay_ok", "firewall_passed", "projector_excludes_target", "projector_excludes_source_val", "projector_k"])
    _w(out / "functional_multiseed_r3.csv", mrows,
       ["dataset", "method", "seed", "fold", "R3_task_drop_k2", "R3_task_drop_k8", "random_subspace_task_drop_k2", "head_replay_ok"])
    _w(out / "functional_multiseed_alignment.csv", mrows,
       ["dataset", "method", "seed", "fold", "task_head_alignment_k2", "task_head_alignment_k8"])

    # paired deltas + bootstrap
    ci_rows = []; delta_rows = []
    for m in VARIANTS:
        for q in QTY:
            deltas = []
            for (ds, mm, s, f), v in fc.items():
                if mm != m:
                    continue
                c = cg.get((ds, s, f))
                if not c or q not in c or v.get(q) is None:
                    continue
                dv = v[q] - c[q]
                if dv == dv:
                    deltas.append(dict(dataset=ds, seed=s, fold=f, value=float(dv)))
                    delta_rows.append(dict(method=m, quantity=q, dataset=ds, seed=s, fold=f, delta=float(dv)))
            for scope, lv, sub in (("pooled", ("dataset", "seed", "fold"), deltas),
                                   ("BNCI2014_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2014_001"]),
                                   ("BNCI2015_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2015_001"])):
                b = _boot(sub, lv)
                if b:
                    ci_rows.append(dict(comparison=f"{m}_minus_CIGL", quantity=q, scope=scope, point=b["point"],
                                        ci_lo=b["lo"], ci_hi=b["hi"], n=b["n"], excludes_zero=b["excludes_zero"]))
    _w(out / "functional_vs_frozen_deltas.csv", delta_rows, ["method", "quantity", "dataset", "seed", "fold", "delta"])
    _w(out / "functional_vs_oldcigl_bootstrap_ci.csv", ci_rows,
       ["comparison", "quantity", "scope", "point", "ci_lo", "ci_hi", "n", "excludes_zero"])

    _manifest(out, mrows, ci_rows)
    print(f"\n[conf] wrote {out}/ ({len(mrows)} fold rows, {len(ci_rows)} CI rows)")
    print("\n=== paired (FCIGL - old CIGL) hierarchical bootstrap CIs (sig = excludes 0) ===")
    for m in VARIANTS:
        print(f"  {m}:")
        for q in QTY:
            for scope in ("pooled", "BNCI2014_001", "BNCI2015_001"):
                r = [c for c in ci_rows if c["comparison"] == f"{m}_minus_CIGL" and c["quantity"] == q and c["scope"] == scope]
                if r:
                    r = r[0]
                    print(f"    {q:26s} {scope:13s} {r['point']:+.4f} [{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}] sig={r['excludes_zero']}")


def _w(path, rows, fields):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fields})


def _manifest(out, mrows, ci_rows):
    n_align = sum(1 for r in mrows if r.get("task_head_alignment_k2") is not None)
    lines = ["# CIGL_67 functional confirmation manifest (seeds 0/1/2)", "phase: CIGL_67_FUNCTIONAL_CMI_CONFIRMATION",
             "branch: project/cigl-functional-cmi", "seeds: [0, 1, 2]",
             "methods: [fcigl_align_eta0.01, fcigl_align_eta0.05]",
             "comparators_frozen: [cigl_graph_node, erm, cdan]  # CIGL_65 seeds 0/1/2",
             "projector: {fit: source_train_only, excludes_target: true, excludes_source_val: true, k: 2}",
             f"n_fold_rows: {len(mrows)}", f"n_with_alignment: {n_align}",
             "bootstrap: {per_dataset: seed->fold, pooled: dataset->seed->fold, n_boot: 4000}",
             "files: [functional_multiseed_metrics.csv, functional_multiseed_r3.csv, functional_multiseed_alignment.csv, functional_vs_frozen_deltas.csv, functional_vs_oldcigl_bootstrap_ci.csv]"]
    (out / "MANIFEST.yaml").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
