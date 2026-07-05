#!/usr/bin/env python
"""CIGL R1 hardened leakage audit (n_perm=1000) recomputed from the gate's SEED0 saved features — NO retrain,
NO target-label use. Confirmatory tightening of the in-run n_perm=50 leakage p-values. Per PM priority:
ERM, CIGL graph+node, then CDAN. For each (dataset, method, fold): select SOURCE rows from the .audit.npz,
recompute the within-label-permutation graph/node KL-proxy audit at n_perm=1000 (same probe/epochs/split as the
gate), plus the label-conditional subject bAcc. Aggregate BH-FDR across folds. Parallel over folds (CPU pool).

    python scripts/r1_hardened_audit.py --gate_dir results/cigl/r2_seed0_gate --n_perm 1000 --workers 8
"""
from __future__ import annotations
import argparse, glob, json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

DATASETS = ["BNCI2014_001", "BNCI2015_001"]
METHODS = ["erm", "cigl_graph_node", "cdan"]        # PM priority: erm, cigl first; cdan the key comparator


def _one_fold(args):
    dataset, method, path, n_perm, epochs = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    import torch; torch.set_num_threads(1)
    sys.path.insert(0, REPO)
    from cmi.eval.audit_npz import load_audit_npz
    from cmi.eval.graph_leakage import audit_graph_node_objects
    from cmi.eval.probe_splits import stratified_trial_split_by_y_d
    from cmi.eval.leakage_removal import _subject_bacc
    d = load_audit_npz(path)
    si = np.asarray(d["source_indices"]).ravel()               # SOURCE rows only (leakage is among source subjects)
    gz, nz = np.asarray(d["graph_z"])[si], np.asarray(d["node_z"])[si]
    y, dom = np.asarray(d["y"])[si], np.asarray(d["d"])[si]
    n_cls, n_dom = int(y.max() + 1), int(dom.max() + 1)
    tr, va, _ = stratified_trial_split_by_y_d(y, dom, train_frac=0.7, seed=0, min_per_cell=2)
    au = audit_graph_node_objects(gz, nz, y, dom, n_cls, n_dom, n_perm=n_perm, seed=0, device="cpu",
                                  epochs=epochs, train_idx=tr, val_idx=va)
    subj = _subject_bacc(gz, dom, y, seed=0)                    # label-conditional subject bAcc on source
    fold = int(np.asarray(d.get("fold", -1)))
    return dict(dataset=dataset, method=method, fold=fold, fold_file=Path(path).name,
                graph_kl=float(au["graph"]["kl_mean"]), node_kl=float(au["node"]["kl_mean"]),
                graph_perm_mean=float(au["graph"]["permutation_mean"]), node_perm_mean=float(au["node"]["permutation_mean"]),
                graph_perm_p=float(au["graph"]["permutation_p"]), node_perm_p=float(au["node"]["permutation_p"]),
                subject_bacc_labelcond=float(subj) if subj == subj else None, n_perm=int(n_perm))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    tasks = []
    for ds in DATASETS:
        for m in args.methods:
            for p in sorted(glob.glob(str(Path(args.gate_dir) / ds / "audit" / f"{ds}_fold*_{m}_seed{args.seed}.audit.npz"))):
                tasks.append((ds, m, p, args.n_perm, args.epochs))
    print(f"[r1-hardened] {len(tasks)} fold-audits (n_perm={args.n_perm}, {args.workers} workers)", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_one_fold, t): t for t in tasks}
        for i, f in enumerate(as_completed(futs)):
            r = f.result(); results.append(r)
            print(f"  [{i+1}/{len(tasks)}] {r['dataset']} {r['method']} fold{r['fold']} "
                  f"gKL={r['graph_kl']:.3f}(p={r['graph_perm_p']:.4g}) nKL={r['node_kl']:.3f}(p={r['node_perm_p']:.4g})", flush=True)

    # BH-FDR across folds per (dataset, method, object)
    from cmi.eval.evidence_hardening import benjamini_hochberg
    agg = {}
    for ds in DATASETS:
        for m in args.methods:
            rows = [r for r in results if r["dataset"] == ds and r["method"] == m]
            if not rows:
                continue
            for obj, pk in (("graph", "graph_perm_p"), ("node", "node_perm_p")):
                ps = [r[pk] for r in rows]
                bh = benjamini_hochberg(ps, alpha=0.05)
                agg[f"{ds}|{m}|{obj}"] = dict(
                    n_folds=len(rows), kl_mean=float(np.mean([r[f"{obj}_kl"] for r in rows])),
                    perm_p=ps, n_significant=int(np.sum(bh["rejected"])), fdr_critical_p=float(bh["critical_p"]),
                    subject_bacc_labelcond=float(np.nanmean([r["subject_bacc_labelcond"] for r in rows
                                                            if r["subject_bacc_labelcond"] is not None])))
    outp = args.out or str(Path(args.gate_dir) / f"R1_hardened_nperm{args.n_perm}_seed{args.seed}.json")
    json.dump({"n_perm": args.n_perm, "seed": args.seed, "per_fold": results, "aggregate": agg}, open(outp, "w"), indent=2)
    print(f"\n[r1-hardened] wrote {outp}")
    for k, v in sorted(agg.items()):
        print(f"  {k:44s} kl={v['kl_mean']:.3f}  sig_folds={v['n_significant']}/{v['n_folds']}  subj_bacc={v['subject_bacc_labelcond']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
