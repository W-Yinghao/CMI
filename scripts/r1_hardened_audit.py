#!/usr/bin/env python
"""CIGL R1 hardened leakage audit (n_perm=1000) recomputed from the gate's SEED0 saved features -- NO retrain,
NO target-label use. Confirmatory tightening of the in-run n_perm=50 p-values. Per (dataset, method, fold):
select SOURCE rows, recompute the within-label-permutation graph/node KL-proxy audit (same probe/epochs/split as
the gate) at n_perm=1000, emit one row per representation (graph, node) with the fields the evidence-freeze needs.
BH-FDR is computed across folds per (dataset, method, representation). Parallel over folds; per-fold rows are
flushed to the CSV as they finish (crash-safe partial output). num_exceed is derived from the (+1)-smoothed
exact p: exact_p = (1+num_exceed)/(1+n_perm).

    python scripts/r1_hardened_audit.py --n_perm 1000 --workers 32 --methods erm cigl_graph_node cdan
"""
from __future__ import annotations
import argparse, csv, glob, json, os, sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
DATASETS = ["BNCI2014_001", "BNCI2015_001"]
METHODS = ["erm", "cigl_graph_node", "cdan"]
FIELDS = ["method", "dataset", "fold", "seed", "representation", "observed_kl", "perm_mean", "perm_std",
          "exact_p", "num_exceed", "n_perm", "bh_fdr_q"]


def _one_fold(a):
    dataset, method, path, seed, n_perm, epochs = a
    os.environ.setdefault("OMP_NUM_THREADS", "1"); os.environ.setdefault("MKL_NUM_THREADS", "1")
    import torch; torch.set_num_threads(1); sys.path.insert(0, REPO)
    from cmi.eval.audit_npz import load_audit_npz
    from cmi.eval.graph_leakage import audit_graph_node_objects
    from cmi.eval.probe_splits import stratified_trial_split_by_y_d
    from cmi.eval.leakage_removal import _subject_bacc
    d = load_audit_npz(path)
    si = np.asarray(d["source_indices"]).ravel()
    gz, nz = np.asarray(d["graph_z"])[si], np.asarray(d["node_z"])[si]
    y, dom = np.asarray(d["y"])[si], np.asarray(d["d"])[si]
    n_cls, n_dom = int(y.max() + 1), int(dom.max() + 1)
    tr, va, _ = stratified_trial_split_by_y_d(y, dom, train_frac=0.7, seed=0, min_per_cell=2)
    au = audit_graph_node_objects(gz, nz, y, dom, n_cls, n_dom, n_perm=n_perm, seed=0, device="cpu",
                                  epochs=epochs, train_idx=tr, val_idx=va)
    fold = int(np.asarray(d.get("fold", -1)))
    subj = _subject_bacc(gz, dom, y, seed=0)
    out = []
    for rep, obj in (("graph", "graph"), ("node", "node")):
        b = au[obj]
        p = float(b["permutation_p"])
        out.append(dict(method=method, dataset=dataset, fold=fold, seed=int(seed), representation=rep,
                        observed_kl=float(b["kl_mean"]), perm_mean=float(b["permutation_mean"]),
                        perm_std=float(b.get("permutation_std", float("nan"))), exact_p=p,
                        num_exceed=int(round(p * (1 + n_perm) - 1)), n_perm=int(n_perm), bh_fdr_q=None,
                        subject_bacc_labelcond=(float(subj) if subj == subj else None)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_perm", type=int, default=1000)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--methods", nargs="+", default=METHODS)
    ap.add_argument("--out_csv", default=None)
    ap.add_argument("--out_json", default=None)
    args = ap.parse_args()

    tasks = []
    for ds in DATASETS:
        for m in args.methods:
            for p in sorted(glob.glob(str(Path(args.gate_dir) / ds / "audit" / f"{ds}_fold*_{m}_seed{args.seed}.audit.npz"))):
                tasks.append((ds, m, p, args.seed, args.n_perm, args.epochs))
    print(f"[r1-hardened] {len(tasks)} fold-audits x2 reps (n_perm={args.n_perm}, {args.workers} workers)", flush=True)

    out_csv = args.out_csv or str(Path(args.gate_dir) / f"R1_hardened_nperm{args.n_perm}_seed{args.seed}.csv")
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    rows = []
    # stream partial rows to a .partial CSV as folds finish (crash-safe)
    with open(out_csv + ".partial", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["subject_bacc_labelcond"]); w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(_one_fold, t): t for t in tasks}
            for i, f in enumerate(as_completed(futs)):
                two = f.result(); rows.extend(two)
                for r in two:
                    w.writerow(r)
                fh.flush()
                g = [r for r in two if r["representation"] == "graph"][0]
                print(f"  [{i+1}/{len(tasks)}] {g['dataset']} {g['method']} fold{g['fold']} "
                      f"gKL={g['observed_kl']:.3f}(p={g['exact_p']:.4g},#>={g['num_exceed']})", flush=True)

    # BH-FDR across folds per (dataset, method, representation)
    from cmi.eval.evidence_hardening import benjamini_hochberg
    for ds in DATASETS:
        for m in args.methods:
            for rep in ("graph", "node"):
                grp = [r for r in rows if r["dataset"] == ds and r["method"] == m and r["representation"] == rep]
                if not grp:
                    continue
                bh = benjamini_hochberg([r["exact_p"] for r in grp], alpha=0.05)
                for r, q in zip(grp, bh["adjusted_p"]):
                    r["bh_fdr_q"] = float(q)
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS + ["subject_bacc_labelcond"]); w.writeheader()
        for r in rows:
            w.writerow(r)
    out_json = args.out_json or out_csv.replace(".csv", ".json")
    json.dump({"n_perm": args.n_perm, "seed": args.seed, "rows": rows}, open(out_json, "w"), indent=2)
    print(f"\n[r1-hardened] wrote {out_csv} ({len(rows)} rows)")
    # summary
    for ds in DATASETS:
        for m in args.methods:
            for rep in ("graph", "node"):
                grp = [r for r in rows if r["dataset"] == ds and r["method"] == m and r["representation"] == rep]
                if grp:
                    nsig = sum(1 for r in grp if r["bh_fdr_q"] is not None and r["bh_fdr_q"] <= 0.05)
                    print(f"  {ds:14s} {m:16s} {rep:5s}: kl={np.mean([r['observed_kl'] for r in grp]):.3f} "
                          f"FDR-sig {nsig}/{len(grp)}  min_q={min(r['bh_fdr_q'] for r in grp):.4g}")


if __name__ == "__main__":
    sys.exit(main())
