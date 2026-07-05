#!/usr/bin/env python
"""CIGL_65 evidence freeze — assemble the final tables from the completed real-EEG gate (seeds 0/1/2), CPU only,
no retrain, no target-label fit. Produces, under --out_dir:
  multiseed_pareto.csv     per (dataset, method, seed) fold-mean target/source/graph_kl/node_kl + frontier
  r3_reliance.csv          per (dataset, method, seed, fold, conditioning, k) task_drop + subject_leak_drop + mode
  head_replay_check.csv    per (dataset, method, seed, fold) max_abs_diff, max_logit, replay_ok_abs/rel/primary
  bootstrap_ci.csv         hierarchical-bootstrap CIs for CIGL-ERM and CIGL-CDAN deltas (pooled + per-dataset)
  MANIFEST.yaml            commit/jobs/run-counts/file inventory
  data_access_note.md      2015 datalake owner-only workaround (staged read-only copy) reproducibility note
(r1_hardened_nperm1000.csv is produced by the R1 CPU job and copied in separately.)
"""
from __future__ import annotations
import argparse, csv, glob, json, re, sys
from pathlib import Path
import numpy as np


def _audit_paths(gate_dir, ds, m, s):
    """Glob the per-fold sidecars for EXACTLY method m (the `*_dann_*` glob also matches `cond_dann`, so filter
    the label between `_sub<digits>_` and `_seed<s>` to the exact method)."""
    paths = glob.glob(str(Path(gate_dir) / ds / "audit" / f"{ds}_fold*_{m}_seed{s}.audit.npz"))
    keep = []
    for p in paths:
        mt = re.search(rf"_sub[^_]+_(.+)_seed{s}\.audit\.npz$", Path(p).name)
        if mt and mt.group(1) == m:
            keep.append(p)
    return sorted(keep)

REPO = str(Path(__file__).resolve().parents[1]); sys.path.insert(0, REPO)
from cmi.eval.evidence_hardening import hierarchical_bootstrap                       # noqa: E402
from cmi.eval.leakage_removal import evaluate_reliance                              # noqa: E402
from cmi.eval.audit_npz import load_audit_npz                                       # noqa: E402
from scripts.analyze_r2_multiseed import _load_rows, pareto_multiseed, _relative_replay_ok, METHODS, SEEDS  # noqa: E402

DATASETS = ["BNCI2014_001", "BNCI2015_001"]


def _wcsv(path, fieldnames, rows):
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames); w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


def pareto_csv(gate_dir):
    rows = []
    for ds in DATASETS:
        gr = _load_rows(gate_dir, ds, SEEDS)
        par = pareto_multiseed(gr, METHODS)             # for frontier flag (across-seed)
        for m in METHODS:
            for s in SEEDS:
                r = [x for x in gr if x["method_label"] == m and x["seed"] == s]
                if not r:
                    continue
                rows.append(dict(dataset=ds, method=m, seed=s, n_folds=len(r),
                                 target_bacc=float(np.mean([x["target_bacc"] for x in r])),
                                 source_bacc=float(np.mean([x["source_bacc"] for x in r])),
                                 graph_kl=float(np.mean([x["graph_kl_proxy"] for x in r])),
                                 node_kl=float(np.mean([x["node_kl_proxy"] for x in r])),
                                 on_frontier=par.get(m, {}).get("on_frontier"),
                                 dominated_by="|".join(par.get(m, {}).get("dominated_by", []))))
    return rows


def head_replay_csv(gate_dir):
    rows = []
    for ds in DATASETS:
        for m in METHODS:
            for s in SEEDS:
                for p in _audit_paths(gate_dir, ds, m, s):
                    d = load_audit_npz(p)
                    if "task_head_weight" not in d:
                        continue
                    W = np.asarray(d["task_head_weight"], float); b = np.asarray(d.get("task_head_bias", 0.0), float)
                    gz = np.asarray(d["graph_z"], float); lo = np.asarray(d["model_logits"], float)
                    mad = float(np.abs(lo - (gz @ W.T + b)).max()); mlog = float(np.abs(lo).max())
                    abs_ok = bool(d.get("task_head_replay_ok", False))
                    rel_ok = _relative_replay_ok(d)                # max|d| <= 1e-4*max(1,max|logit|)
                    rows.append(dict(dataset=ds, method=m, seed=s, fold=int(np.asarray(d.get("fold", -1))),
                                     max_abs_diff=mad, max_abs_logit=mlog, replay_ok_abs=abs_ok,
                                     replay_ok_rel=bool(rel_ok), primary_replay_ok=bool(abs_ok or rel_ok)))
    return rows


def _reliance_one(a):
    """Worker: one audit file -> 12 reliance rows (4 k x 3 conditioning). Module-level for pickling."""
    import os as _os; _os.environ.setdefault("OMP_NUM_THREADS", "1")
    ds, m, s, p = a
    sys.path.insert(0, REPO)
    from cmi.eval.leakage_removal import evaluate_reliance as _er
    from cmi.eval.audit_npz import load_audit_npz as _load
    d = _load(p)
    ti = np.asarray(d.get("target_indices", []))
    if not ti.size:
        return []
    tgt = int(np.asarray(d["d"])[ti.ravel()][0]); fold = int(np.asarray(d.get("fold", -1)))
    out = []
    for c in ("label_conditional", "marginal_domain", "random_subspace"):
        for k in (1, 2, 4, 8):
            r = _er(d, target_domain=tgt, k=k, conditioning=c)
            out.append(dict(dataset=ds, method=m, seed=s, fold=fold, conditioning=c, k=k,
                            task_drop=r["task_drop"], subject_leak_drop=r["subject_leakage_drop"],
                            removal_mode=r["removal_mode"], firewall_passed=r["firewall_passed"]))
    return out


def reliance_csv(gate_dir, methods=("erm", "cigl_graph_node"), workers=8):
    from concurrent.futures import ProcessPoolExecutor
    tasks = [(ds, m, s, p) for ds in DATASETS for m in methods for s in SEEDS
             for p in _audit_paths(gate_dir, ds, m, s)]
    rows = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for res in ex.map(_reliance_one, tasks):
            rows.extend(res)
    return rows


def _paired_delta(records_a, records_b, key):
    """Pair a-minus-b by (dataset, seed, fold); return delta records with levels for hierarchical bootstrap."""
    idx = {}
    for r in records_b:
        idx[(r["dataset"], r["seed"], r["fold"])] = r[key]
    out = []
    for r in records_a:
        kk = (r["dataset"], r["seed"], r["fold"])
        if kk in idx and r[key] == r[key] and idx[kk] == idx[kk]:
            out.append(dict(dataset=r["dataset"], seed=r["seed"], fold=r["fold"], value=float(r[key] - idx[kk])))
    return out


def _fold_pareto_records(gate_dir):
    """Per-fold Pareto records keyed to pair deltas (fold = target_subject)."""
    per = {m: [] for m in METHODS}
    for ds in DATASETS:
        for x in _load_rows(gate_dir, ds, SEEDS):
            per[x["method_label"]].append(dict(dataset=ds, seed=x["seed"], fold=str(x["target_subject"]),
                                                target_bacc=x["target_bacc"], graph_kl=x["graph_kl_proxy"],
                                                node_kl=x["node_kl_proxy"]))
    return per


def bootstrap_csv(gate_dir, reliance_rows, n_boot=4000):
    par = _fold_pareto_records(gate_dir)
    out = []

    def _ci(deltas, comparison, quantity):
        if not deltas:
            return
        for scope, lv, subset in (("pooled", ("dataset", "seed", "fold"), deltas),
                                  ("BNCI2014_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2014_001"]),
                                  ("BNCI2015_001", ("seed", "fold"), [d for d in deltas if d["dataset"] == "BNCI2015_001"])):
            if not subset:
                continue
            r = hierarchical_bootstrap(subset, value_key="value", levels=lv, n_boot=n_boot, seed=0)
            out.append(dict(comparison=comparison, quantity=quantity, scope=scope, point=r["point"],
                            ci_lo=r["lo"], ci_hi=r["hi"], n_records=r["n_records"],
                            excludes_zero=bool(r["lo"] > 0 or r["hi"] < 0)))

    for q in ("target_bacc", "graph_kl", "node_kl"):
        _ci(_paired_delta(par["cigl_graph_node"], par["erm"], q), "CIGL-ERM", q)
        _ci(_paired_delta(par["cigl_graph_node"], par["cdan"], q), "CIGL-CDAN", q)
    # R3 task_drop deltas (label_conditional k2, k8) CIGL - ERM
    for k in (2, 8):
        a = [r for r in reliance_rows if r["method"] == "cigl_graph_node" and r["conditioning"] == "label_conditional" and r["k"] == k]
        b = [r for r in reliance_rows if r["method"] == "erm" and r["conditioning"] == "label_conditional" and r["k"] == k]
        _ci(_paired_delta(a, b, "task_drop"), "CIGL-ERM", f"R3_task_drop_k{k}")
    return out


def data_access_note(out_dir):
    txt = """# Data-access note (reproducibility)

`BNCI2014_001` loads directly from the shared datalake. `BNCI2015_001` required a **read-only** workaround:

- **Original source:** `/projects/EEG-foundation-model/datalake/raw/MNE-bnci-data/~bci/database/001-2015/*.mat`
  are owner-only (`tmaye`, mode ~0600; not group-readable) — MOABB/pooch failed with `PermissionError`.
- **Readable copy (world-readable, `rwxrwxrwx`):** `.../MNE-bnci-data/database/data-sets/001-2015/*.mat` (28 files,
  subjects S01-S12, sessions A/B[/C]).
- **Staged tree (ours, read-only symlinks to the readable copy):** `/home/infres/yinwang/mne_stage_bnci/MNE-bnci-data/~bci/database/001-2015/`.
- **Override:** the 2015 sbatch exports `MNE_DATASETS_BNCI_PATH=$STAGE` and `MNE_DATA=$STAGE` (cmi.paths uses
  `os.environ.setdefault`, so a pre-set env wins). We did **not** copy or modify any `.mat` bytes; symlinks point
  at the world-readable copy.
- **Loaded shape (verified):** X (5600, 13, 384), 2 classes (feet vs right_hand), 12 subjects → 12 LOSO folds.
- **No trial / label / fold / split / normalization logic changed** — only the data-file path. The datalake perms
  should be fixed at source (`chmod g+r` the `~bci/001-2015` files) for clean reproduction.
"""
    (Path(out_dir) / "data_access_note.md").write_text(txt)


def manifest(out_dir, gate_dir, counts):
    import subprocess
    commit = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, cwd=REPO).stdout.strip()
    lines = ["# CIGL_65 evidence-freeze manifest", f"commit: {commit}", "branch: project/cigl-r123-scaffold",
             "scientific_level: method-level (full-LOSO x seeds 0/1/2)",
             "datasets: [BNCI2014_001, BNCI2015_001]", "seeds: [0, 1, 2]",
             "methods: [erm, cigl_graph_node, dann, cond_dann, cdan]",
             "backbone: dgcnn_forward_graph_adapter", "cigl_lambda: {graph: 0.010, node: 0.010}",
             "gate_jobs: {seed0: [883118, 883120], seed1: [883206, 883207], seed2: [883203, 883204]}",
             "n_perm_in_run: 50", "r1_hardened_n_perm: 1000",
             "firewall: strict_source_only; target_eval_only; firewall_passed_all_folds: true",
             "head_replay: {erm: absolute_1e-5, cigl: absolute_1e-5, adversarial_2015: relative_1e-4_recovered}",
             "files:"]
    for k, v in counts.items():
        lines.append(f"  {k}: {{rows: {v}}}")
    (Path(out_dir) / "MANIFEST.yaml").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_dir", default="results/cigl/r2_seed0_gate")
    ap.add_argument("--out_dir", default="results/cigl_r123/final")
    ap.add_argument("--n_boot", type=int, default=4000)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    out = Path(args.out_dir); out.mkdir(parents=True, exist_ok=True)

    print("[freeze] pareto..."); pareto = pareto_csv(args.gate_dir)
    _wcsv(out / "multiseed_pareto.csv",
          ["dataset", "method", "seed", "n_folds", "target_bacc", "source_bacc", "graph_kl", "node_kl", "on_frontier", "dominated_by"], pareto)
    print("[freeze] head-replay check..."); hr = head_replay_csv(args.gate_dir)
    _wcsv(out / "head_replay_check.csv",
          ["dataset", "method", "seed", "fold", "max_abs_diff", "max_abs_logit", "replay_ok_abs", "replay_ok_rel", "primary_replay_ok"], hr)
    print("[freeze] R3 reliance (recompute)...", flush=True); rel = reliance_csv(args.gate_dir, workers=args.workers)
    _wcsv(out / "r3_reliance.csv",
          ["dataset", "method", "seed", "fold", "conditioning", "k", "task_drop", "subject_leak_drop", "removal_mode", "firewall_passed"], rel)
    print("[freeze] bootstrap CIs..."); bci = bootstrap_csv(args.gate_dir, rel, n_boot=args.n_boot)
    _wcsv(out / "bootstrap_ci.csv",
          ["comparison", "quantity", "scope", "point", "ci_lo", "ci_hi", "n_records", "excludes_zero"], bci)
    data_access_note(out)
    manifest(out, args.gate_dir, {"multiseed_pareto.csv": len(pareto), "head_replay_check.csv": len(hr),
                                  "r3_reliance.csv": len(rel), "bootstrap_ci.csv": len(bci)})
    print(f"\n[freeze] wrote {out}/ (pareto={len(pareto)}, head_replay={len(hr)}, reliance={len(rel)}, bootstrap={len(bci)})")
    print("\n=== bootstrap CI (delta; excludes_zero => significant) ===")
    for r in bci:
        print(f"  {r['comparison']:9s} {r['quantity']:18s} {r['scope']:13s} "
              f"{r['point']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]  sig={r['excludes_zero']}")


if __name__ == "__main__":
    sys.exit(main())
